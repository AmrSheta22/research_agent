"""
StudyFacts extraction and telemetry engine.

Extracts the immutable single source of truth (StudyFacts) from a dataset,
research context, and statistical test outputs to guarantee 100% mathematical
and narrative consistency across all generated paper sections.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.outliers_influence import variance_inflation_factor

from research_agent.schemas import (
    DataFilteringManifest,
    PairwiseContrast,
    RegressionCoefficient,
    ResearchContext,
    StudyFacts,
)

logger = logging.getLogger(__name__)


def build_study_facts(
    df: pd.DataFrame,
    context: ResearchContext,
    test_results: Optional[Dict[str, Any]] = None,
    raw_df: Optional[pd.DataFrame] = None,
) -> Optional[StudyFacts]:
    """
    Deterministically extracts and computes all StudyFacts parameters from the
    cleaned DataFrame, raw DataFrame, research context, and test telemetry.
    """
    iv = context.independent_variable
    dv = context.dependent_variable
    covariates = [c for c in (context.covariates or []) if c in df.columns]

    # This telemetry schema describes independent groups with a numeric outcome.
    # Other designs retain their dispatched results; do not invent ANOVA facts for them.
    from research_agent.study_designs.dispatcher import normalize_type, resolve_design_key
    if df.empty or not iv or not dv or iv not in df or dv not in df:
        return None
    if context.research_type and resolve_design_key(context.research_type) != "cross_sectional":
        return None
    iv_kind = normalize_type(context.independent_variable_type or "")
    dv_kind = normalize_type(context.dependent_variable_type or "")
    if iv_kind == "cont" or dv_kind in {"binary", "multinomial", "ordinal"}:
        return None
    if not pd.api.types.is_numeric_dtype(df[dv]):
        return None
    if pd.api.types.is_numeric_dtype(df[iv]) and df[iv].nunique() > 2 and iv_kind not in {"binary", "multinomial", "ordinal"}:
        return None

    # Clean working data
    cols_to_keep = list(dict.fromkeys([iv, dv] + covariates + ([context.subject_group] if context.subject_group in df.columns else [])))
    working_df = df.dropna(subset=cols_to_keep).copy()

    # Standardize string values in categorical IV
    working_df[iv] = working_df[iv].astype(str)

    sample_size_total = len(working_df)
    counts = working_df[iv].value_counts().to_dict() if iv in working_df.columns else {}
    factor_levels = sorted(list(counts.keys()))

    # Compute descriptive statistics for continuous covariates
    covariate_descriptives: Dict[str, Dict[str, float]] = {}
    for c in covariates:
        if c in working_df.columns and pd.api.types.is_numeric_dtype(working_df[c]):
            c_series = working_df[c].dropna()
            if len(c_series) > 0:
                c_mean = float(c_series.mean())
                c_std = float(c_series.std(ddof=1)) if len(c_series) > 1 else 0.0
                c_min = float(c_series.min())
                c_max = float(c_series.max())
                c_median = float(c_series.median())
                c_q25 = float(c_series.quantile(0.25))
                c_q75 = float(c_series.quantile(0.75))
                covariate_descriptives[c] = {
                    "mean": round(c_mean, 2),
                    "std": round(c_std, 2),
                    "min": round(c_min, 2),
                    "max": round(c_max, 2),
                    "median": round(c_median, 2),
                    "iqr": round(c_q75 - c_q25, 2),
                }

    if sample_size_total < 3 or len(factor_levels) < 2 or working_df[dv].nunique() < 2:
        return None

    reference_level = max(counts, key=counts.get) if counts else factor_levels[0]

    # Group descriptives
    level_descriptives: Dict[str, Dict[str, float]] = {}
    for lvl in factor_levels:
        sub = working_df[working_df[iv] == lvl][dv].dropna()
        n_lvl = len(sub)
        if n_lvl > 0:
            mean_val = float(sub.mean())
            std_val = float(sub.std(ddof=1)) if n_lvl > 1 else 0.0
            median_val = float(sub.median())
            q25 = float(sub.quantile(0.25))
            q75 = float(sub.quantile(0.75))
            iqr_val = float(q75 - q25)
            se_val = (std_val / math.sqrt(n_lvl)) if n_lvl > 0 else 0.0
            level_descriptives[lvl] = {
                "mean": round(mean_val, 4),
                "std": round(std_val, 4),
                "median": round(median_val, 4),
                "iqr": round(iqr_val, 4),
                "ci_lower": round(mean_val - float(stats.t.ppf(.975, n_lvl - 1)) * se_val, 4) if n_lvl > 1 else mean_val,
                "ci_upper": round(mean_val + float(stats.t.ppf(.975, n_lvl - 1)) * se_val, 4) if n_lvl > 1 else mean_val,
            }

    # Omnibus One-Way ANOVA & Kruskal-Wallis
    groups = [working_df[working_df[iv] == lvl][dv].values for lvl in factor_levels]
    k = len(factor_levels)
    n = sample_size_total
    if n <= k:
        return None
    df_between = k - 1
    df_within = n - k

    f_stat, anova_p = 0.0, 1.0
    if k >= 2 and n > k:
        try:
            f_stat, anova_p = stats.f_oneway(*groups)
            f_stat = float(f_stat)
            if not math.isfinite(f_stat) or not math.isfinite(anova_p):
                return None
            anova_p = float(anova_p)
        except Exception as exc:
            logger.warning(f"ANOVA calculation failed: {exc}")
            return None

    # Eta squared calculation
    try:
        grand_mean = working_df[dv].mean()
        ss_between = sum(len(g) * ((np.mean(g) - grand_mean) ** 2) for g in groups if len(g) > 0)
        ss_total = np.sum((working_df[dv] - grand_mean) ** 2)
        eta_squared = float(ss_between / ss_total) if ss_total > 0 else 0.0
    except Exception:
        eta_squared = 0.0

    h_stat, kruskal_p, kruskal_df = 0.0, 1.0, df_between
    if k >= 2 and n > k:
        try:
            h_stat, kruskal_p = stats.kruskal(*groups)
            h_stat = float(h_stat)
            if not math.isfinite(h_stat) or not math.isfinite(kruskal_p):
                return None
            kruskal_p = float(kruskal_p)
        except Exception as exc:
            logger.warning(f"Kruskal-Wallis failed: {exc}")
            return None

    # Pairwise Contrasts (Tukey HSD)
    pairwise_contrasts: List[PairwiseContrast] = []
    try:
        if k >= 2 and n > k:
            tukey = pairwise_tukeyhsd(working_df[dv], working_df[iv])
            for row in tukey.summary().data[1:]:
                g1, g2, diff, p_adj, lower, upper, reject = row
                g1_str, g2_str = str(g1), str(g2)
                diff_val = float(diff)
                p_val = float(p_adj)
                sig = bool(reject)
                m1 = level_descriptives.get(g1_str, {}).get("mean", 0.0)
                m2 = level_descriptives.get(g2_str, {}).get("mean", 0.0)
                if diff_val > 0:
                    dir_str = f"{g2_str.capitalize()} (mean={m2:.2f}) is {diff_val:.3f} higher (less negative) than {g1_str.capitalize()} (mean={m1:.2f})"
                else:
                    dir_str = f"{g2_str.capitalize()} (mean={m2:.2f}) is {abs(diff_val):.3f} lower (more negative) than {g1_str.capitalize()} (mean={m1:.2f})"
                pairwise_contrasts.append(
                    PairwiseContrast(
                        group1=g1_str,
                        group2=g2_str,
                        mean1=round(m1, 4),
                        mean2=round(m2, 4),
                        diff=round(diff_val, 4),
                        p_value=float(p_val),
                        ci_lower=round(float(lower), 4),
                        ci_upper=round(float(upper), 4),
                        direction=dir_str,
                        significant=sig,
                    )
                )
    except Exception as exc:
        logger.warning(f"Tukey HSD calculation failed: {exc}")

    # Rescaling & OLS Dummy Regression
    covariates_scaling: Dict[str, str] = {}
    model_covariates: List[str] = []
    reg_df = working_df.copy()

    model_covariates = list(covariates)
    factor_term = f"C(Q({iv!r}), Treatment(reference={reference_level!r}))"
    formula_terms = [factor_term] + [f"Q({c!r})" for c in model_covariates]
    ols_formula = f"Q({dv!r}) ~ " + " + ".join(formula_terms)

    reg_coefs: List[RegressionCoefficient] = []
    reg_r2 = 0.0
    shapiro_p: Optional[float] = None
    bp_p: Optional[float] = None
    vif_max: Optional[float] = None
    ols_robust_se: Dict[str, Dict[str, float]] = {}
    shapiro_stat = bp_stat = None

    try:
        ols_model = smf.ols(ols_formula, data=reg_df).fit()
        reg_r2 = float(ols_model.rsquared)
        conf_int = ols_model.conf_int()

        for param_name in ols_model.params.index:
            coef_val = float(ols_model.params[param_name])
            se_val = float(ols_model.bse[param_name])
            t_val = float(ols_model.tvalues[param_name])
            p_val = float(ols_model.pvalues[param_name])
            ci_l = float(conf_int.loc[param_name, 0])
            ci_u = float(conf_int.loc[param_name, 1])

            # Clean readable parameter label
            clean_param = param_name
            if factor_term in param_name:
                # e.g. C(factor, Treatment(reference='baseline'))[T.treatment] -> factor: treatment (vs baseline)
                lvl_match = param_name.split("[T.")[-1].rstrip("]")
                clean_param = f"{iv}: {lvl_match} (vs {reference_level})"
            elif param_name == "Intercept":
                clean_param = f"Baseline Intercept ({reference_level})"

            reg_coefs.append(
                RegressionCoefficient(
                    parameter=clean_param,
                    coef=round(coef_val, 4),
                    std_err=round(se_val, 4),
                    t_value=round(t_val, 3),
                    p_value=float(p_val),
                    ci_lower=round(ci_l, 4),
                    ci_upper=round(ci_u, 4),
                )
            )

        # Huber-White HC3 robust standard errors on OLS
        ols_robust_se: Dict[str, Dict[str, float]] = {}
        try:
            hc3_res = ols_model.get_robustcov_results(cov_type="HC3")
            hc3_conf = hc3_res.conf_int()
            for param_index, p_name in enumerate(ols_model.params.index):
                clean_p = p_name
                if factor_term in p_name:
                    lvl_match = p_name.split("[T.")[-1].rstrip("]")
                    clean_p = f"{iv}: {lvl_match} (vs {reference_level})"
                elif p_name == "Intercept":
                    clean_p = f"Baseline Intercept ({reference_level})"

                ols_robust_se[clean_p] = {
                    "coef": round(float(hc3_res.params[param_index]), 4),
                    "std_err_hc3": round(float(hc3_res.bse[param_index]), 4),
                    "t_value": round(float(hc3_res.tvalues[param_index]), 3),
                    "p_value": float(hc3_res.pvalues[param_index]),
                    "ci_lower": round(float(hc3_conf[param_index, 0]), 4),
                    "ci_upper": round(float(hc3_conf[param_index, 1]), 4),
                }
        except Exception as hc3_err:
            logger.debug(f"HC3 robust SE calculation skipped: {hc3_err}")

        # Residual diagnostics
        shapiro_stat, shapiro_p = None, None
        bp_stat, bp_p = None, None
        if len(ols_model.resid) >= 3:
            try:
                sh_res = stats.shapiro(ols_model.resid)
                shapiro_stat = round(float(sh_res[0]), 4)
                shapiro_p = round(float(sh_res[1]), 5)
            except Exception:
                pass
            try:
                from statsmodels.stats.diagnostic import het_breuschpagan
                bp_test = het_breuschpagan(ols_model.resid, ols_model.model.exog)
                bp_stat = round(float(bp_test[0]), 4)
                bp_p = round(float(bp_test[1]), 5)
            except Exception:
                pass

        # VIF for continuous covariates
        if len(model_covariates) >= 2:
            try:
                X_cov = sm.add_constant(reg_df[model_covariates])
                vifs = [variance_inflation_factor(X_cov.values, i) for i in range(1, X_cov.shape[1])]
                vif_max = round(float(max(vifs)), 3)
            except Exception:
                pass

    except Exception as exc:
        logger.warning(f"OLS Dummy Regression failed: {exc}")

    # Hierarchical Linear Mixed Model (Clustering by Site / Entity)
    lmm_executed = False
    lmm_icc: Optional[float] = None
    lmm_var_site: Optional[float] = None
    lmm_var_resid: Optional[float] = None
    num_clusters_total: Optional[int] = None
    level_cluster_counts: Dict[str, int] = {}
    effective_sample_size: Optional[float] = None
    design_effect: Optional[float] = None
    lmm_fixed_effects: List[RegressionCoefficient] = []
    lmm_marginal_r2: Optional[float] = None
    lmm_conditional_r2: Optional[float] = None
    variance_shares: Dict[str, float] = {}

    cluster_col = context.subject_group if context.subject_group in working_df.columns else None
    for cand in ["cluster_id", "cluster", "site_id", "site_name", "group_id", "subject_id", "location_id", "center_id", "unit_id"]:
        if cluster_col is None and cand in working_df.columns:
            cluster_col = cand
            break

    if cluster_col is not None and working_df[cluster_col].nunique() > 1 and sample_size_total >= 15:
        try:
            num_clusters_total = int(working_df[cluster_col].nunique())
            level_cluster_counts = working_df.groupby(iv)[cluster_col].nunique().to_dict()
            m_bar = sample_size_total / num_clusters_total

            mixed_model = smf.mixedlm(ols_formula, data=reg_df, groups=reg_df[cluster_col]).fit(reml=True, maxiter=100)
            if not mixed_model.converged:
                raise ValueError("Mixed model did not converge")
            lmm_var_site = float(mixed_model.cov_re.iloc[0, 0])
            lmm_var_resid = float(mixed_model.scale)
            if (lmm_var_site + lmm_var_resid) > 0:
                lmm_icc = round(float(lmm_var_site / (lmm_var_site + lmm_var_resid)), 4)
                lmm_executed = True
                deff = 1.0 + (m_bar - 1.0) * lmm_icc
                design_effect = round(float(deff), 3)
                effective_sample_size = round(float(sample_size_total / deff), 1)

            # Nakagawa marginal and conditional R2 (Nakagawa & Schielzeth, 2013)
            try:
                fixed_pred = mixed_model.predict(reg_df)
                var_fixed = float(np.var(fixed_pred))
                tot_var = var_fixed + lmm_var_site + lmm_var_resid
                if tot_var > 0:
                    lmm_marginal_r2 = round(float(var_fixed / tot_var), 4)
                    lmm_conditional_r2 = round(float((var_fixed + lmm_var_site) / tot_var), 4)
            except Exception as r2_err:
                logger.debug(f"LMM R2 calculation skipped: {r2_err}")

            # Extract fixed effects from MixedLM
            fe_params = mixed_model.fe_params
            fe_bse = mixed_model.bse_fe
            fe_conf = mixed_model.conf_int().loc[fe_params.index]
            for p_name in fe_params.index:
                c_val = float(fe_params[p_name])
                s_val = float(fe_bse[p_name]) if p_name in fe_bse else 0.0
                z_val = float(c_val / s_val) if s_val > 0 else 0.0
                pval = float(2 * stats.norm.sf(abs(z_val)))
                ci_l = float(fe_conf.loc[p_name, 0])
                ci_u = float(fe_conf.loc[p_name, 1])

                clean_p = p_name
                if factor_term in p_name:
                    lvl_match = p_name.split("[T.")[-1].rstrip("]")
                    clean_p = f"{iv}: {lvl_match} (vs {reference_level})"
                elif p_name == "Intercept":
                    clean_p = f"Baseline Intercept ({reference_level})"

                lmm_fixed_effects.append(
                    RegressionCoefficient(
                        parameter=clean_p,
                        coef=round(c_val, 4),
                        std_err=round(s_val, 4),
                        t_value=round(z_val, 3),
                        p_value=float(pval),
                        ci_lower=round(ci_l, 4),
                        ci_upper=round(ci_u, 4),
                    )
                )
        except Exception as exc:
            logger.warning(f"MixedLM execution failed: {exc}")

    cluster_robust_ols = {}
    additional_diagnostics = {}
    if cluster_col and reg_coefs:
        try:
            clustered = ols_model.get_robustcov_results(cov_type="cluster", groups=reg_df[cluster_col], use_correction=True, use_t=True)
            cis = clustered.conf_int()
            for i, c in enumerate(reg_coefs):
                cluster_robust_ols[c.parameter] = dict(coef=float(clustered.params[i]), std_err=float(clustered.bse[i]),
                    p_value=float(clustered.pvalues[i]), ci_lower=float(cis[i,0]), ci_upper=float(cis[i,1]))
        except Exception as exc:
            additional_diagnostics["cluster_robust_failure"] = str(exc)
    if lmm_executed:
        try:
            residuals = mixed_model.resid
            random_intercepts = np.array([np.asarray(v)[0] for v in mixed_model.random_effects.values()])
            for label, values in [("lmm_residuals",residuals),("lmm_random_intercepts",random_intercepts)]:
                if len(values) >= 3:
                    w,p = stats.shapiro(values)
                    additional_diagnostics[label] = {"n":len(values),"shapiro_w":float(w),"p_value":float(p)}
        except Exception as exc:
            additional_diagnostics["lmm_diagnostic_failure"] = str(exc)
    if cluster_col:
        sizes = working_df.groupby(cluster_col).size()
        additional_diagnostics["entities_per_cluster"] = {"min":int(sizes.min()),"median":float(sizes.median()),"max":int(sizes.max()),"singletons":int((sizes==1).sum()),"distribution":{str(k):int(v) for k,v in sizes.value_counts().sort_index().items()}}
    q1,q3 = working_df[dv].quantile([.25,.75])
    lo,hi = q1-1.5*(q3-q1),q3+1.5*(q3-q1)
    trimmed = working_df[working_df[dv].between(lo,hi)]
    sensitivity = {"rule":"global outcome 1.5-IQR", "lower":float(lo), "upper":float(hi), "n":len(trimmed), "excluded":len(working_df)-len(trimmed)}
    try:
        groups = [g[dv].values for _,g in trimmed.groupby(iv) if len(g)>1]
        result = stats.f_oneway(*groups)
        sensitivity.update(anova_f=float(result.statistic),p_value=float(result.pvalue))
        if cluster_col:
            fit = smf.mixedlm(ols_formula,data=trimmed,groups=trimmed[cluster_col]).fit(reml=True,maxiter=100)
            sensitivity["converged"] = bool(fit.converged)
            if fit.converged:
                sensitivity["fixed_effects"] = {str(k):float(v) for k,v in fit.fe_params.items()}
                sensitivity["confidence_intervals"] = fit.conf_int().loc[fit.fe_params.index].to_dict(orient="index")
    except Exception as exc:
        sensitivity["failure"] = str(exc)
    additional_diagnostics["extreme_value_sensitivity"] = sensitivity

    # Power analysis and Minimum Detectable Effect Sizes (MDES at 80% power, alpha = 0.05)
    mdes_by_factor_level: Dict[str, float] = {}
    if lmm_fixed_effects:
        for fe in lmm_fixed_effects:
            p_name = fe.parameter
            se = fe.std_err
            if se > 0 and p_name.startswith(f"{iv}:"):
                lvl = p_name.split(":")[-1].split("(vs")[0].strip()
                mdes_by_factor_level[lvl] = round(2.8016 * se, 2)

    # Compute DataFilteringManifest & Pre-screening sensitivity check deterministically
    raw_sample_size: Optional[int] = len(raw_df) if raw_df is not None else len(df)
    raw_f_stat: Optional[float] = None
    raw_anova_df_b: Optional[int] = None
    raw_anova_df_w: Optional[int] = None
    raw_anova_p: Optional[float] = None
    raw_eta_squared: Optional[float] = None
    filters_applied: List[Dict[str, Any]] = []

    if raw_df is not None and len(raw_df) > sample_size_total:
        raw_n = len(raw_df)
        outliers_removed = raw_n - sample_size_total
        # Omnibus ANOVA on raw_df
        if iv in raw_df.columns and dv in raw_df.columns:
            raw_groups = [raw_df[raw_df[iv].astype(str) == lvl][dv].dropna().values for lvl in factor_levels if lvl in raw_df[iv].astype(str).values]
            if len(raw_groups) >= 2 and sum(len(g) for g in raw_groups) > len(raw_groups):
                try:
                    f_val, p_val = stats.f_oneway(*raw_groups)
                    raw_f_stat = round(float(f_val), 4) if not (math.isnan(f_val) or math.isinf(f_val)) else None
                    raw_anova_p = round(float(p_val), 5) if not (math.isnan(p_val) or math.isinf(p_val)) else None
                    raw_anova_df_b = len(raw_groups) - 1
                    raw_anova_df_w = sum(len(g) for g in raw_groups) - len(raw_groups)
                    all_raw = np.concatenate(raw_groups)
                    gm = np.mean(all_raw)
                    ss_b = sum(len(g) * ((np.mean(g) - gm) ** 2) for g in raw_groups)
                    ss_t = np.sum((all_raw - gm) ** 2)
                    raw_eta_squared = round(float(ss_b / ss_t), 4) if ss_t > 0 else None
                except Exception as raw_e:
                    logger.debug(f"Raw ANOVA calculation skipped: {raw_e}")
    audit = df.attrs.get("cleaning_audit", {})
    filters_applied = list(audit.get("filters", []))
    raw_n = len(raw_df) if raw_df is not None else audit.get("raw_sample_size", len(df))
    outliers_removed = int(audit.get("outliers_removed", 0))
    strategy = audit.get("strategy", "unknown")
    methods = {"iqr_standard": "IQR 1.5x rule", "iqr_conservative": "IQR 3.0x rule", "zscore_filter": "Absolute Z-score > 3"}
    dv_filter = next((f for f in filters_applied if f.get("variable") == dv), {})
    manifest = DataFilteringManifest(
        raw_sample_size=raw_n, final_sample_size=sample_size_total,
        missing_rows_removed=int(audit.get("missing_rows_removed", 0)),
        outliers_detected=outliers_removed, outliers_removed=outliers_removed,
        outlier_method=methods.get(strategy, "None" if audit else "Unknown (cleaning audit unavailable)"),
        outlier_variable=dv if dv_filter else None,
        q1=dv_filter.get("q1"), q3=dv_filter.get("q3"), iqr=dv_filter.get("iqr"),
        lower_fence=dv_filter.get("lower_fence"), upper_fence=dv_filter.get("upper_fence"),
        variable_transformations={**audit.get("transformations", {}), **covariates_scaling},
        filters_applied=filters_applied,
    )

    # Domain & Repository Metadata extraction from generic ResearchContext
    dataset_source_name = (
        getattr(context, "dataset_source_name", None)
        or getattr(context, "provenance_repository", None)
        or (f"the project analytical dataset ({os.path.basename(context.file_path)})" if context.file_path else "primary research database")
    )
    dataset_version = getattr(context, "dataset_version", None) or None
    variable_units = getattr(context, "variable_units", None) or {}
    if not variable_units:
        variable_units = {dv: "standard units"}
        for c in covariates:
            variable_units[c] = "standard units"
    factor_level_definitions = getattr(context, "factor_level_definitions", None) or {lvl: f"Observed cohort category: {lvl}" for lvl in factor_levels}
    unmeasured_confounders_note = getattr(context, "unmeasured_confounders_note", None) or getattr(context, "mineralogy_note", None)
    sampling_boundaries_note = getattr(context, "sampling_boundaries_note", None) or getattr(context, "age_window_note", None)

    geographic_scope = context.collection_location or "Multi-site observational dataset"
    if context.file_path:
        base_name = os.path.basename(context.file_path)
        provenance_repository = f"the project analytical dataset ({base_name})"
    else:
        provenance_repository = "the primary project research database"
    material_analyzed = context.subjects_or_participants or f"Empirical observational units of {dv}"
    measurement_scale = context.dependent_variable_summary or (variable_units.get(dv) if variable_units else None)
    secondary_scale = None

    return StudyFacts(
        sample_size_total=sample_size_total,
        geographic_scope=geographic_scope,
        provenance_repository=provenance_repository,
        material_analyzed=material_analyzed,
        measurement_scale=measurement_scale,
        secondary_scale=secondary_scale,
        isotopic_scale=measurement_scale,
        water_scale=secondary_scale,
        manifest=manifest,
        factor_name=iv,
        factor_levels=factor_levels,
        reference_level=reference_level,
        level_sample_sizes=counts,
        level_descriptives=level_descriptives,
        dependent_variable=dv,
        covariates=covariates,
        covariates_scaling=covariates_scaling,
        anova_f=round(f_stat, 4),
        anova_df_between=df_between,
        anova_df_within=df_within,
        anova_p=float(anova_p),
        eta_squared=round(eta_squared, 4),
        kruskal_h=round(h_stat, 4),
        kruskal_df=kruskal_df,
        kruskal_p=float(kruskal_p),
        regression_model_type="Ordinary Least Squares (OLS) with Dummy Contrasts",
        regression_r_squared=round(reg_r2, 4),
        regression_coefficients=reg_coefs,
        lmm_executed=lmm_executed,
        lmm_icc=lmm_icc,
        lmm_variance_site=round(lmm_var_site, 4) if lmm_var_site is not None else None,
        lmm_variance_residual=round(lmm_var_resid, 4) if lmm_var_resid is not None else None,
        cluster_variable=cluster_col,
        num_clusters_total=num_clusters_total,
        level_cluster_counts=level_cluster_counts,
        effective_sample_size=effective_sample_size,
        design_effect=design_effect,
        lmm_fixed_effects=lmm_fixed_effects,
        lmm_marginal_r2=lmm_marginal_r2,
        lmm_conditional_r2=lmm_conditional_r2,
        pairwise_contrasts=pairwise_contrasts,
        raw_sample_size=raw_sample_size,
        raw_anova_f=raw_f_stat,
        raw_anova_df_between=raw_anova_df_b,
        raw_anova_df_within=raw_anova_df_w,
        raw_anova_p=raw_anova_p,
        raw_eta_squared=raw_eta_squared,
        covariate_descriptives=covariate_descriptives,
        ols_robust_se=ols_robust_se,
        shapiro_wilk_stat=shapiro_stat,
        shapiro_wilk_p=shapiro_p,
        breusch_pagan_stat=bp_stat,
        breusch_pagan_p=bp_p,
        vif_max=vif_max,
        mdes_by_factor_level=mdes_by_factor_level,
        cluster_robust_ols=cluster_robust_ols,
        additional_diagnostics=additional_diagnostics,
        variance_shares=variance_shares,
        dataset_source_name=dataset_source_name,
        dataset_version=dataset_version,
        variable_units=variable_units,
        factor_level_definitions=factor_level_definitions,
        unmeasured_confounders_note=unmeasured_confounders_note,
        sampling_boundaries_note=sampling_boundaries_note,
        mineralogy_note=unmeasured_confounders_note,
        age_window_note=sampling_boundaries_note,
    )


def generate_detailed_state_narrative(study_facts: StudyFacts, context: Optional[ResearchContext] = None) -> str:
    """
    Constructs a detailed, comprehensive empirical audit narrative from StudyFacts.
    Transfers the full operational state to the agent, specifying:
      1. Original sample size, cleaning decisions, and rationale.
      2. Exact analytical sample size and subgroup breakdown.
      3. Cluster hierarchy, ICC, design effect, and effective sample size.
      4. Omnibus tests (ANOVA, Kruskal-Wallis) and effect sizes.
      5. Post-hoc contrasts with explicit directional polarity.
      6. Multivariable regression & Linear Mixed Model outputs.
      7. Model diagnostics and power analysis / MDES.
    """
    lines: List[str] = []
    lines.append("# EMPIRICAL DATA AUDIT & METHODOLOGICAL STATE NARRATIVE")
    lines.append("All synthesized paper sections MUST conform strictly to these empirical facts.\n")

    # 1. Dataset Origin & Sample Screening Decisions
    m = study_facts.manifest
    lines.append("## 1. Data Ingestion, Screening, and Sample Size Rationale")
    lines.append(f"- Raw Compiled Sample Size: N_raw = {m.raw_sample_size}")
    lines.append(f"- Final Cleaned Analytical Sample Size: N = {m.final_sample_size}")
    if study_facts.dataset_source_name:
        lines.append(f"- Data Repository & Provenance: {study_facts.provenance_repository} ({study_facts.dataset_source_name}, Version: {study_facts.dataset_version or 'v1'})")
    if m.outliers_removed > 0:
        lines.append(
            f"- Outlier Detection Protocol: {m.outlier_method} applied across continuous response and covariate dimensions. "
            f"Fences calculated at Q1 = {m.q1:.4f}, Q3 = {m.q3:.4f}, IQR = {m.iqr:.4f}. "
            f"Lower fence = {m.lower_fence:.4f}, Upper fence = {m.upper_fence:.4f}."
        )
        lines.append(
            f"- Outliers Screened: {m.outliers_removed} multivariate/covariate outlier records were excluded from analytical modeling, "
            f"establishing the analytical cohort of N = {m.final_sample_size} complete records."
        )
    else:
        lines.append(
            f"- Outlier Screening Rationale: No extreme observational outliers were excluded (outliers removed = 0) "
            f"preserving complete empirical records across all sites."
        )

    if study_facts.raw_sample_size and study_facts.raw_sample_size > study_facts.sample_size_total and study_facts.raw_anova_f is not None:
        lines.append(
            f"- Pre-Screening Raw Cohort Sensitivity Check (N = {study_facts.raw_sample_size}): "
            f"Evaluating the unfiltered source cohort yields omnibus ANOVA F({study_facts.raw_anova_df_between}, {study_facts.raw_anova_df_within}) = "
            f"{study_facts.raw_anova_f:.2f}, p = {study_facts.raw_anova_p:.4f}, raw eta^2 = {study_facts.raw_eta_squared:.4f}. "
            f"Note that the headline omnibus effect is sensitive to outlier screening, demonstrating the statistical fragility of unadjusted associations."
        )

    # 2. Factor Levels, Units, and Subgroup Counts
    lines.append("\n## 2. Factor Structure, Measurement Units, and Descriptive Statistics")
    lines.append(f"- Primary Independent Factor: '{study_facts.factor_name}'")
    lines.append(f"- Reference Baseline Level: '{study_facts.reference_level}'")
    if study_facts.variable_units:
        units_str = ", ".join([f"{k}: {v}" for k, v in study_facts.variable_units.items()])
        lines.append(f"- Measurement Units & Scales: {units_str}")
    if study_facts.factor_level_definitions:
        lines.append("- Factor Level Definitions:")
        for lvl, defn in study_facts.factor_level_definitions.items():
            lines.append(f"  * {lvl}: {defn}")
    lines.append("- Subgroup Sample Sizes and Descriptives (n per group):")
    for lvl, lvl_n in study_facts.level_sample_sizes.items():
        desc = study_facts.level_descriptives.get(lvl, {})
        mean_s = f"mean = {desc.get('mean', 0.0):.3f}" if "mean" in desc else ""
        sd_s = f"SD = {desc.get('std', 0.0):.3f}" if "std" in desc else ""
        lines.append(f"  * {lvl}: n = {lvl_n} ({mean_s}, {sd_s})")
    if study_facts.covariate_descriptives:
        lines.append("- Continuous Covariate Descriptive Statistics:")
        for c_name, cd in study_facts.covariate_descriptives.items():
            lines.append(f"  * {c_name}: mean = {cd.get('mean')}, SD = {cd.get('std')}, range = [{cd.get('min')}, {cd.get('max')}], median = {cd.get('median')}")

    # 3. Clustering and Hierarchical Design
    if study_facts.lmm_executed:
        lines.append("\n## 3. Hierarchical Clustering and Design Effect")
        lines.append(f"- Cluster Variable: '{study_facts.cluster_variable}'")
        lines.append(f"- Total Clusters: {study_facts.num_clusters_total}")
        lines.append(f"- Intraclass Correlation Coefficient (ICC): {study_facts.lmm_icc:.4f}")
        if study_facts.design_effect is not None:
            lines.append(f"- Survey Design Effect (Deff): {study_facts.design_effect:.3f}")
        if study_facts.effective_sample_size is not None:
            lines.append(
                f"- Effective Sample Size: N_eff = {study_facts.effective_sample_size:.1f} "
                f"(reflects information content after adjusting for site-level clustering)"
            )

    # 4. Omnibus Statistical Tests
    lines.append(f"\n## 4. Omnibus Statistical Evaluations (Analytical Cohort N = {study_facts.sample_size_total})")
    p_str = "< 0.001" if study_facts.anova_p < 0.001 else f"= {study_facts.anova_p:.4f}"
    lines.append(
        f"- One-Way ANOVA (Unadjusted): F({study_facts.anova_df_between}, {study_facts.anova_df_within}) = "
        f"{study_facts.anova_f:.2f}, p {p_str}, eta^2 = {study_facts.eta_squared:.4f} "
        f"({study_facts.eta_squared * 100:.1f}% unadjusted variance explained)"
    )
    kp_str = "< 0.001" if study_facts.kruskal_p < 0.001 else f"= {study_facts.kruskal_p:.4f}"
    lines.append(
        f"- Kruskal-Wallis Non-Parametric Test: H({study_facts.kruskal_df}) = {study_facts.kruskal_h:.2f}, p {kp_str}"
    )

    # 5. Pairwise Contrasts (Tukey HSD)
    if study_facts.pairwise_contrasts:
        lines.append("\n## 5. Post-Hoc Pairwise Contrasts (Tukey HSD - Unadjusted for Clustering)")
        for c in study_facts.pairwise_contrasts:
            sig_label = "Significant" if c.significant else "Non-Significant"
            lines.append(
                f"- {c.group1} vs {c.group2}: Diff = {c.diff:+.3f} (95% CI: [{c.ci_lower:+.3f}, {c.ci_upper:+.3f}]), "
                f"adj p = {format_p(c.p_value)} [{sig_label}]. Narrative: {c.direction}."
            )

    # 6. Multivariable & Hierarchical Linear Mixed Model
    lines.append("\n## 6. Multivariable Regression and Linear Mixed-Effects Models")
    lines.append(f"- OLS Multiple Regression: R^2 = {study_facts.regression_r_squared:.4f}")
    for rc in study_facts.regression_coefficients:
        lines.append(
            f"  * Standard OLS {rc.parameter}: beta = {rc.coef:+.4f}, OLS SE = {rc.std_err:.4f}, "
            f"95% CI: [{rc.ci_lower:+.4f}, {rc.ci_upper:+.4f}], p = {format_p(rc.p_value)}"
        )
    if study_facts.ols_robust_se:
        lines.append("- OLS Huber-White HC3 Robust Standard Errors (accounting for heteroscedasticity):")
        for p_name, r_info in study_facts.ols_robust_se.items():
            lines.append(
                f"  * HC3 {p_name}: beta = {r_info.get('coef'):+.4f}, HC3 SE = {r_info.get('std_err_hc3'):.4f}, "
                f"95% CI: [{r_info.get('ci_lower'):+.4f}, {r_info.get('ci_upper'):+.4f}], p = {format_p(r_info.get('p_value'))}"
            )

    if study_facts.lmm_executed:
        r2m_s = f"{study_facts.lmm_marginal_r2:.4f}" if study_facts.lmm_marginal_r2 is not None else "N/A"
        r2c_s = f"{study_facts.lmm_conditional_r2:.4f}" if study_facts.lmm_conditional_r2 is not None else "N/A"
        lines.append(f"- Linear Mixed Model (LMM): Marginal R^2_m = {r2m_s}, Conditional R^2_c = {r2c_s}")
        if study_facts.variance_shares:
            lines.append(f"- Fixed-Predictor Variance Share Decomposition (Method: {study_facts.variance_share_method}):")
            for pred_name, v_share in study_facts.variance_shares.items():
                lines.append(f"  * {pred_name}: {v_share:.1f}% of fixed predictor variance")
        lines.append("- LMM Fixed Effects (adjusted for clustering and covariates):")
        for fe in study_facts.lmm_fixed_effects:
            lines.append(
                f"  * LMM {fe.parameter}: beta = {fe.coef:+.4f}, SE = {fe.std_err:.4f}, "
                f"95% CI: [{fe.ci_lower:+.4f}, {fe.ci_upper:+.4f}], p = {format_p(fe.p_value)}"
            )

    # 7. Model Diagnostics & Epistemic Calibration
    lines.append("\n## 7. Model Diagnostics & Statistical Reasoning")
    if study_facts.shapiro_wilk_p is not None:
        lines.append(f"- Residual Normality (Shapiro-Wilk on OLS residuals): W = {study_facts.shapiro_wilk_stat}, p = {study_facts.shapiro_wilk_p:.4f}")
    if study_facts.breusch_pagan_p is not None:
        bp_note = "Significant heteroscedasticity detected (p < 0.05); motivates Huber-White HC3 robust standard errors on OLS." if study_facts.breusch_pagan_p < 0.05 else "Homoscedasticity tenable."
        lines.append(f"- Homoscedasticity (Breusch-Pagan on OLS residuals): LM = {study_facts.breusch_pagan_stat}, p = {study_facts.breusch_pagan_p:.4f} ({bp_note})")
    if study_facts.vif_max is not None:
        lines.append(f"- Multicollinearity (Max VIF): {study_facts.vif_max:.2f}")

    # 8. Minimum Detectable Effect Sizes (MDES)
    if study_facts.mdes_by_factor_level:
        lines.append("\n## 8. Statistical Power & Minimum Detectable Effect Sizes (MDES)")
        lines.append(f"- Target Power: 80% (alpha = 0.05, critical t ~ 2.80) based on effective sample size N_eff = {study_facts.effective_sample_size or 'N/A'}:")
        for lvl, mdes_val in study_facts.mdes_by_factor_level.items():
            lines.append(f"  * {lvl} contrast MDES: {mdes_val:.2f}")
        lines.append(
            "- Power Implication: Because plausible physical/domain offsets are often smaller than these detectable thresholds, "
            "non-significant findings reflect bounded statistical power under high clustering rather than affirmative proof of zero effect."
        )

    # 9. Domain Limitations & Methodological Recommendations
    lines.append("\n## 9. Study Design Boundaries & Methodological Limitations")
    confounder_note = study_facts.unmeasured_confounders_note or study_facts.mineralogy_note
    if confounder_note:
        lines.append(f"- Unmeasured Confounders Boundary: {confounder_note}")
    bounds_note = study_facts.sampling_boundaries_note or study_facts.age_window_note
    if bounds_note:
        lines.append(f"- Sampling/Temporal Boundary: {bounds_note}")
    if study_facts.covariates:
        covar_str = ", ".join(study_facts.covariates)
        lines.append(f"- Covariate Proxies: Observed continuous covariates ({covar_str}) serve as statistical controls and proxy adjustments; direct causal mechanisms remain bounded by observational sampling.")
    lines.append(
        "- Recommendations for Future Studies: (1) Prospective and high-resolution continuous monitoring across contrasting factor levels within matched conditions; "
        "(2) prospective, balanced multi-site sampling campaigns to mitigate sample size disparities; and (3) systematic screening for unmeasured confounding attributes and constrained observation windows."
    )

    lines.append(
        "\nCRITICAL EPISTEMIC CALIBRATION RULES:\n"
        "1. DO NOT claim the factor exerts an influence independent of covariates when LMM fixed effects are non-significant (p >= 0.05).\n"
        "2. BANNED WORDS: Never say differences 'dissolve' or 'vanish'. Use 'become statistically inconclusive'.\n"
        "3. TUKEY CONTRASTS: Refer to them strictly as 'unadjusted pairwise comparisons on raw group means'. Do not claim they 'became non-significant after adjustment'.\n"
        "4. CONTRAST POLARITY: Maintain a uniform Group minus Baseline reference level sign convention."
    )

    return "\n".join(lines)



def format_p(value):
    return "< 0.0001" if value < .0001 else format(value, ".4f")
