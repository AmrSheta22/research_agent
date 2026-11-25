"""
Autonomous Analysis & Data Manipulation Tool:
Empowers the agent to dynamically manipulate data, perform secondary and sensitivity analyses,
calculate statistical power / MDES, execute variance partitioning, and run equivalence tests
based on empirical results. Completely domain-agnostic.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any, Dict, List, Optional, Type

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

from research_agent.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class AnalysisToolInput(BaseModel):
    operation: str = Field(
        default="power_and_mdes",
        description=(
            "Analysis operation to execute: "
            "'power_and_mdes' (computes statistical power and minimum detectable effect sizes), "
            "'variance_partitioning' (calculates exact Nakagawa R2_m, R2_c, and predictor variance shares), "
            "'sensitivity_analysis' (leave-one-cluster-out jackknife across primary sites/clusters), "
            "'cohort_and_covariates' (computes cohort sampling and covariate distributional metrics), "
            "'equivalence_testing' (TOST two one-sided tests against equivalence margins), "
            "or 'custom_query' (custom filtering and grouping)."
        ),
    )
    df_path: Optional[str] = Field(default=None, description="Path to CSV dataset.")
    independent_var: Optional[str] = Field(default=None, description="Independent variable column.")
    dependent_var: Optional[str] = Field(default=None, description="Dependent variable column.")
    cluster_var: Optional[str] = Field(default=None, description="Cluster / site column.")
    covariates: Optional[List[str]] = Field(default_factory=list, description="Covariates to control.")
    equivalence_margin: Optional[float] = Field(default=0.5, description="Equivalence bound (Delta) for TOST.")
    custom_filter: Optional[str] = Field(default=None, description="Pandas query string for custom filtering.")


class AutonomousAnalysisTool(BaseTool):
    """Executes dynamic secondary analyses, sensitivity checks, variance partitioning, and power analysis."""

    name = "autonomous_data_analysis"
    description = (
        "Enables autonomous data manipulation and secondary statistical exploration: "
        "computes statistical power and Minimum Detectable Effect Sizes (MDES), "
        "calculates formal variance partitioning (Nakagawa marginal and conditional R2), "
        "performs leave-one-cluster-out sensitivity analyses, and runs equivalence testing (TOST)."
    )
    input_schema: Optional[Type[BaseModel]] = AnalysisToolInput

    def execute(self, **kwargs) -> ToolResult:
        # 1. Resolve DataFrame
        df: Optional[pd.DataFrame] = kwargs.get("df")
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            df_path = kwargs.get("df_path")
            if df_path and os.path.exists(df_path):
                try:
                    df = pd.read_csv(df_path)
                except Exception as exc:
                    return ToolResult.fail(f"Could not load dataset from {df_path}: {exc}")
            else:
                return ToolResult.fail("No valid dataset DataFrame or CSV path provided.")

        if df is None or df.empty:
            return ToolResult.fail("Dataset is empty.")

        # 2. Resolve IV, DV, cluster_var, covars dynamically
        iv = kwargs.get("independent_var") or kwargs.get("iv")
        if not iv or iv not in df.columns:
            cat_cols = [c for c in df.columns if df[c].dtype == object or df[c].dtype.name == "category"]
            iv = cat_cols[0] if cat_cols else df.columns[0]

        dv = kwargs.get("dependent_var") or kwargs.get("dv")
        if not dv or dv not in df.columns:
            num_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number) and c != iv]
            dv = num_cols[0] if num_cols else (df.columns[1] if len(df.columns) > 1 else df.columns[0])

        cluster_var = kwargs.get("cluster_var")
        if not cluster_var or cluster_var not in df.columns:
            cluster_var = next((c for c in ["cluster_id", "cluster", "site_id", "site_name", "site", "group_id", "subject_id", "location_id"] if c in df.columns), None)

        raw_covars = kwargs.get("covariates") or kwargs.get("covars") or []
        covars = [c for c in raw_covars if c in df.columns]
        margin = kwargs.get("equivalence_margin", 0.5)

        # Standardize IV
        working_df = df.dropna(subset=[iv, dv]).copy()
        if working_df[iv].dtype == object or working_df[iv].dtype.name == "category":
            working_df[iv] = working_df[iv].astype(str).str.strip().str.lower()

        # Respect the variables as supplied; do not infer scientific transformations
        # from column names (e.g. replacing latitude by its absolute value).
        model_covars = covars

        # 3. Check if multiple analyses requested (e.g. from autonomous_analysis_state node)
        analyses_list = kwargs.get("analyses")
        if analyses_list and isinstance(analyses_list, list):
            aggregated: Dict[str, Any] = {}
            for op_name in analyses_list:
                res = self._dispatch_operation(op_name, working_df, iv, dv, cluster_var, model_covars, margin, kwargs)
                if res.success and isinstance(res.data, dict):
                    aggregated[op_name] = res.data
            return ToolResult.ok(aggregated)

        # Single operation dispatch
        operation = kwargs.get("operation", "power_and_mdes").lower().strip()
        return self._dispatch_operation(operation, working_df, iv, dv, cluster_var, model_covars, margin, kwargs)

    def _dispatch_operation(
        self,
        operation: str,
        df: pd.DataFrame,
        iv: str,
        dv: str,
        cluster_var: Optional[str],
        model_covars: List[str],
        margin: float,
        kwargs: Dict[str, Any],
    ) -> ToolResult:
        if operation == "power_and_mdes":
            return self._compute_power_and_mdes(df, iv, dv, cluster_var, model_covars)
        elif operation == "variance_partitioning":
            return self._compute_variance_partitioning(df, iv, dv, cluster_var, model_covars)
        elif operation in ("sensitivity_analysis", "leave_one_cluster_out"):
            return self._compute_leave_one_cluster_out(df, iv, dv, cluster_var, model_covars)
        elif operation == "cohort_and_covariates":
            return self._compute_cohort_and_covariates(df, iv, dv, cluster_var, model_covars)
        elif operation == "mineralogy_and_age_controls":
            return ToolResult.fail("Mineralogy and age control models require measured phase/chronology metadata and are not implemented by descriptive cohort summaries.")
        elif operation == "equivalence_testing":
            return self._compute_equivalence_tost(df, iv, dv, cluster_var, model_covars, margin=margin)
        elif operation == "custom_query":
            custom_filter = kwargs.get("custom_filter")
            if custom_filter:
                filtered_df = df.query(custom_filter)
                return ToolResult.ok({
                    "filtered_sample_size": len(filtered_df),
                    "descriptives": filtered_df.groupby(iv)[dv].agg(["count", "mean", "std", "median"]).to_dict(orient="index"),
                })
            return ToolResult.ok({"sample_size": len(df)})
        else:
            return ToolResult.fail(f"Unknown operation: {operation}")

    def _compute_power_and_mdes(self, df: pd.DataFrame, iv: str, dv: str, cluster_var: Optional[str], covars: list[str]) -> ToolResult:
        """Calculate standard errors, statistical power at 80%, and Minimum Detectable Effect Size (MDES)."""
        counts = df[iv].value_counts()
        if len(counts) == 0:
            return ToolResult.fail("No factor levels present.")
        ref = counts.index[0]
        covar_terms = " + " + " + ".join(covars) if covars else ""
        formula = f"{dv} ~ C({iv}, Treatment(reference='{ref}')){covar_terms}"

        model = None
        if cluster_var and cluster_var in df.columns and df[cluster_var].nunique() > 1 and len(df) >= 10:
            try:
                model = smf.mixedlm(formula, data=df, groups=df[cluster_var]).fit(reml=True)
            except Exception:
                pass

        if model is None:
            try:
                model = smf.ols(formula, data=df).fit()
            except Exception as exc:
                return ToolResult.ok({
                    "analysis": "Statistical Power & Minimum Detectable Effect Size (MDES)",
                    "status": f"Model fitting skipped: {exc}",
                    "nominal_sample_size": len(df),
                })

        mdes_dict = {}
        for param, se in model.bse.items():
            if iv in param:
                mdes_80 = 2.80 * float(se)
                mdes_90 = 3.24 * float(se)
                coef = float(model.params.get(param, 0.0))
                pval = float(model.pvalues.get(param, 1.0))
                clean_name = param.split("[T.")[-1].rstrip("]") if "[T." in param else param
                mdes_dict[clean_name] = {
                    "coefficient": round(coef, 4),
                    "std_err": round(float(se), 4),
                    "p_value": round(pval, 5),
                    "mdes_80_power": round(mdes_80, 3),
                    "mdes_90_power": round(mdes_90, 3),
                    "interpretation": (
                        f"At 80% power, the study can only reliably detect offsets >= {mdes_80:.2f} units. "
                        f"True effects below {mdes_80:.2f} units will frequently yield non-significant p-values (p > 0.05) "
                        f"due to bounded statistical power."
                    ),
                }

        clusters_count = df[cluster_var].nunique() if (cluster_var and cluster_var in df.columns) else len(df)
        return ToolResult.ok({
            "analysis": "Statistical Power & Minimum Detectable Effect Size (MDES)",
            "nominal_sample_size": len(df),
            "clusters": clusters_count,
            "alpha": 0.05,
            "target_power": 0.80,
            "mdes_by_factor_level": mdes_dict,
            "synthesis": (
                "The MDES values are normal-approximation planning estimates based on fitted standard errors. "
                "They do not by themselves establish that a non-significant result is caused by low power."
            ),
        })

    def _compute_variance_partitioning(self, df: pd.DataFrame, iv: str, dv: str, cluster_var: Optional[str], covars: list[str]) -> ToolResult:
        """Compute formal Nakagawa marginal (R2_m) and conditional (R2_c) variance partitioning."""
        counts = df[iv].value_counts()
        if len(counts) == 0:
            return ToolResult.fail("No factor levels present.")
        ref = counts.index[0]
        covar_terms = " + " + " + ".join(covars) if covars else ""
        formula = f"{dv} ~ C({iv}, Treatment(reference='{ref}')){covar_terms}"

        if not cluster_var or cluster_var not in df.columns or df[cluster_var].nunique() <= 1 or len(df) < 10:
            return ToolResult.ok({
                "analysis": "Variance Partitioning",
                "status": "Hierarchical clustering not applicable or insufficient sample size.",
                "sample_size": len(df),
            })

        try:
            lmm = smf.mixedlm(formula, data=df, groups=df[cluster_var]).fit(reml=True)
            var_site = float(lmm.cov_re.iloc[0, 0])
            var_resid = float(lmm.scale)

            fixed_fitted = np.dot(lmm.model.exog, lmm.fe_params)
            var_fixed = float(np.var(fixed_fitted, ddof=1))
            var_total = var_fixed + var_site + var_resid
            r2_m = var_fixed / var_total if var_total > 0 else 0.0
            r2_c = (var_fixed + var_site) / var_total if var_total > 0 else 0.0

            # Predictor variances alone are not an additive decomposition when
            # predictors covary. Do not assign a residual share to the factor.
            covar_models = {}

            return ToolResult.ok({
                "analysis": "Nakagawa & Schielzeth (2013) Variance Partitioning",
                "marginal_r2": round(r2_m, 4),
                "conditional_r2": round(r2_c, 4),
                "variance_components": {
                    "fixed_effects_variance": round(var_fixed, 4),
                    "random_site_variance": round(var_site, 4),
                    "residual_variance": round(var_resid, 4),
                    "total_variance": round(var_total, 4),
                },
                "variance_shares_percent": {
                    "fixed_predictors_share_of_total": round(var_fixed / var_total * 100, 2) if var_total > 0 else 0.0,
                    "random_clusters_share_of_total": round(var_site / var_total * 100, 2) if var_total > 0 else 0.0,
                    "residual_noise_share_of_total": round(var_resid / var_total * 100, 2) if var_total > 0 else 0.0,
                },
                "fixed_variance_breakdown_percent": covar_models,
                "fixed_variance_breakdown_note": "Not computed: correlated predictors require covariance-aware attribution.",
            })
        except Exception as exc:
            return ToolResult.ok({
                "analysis": "Variance Partitioning",
                "status": f"MixedLM calculation skipped: {exc}",
                "sample_size": len(df),
            })

    def _compute_leave_one_cluster_out(self, df: pd.DataFrame, iv: str, dv: str, cluster_var: Optional[str], covars: list[str]) -> ToolResult:
        """Run leave-one-cluster-out jackknife sensitivity analysis across non-baseline factor levels."""
        counts = df[iv].value_counts()
        if len(counts) == 0:
            return ToolResult.fail("No factor levels present.")
        ref = counts.index[0]
        covar_terms = " + " + " + ".join(covars) if covars else ""
        formula = f"{dv} ~ C({iv}, Treatment(reference='{ref}')){covar_terms}"

        if not cluster_var or cluster_var not in df.columns or df[cluster_var].nunique() <= 2 or len(df) < 15:
            return ToolResult.ok({
                "analysis": "Leave-One-Cluster-Out Sensitivity Analysis",
                "status": "Hierarchical clustering not applicable or insufficient sample size.",
                "sample_size": len(df),
            })

        try:
            base_lmm = smf.mixedlm(formula, data=df, groups=df[cluster_var]).fit(reml=True)
            if not base_lmm.converged:
                raise ValueError("Base model did not converge")
            non_ref_levels = [lvl for lvl in counts.index if lvl != ref]

            sensitivity_by_level = {}
            for lvl in non_ref_levels[:3]:
                lvl_clusters = df[df[iv] == lvl][cluster_var].unique()
                param_key = f"C({iv}, Treatment(reference='{ref}'))[T.{lvl}]"
                base_slope = float(base_lmm.fe_params.get(param_key, 0.0))
                jackknife_slopes = []
                failures = []
                for c_id in lvl_clusters:
                    sub = df[df[cluster_var] != c_id]
                    try:
                        fit = smf.mixedlm(formula, data=sub, groups=sub[cluster_var]).fit(reml=True)
                        if not fit.converged:
                            raise ValueError("Exclusion fit did not converge")
                        m_val = float(fit.fe_params[param_key])
                        jackknife_slopes.append(m_val)
                    except Exception as exc:
                        failures.append({"cluster":str(c_id),"reason":str(exc)})

                min_s = min(jackknife_slopes) if jackknife_slopes else base_slope
                max_s = max(jackknife_slopes) if jackknife_slopes else base_slope
                sensitivity_by_level[str(lvl)] = {
                    "clusters_tested": len(lvl_clusters),
                    "successful_fits": len(jackknife_slopes),
                    "failed_fits": failures,
                    "cluster_variable": cluster_var,
                    "baseline_coefficient": round(base_slope, 4),
                    "jackknife_coefficient_range": [round(min_s, 4), round(max_s, 4)] if jackknife_slopes else None,
                    "stability": ("No converged exclusion fits" if not jackknife_slopes else "Directionally stable across successful exclusion fits" if min_s * max_s >= 0 else "Direction changes across successful exclusion fits"),
                }

            return ToolResult.ok({
                "analysis": "Leave-One-Cluster-Out Jackknife Sensitivity Analysis",
                "sensitivity_by_level": sensitivity_by_level,
                "interpretation": (
                    "The reported coefficient ranges describe successful cluster-exclusion fits. "
                    "They do not establish stability of statistical significance."
                ),
            })
        except Exception as exc:
            return ToolResult.ok({
                "analysis": "Leave-One-Cluster-Out Jackknife Sensitivity Analysis",
                "status": f"Jackknife calculation skipped: {exc}",
                "sample_size": len(df),
            })

    def _compute_cohort_and_covariates(self, df: pd.DataFrame, iv: str, dv: str, cluster_var: Optional[str], covars: list[str]) -> ToolResult:
        """Compute empirical sample retention and continuous covariate distributional properties."""
        n_total = len(df)
        clusters_total = df[cluster_var].nunique() if (cluster_var and cluster_var in df.columns) else n_total

        covar_stats = {}
        for c in covars:
            if c in df.columns and np.issubdtype(df[c].dtype, np.number):
                series = df[c].dropna()
                if len(series) > 0:
                    covar_stats[c] = {
                        "min": round(float(series.min()), 3),
                        "max": round(float(series.max()), 3),
                        "mean": round(float(series.mean()), 3),
                        "std": round(float(series.std()), 3),
                    }

        return ToolResult.ok({
            "analysis": "Cohort Sampling and Covariate Distributional Metrics",
            "sample_retention": {
                "analyzed_records": n_total,
                "clusters_or_sites": clusters_total,
            },
            "covariate_distributions": covar_stats,
        })

    def _compute_equivalence_tost(self, df: pd.DataFrame, iv: str, dv: str, cluster_var: Optional[str], covars: list[str], margin: float = 0.5) -> ToolResult:
        """Perform Two One-Sided Tests (TOST) for statistical equivalence within [-margin, +margin]."""
        counts = df[iv].value_counts()
        if len(counts) == 0:
            return ToolResult.fail("No factor levels present.")
        ref = counts.index[0]
        covar_terms = " + " + " + ".join(covars) if covars else ""
        formula = f"{dv} ~ C({iv}, Treatment(reference='{ref}')){covar_terms}"

        model = None
        if cluster_var and cluster_var in df.columns and df[cluster_var].nunique() > 1 and len(df) >= 15:
            try:
                model = smf.mixedlm(formula, data=df, groups=df[cluster_var]).fit(reml=True)
            except Exception:
                pass

        if model is None:
            try:
                model = smf.ols(formula, data=df).fit()
            except Exception as exc:
                return ToolResult.ok({
                    "analysis": f"Two One-Sided Tests (TOST) Equivalence Evaluation (Margin = +/-{margin})",
                    "status": f"Model fitting skipped: {exc}",
                })

        tost_results = {}
        df_resid = getattr(model, "df_resid", len(df) - 5)
        for param, coef in model.params.items():
            if iv in param:
                se = model.bse.get(param, 0.0)
                if se <= 0:
                    continue
                clean_name = param.split("[T.")[-1].rstrip("]") if "[T." in param else param
                t1 = (coef - (-margin)) / se
                p1 = 1 - stats.t.cdf(t1, df=max(df_resid, 1))
                t2 = (coef - margin) / se
                p2 = stats.t.cdf(t2, df=max(df_resid, 1))
                tost_p = max(p1, p2)
                critical = stats.t.ppf(0.95, df=max(df_resid, 1))
                ci_90_lower = coef - critical * se
                ci_90_upper = coef + critical * se

                equivalent = tost_p < 0.05
                tost_results[clean_name] = {
                    "estimate": round(float(coef), 4),
                    "std_err": round(float(se), 4),
                    "equivalence_margin": margin,
                    "ci_90": [round(float(ci_90_lower), 3), round(float(ci_90_upper), 3)],
                    "tost_p_value": round(float(tost_p), 5),
                    "statistically_equivalent": equivalent,
                    "classification": "Statistically Equivalent" if equivalent else "Indeterminate / Inconclusive",
                }

        return ToolResult.ok({
            "analysis": f"Two One-Sided Tests (TOST) Equivalence Evaluation (Margin = +/-{margin})",
            "results": tost_results,
            "conclusion": (
                f"Within the supplied +/-{margin} equivalence margin, "
                f"{sum(bool(r['statistically_equivalent']) for r in tost_results.values())} "
                f"of {len(tost_results)} evaluated contrasts meet the TOST equivalence criterion. "
                "Contrasts failing this criterion do not establish equivalence; this does not test for a zero effect."
            ),
        })
