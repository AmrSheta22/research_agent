"""
Statistical analysis utilities for individual RCT data.

This module provides functions for analyzing different combinations of
independent and dependent variables in randomized controlled trials.

Cases covered:
1. Binary IV, Continuous DV
2. Binary IV, Binary DV
3. Binary IV, Multinomial DV
4. Multinomial IV, Continuous DV
5. Multinomial IV, Binary DV
6. Multinomial IV, Multinomial DV
7. Ordinal IV, Continuous DV
8. Ordinal IV, Binary DV
9. Ordinal IV, Multinomial DV
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import (
    mannwhitneyu,
    spearmanr,
    kendalltau,
    chi2_contingency,
    fisher_exact,
    f_oneway,
    kruskal,
    norm,
    chi2,
)
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.contingency_tables import Table
from .utils_general_functions import safe_execute


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _ensure_covariates_list(covariates):
    """Convert covariates to list format."""
    if covariates is None:
        return []
    if isinstance(covariates, (list, tuple)):
        return list(covariates)
    return [covariates]


def _iv_to_numeric(df, iv, iv_order=None):
    """
    Convert ordinal IV into numeric scores 0,1,2,... preserving order.

    Priority:
      1) If iv_order provided (list), use that ordering.
      2) If df[iv] is pandas.Categorical and ordered, use its codes.
      3) Else use unique values in the order they appear.

    Returns numeric_series (float) and mapping dict.
    """
    s = df[iv].astype(object)
    if iv_order is not None:
        mapping = {cat: i for i, cat in enumerate(iv_order)}
        numeric = s.map(mapping).astype(float)
    else:
        if pd.api.types.is_categorical_dtype(df[iv]) and df[iv].cat.ordered:
            numeric = df[iv].cat.codes.astype(float)
            mapping = {cat: code for code, cat in enumerate(df[iv].cat.categories)}
        else:
            uniq = list(pd.Series(s.unique()).astype(object))
            mapping = {cat: i for i, cat in enumerate(uniq)}
            numeric = s.map(mapping).astype(float)
    return numeric, mapping


def _dv_to_numeric(df, dv, dv_order=None):
    """Map ordinal DV to numeric scores 0,1,2..."""
    s = df[dv].astype(object)
    if dv_order is not None:
        mapping = {cat: i for i, cat in enumerate(dv_order)}
    else:
        uniq = list(pd.Categorical(s).categories)
        mapping = {cat: i for i, cat in enumerate(uniq)}
    return s.map(mapping).astype(float), mapping


def _ensure_binary_y(df, dv):
    """
    Map DV to 0/1 numeric if not already numeric 0/1.
    If DV has exactly two unique values, map first->0, second->1.
    """
    y = df[dv]
    if pd.api.types.is_numeric_dtype(y):
        unique_vals = pd.Series(y.dropna().unique())
        if set(unique_vals.astype(str)) <= set(["0", "1"]):
            return pd.to_numeric(y, errors="coerce").astype(float)

    uniq = pd.Series(y.dropna().unique())
    if len(uniq) != 2:
        return pd.Series(np.nan, index=df.index)

    mapping = {uniq.iloc[0]: 0.0, uniq.iloc[1]: 1.0}
    return y.map(mapping).astype(float)


# =============================================================================
# CASE 1: BINARY IV, CONTINUOUS DV
# =============================================================================


def t_test_unadjusted(df, treatment_col, outcome_col):
    """Two-sample Welch's t-test: outcome ~ treatment"""
    df = df.copy()
    df[treatment_col] = _ensure_binary_y(df, treatment_col)
    grp0 = df.loc[df[treatment_col] == 0, outcome_col]
    grp1 = df.loc[df[treatment_col] == 1, outcome_col]
    tstat, pval = stats.ttest_ind(grp1, grp0, equal_var=False)
    return {"p-value": pval}


def ols_unadjusted(df, treatment_col, outcome_col):
    """OLS regression: outcome ~ treatment"""
    df = df.copy()
    df[treatment_col] = _ensure_binary_y(df, treatment_col)
    formula = f"{outcome_col} ~ {treatment_col}"
    model = smf.ols(formula, data=df).fit()
    pval = model.pvalues[treatment_col]
    return {"p-value": pval}


def ols_adjusted(df, treatment_col, outcome_col, covariates):
    """OLS regression with covariates: outcome ~ treatment + covariates"""
    df = df.copy()
    df[treatment_col] = _ensure_binary_y(df, treatment_col)
    cov_str = ""
    if covariates:
        cov_str = " + " + " + ".join(
            [
                f"C({c})" if str(df[c].dtype) in ["object", "category"] else c
                for c in covariates
            ]
        )
    formula = f"{outcome_col} ~ {treatment_col}"
    if covariates:
        formula += " + " + cov_str
    model = smf.ols(formula, data=df).fit()
    pval = model.pvalues[treatment_col]
    return {"p-value": pval}


def mann_whitney_test(df, group_col, outcome_col):
    """
    Nonparametric alternative to independent t-test
    Binary independent variable -> Continuous dependent variable
    """
    groups = df[group_col].unique()
    if len(groups) != 2:
        raise ValueError(
            "Independent variable must have exactly 2 groups for Mann–Whitney U test."
        )

    group1 = df.loc[df[group_col] == groups[0], outcome_col]
    group2 = df.loc[df[group_col] == groups[1], outcome_col]

    stat, pval = mannwhitneyu(group1, group2, alternative="two-sided")
    return {"p-value": pval}


def run_all_tests_binaryIV_continuousDV(df, iv, dv, covariates=None):
    """Run all tests for binary IV vs continuous DV."""
    results = {}
    results["T-test (unadjusted)"] = safe_execute(t_test_unadjusted, df, iv, dv)
    results["OLS (unadjusted)"] = safe_execute(ols_unadjusted, df, iv, dv)
    results["OLS (adjusted)"] = safe_execute(ols_adjusted, df, iv, dv, covariates)
    results["Mann-Whitney U test"] = safe_execute(mann_whitney_test, df, iv, dv)
    return results


# =============================================================================
# CASE 2: BINARY IV, BINARY DV
# =============================================================================


def chi_square_test(df, treatment_col, outcome_col):
    """Chi-square test of independence: treatment x categorical outcome"""
    table = pd.crosstab(df[treatment_col], df[outcome_col])
    chi2, pval, dof, exp = chi2_contingency(table)
    return {"p-value": pval}


def logistic_regression(df, treatment_col, outcome_col, covariates=None):
    """Logistic regression (binary outcome): outcome ~ treatment [+ covariates]"""
    if len(df[outcome_col].unique()) != 2:
        raise ValueError("Logistic regression requires a binary outcome")

    df = df.copy()
    df[treatment_col] = _ensure_binary_y(df, treatment_col)
    df[outcome_col] = _ensure_binary_y(df, outcome_col)
    cov_str = ""
    if covariates:
        cov_str = " + " + " + ".join(
            [
                f"C({c})" if str(df[c].dtype) in ["object", "category"] else c
                for c in covariates
            ]
        )
    formula = f"{outcome_col} ~ {treatment_col}{cov_str}"
    model = smf.logit(formula, data=df).fit(disp=False)
    pval = model.pvalues[treatment_col]
    return {"p-value": pval}


def fisher_test(df, group_col, outcome_col):
    """Fisher's exact test (2x2 only)."""
    contingency = pd.crosstab(df[group_col], df[outcome_col])
    if contingency.shape != (2, 2):
        raise ValueError("Fisher's exact test is only valid for 2x2 tables.")

    oddsratio, pval = fisher_exact(contingency)
    return {"p-value": pval}


def run_all_tests_binaryIV_binaryDV(df, iv, dv, covariates=None):
    """Run all tests for binary IV vs binary DV."""
    results = {}
    results["Chi-square test"] = safe_execute(chi_square_test, df, iv, dv)
    results["Logistic regression"] = safe_execute(logistic_regression, df, iv, dv, covariates)
    results["Fisher's exact test"] = safe_execute(fisher_test, df, iv, dv)
    return results


# =============================================================================
# CASE 3: BINARY IV, MULTINOMIAL DV
# =============================================================================


def multinomial_logistic_regression(df, treatment_col, outcome_col, covariates=None):
    """Multinomial logistic regression (multi-class outcome): outcome ~ treatment [+ covariates]"""
    if len(df[outcome_col].unique()) < 3:
        raise ValueError(
            "Multinomial logistic regression requires a categorical outcome with >= 3 levels"
        )

    cov_str = ""
    if covariates:
        cov_str = " + " + " + ".join(
            [
                f"C({c})" if str(df[c].dtype) in ["object", "category"] else c
                for c in covariates
            ]
        )
    # convert outcome to numerical category if not already
    df[outcome_col] = df[outcome_col].astype("category").cat.codes
    formula = f"{outcome_col} ~ {treatment_col}{cov_str}"
    model = smf.mnlogit(formula, data=df).fit(disp=False)
    pval = model.pvalues.loc[treatment_col].min()  # min p across outcome levels
    return {"p-value": pval}


def run_all_tests_binaryIV_multinomialDV(df, iv, dv, covariates=None):
    """Run all tests for binary IV vs multinomial DV."""
    results = {}
    results["Chi-square test"] = safe_execute(chi_square_test, df, iv, dv)
    results["Multinomial logistic regression"] = safe_execute(multinomial_logistic_regression, df, iv, dv, covariates)
    return results


# =============================================================================
# CASE 4: CATEGORICAL IV, CONTINUOUS DV
# =============================================================================


def anova_test(df, group_col, outcome_col):
    """One-way ANOVA: continuous outcome ~ categorical group"""
    groups = [df.loc[df[group_col] == g, outcome_col] for g in df[group_col].unique()]
    fstat, pval = f_oneway(*groups)
    return {"p-value": pval}


def ancova(df, group_col, outcome_col, covariates=None):
    """ANCOVA via OLS: outcome ~ group + covariates"""
    cov_str = ""
    if covariates:
        cov_str = " + " + " + ".join(
            [
                f"C({c})" if str(df[c].dtype) in ["object", "category"] else c
                for c in covariates
            ]
        )
    formula = f"{outcome_col} ~ C({group_col}){cov_str}"
    model = smf.ols(formula, data=df).fit()
    anova_table = anova_lm(model, typ=2)
    pval = anova_table.loc[f"C({group_col})", "PR(>F)"]
    return {"p-value": pval}


def kruskal_wallis_test(df, group_col, outcome_col):
    """Kruskal–Wallis test: nonparametric alternative to ANOVA"""
    groups = [df.loc[df[group_col] == g, outcome_col] for g in df[group_col].unique()]
    hstat, pval = kruskal(*groups)
    return {"p-value": pval}


def run_all_tests_multinomialIV_continuousDV(df, iv, dv, covariates=None):
    """Run all tests for multinomial (categorical) IV vs continuous DV."""
    results = {}
    results["ANOVA test"] = safe_execute(anova_test, df, iv, dv)
    results["ANCOVA"] = safe_execute(ancova, df, iv, dv, covariates)
    results["Kruskal-Wallis test"] = safe_execute(kruskal_wallis_test, df, iv, dv)
    return results


# =============================================================================
# CASE 5: MULTINOMIAL IV, BINARY DV
# =============================================================================


def fisher_exact_general(df, group_col, outcome_col):
    """
    Generalized Fisher's exact test for r x c tables.
    Falls back to scipy fisher_exact if 2x2.
    """
    contingency = pd.crosstab(df[group_col], df[outcome_col])

    if contingency.shape == (2, 2):
        _, pval = fisher_exact(contingency)
        return {"p-value": pval}
    else:
        table = Table(contingency)
        res = table.test_nominal_association()
        return {"p-value": res.pvalue}


def logistic_regression_categorical(df, group_col, outcome_col, covariates=None):
    """Logistic regression: outcome ~ group + covariates"""
    cov_str = ""
    if covariates:
        cov_str = " + " + " + ".join(
            [
                f"C({c})" if str(df[c].dtype) in ["object", "category"] else c
                for c in covariates
            ]
        )
    formula = f"{outcome_col} ~ C({group_col}){cov_str}"
    model = smf.logit(formula, data=df).fit(disp=False)
    pval = model.pvalues[1:].max()  # max p-value for group terms
    return {"p-value": pval}


def run_all_tests_multinomialIV_binaryDV(df, iv, dv, covariates=None):
    """Run all tests for multinomial IV vs binary DV."""
    results = {}
    results["Chi-square test"] = safe_execute(chi_square_test, df, iv, dv)
    results["Fisher's exact test"] = safe_execute(fisher_exact_general, df, iv, dv)
    results["Logistic regression"] = safe_execute(logistic_regression_categorical, df, iv, dv, covariates)
    return results


# =============================================================================
# CASE 6: MULTINOMIAL IV, MULTINOMIAL DV
# =============================================================================


def likelihood_ratio_test(df, iv, dv):
    """Likelihood Ratio Test (G-test)"""
    contingency = pd.crosstab(df[iv], df[dv])
    chi2, p, dof, expected = chi2_contingency(contingency, lambda_="log-likelihood")
    return {"p-value": p}


def multinomial_logistic_lrtest(df, iv, dv, covariates=None):
    """Multinomial Logistic Regression (Likelihood Ratio Test)"""
    if covariates:
        predictors = f"{iv} + " + " + ".join(covariates)
    else:
        predictors = iv
    df[iv] = df[iv].astype("category").cat.codes
    df[dv] = df[dv].astype("category").cat.codes
    formula = f"{dv} ~ {predictors}"
    model = smf.mnlogit(formula, data=df).fit(disp=False)
    lr_test = model.llr_pvalue
    return {"p-value": lr_test}


def run_all_tests_multinomialIV_multinomialDV(df, iv, dv, covariates=None):
    """Run all tests for multinomial IV vs multinomial DV."""
    results = {}
    results["Chi-square test"] = safe_execute(chi_square_test, df, iv, dv)
    results["Likelihood ratio test"] = safe_execute(likelihood_ratio_test, df, iv, dv)
    results["Multinomial logistic regression (LR test)"] = safe_execute(multinomial_logistic_lrtest, df, iv, dv, covariates)
    return results


# =============================================================================
# CASE 7: ORDINAL IV, CONTINUOUS DV
# =============================================================================


def spearman_test(df, iv, dv, iv_order=None):
    """Spearman rank correlation between ordinal IV and continuous DV."""
    x, mapping = _iv_to_numeric(df, iv, iv_order)
    y = pd.to_numeric(df[dv], errors="coerce")
    mask = x.notna() & y.notna()
    if mask.sum() < 3:
        return {"p-value": np.nan}
    rho, p = stats.spearmanr(x[mask], y[mask])
    return {"p-value": float(p)}


def ols_trend_test(df, iv, dv, iv_order=None):
    """OLS regression treating IV as numeric (trend test)."""
    df2 = df[[iv, dv]].copy()
    x, mapping = _iv_to_numeric(df2, iv, iv_order)
    df2["_iv_num"] = x
    df2["_dv_num"] = pd.to_numeric(df2[dv], errors="coerce")
    df2 = df2[["_iv_num", "_dv_num"]].dropna()
    if len(df2) < 3:
        return {"p-value": np.nan}
    model = sm.OLS(df2["_dv_num"], sm.add_constant(df2["_iv_num"])).fit()
    p = model.pvalues.get("_iv_num", np.nan)
    return {"p-value": float(p)}


def ols_with_covariates_trend(df, iv, dv, covariates=None, iv_order=None):
    """OLS regression treating IV as numeric and adjusting for covariates."""
    covs = _ensure_covariates_list(covariates)
    df2 = df[[iv, dv] + covs].copy()
    iv_num, mapping = _iv_to_numeric(df2, iv, iv_order)
    df2["_iv_num"] = iv_num
    df2["_dv_num"] = pd.to_numeric(df2[dv], errors="coerce")
    df2 = df2.dropna(subset=["_iv_num", "_dv_num"])
    if len(df2) < 3:
        return {"p-value": np.nan}

    # Build formula
    rhs = ["_iv_num"] + [
        (
            f"C({c})"
            if pd.api.types.is_object_dtype(df2[c])
            or pd.api.types.is_categorical_dtype(df2[c])
            else c
        )
        for c in covs
    ]
    formula = "_dv_num ~ " + " + ".join(rhs) if rhs else "_dv_num ~ _iv_num"

    try:
        model = smf.ols(formula, data=df2).fit()
    except Exception:
        # fallback: encode categorical covariates manually
        df_enc = pd.get_dummies(df2.drop(columns=[dv, iv]), drop_first=True)
        X = pd.concat(
            [df2[["_iv_num"]].reset_index(drop=True), df_enc.reset_index(drop=True)],
            axis=1,
        )
        model = sm.OLS(df2["_dv_num"], sm.add_constant(X)).fit()

    p = model.pvalues.get("_iv_num", np.nan)
    return {"p-value": float(p)}


def anova_iv_categorical_with_covariates(df, iv, dv, covariates=None):
    """
    Treat IV as categorical. Test overall IV effect adjusting for covariates
    by comparing full model (with C(iv)) vs reduced model (without C(iv)).
    """
    covs = _ensure_covariates_list(covariates)
    df2 = df[[iv, dv] + covs].copy()
    df2["_dv_num"] = pd.to_numeric(df2[dv], errors="coerce")
    df2 = df2.dropna(subset=["_dv_num"])
    if len(df2) < 3:
        return {"p-value": np.nan}

    # Build formulas
    rhs_full = [f"C({iv})"] + [
        (
            f"C({c})"
            if pd.api.types.is_object_dtype(df2[c])
            or pd.api.types.is_categorical_dtype(df2[c])
            else c
        )
        for c in covs
    ]
    rhs_reduced = [
        (
            f"C({c})"
            if pd.api.types.is_object_dtype(df2[c])
            or pd.api.types.is_categorical_dtype(df2[c])
            else c
        )
        for c in covs
    ]

    formula_full = "_dv_num ~ " + " + ".join(rhs_full)
    if rhs_reduced:
        formula_reduced = "_dv_num ~ " + " + ".join(rhs_reduced)
    else:
        formula_reduced = "_dv_num ~ 1"

    try:
        model_full = smf.ols(formula_full, data=df2).fit()
        model_reduced = smf.ols(formula_reduced, data=df2).fit()
        anova_res = anova_lm(model_reduced, model_full)
        p = anova_res["Pr(>F)"].iloc[-1]
    except Exception:
        p = np.nan

    return {"p-value": float(p) if not pd.isna(p) else np.nan}


def partial_correlation_controling_covariates(
    df, iv, dv, covariates=None, iv_order=None
):
    """Partial correlation between IV (ordinal numeric) and DV controlling for covariates."""
    covs = _ensure_covariates_list(covariates)
    df2 = df[[iv, dv] + covs].copy()
    df2["_iv_num"], mapping = _iv_to_numeric(df2, iv, iv_order)
    df2["_dv_num"] = pd.to_numeric(df2[dv], errors="coerce")
    df2 = df2.dropna(subset=["_iv_num", "_dv_num"])
    if len(df2) < 4:
        return {"p-value": np.nan}

    if covs:
        # One-hot encode categoricals
        X = pd.get_dummies(df2[covs], drop_first=True)
        X = X.apply(pd.to_numeric, errors="coerce").astype(float)
        X = sm.add_constant(X, has_constant="add")

        # Regress dv on X
        model_y = sm.OLS(df2["_dv_num"].astype(float), X).fit()
        ry = model_y.resid

        # Regress iv on X
        model_x = sm.OLS(df2["_iv_num"].astype(float), X).fit()
        rx = model_x.resid
    else:
        rx = df2["_iv_num"].astype(float)
        ry = df2["_dv_num"].astype(float)

    # Pearson correlation with p-value
    r, p = stats.pearsonr(rx, ry)
    return {"p-value": float(p)}


def permutation_trend_test(
    df, iv, dv, iv_order=None, n_permutations=2000, random_state=0
):
    """Permutation test for trend (slope) treating IV as numeric."""
    rng = np.random.default_rng(random_state)
    iv_num, mapping = _iv_to_numeric(df, iv, iv_order)
    dv_num = pd.to_numeric(df[dv], errors="coerce")
    mask = iv_num.notna() & dv_num.notna()
    x = iv_num[mask].values
    y = dv_num[mask].values
    n = len(x)
    if n < 3:
        return {"p-value": np.nan}

    # observed slope (linear regression)
    slope_obs = np.polyfit(x, y, 1)[0]
    perm_slopes = np.empty(n_permutations)
    for i in range(n_permutations):
        perm_x = rng.permutation(x)
        perm_slopes[i] = np.polyfit(perm_x, y, 1)[0]

    # two-sided p-value
    p = (np.sum(np.abs(perm_slopes) >= abs(slope_obs)) + 1) / (n_permutations + 1)
    return {"p-value": float(p)}


def run_all_tests_ordinalIV_continuousDV(
    df, iv, dv, covariates=None, iv_order=None, n_permutations=2000
):
    """Run all relevant tests for ordinal IV (3+ levels) vs continuous DV."""
    results = {}
    covs = _ensure_covariates_list(covariates)

    # Basic tests (no covariates)
    results["Spearman test"] = safe_execute(spearman_test, df, iv, dv, iv_order=iv_order)
    results["Kruskal-Wallis test"] = safe_execute(kruskal_wallis_test, df, iv, dv)
    results["OLS trend test"] = safe_execute(ols_trend_test, df, iv, dv, iv_order=iv_order)

    # With covariates (if any) we run adjusted tests
    if covs:
        results["OLS trend test (adjusted)"] = safe_execute(
            ols_with_covariates_trend,
            df,
            iv,
            dv,
            covariates=covs,
            iv_order=iv_order,
        )
        results["ANOVA (adjusted)"] = safe_execute(
            anova_iv_categorical_with_covariates, df, iv, dv, covariates=covs
        )
        results["Partial correlation (adjusted)"] = safe_execute(
            partial_correlation_controling_covariates,
            df,
            iv,
            dv,
            covariates=covs,
            iv_order=iv_order,
        )
    else:
        # partial correlation without covariates reduces to Pearson between numeric IV and DV
        results["Partial Correlation (no covariates) – equals Pearson between IV numeric & DV"] = safe_execute(spearman_test, df, iv, dv, iv_order=iv_order)[1]

    # Permutation trend test (robust)
    results["Permutation trend test (robust)"] = safe_execute(
                permutation_trend_test,
                df,
                iv,
                dv,
                iv_order=iv_order,
                n_permutations=n_permutations,
            )

    return results


# =============================================================================
# CASE 8: ORDINAL IV, BINARY DV
# =============================================================================


def cochran_armitage_test(df, iv, dv, iv_order=None):
    """Cochran-Armitage test for linear trend in proportions across ordered groups."""
    df2 = df[[iv, dv]].copy()
    y = _ensure_binary_y(df2, dv)
    df2["_y"] = y
    df2 = df2.dropna(subset=["_y"])
    if df2["_y"].nunique() == 0:
        return {"p-value": np.nan}

    scores, mapping = _iv_to_numeric(df2, iv, iv_order)
    df2["_score"] = scores
    counts = (
        df2.groupby(iv)["_y"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "successes", "count": "n"})
    )
    if len(counts) < 2:
        return {"p-value": np.nan}

    successes = counts["successes"].values.astype(float)
    ns = counts["n"].values.astype(float)
    scores_arr = np.array([mapping[level] for level in counts.index])
    N = ns.sum()
    if N == 0:
        return {"p-value": np.nan}

    p_hat = successes.sum() / N
    numer = np.sum(scores_arr * (successes - ns * p_hat))
    denom = np.sqrt(
        p_hat
        * (1 - p_hat)
        * (np.sum(ns * (scores_arr**2)) - (np.sum(ns * scores_arr) ** 2) / N)
    )
    if denom == 0:
        return {"p-value": np.nan}

    z = numer / denom
    p_two = 2 * (1 - norm.cdf(abs(z)))
    return {"p-value": float(p_two)}


def logistic_trend_test(df, iv, dv, iv_order=None):
    """Logistic regression treating IV as numeric (trend)."""
    df2 = df[[iv, dv]].copy()
    y = _ensure_binary_y(df2, dv)
    df2["_y"] = y
    df2 = df2.dropna(subset=["_y"])
    if df2.shape[0] < 10:
        return {"p-value": np.nan}

    ivnum, mapping = _iv_to_numeric(df2, iv, iv_order)
    df2["_ivnum"] = ivnum
    df2 = df2.dropna(subset=["_ivnum"])
    X = sm.add_constant(df2["_ivnum"].astype(float))

    try:
        model = sm.Logit(df2["_y"].astype(float), X).fit(disp=False)
        p = model.pvalues.get("_ivnum", np.nan)
    except Exception:
        p = np.nan

    return {"p-value": float(p) if not pd.isna(p) else np.nan}


def logistic_with_covariates_trend(df, iv, dv, covariates=None, iv_order=None):
    """Logistic regression: DV ~ IV_numeric + covariates."""
    covs = _ensure_covariates_list(covariates)
    df2 = df[[iv, dv] + covs].copy()
    df2["_y"] = _ensure_binary_y(df2, dv)
    df2 = df2.dropna(subset=["_y"])
    if df2.shape[0] < 10:
        return {"p-value": np.nan}

    df2["_ivnum"], mapping = _iv_to_numeric(df2, iv, iv_order)
    df2 = df2.dropna(subset=["_ivnum"])

    # Build formula
    rhs = ["_ivnum"] + [
        (
            f"C({c})"
            if df2[c].dtype == "O" or pd.api.types.is_categorical_dtype(df2[c])
            else c
        )
        for c in covs
    ]
    formula = "_y ~ " + " + ".join(rhs) if rhs else "_y ~ _ivnum"

    try:
        model = smf.logit(formula, data=df2).fit(disp=False)
        p = model.pvalues.get("_ivnum", np.nan)
    except Exception:
        # fallback: manual dummies
        X = pd.get_dummies(df2[covs], drop_first=True)
        X = pd.concat(
            [df2[["_ivnum"]].reset_index(drop=True), X.reset_index(drop=True)], axis=1
        )
        X = sm.add_constant(X)
        try:
            model = sm.Logit(df2["_y"].astype(float), X).fit(disp=False)
            p = model.pvalues.get("_ivnum", np.nan)
        except Exception:
            p = np.nan

    return {"p-value": float(p) if not pd.isna(p) else np.nan}


def logistic_categorical_lrtest(df, iv, dv, covariates=None):
    """Likelihood ratio test for IV effect treating IV as categorical."""
    covs = _ensure_covariates_list(covariates)
    df2 = df[[iv, dv] + covs].copy()
    df2["_y"] = _ensure_binary_y(df2, dv)
    df2 = df2.dropna(subset=["_y"])
    if df2.shape[0] < 10:
        return {"p-value": np.nan}

    # Build formulas
    rhs_covs = [
        (
            f"C({c})"
            if df2[c].dtype == "O" or pd.api.types.is_categorical_dtype(df2[c])
            else c
        )
        for c in covs
    ]
    formula_reduced = "_y ~ " + " + ".join(rhs_covs) if rhs_covs else "_y ~ 1"
    formula_full = "_y ~ C({})".format(iv) + (
        " + " + " + ".join(rhs_covs) if rhs_covs else ""
    )

    try:
        m_reduced = smf.logit(formula_reduced, data=df2).fit(disp=False)
        m_full = smf.logit(formula_full, data=df2).fit(disp=False)
        llr = 2.0 * (m_full.llf - m_reduced.llf)
        df_diff = int(m_full.df_model - m_reduced.df_model)
        p = chi2.sf(llr, df_diff) if df_diff > 0 else np.nan
    except Exception:
        p = np.nan

    return {"p-value": float(p) if not pd.isna(p) else np.nan}


def permutation_cochran_armitage_test(
    df, iv, dv, iv_order=None, n_permutations=2000, random_state=0
):
    """Permutation test for Cochran-Armitage statistic (trend)."""
    rng = np.random.default_rng(random_state)
    df2 = df[[iv, dv]].copy()
    df2["_y"] = _ensure_binary_y(df2, dv)
    df2 = df2.dropna(subset=["_y"])
    if df2.shape[0] < 3:
        return {"p-value": np.nan}

    # observed stat (use z)
    ivnum, mapping = _iv_to_numeric(df2, iv, iv_order)
    df2["_score"] = ivnum

    # compute observed z (use same code as Cochran-Armitage)
    counts = df2.groupby(iv)[["_y"]].agg(["sum", "count"])
    counts.columns = counts.columns.droplevel(0)
    successes = counts["sum"].values.astype(float)
    ns = counts["count"].values.astype(float)
    scores_arr = np.array([mapping[level] for level in counts.index])
    N = ns.sum()
    p_hat = successes.sum() / N
    denom = np.sqrt(
        p_hat
        * (1 - p_hat)
        * (np.sum(ns * (scores_arr**2)) - (np.sum(ns * scores_arr) ** 2) / N)
    )
    if denom == 0:
        return {"p-value": np.nan}

    numer = np.sum(scores_arr * (successes - ns * p_hat))
    z_obs = numer / denom
    perm_stats = 0
    n_valid = 0

    for i in range(n_permutations):
        # permute the iv labels instead of scores
        perm_iv = rng.permutation(df2[iv].values)
        df2["_perm_iv"] = perm_iv
        counts_p2 = df2.groupby("_perm_iv")["_y"].agg(["sum", "count"])

        # align with original levels ordering
        ordered_levels = list(mapping.keys())
        succs = []
        ns_p = []
        for lvl in ordered_levels:
            if lvl in counts_p2.index:
                succs.append(counts_p2.loc[lvl, "sum"])
                ns_p.append(counts_p2.loc[lvl, "count"])
            else:
                succs.append(0.0)
                ns_p.append(0.0)

        succs = np.array(succs, dtype=float)
        ns_p = np.array(ns_p, dtype=float)
        scores_p = np.array([mapping[l] for l in ordered_levels], dtype=float)
        Np = ns_p.sum()
        if Np == 0:
            continue

        p_hat_p = succs.sum() / Np
        denom_p = np.sqrt(
            p_hat_p
            * (1 - p_hat_p)
            * (np.sum(ns_p * (scores_p**2)) - (np.sum(ns_p * scores_p) ** 2) / Np)
        )
        if denom_p == 0:
            continue

        numer_p = np.sum(scores_p * (succs - ns_p * p_hat_p))
        z_p = numer_p / denom_p
        n_valid += 1
        if abs(z_p) >= abs(z_obs):
            perm_stats += 1

    if n_valid == 0:
        return {"p-value": np.nan}

    p = (perm_stats + 1) / (n_valid + 1)
    return {"p-value": float(p)}


def run_all_tests_ordinalIV_binaryDV(
    df, iv, dv, covariates=None, iv_order=None, n_permutations=2000
):
    """Run all tests appropriate for ordinal IV (3+ levels) vs binary DV."""
    results = {}
    covs = _ensure_covariates_list(covariates)

    # Basic tests (no covariates)
    results["Cochran-Armitage test"] = safe_execute(cochran_armitage_test, df, iv, dv, iv_order=iv_order)
    results["Chi-square test"] = safe_execute(chi_square_test, df, iv, dv)
    results["Logistic trend test"] = safe_execute(logistic_trend_test, df, iv, dv, iv_order=iv_order)

    # With covariates (if any) we run adjusted tests
    if covs:
        results["Logistic with Covariates (IV numeric)"] = safe_execute(logistic_with_covariates_trend, df, iv, dv, covariates=covs, iv_order=iv_order)
        results["Logistic with Covariates (IV categorical)"] = safe_execute(logistic_categorical_lrtest, df, iv, dv, covariates=covs)
    else:
        results["Logistic with Covariates (IV numeric)"] = "N/A (no covariates)"
        # include a basic point-biserial (Pearson between iv numeric and binary)
        results["Point-biserial (Pearson between IV numeric & binary DV)"] = safe_execute(
                    lambda d, i, v, io: (
                        "Point-biserial (Pearson between IV numeric & binary DV)",
                        stats.pearsonr(
                            _iv_to_numeric(d, i, io)[0].dropna(),
                            _ensure_binary_y(d, v).dropna(),
                        )[1]
                        if _iv_to_numeric(d, i, io)[0].notna().sum() >= 3
                        and _ensure_binary_y(d, v).notna().sum() >= 3
                        else np.nan,
                    ),
                    df,
                    iv,
                    dv,
                    iv_order,
                )

    # Permutation trend test (robust)
    results["Permutation trend test (robust)"] = safe_execute(permutation_cochran_armitage_test,
                df, iv, dv, iv_order=iv_order, n_permutations=n_permutations
            )

    return results


# =============================================================================
# CASE 9: ORDINAL IV, MULTINOMIAL DV
# =============================================================================


def linear_by_linear_association(df, iv, dv, iv_order=None, dv_order=None):
    """Linear-by-Linear Association test for ordinal IV vs ordinal DV."""
    iv_num, _ = _iv_to_numeric(df, iv, iv_order)
    dv_num, _ = _dv_to_numeric(df, dv, dv_order)
    mask = iv_num.notna() & dv_num.notna()
    if mask.sum() < 3:
        return {"p-value": np.nan}

    corr = np.corrcoef(iv_num[mask], dv_num[mask])[0, 1]
    n = mask.sum()
    chi2_stat = n * corr**2
    p = 1 - chi2.cdf(chi2_stat, 1)
    return {"p-value": float(p)}


def ordinal_logistic_regression(
    df, iv, dv, iv_order=None, dv_order=None, covariates=None
):
    """Ordinal Logistic Regression (using multinomial as fallback)."""
    covs = _ensure_covariates_list(covariates)
    df2 = df[[iv, dv] + covs].dropna()
    if df2.shape[0] < 10:
        return {"p-value": np.nan}

    iv_num, _ = _iv_to_numeric(df2, iv, iv_order)
    df2["_ivnum"] = iv_num

    if dv_order is not None:
        cat_type = pd.CategoricalDtype(categories=dv_order, ordered=True)
        df2["_dv_ord"] = df2[dv].astype(cat_type)
    else:
        df2["_dv_ord"] = pd.Categorical(df2[dv], ordered=True)

    rhs = ["_ivnum"] + [
        (
            f"C({c})"
            if df2[c].dtype == "O" or pd.api.types.is_categorical_dtype(df2[c])
            else c
        )
        for c in covs
    ]
    formula = "_dv_ord ~ " + " + ".join(rhs)
    df2["_dv_ord"] = df2["_dv_ord"].cat.codes  # convert to integer codes for MNLogit

    try:
        model = smf.mnlogit(formula, data=df2).fit(
            disp=False
        )  # fallback if ordinal_logit unavailable
        p = model.pvalues.loc["_ivnum"].min()
    except Exception:
        p = np.nan

    return {"p-value": float(p) if not pd.isna(p) else np.nan}


def run_all_tests_ordinalIV_categoricalDV(
    df, iv, dv, covariates=None, iv_order=None, dv_order=None
):
    """Run all tests for ordinal IV vs multinomial DV."""
    results = {}
    covs = _ensure_covariates_list(covariates) # Added this line
    results["Chi-square test"] = safe_execute(chi_square_test, df, iv, dv)
    results["Linear-by-Linear Association"] = safe_execute(linear_by_linear_association, df, iv, dv, iv_order, dv_order)
    results["Ordinal Logistic Regression"] = safe_execute(ordinal_logistic_regression, df, iv, dv, iv_order, dv_order, covariates)
    results["Multinomial Logistic Regression"] = safe_execute(multinomial_logistic_lrtest, df, iv, dv, covariates)
    return results