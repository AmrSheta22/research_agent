"""
Statistical analysis utilities for Retrospective study designs.

This module provides functions for analyzing different combinations of
independent and dependent variables in retrospective studies.

Cases covered:
1. Continuous IV, Continuous DV
2. Continuous IV, Binary DV
3. Continuous IV, Multinomial DV
4. Binary IV, Continuous DV
5. Binary IV, Binary DV
6. Binary IV, Multinomial DV
7. Multinomial IV, Continuous DV
8. Multinomial IV, Binary DV
9. Multinomial IV, Multinomial DV
"""

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
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


def _ensure_binary_y(df, dv):
    """
    Ensure DV is 0/1 numeric.
    """
    y = df[dv]
    if pd.api.types.is_numeric_dtype(y):
        unique_vals = pd.Series(y.dropna().unique())
        if set(unique_vals.astype(str)) <= set(["0", "1", "0.0", "1.0"]):
            return pd.to_numeric(y, errors="coerce").astype(float)

    # Map to 0/1 if str/category
    uniq = pd.Series(y.dropna().unique())
    if len(uniq) != 2:
        # Fallback or error? For now return None or handle downstream
        return None

    # Simple mapping: first alphabetic/numeric -> 0, second -> 1 (or other way)
    # Ideally user knows which is which, but standardized mapping:
    # We sort to be deterministic
    uniq = sorted(uniq)
    mapping = {uniq[0]: 0.0, uniq[1]: 1.0}
    return y.map(mapping).astype(float)


# =============================================================================
# CASE 1: CONTINUOUS IV, CONTINUOUS DV
# =============================================================================


def pearson_correlation(df, iv, dv):
    """Pearson correlation coefficient."""
    df_clean = df[[iv, dv]].dropna()
    if len(df_clean) < 2:
        return ("Pearson Correlation", np.nan)
    r, p = stats.pearsonr(df_clean[iv], df_clean[dv])
    return {"p-value": float(p)}


def spearman_correlation(df, iv, dv):
    """Spearman rank correlation."""
    df_clean = df[[iv, dv]].dropna()
    if len(df_clean) < 2:
        return ("Spearman Correlation", np.nan)
    r, p = stats.spearmanr(df_clean[iv], df_clean[dv])
    return {"p-value": float(p)}


def linear_regression_continuous(df, iv, dv, covariates=None):
    """Linear Regression (OLS)."""
    covs = _ensure_covariates_list(covariates)
    formula = f"{dv} ~ {iv}"
    if covs:
        # Add covariates, wrapping categoricals in C()
        cov_terms = []
        for c in covs:
            if df[c].dtype == "O" or pd.api.types.is_categorical_dtype(df[c]):
                cov_terms.append(f"C({c})")
            else:
                cov_terms.append(c)
        formula += " + " + " + ".join(cov_terms)

    try:
        model = smf.ols(formula, data=df).fit()
        p = model.pvalues[iv]
        return {"p-value": float(p)}
    except Exception:
        return {"p-value": np.nan}


def run_all_tests_continuousIV_continuousDV(df, iv, dv, covariates=None):
    results = {}
    results["Person's Correlation"] = safe_execute(pearson_correlation, df, iv, dv)
    results["Spearman Correlation"] = safe_execute(spearman_correlation, df, iv, dv)
    results["Linear Regression"] = safe_execute(linear_regression_continuous, df, iv, dv, covariates)
    return results


# =============================================================================
# CASE 2: CONTINUOUS IV, BINARY DV
# =============================================================================


def logistic_regression_continuous_iv(df, iv, dv, covariates=None):
    """Logistic Regression for Continuous IV."""
    df_clean = df.copy()
    y = _ensure_binary_y(df_clean, dv)
    if y is None:
        return {"p-value": np.nan}
    df_clean["_y_bin"] = y

    covs = _ensure_covariates_list(covariates)
    formula = f"_y_bin ~ {iv}"
    if covs:
        cov_terms = []
        for c in covs:
            if df[c].dtype == "O" or pd.api.types.is_categorical_dtype(df[c]):
                cov_terms.append(f"C({c})")
            else:
                cov_terms.append(c)
        formula += " + " + " + ".join(cov_terms)

    try:
        model = smf.logit(formula, data=df_clean).fit(disp=False)
        p = model.pvalues[iv]
        return {"p-value": float(p)}
    except Exception:
        return {"p-value": np.nan}


def point_biserial(df, iv, dv):
    """Point-Biserial Correlation (equivalent to Pearson for Binary x Continuous)."""
    df_clean = df.copy()
    y = _ensure_binary_y(df_clean, dv)
    if y is None:
        return {"p-value": np.nan}

    mask = df_clean[iv].notna() & y.notna()
    if mask.sum() < 2:
        return {"p-value": np.nan}

    r, p = stats.pearsonr(df_clean.loc[mask, iv], y[mask])
    return {"p-value": float(p)}


def mann_whitney_continuous_iv(df, iv, dv):
    """Mann-Whitney U Test (Continuous IV split by Binary DV)."""
    df_clean = df.copy()
    y = _ensure_binary_y(df_clean, dv)
    if y is None:
        return {"p-value": np.nan}

    groups = y.unique()
    if len(groups) != 2:
        return {"p-value": np.nan}

    g1 = df_clean.loc[y == groups[0], iv].dropna()
    g2 = df_clean.loc[y == groups[1], iv].dropna()

    stat, p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
    return {"p-value": float(p)}


def run_all_tests_continuousIV_binaryDV(df, iv, dv, covariates=None):
    results = {}
    results["Logistic Regression"] = safe_execute(logistic_regression_continuous_iv, df, iv, dv, covariates)
    results["Point Biserial"] = safe_execute(point_biserial, df, iv, dv)
    results["Mann-Whitney"] = safe_execute(mann_whitney_continuous_iv, df, iv, dv)
    return results


# =============================================================================
# CASE 3: CONTINUOUS IV, MULTINOMIAL DV
# =============================================================================


def multinomial_logistic_continuous_iv(df, iv, dv, covariates=None):
    """Multinomial Logistic Regression."""
    df_clean = df.copy()
    # Ensure DV is categorical codes
    df_clean[dv] = df_clean[dv].astype("category").cat.codes

    covs = _ensure_covariates_list(covariates)
    formula = f"{dv} ~ {iv}"
    if covs:
        cov_terms = []
        for c in covs:
            if df[c].dtype == "O" or pd.api.types.is_categorical_dtype(df[c]):
                cov_terms.append(f"C({c})")
            else:
                cov_terms.append(c)
        formula += " + " + " + ".join(cov_terms)

    try:
        model = smf.mnlogit(formula, data=df_clean).fit(disp=False)
        # p-value for IV. mnlogit gives p-values for each level vs base.
        # We can take the min p-value or do a Likelihood Ratio Test.
        # Taking min is a simple heuristic effectively testing "is it significant for ANY level?"
        # A more rigorous approach is LR test, but similar to other files, we'll return p-value(s).
        # However, to return a SINGLE scalar p-value for the framework:
        # We will use the Likelihood Ratio Test if possible, or min p-value.
        # Let's use the min p-value of the IV coefficient across all equations.
        p = model.pvalues.loc[iv].min()
        return {"p-value": float(p)}
    except Exception:
        return {"p-value": np.nan}


def lda_anova_approx(df, iv, dv):
    """
    Linear Discriminant Analysis approach (ANOVA on IV across DV groups).
    Testing if IV means differ across DV categories.
    """
    groups = []
    for g in df[dv].unique():
        groups.append(df.loc[df[dv] == g, iv].dropna())

    if len(groups) < 2:
        return {"p-value": np.nan}

    f, p = stats.f_oneway(*groups)
    return {"p-value": float(p)}


def run_all_tests_continuousIV_multinomialDV(df, iv, dv, covariates=None):
    results = {}
    results["Multinomial Logistic Regression"] = safe_execute(multinomial_logistic_continuous_iv, df, iv, dv, covariates)
    results["LDA ANOVA Approx"] = safe_execute(lda_anova_approx, df, iv, dv)
    return results


# =============================================================================
# CASE 4: BINARY IV, CONTINUOUS DV
# =============================================================================


def t_test_binary_iv(df, iv, dv):
    """Independent T-Test."""
    groups = df[iv].unique()
    if len(groups) != 2:
        return {"p-value": np.nan}

    g1 = df.loc[df[iv] == groups[0], dv].dropna()
    g2 = df.loc[df[iv] == groups[1], dv].dropna()

    t, p = stats.ttest_ind(g1, g2, equal_var=False)  # Welch's
    return {"p-value": float(p)}


def mann_whitney_binary_iv(df, iv, dv):
    """Mann-Whitney U Test."""
    groups = df[iv].unique()
    if len(groups) != 2:
        return {"p-value": np.nan}

    g1 = df.loc[df[iv] == groups[0], dv].dropna()
    g2 = df.loc[df[iv] == groups[1], dv].dropna()

    u, p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
    return {"p-value": float(p)}


def linear_regression_binary_iv(df, iv, dv, covariates=None):
    """Linear Regression with Binary IV."""
    # Logic is same as continuous IV basically, but IV is 0/1 or categorical
    covs = _ensure_covariates_list(covariates)
    formula = f"{dv} ~ C({iv})"
    if covs:
        cov_terms = []
        for c in covs:
            if df[c].dtype == "O" or pd.api.types.is_categorical_dtype(df[c]):
                cov_terms.append(f"C({c})")
            else:
                cov_terms.append(c)
        formula += " + " + " + ".join(cov_terms)

    try:
        model = smf.ols(formula, data=df).fit()
        # p-value for the IV term. It will be "C(iv)[T.val]"
        # Find the term starting with C(iv)
        iv_term = [x for x in model.pvalues.index if f"C({iv})" in x]
        if iv_term:
            p = model.pvalues[iv_term[0]]
            return {"p-value": float(p)}
        return {"p-value": np.nan}
    except Exception:
        return {"p-value": np.nan}


def run_all_tests_binaryIV_continuousDV(df, iv, dv, covariates=None):
    results = {}
    results["T-Test"] = safe_execute(t_test_binary_iv, df, iv, dv)
    results["Mann-Whitney"] = safe_execute(mann_whitney_binary_iv, df, iv, dv)
    results["Linear Regression"] = safe_execute(linear_regression_binary_iv, df, iv, dv, covariates)
    return results


# =============================================================================
# CASE 5: BINARY IV, BINARY DV
# =============================================================================


def chi_square_binary(df, iv, dv):
    """Chi-Square Test."""
    ct = pd.crosstab(df[iv], df[dv])
    if ct.size == 0:
        return {"p-value": np.nan}
    chi2, p, dof, ex = stats.chi2_contingency(ct)
    return {"p-value": float(p)}


def fisher_exact_binary(df, iv, dv):
    """Fisher's Exact Test."""
    ct = pd.crosstab(df[iv], df[dv])
    if ct.shape != (2, 2):
        return {"p-value": np.nan}
    odd, p = stats.fisher_exact(ct)
    return {"p-value": float(p)}


def odds_ratio_test(df, iv, dv):
    """Odds Ratio (and Confidence Interval check significance, returned as p-value placeholder or just description).
    For uniformity, we'll return the p-value from the Wald test of the Odds Ratio (log-odds).
    Actually, Fisher's Exact Test p-value is practically the significance of the OR.
    Let's return the OR value itself in the name if possible or just rely on Fisher/Logistic.
    Here we'll calculate OR and return it as a metric, but the function signature expects (test_name, p_value).
    So we'll stick to statistical tests.
    We can add a 'Logistic Regression' which gives significance of the OR.
    """
    # Just returning Logistic Regression p-value is better for 'significance'.
    # But let's keep consistency.
    return ("Odds Ratio Significance (via Fisher)", fisher_exact_binary(df, iv, dv)[1])


def logistic_regression_binary_iv(df, iv, dv, covariates=None):
    """Logistic Regression."""
    df_clean = df.copy()
    y = _ensure_binary_y(df_clean, dv)
    if y is None:
        return {"p-value": np.nan}
    df_clean["_y_bin"] = y

    covs = _ensure_covariates_list(covariates)
    formula = f"_y_bin ~ C({iv})"
    if covs:
        cov_terms = []
        for c in covs:
            if df[c].dtype == "O" or pd.api.types.is_categorical_dtype(df[c]):
                cov_terms.append(f"C({c})")
            else:
                cov_terms.append(c)
        formula += " + " + " + ".join(cov_terms)

    try:
        model = smf.logit(formula, data=df_clean).fit(disp=False)
        iv_term = [x for x in model.pvalues.index if f"C({iv})" in x]
        if iv_term:
            p = model.pvalues[iv_term[0]]
            return {"p-value": float(p)}
        return {"p-value": np.nan}
    except Exception:
        return {"p-value": np.nan}


def run_all_tests_binaryIV_binaryDV(df, iv, dv, covariates=None):
    results = {}
    results["Chi-Square"] = safe_execute(chi_square_binary, df, iv, dv)
    results["Fisher's Exact"] = safe_execute(fisher_exact_binary, df, iv, dv)
    results["Logistic Regression"] = safe_execute(logistic_regression_binary_iv, df, iv, dv, covariates)
    return results


# =============================================================================
# CASE 6: BINARY IV, MULTINOMIAL DV
# =============================================================================


def chi_square_multinomial_dv(df, iv, dv):
    """Chi-Square Test."""
    ct = pd.crosstab(df[iv], df[dv])
    if ct.size == 0:
        return {"p-value": np.nan}
    chi2, p, dof, ex = stats.chi2_contingency(ct)
    return {"p-value": float(p)}


def cramers_v_binary_iv(df, iv, dv):
    """
    Cramer's V for nominal association.
    """
    contingency_table = pd.crosstab(df[iv], df[dv])
    chi2, _, _, _ = stats.chi2_contingency(contingency_table)
    n = contingency_table.sum().sum()
    phi2 = chi2 / n
    r, k = contingency_table.shape
    with np.errstate(divide="ignore", invalid="ignore"):
        phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
        rcorr = r - ((r - 1) ** 2) / (n - 1)
        kcorr = k - ((k - 1) ** 2) / (n - 1)
        v = np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))
    return {"V": v}

def multinomial_logistic_binary_iv(df, iv, dv, covariates=None):
    """Multinomial Logistic Regression."""
    df_clean = df.copy()
    df_clean[dv] = df_clean[dv].astype("category").cat.codes

    covs = _ensure_covariates_list(covariates)
    formula = f"{dv} ~ C({iv})"
    if covs:
        cov_terms = []
        for c in covs:
            if df[c].dtype == "O" or pd.api.types.is_categorical_dtype(df[c]):
                cov_terms.append(f"C({c})")
            else:
                cov_terms.append(c)
        formula += " + " + " + ".join(cov_terms)

    try:
        model = smf.mnlogit(formula, data=df_clean).fit(disp=False)
        # Check IV terms
        iv_terms = [x for x in model.pvalues.index if f"C({iv})" in x]
        if iv_terms:
            # We have one term per variety (k-1).
            # We can return the min p-value
            p = model.pvalues.loc[iv_terms].min().min()  # min across levels and terms
            return {"p-value": float(p)}
        return {"p-value": np.nan}
    except Exception:
        return {"p-value": np.nan}


def run_all_tests_binaryIV_multinomialDV(df, iv, dv, covariates=None):
    results = {}
    results["Chi-Square"] = safe_execute(chi_square_multinomial_dv, df, iv, dv)
    results["Multinomial Logistic Regression"] = safe_execute(multinomial_logistic_binary_iv, df, iv, dv, covariates)
    return results


# =============================================================================
# CASE 7: MULTINOMIAL IV, CONTINUOUS DV
# =============================================================================


def anova_multinomial_iv(df, iv, dv):
    """One-Way ANOVA."""
    groups = []
    for g in df[iv].unique():
        groups.append(df.loc[df[iv] == g, dv].dropna())

    if len(groups) < 2:
        print("ANOVA: Not enough groups")
        return {"p-value": np.nan}

    f, p = stats.f_oneway(*groups)
    print("ANOVA: p-value ", p)
    return {"p-value": float(p)}


def kruskal_wallis_multinomial_iv(df, iv, dv):
    """Kruskal-Wallis Test."""
    groups = []
    for g in df[iv].unique():
        groups.append(df.loc[df[iv] == g, dv].dropna())

    if len(groups) < 2:
        return {"p-value": np.nan}

    h, p = stats.kruskal(*groups)
    return {"p-value": float(p)}


def linear_regression_multinomial_iv(df, iv, dv, covariates=None):
    """Linear Regression (ANCOVA)."""
    covs = _ensure_covariates_list(covariates)
    formula = f"{dv} ~ C({iv})"
    if covs:
        cov_terms = []
        for c in covs:
            if df[c].dtype == "O" or pd.api.types.is_categorical_dtype(df[c]):
                cov_terms.append(f"C({c})")
            else:
                cov_terms.append(c)
        formula += " + " + " + ".join(cov_terms)

    try:
        model = smf.ols(formula, data=df).fit()
        # F-test for the categorical IV as a block is better, but simple approach:
        # Check min p-value of the dummy variables for IV
        iv_terms = [x for x in model.pvalues.index if f"C({iv})" in x]
        if iv_terms:
            p = model.pvalues[iv_terms].min()
            return {"p-value": float(p)}
        return {"p-value": np.nan}
    except Exception:
        return {"p-value": np.nan}


def run_all_tests_multinomialIV_continuousDV(df, iv, dv, covariates=None):
    results = {}
    results["ANOVA"] = safe_execute(anova_multinomial_iv, df, iv, dv)
    results["Kruskal-Wallis"] = safe_execute(kruskal_wallis_multinomial_iv, df, iv, dv)
    results["Linear Regression"] = safe_execute(linear_regression_multinomial_iv, df, iv, dv, covariates)
    return results


# =============================================================================
# CASE 8: MULTINOMIAL IV, BINARY DV
# =============================================================================


def run_all_tests_multinomialIV_binaryDV(df, iv, dv, covariates=None):
    # Same logic as Binary IV x Binary DV, basically Chi2 and Logistic works for any categorical IV.
    # We rename to fit the case.
    results = {}

    # helper wrapper for renaming
    def renamed_chi2(d, i, v):
        res = chi_square_binary(d, i, v)
        return {"p-value": res["p-value"]}

    results["Chi-square test"] = safe_execute(renamed_chi2, df, iv, dv)
    results["Logistic Regression"] = safe_execute(logistic_regression_binary_iv, df, iv, dv, covariates)
    return results


# =============================================================================
# CASE 9: MULTINOMIAL IV, MULTINOMIAL DV
# =============================================================================


def run_all_tests_multinomialIV_multinomialDV(df, iv, dv, covariates=None):
    results = {}

    # helper wrapper for renaming
    def renamed_chi2(d, i, v):
        res = chi_square_multinomial_dv(d, i, v)
        return {"p-value": res["p-value"]}

    results["Chi-square test"] = safe_execute(renamed_chi2, df, iv, dv)
    results["Multinomial Logistic Regression"] = safe_execute(multinomial_logistic_binary_iv, df, iv, dv, covariates)
    return results


if __name__ == "__main__":
    # Minimal verification
    # 1. Continuous IV, Continuous DV
    df1 = pd.DataFrame(
        {"iv": np.random.normal(0, 1, 100), "dv": np.random.normal(0, 1, 100)}
    )
    print("Case 1:", run_all_tests_continuousIV_continuousDV(df1, "iv", "dv"))

    # 2. Continuous IV, Binary DV
    df2 = pd.DataFrame(
        {"iv": np.random.normal(0, 1, 100), "dv": np.random.randint(0, 2, 100)}
    )
    print("Case 2:", run_all_tests_continuousIV_binaryDV(df2, "iv", "dv"))

    # 4. Binary IV, Continuous DV
    df4 = pd.DataFrame(
        {"iv": np.random.randint(0, 2, 100), "dv": np.random.normal(0, 1, 100)}
    )
    print("Case 4:", run_all_tests_binaryIV_continuousDV(df4, "iv", "dv"))

    print("Verification complete.")
