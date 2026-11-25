"""
General statistical analysis utilities for all study designs.

This module provides reusable functions that can be used across different study designs:
- Individual RCT
- Case-Control
- Cross-Sectional
- Controlled Before-After
- Crossover RCT
- Pre-Post
- Prospective
- Retrospective

All functions are designed to be flexible and adaptable to different variable naming conventions.
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


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _ensure_covariates_list(covariates):
    """
    Convert covariates to list format.

    Args:
        covariates: None, string, list, or tuple of covariate names

    Returns:
        list: List of covariate names (empty if None)
    """
    if covariates is None:
        return []
    if isinstance(covariates, (list, tuple)):
        return list(covariates)
    return [covariates]


def safe_execute(func, *args, **kwargs):
    """
    Executes a function and returns its result.
    If an exception occurs, returns a tuple with the function name (or custom name) and the error message.

    Args:
        func: The function to execute.
        *args: Positional arguments for the function.
        **kwargs: Keyword arguments for the function.

    Returns:
        The result of the function call, or ("Test Name", "Error: <error_message>").
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        # Try to infer a readable test name if possible
        name = getattr(func, "__name__", "Unknown Test").replace("_", " ").title()
        return (name, f"Error: {str(e)}")


def _iv_to_numeric(df, iv, iv_order=None):
    """
    Convert ordinal IV into numeric scores 0,1,2,... preserving order.

    Priority:
      1) If iv_order provided (list), use that ordering.
      2) If df[iv] is pandas.Categorical and ordered, use its codes.
      3) Else use unique values in the order they appear.

    Args:
        df: DataFrame containing the independent variable
        iv: Name of the independent variable column
        iv_order: Optional list specifying the order of categories

    Returns:
        tuple: (numeric_series (float), mapping dict)
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
    """
    Map ordinal DV to numeric scores 0,1,2...

    Args:
        df: DataFrame containing the dependent variable
        dv: Name of the dependent variable column
        dv_order: Optional list specifying the order of categories

    Returns:
        tuple: (numeric_series (float), mapping dict)
    """
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

    Args:
        df: DataFrame containing the dependent variable
        dv: Name of the dependent variable column

    Returns:
        Series: Binary variable as float (0.0 or 1.0)
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


def _build_covariate_formula_string(df, covariates):
    """
    Build a formula string for covariates, handling categorical variables.

    Args:
        df: DataFrame containing the covariates
        covariates: List of covariate column names

    Returns:
        str: Formula string for covariates (e.g., "C(age) + gender")
    """
    if not covariates:
        return ""

    cov_terms = []
    for c in covariates:
        if str(df[c].dtype) in ["object", "category"]:
            cov_terms.append(f"C({c})")
        else:
            cov_terms.append(c)

    return " + ".join(cov_terms)


# =============================================================================
# CATEGORICAL TESTS (Binary/Multinomial IVs)
# =============================================================================


def chi_square_test(df, iv_col, dv_col):
    """
    Chi-square test of independence for categorical variables.

    Works for any categorical IV x categorical DV combination.

    Args:
        df: DataFrame
        iv_col: Independent variable column name
        dv_col: Dependent variable column name

    Returns:
        tuple: (test_name, p_value)
    """
    table = pd.crosstab(df[iv_col], df[dv_col])
    chi2_stat, pval, dof, exp = chi2_contingency(table)
    return "Chi-square test of independence", pval


def fisher_test(df, iv_col, dv_col):
    """
    Fisher's exact test for 2x2 contingency tables.

    Args:
        df: DataFrame
        iv_col: Independent variable column name (must be binary)
        dv_col: Dependent variable column name (must be binary)

    Returns:
        tuple: (test_name, p_value)

    Raises:
        ValueError: If table is not 2x2
    """
    contingency = pd.crosstab(df[iv_col], df[dv_col])
    if contingency.shape != (2, 2):
        raise ValueError("Fisher's exact test is only valid for 2x2 tables.")

    oddsratio, pval = fisher_exact(contingency)
    return "Fisher's exact test (2x2)", pval


def fisher_exact_general(df, iv_col, dv_col):
    """
    Generalized Fisher's exact test for r x c tables.
    Falls back to scipy fisher_exact if 2x2.

    Args:
        df: DataFrame
        iv_col: Independent variable column name
        dv_col: Dependent variable column name

    Returns:
        tuple: (test_name, p_value)
    """
    contingency = pd.crosstab(df[iv_col], df[dv_col])

    if contingency.shape == (2, 2):
        _, pval = fisher_exact(contingency)
        return "Fisher's exact test (2x2)", pval
    else:
        table = Table(contingency)
        res = table.test_nominal_association()
        return "Generalized Fisher's exact test (r×c)", res.pvalue


def likelihood_ratio_test(df, iv, dv):
    """
    Likelihood Ratio Test (G-test) for categorical independence.

    Args:
        df: DataFrame
        iv: Independent variable column name
        dv: Dependent variable column name

    Returns:
        tuple: (test_name, p_value)
    """
    contingency = pd.crosstab(df[iv], df[dv])
    chi2_stat, p, dof, expected = chi2_contingency(
        contingency, lambda_="log-likelihood"
    )
    return "Likelihood Ratio Test (G-test)", p


# =============================================================================
# CONTINUOUS VARIABLE TESTS
# =============================================================================


def t_test(df, group_col, outcome_col):
    """
    Two-sample t-test (Welch's) for comparing means between two groups.

    Args:
        df: DataFrame
        group_col: Grouping variable column name (must be binary)
        outcome_col: Continuous outcome variable column name

    Returns:
        tuple: (test_name, p_value)

    Raises:
        ValueError: If group_col doesn't have exactly 2 unique values
    """
    groups = df[group_col].unique()
    if len(groups) != 2:
        raise ValueError("t-test requires exactly two groups")

    grp0 = df.loc[df[group_col] == groups[0], outcome_col]
    grp1 = df.loc[df[group_col] == groups[1], outcome_col]

    tstat, pval = stats.ttest_ind(grp1, grp0, equal_var=False, nan_policy="omit")
    return "Welch's t-test", pval


def mann_whitney_u_test(df, group_col, outcome_col):
    """
    Mann-Whitney U test (non-parametric alternative to t-test).

    Args:
        df: DataFrame
        group_col: Grouping variable column name (must be binary)
        outcome_col: Continuous outcome variable column name

    Returns:
        tuple: (test_name, p_value)

    Raises:
        ValueError: If group_col doesn't have exactly 2 unique values
    """
    groups = df[group_col].unique()
    if len(groups) != 2:
        raise ValueError("Mann-Whitney U test requires exactly two groups")

    group1 = df.loc[df[group_col] == groups[0], outcome_col]
    group2 = df.loc[df[group_col] == groups[1], outcome_col]

    stat, pval = mannwhitneyu(group1, group2, alternative="two-sided")
    return "Mann–Whitney U test", pval


def anova_test(df, group_col, outcome_col):
    """
    One-way ANOVA for comparing means across multiple groups.

    Args:
        df: DataFrame
        group_col: Grouping variable column name (2+ categories)
        outcome_col: Continuous outcome variable column name

    Returns:
        tuple: (test_name, p_value)
    """
    groups = [df.loc[df[group_col] == g, outcome_col] for g in df[group_col].unique()]
    fstat, pval = f_oneway(*groups)
    return "One-way ANOVA", pval


def kruskal_wallis_test(df, group_col, outcome_col):
    """
    Kruskal–Wallis test (non-parametric alternative to ANOVA).

    Args:
        df: DataFrame
        group_col: Grouping variable column name (2+ categories)
        outcome_col: Continuous outcome variable column name

    Returns:
        tuple: (test_name, p_value)
    """
    groups = [df.loc[df[group_col] == g, outcome_col] for g in df[group_col].unique()]
    hstat, pval = kruskal(*groups)
    return "Kruskal–Wallis test", pval


# =============================================================================
# CORRELATION TESTS
# =============================================================================


def pearson_correlation(df, iv, dv):
    """
    Pearson correlation for two continuous variables.

    Args:
        df: DataFrame
        iv: Independent variable column name
        dv: Dependent variable column name

    Returns:
        tuple: (test_name, p_value)
    """
    r, p = stats.pearsonr(df[iv], df[dv])
    return "Pearson correlation", p


def spearman_correlation(df, iv, dv):
    """
    Spearman rank correlation for ordinal or continuous variables.

    Args:
        df: DataFrame
        iv: Independent variable column name
        dv: Dependent variable column name

    Returns:
        tuple: (test_name, p_value)
    """
    rho, p = stats.spearmanr(df[iv], df[dv])
    return "Spearman correlation", p


def kendall_tau(df, iv, dv):
    """
    Kendall's tau correlation for ordinal or continuous variables.

    Args:
        df: DataFrame
        iv: Independent variable column name
        dv: Dependent variable column name

    Returns:
        tuple: (test_name, p_value)
    """
    tau, p = stats.kendalltau(df[iv], df[dv])
    return "Kendall's tau", p


def partial_correlation(df, iv, dv, covariates):
    """
    Partial correlation controlling for covariates.

    Args:
        df: DataFrame
        iv: Independent variable column name
        dv: Dependent variable column name
        covariates: List of covariate column names

    Returns:
        tuple: (test_name, p_value)
    """
    X = sm.add_constant(df[covariates])
    iv_resid = sm.OLS(df[iv], X).fit().resid
    dv_resid = sm.OLS(df[dv], X).fit().resid
    r, p = stats.pearsonr(iv_resid, dv_resid)
    return "Partial correlation", p


# =============================================================================
# REGRESSION MODELS
# =============================================================================


def ols_regression(df, iv_col, dv_col, covariates=None):
    """
    Ordinary Least Squares regression.

    Args:
        df: DataFrame
        iv_col: Independent variable column name
        dv_col: Dependent variable column name
        covariates: Optional list of covariate column names

    Returns:
        tuple: (test_name, p_value for IV)
    """
    covs = _ensure_covariates_list(covariates)
    cov_str = _build_covariate_formula_string(df, covs)

    formula = f"{dv_col} ~ {iv_col}"
    if cov_str:
        formula += " + " + cov_str

    model = smf.ols(formula, data=df).fit()
    pval = model.pvalues[iv_col]

    test_name = "OLS regression (adjusted)" if covs else "OLS regression (unadjusted)"
    return test_name, pval


def logistic_regression(df, iv_col, dv_col, covariates=None):
    """
    Logistic regression for binary dependent variable.

    Args:
        df: DataFrame
        iv_col: Independent variable column name
        dv_col: Dependent variable column name (must be binary)
        covariates: Optional list of covariate column names

    Returns:
        tuple: (test_name, p_value for IV)

    Raises:
        ValueError: If dv_col is not binary
    """
    if len(df[dv_col].unique()) != 2:
        raise ValueError("Logistic regression requires a binary outcome")

    covs = _ensure_covariates_list(covariates)
    cov_str = _build_covariate_formula_string(df, covs)

    formula = f"{dv_col} ~ {iv_col}"
    if cov_str:
        formula += " + " + cov_str

    model = smf.logit(formula, data=df).fit(disp=False)
    pval = model.pvalues[iv_col]

    return "Logistic regression", pval


def logistic_regression_categorical_iv(df, iv_col, dv_col, covariates=None):
    """
    Logistic regression with categorical IV (treats IV as categorical).

    Args:
        df: DataFrame
        iv_col: Independent variable column name (categorical)
        dv_col: Dependent variable column name (must be binary)
        covariates: Optional list of covariate column names

    Returns:
        tuple: (test_name, max p_value for IV terms)
    """
    covs = _ensure_covariates_list(covariates)
    cov_str = _build_covariate_formula_string(df, covs)

    formula = f"{dv_col} ~ C({iv_col})"
    if cov_str:
        formula += " + " + cov_str

    model = smf.logit(formula, data=df).fit(disp=False)
    # Get max p-value for IV terms (excluding intercept and covariates)
    pval = model.pvalues[1:].max()

    return "Logistic regression (categorical IV)", pval


def multinomial_logistic_regression(df, iv_col, dv_col, covariates=None):
    """
    Multinomial logistic regression for multinomial dependent variable.

    Args:
        df: DataFrame
        iv_col: Independent variable column name
        dv_col: Dependent variable column name (3+ categories)
        covariates: Optional list of covariate column names

    Returns:
        tuple: (test_name, min p_value across outcome levels)

    Raises:
        ValueError: If dv_col has fewer than 3 categories
    """
    if len(df[dv_col].unique()) < 3:
        raise ValueError("Multinomial logistic regression requires >= 3 outcome levels")

    covs = _ensure_covariates_list(covariates)
    cov_str = _build_covariate_formula_string(df, covs)

    # Convert outcome to numerical category if not already
    df_copy = df.copy()
    df_copy[dv_col] = df_copy[dv_col].astype("category").cat.codes

    formula = f"{dv_col} ~ {iv_col}"
    if cov_str:
        formula += " + " + cov_str

    model = smf.mnlogit(formula, data=df_copy).fit(disp=False)
    pval = model.pvalues.loc[iv_col].min()  # min p across outcome levels

    return "Multinomial logistic regression", pval


def anova_with_covariates(df, iv_col, dv_col, covariates):
    """
    ANCOVA (ANOVA with covariates) using OLS regression.

    Args:
        df: DataFrame
        iv_col: Independent variable column name (categorical)
        dv_col: Dependent variable column name (continuous)
        covariates: List of covariate column names

    Returns:
        tuple: (test_name, p_value for IV effect)
    """
    covs = _ensure_covariates_list(covariates)
    cov_str = _build_covariate_formula_string(df, covs)

    formula = f"{dv_col} ~ C({iv_col})"
    if cov_str:
        formula += " + " + cov_str

    model = smf.ols(formula, data=df).fit()
    anova_table = anova_lm(model, typ=2)
    pval = anova_table.loc[f"C({iv_col})", "PR(>F)"]

    return "ANCOVA (adjusted)", pval


def robust_regression(df, iv_col, dv_col, covariates=None):
    """
    Robust linear regression using Huber's M-estimator.

    Args:
        df: DataFrame
        iv_col: Independent variable column name
        dv_col: Dependent variable column name (continuous)
        covariates: Optional list of covariate column names

    Returns:
        tuple: (test_name, p_value for IV)
    """
    covs = _ensure_covariates_list(covariates)

    # Handle categorical variables
    df_copy = df.copy()
    for col in [iv_col] + covs:
        if df_copy[col].dtype.name in ["category", "object"]:
            df_copy[col] = df_copy[col].astype("category").cat.codes

    predictors = [iv_col] + covs
    X = sm.add_constant(df_copy[predictors])
    y = df_copy[dv_col]

    model = sm.RLM(y, X, M=sm.robust.norms.HuberT()).fit()
    pval = model.pvalues[iv_col]

    return "Robust regression (Huber)", pval


def quantile_regression(df, iv_col, dv_col, covariates=None, quantile=0.5):
    """
    Quantile regression (default: median regression).

    Args:
        df: DataFrame
        iv_col: Independent variable column name
        dv_col: Dependent variable column name (continuous)
        covariates: Optional list of covariate column names
        quantile: Quantile to estimate (default 0.5 for median)

    Returns:
        tuple: (test_name, p_value for IV)
    """
    from statsmodels.regression.quantile_regression import QuantReg

    covs = _ensure_covariates_list(covariates)

    # Handle categorical variables
    df_copy = df.copy()
    for col in [iv_col] + covs:
        if df_copy[col].dtype.name in ["category", "object"]:
            df_copy[col] = df_copy[col].astype("category").cat.codes

    predictors = [iv_col] + covs
    formula = f"{dv_col} ~ {' + '.join(predictors)}"

    model = QuantReg.from_formula(formula, df_copy)
    res = model.fit(q=quantile)
    pval = res.pvalues[iv_col]

    return f"Quantile regression (q={quantile})", pval


# =============================================================================
# ORDINAL VARIABLE TESTS
# =============================================================================


def spearman_test(df, iv, dv, iv_order=None):
    """
    Spearman rank correlation between ordinal IV and continuous DV.

    Args:
        df: DataFrame
        iv: Independent variable column name (ordinal)
        dv: Dependent variable column name (continuous)
        iv_order: Optional list specifying order of IV categories

    Returns:
        tuple: (test_name, p_value)
    """
    x, mapping = _iv_to_numeric(df, iv, iv_order)
    y = pd.to_numeric(df[dv], errors="coerce")
    mask = x.notna() & y.notna()

    if mask.sum() < 3:
        return ("Spearman Rank Correlation", np.nan)

    rho, p = stats.spearmanr(x[mask], y[mask])
    return ("Spearman Rank Correlation", float(p))


def cochran_armitage_test(df, iv, dv, iv_order=None):
    """
    Cochran-Armitage test for linear trend in proportions across ordered groups.

    Args:
        df: DataFrame
        iv: Independent variable column name (ordinal)
        dv: Dependent variable column name (binary)
        iv_order: Optional list specifying order of IV categories

    Returns:
        tuple: (test_name, p_value)
    """
    df2 = df[[iv, dv]].copy()
    y = _ensure_binary_y(df2, dv)
    df2["_y"] = y
    df2 = df2.dropna(subset=["_y"])

    if df2["_y"].nunique() == 0:
        return ("Cochran-Armitage Trend Test", np.nan)

    scores, mapping = _iv_to_numeric(df2, iv, iv_order)
    df2["_score"] = scores
    counts = (
        df2.groupby(iv)["_y"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "successes", "count": "n"})
    )

    if len(counts) < 2:
        return ("Cochran-Armitage Trend Test", np.nan)

    successes = counts["successes"].values.astype(float)
    ns = counts["n"].values.astype(float)
    scores_arr = np.array([mapping[level] for level in counts.index])
    N = ns.sum()

    if N == 0:
        return ("Cochran-Armitage Trend Test", np.nan)

    p_hat = successes.sum() / N
    numer = np.sum(scores_arr * (successes - ns * p_hat))
    denom = np.sqrt(
        p_hat
        * (1 - p_hat)
        * (np.sum(ns * (scores_arr**2)) - (np.sum(ns * scores_arr) ** 2) / N)
    )

    if denom == 0:
        return ("Cochran-Armitage Trend Test", np.nan)

    z = numer / denom
    p_two = 2 * (1 - norm.cdf(abs(z)))

    return ("Cochran-Armitage Trend Test", float(p_two))


def linear_by_linear_association(df, iv, dv, iv_order=None, dv_order=None):
    """
    Linear-by-Linear Association test for ordinal IV vs ordinal DV.

    Args:
        df: DataFrame
        iv: Independent variable column name (ordinal)
        dv: Dependent variable column name (ordinal)
        iv_order: Optional list specifying order of IV categories
        dv_order: Optional list specifying order of DV categories

    Returns:
        tuple: (test_name, p_value)
    """
    iv_num, _ = _iv_to_numeric(df, iv, iv_order)
    dv_num, _ = _dv_to_numeric(df, dv, dv_order)
    mask = iv_num.notna() & dv_num.notna()

    if mask.sum() < 3:
        return ("Linear-by-Linear Association", np.nan)

    corr = np.corrcoef(iv_num[mask], dv_num[mask])[0, 1]
    n = mask.sum()
    chi2_stat = n * corr**2
    p = 1 - chi2.cdf(chi2_stat, 1)

    return ("Linear-by-Linear Association", float(p))


# =============================================================================
# UTILITY FUNCTIONS FOR DATA GENERATION
# =============================================================================


def generate_binary_data(n=1000, effect_size=0.4, seed=42):
    """
    Generate synthetic binary IV and binary DV data.

    Args:
        n: Sample size
        effect_size: Size of the treatment effect
        seed: Random seed for reproducibility

    Returns:
        DataFrame with binary IV and DV
    """
    np.random.seed(seed)

    iv = np.random.binomial(1, 0.5, n)
    dv = np.random.binomial(1, 0.3 + effect_size * iv, n)

    return pd.DataFrame(
        {
            "treatment": iv,
            "outcome": dv,
            "age": np.random.randint(20, 70, n),
            "sex": np.random.choice(["M", "F"], n),
        }
    )


def generate_continuous_data(n=1000, effect_size=2.0, seed=42):
    """
    Generate synthetic binary IV and continuous DV data.

    Args:
        n: Sample size
        effect_size: Size of the treatment effect
        seed: Random seed for reproducibility

    Returns:
        DataFrame with binary IV and continuous DV
    """
    np.random.seed(seed)

    iv = np.random.binomial(1, 0.5, n)
    dv = np.random.normal(loc=5 + effect_size * iv, scale=1, size=n)

    return pd.DataFrame(
        {
            "treatment": iv,
            "outcome": dv,
            "age": np.random.randint(20, 70, n),
            "sex": np.random.choice(["M", "F"], n),
        }
    )


def generate_categorical_data(n=1000, n_categories=3, seed=42):
    """
    Generate synthetic categorical IV and binary or categorical DV data.

    Args:
        n: Sample size
        n_categories: Number of categories for IV
        seed: Random seed for reproducibility

    Returns:
        DataFrame with categorical IV and outcome
    """
    np.random.seed(seed)

    iv = np.random.choice(range(n_categories), n)
    dv = np.random.choice(["A", "B", "C"], n)

    return pd.DataFrame(
        {
            "treatment": iv,
            "outcome": dv,
            "age": np.random.randint(20, 70, n),
            "sex": np.random.choice(["M", "F"], n),
        }
    )


# =============================================================================
# MAIN TEST DISPATCHER
# =============================================================================


def select_appropriate_test(
    df,
    iv_col,
    dv_col,
    iv_type="binary",
    dv_type="continuous",
    covariates=None,
    iv_order=None,
    dv_order=None,
):
    """
    Automatically select and run appropriate statistical test based on variable types.

    Args:
        df: DataFrame
        iv_col: Independent variable column name
        dv_col: Dependent variable column name
        iv_type: Type of IV ('binary', 'categorical', 'ordinal', 'continuous')
        dv_type: Type of DV ('binary', 'categorical', 'ordinal', 'continuous')
        covariates: Optional list of covariate column names
        iv_order: Optional ordering for ordinal IV
        dv_order: Optional ordering for ordinal DV

    Returns:
        dict: Dictionary of test results
    """
    results = {}

    # Binary IV, Continuous DV
    if iv_type == "binary" and dv_type == "continuous":
        results["t_test"] = t_test(df, iv_col, dv_col)
        results["mann_whitney"] = mann_whitney_u_test(df, iv_col, dv_col)
        results["ols"] = ols_regression(df, iv_col, dv_col, covariates)

    # Binary IV, Binary DV
    elif iv_type == "binary" and dv_type == "binary":
        results["chi_square"] = chi_square_test(df, iv_col, dv_col)
        results["fisher"] = fisher_test(df, iv_col, dv_col)
        results["logistic"] = logistic_regression(df, iv_col, dv_col, covariates)

    # Categorical IV, Continuous DV
    elif iv_type == "categorical" and dv_type == "continuous":
        results["anova"] = anova_test(df, iv_col, dv_col)
        results["kruskal_wallis"] = kruskal_wallis_test(df, iv_col, dv_col)
        if covariates:
            results["ancova"] = anova_with_covariates(df, iv_col, dv_col, covariates)

    # Categorical IV, Binary DV
    elif iv_type == "categorical" and dv_type == "binary":
        results["chi_square"] = chi_square_test(df, iv_col, dv_col)
        results["fisher_general"] = fisher_exact_general(df, iv_col, dv_col)
        results["logistic_cat"] = logistic_regression_categorical_iv(
            df, iv_col, dv_col, covariates
        )

    # Continuous IV, Continuous DV
    elif iv_type == "continuous" and dv_type == "continuous":
        results["pearson"] = pearson_correlation(df, iv_col, dv_col)
        results["spearman"] = spearman_correlation(df, iv_col, dv_col)
        results["ols"] = ols_regression(df, iv_col, dv_col, covariates)
        if covariates:
            results["partial_corr"] = partial_correlation(
                df, iv_col, dv_col, covariates
            )

    # Ordinal IV, Continuous DV
    elif iv_type == "ordinal" and dv_type == "continuous":
        results["spearman"] = spearman_test(df, iv_col, dv_col, iv_order)
        results["kruskal_wallis"] = kruskal_wallis_test(df, iv_col, dv_col)

    # Ordinal IV, Binary DV
    elif iv_type == "ordinal" and dv_type == "binary":
        results["cochran_armitage"] = cochran_armitage_test(
            df, iv_col, dv_col, iv_order
        )
        results["chi_square"] = chi_square_test(df, iv_col, dv_col)

    # Ordinal IV, Ordinal DV
    elif iv_type == "ordinal" and dv_type == "ordinal":
        results["linear_by_linear"] = linear_by_linear_association(
            df, iv_col, dv_col, iv_order, dv_order
        )
        results["spearman"] = spearman_correlation(df, iv_col, dv_col)

    # Multinomial cases
    elif dv_type in ["categorical", "multinomial"]:
        results["chi_square"] = chi_square_test(df, iv_col, dv_col)
        if len(df[dv_col].unique()) >= 3:
            results["multinomial_logistic"] = multinomial_logistic_regression(
                df, iv_col, dv_col, covariates
            )

    return results


if __name__ == "__main__":
    # Example usage
    print("=" * 70)
    print("GENERAL STATISTICAL UTILITIES - EXAMPLE USAGE")
    print("=" * 70)

    # Generate sample data
    df_binary = generate_binary_data(n=200)
    df_continuous = generate_continuous_data(n=200)

    print("\n1. Binary IV × Binary DV:")
    print("-" * 50)
    results = select_appropriate_test(
        df_binary, "treatment", "outcome", iv_type="binary", dv_type="binary"
    )
    for test_name, (desc, pval) in results.items():
        print(f"{desc}: p = {pval:.4f}")

    print("\n2. Binary IV × Continuous DV:")
    print("-" * 50)
    results = select_appropriate_test(
        df_continuous,
        "treatment",
        "outcome",
        iv_type="binary",
        dv_type="continuous",
        covariates=["age"],
    )
    for test_name, (desc, pval) in results.items():
        print(f"{desc}: p = {pval:.4f}")
