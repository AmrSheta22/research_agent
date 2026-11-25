import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon, binomtest, chi2, norm
from sklearn.utils import shuffle
import statsmodels.formula.api as smf
import statsmodels.api as sm
from itertools import permutations
from .utils_general_functions import safe_execute

# 4 Cases
# Case 1: Continuous IV vs Continuous DV
# Case 2: Binary IV vs Binary DV
# Case 3: Multinomial IV vs Multinomial DV
# Case 4: Ordinal IV vs Ordinal DV


# Case 1: Continuous IV vs Continuous DV
def paired_t_test(df, pre, post, covariates=None):
    x = df[pre].astype(float)
    y = df[post].astype(float)
    stat, p = ttest_rel(y, x, nan_policy="omit")
    return {"p-value": p, "statistic": stat}


def wilcoxon_signed_rank_test(df, pre, post, covariates=None):
    x = df[pre].astype(float)
    y = df[post].astype(float)
    stat, p = wilcoxon(y, x, zero_method="wilcox", correction=False)
    return {"p-value": p, "statistic": stat}


def sign_test(df, pre, post, covariates=None):
    diff = (df[post] - df[pre]).dropna()
    diff = diff[diff != 0]  # Drop ties
    n_pos = np.sum(diff > 0)
    n = len(diff)
    res = binomtest(n_pos, n, p=0.5, alternative="two-sided")
    return {"p-value": res.pvalue, "statistic": res.statistic}


def paired_permutation_test(df, pre, post, covariates=None, n_permutations=10000):
    diff = (df[post] - df[pre]).dropna().values
    observed = np.mean(diff)
    count = 0
    for _ in range(n_permutations):
        signs = np.random.choice([1, -1], size=len(diff))
        perm_stat = np.mean(signs * diff)
        if abs(perm_stat) >= abs(observed):
            count += 1
    p = count / n_permutations
    return {"p-value": p, "statistic": observed}


def run_all_tests_continuous_prepost(df, pre, post, convariates=None):
    return {
        "paired t-test": safe_execute(paired_t_test, df, pre, post, convariates),
        "wilcoxon signed-rank": safe_execute(
            wilcoxon_signed_rank_test, df, pre, post, convariates
        ),
        "sign test": safe_execute(sign_test, df, pre, post, convariates),
        "paired permutation test": safe_execute(
            paired_permutation_test, df, pre, post, convariates
        ),
    }


# Case 2: Binary IV vs Binary DV
def mcnemar_test(df, pre, post, covariates=None):
    x = df[pre].astype(int)
    y = df[post].astype(int)
    b = np.sum((x == 1) & (y == 0))
    c = np.sum((x == 0) & (y == 1))
    stat = (abs(b - c) ** 2) / (b + c)
    p = 1 - chi2.cdf(stat, df=1)
    return {"p-value": p, "statistic": stat}


def exact_mcnemar_test(df, pre, post, covariates=None):
    x = df[pre].astype(int)
    y = df[post].astype(int)
    b = np.sum((x == 1) & (y == 0))
    c = np.sum((x == 0) & (y == 1))
    n = b + c
    k = min(b, c)
    res = binomtest(k, n, p=0.5, alternative="two-sided")
    return {"p-value": res.pvalue, "statistic": res.statistic}


def mcnemar_cc_test(df, pre, post, covariates=None):
    x = df[pre].astype(int)
    y = df[post].astype(int)
    b = np.sum((x == 1) & (y == 0))
    c = np.sum((x == 0) & (y == 1))
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    p = 1 - chi2.cdf(stat, df=1)
    return {"p-value": p, "statistic": stat}


def paired_proportions_z_test(df, pre, post, covariates=None):
    x = df[pre].astype(int)
    y = df[post].astype(int)
    diff = y - x
    d_mean = diff.mean()
    d_var = diff.var(ddof=1)
    z = d_mean / np.sqrt(d_var / len(diff))
    p = 2 * (1 - norm.cdf(abs(z)))
    return {"p-value": p, "statistic": z}


def conditional_logistic_regression(df, pre, post, covariates):
    # Reshape to long
    long_df = pd.DataFrame(
        {
            "outcome": pd.concat([df[pre], df[post]]),
            "time": np.concatenate([np.zeros(len(df)), np.ones(len(df))]),
            "subject": np.concatenate([np.arange(len(df)), np.arange(len(df))]),
        }
    )

    # Use GEE for marginal model (Outcome ~ Time, grouped by Subject)
    # This accounts for correlation within subject
    try:
        model = smf.gee(
            "outcome ~ time",
            groups="subject",
            data=long_df,
            family=sm.families.Binomial(),
        ).fit()
        p = model.pvalues.get("time", np.nan)
    except:
        p = np.nan

    return {"p-value": p}


def run_all_tests_binary_prepost(df, pre, post, convariates=None):
    return {
        "McNemar chi-square": safe_execute(mcnemar_test, df, pre, post, convariates),
        "Exact McNemar (binomial)": safe_execute(
            exact_mcnemar_test, df, pre, post, convariates
        ),
        "McNemar continuity corrected": safe_execute(
            mcnemar_cc_test, df, pre, post, convariates
        ),
        "Paired proportions z-test": safe_execute(
            paired_proportions_z_test, df, pre, post, convariates
        ),
        "Conditional logistic regression": safe_execute(
            conditional_logistic_regression, df, pre, post, convariates
        ),
    }


# Case 3: Multinomial IV vs Multinomial DV


def stuart_maxwell_test(df, pre, post, covariates=None):
    x = df[pre].astype(str)
    y = df[post].astype(str)
    categories = sorted(set(x) | set(y))
    k = len(categories)

    tab = (
        pd.crosstab(x, y)
        .reindex(index=categories, columns=categories, fill_value=0)
        .values
    )
    diffs = tab.sum(axis=1) - tab.sum(axis=0)

    v = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            if i == j:
                v[i, j] = tab[i, :].sum() + tab[:, i].sum() - 2 * tab[i, i]
            else:
                v[i, j] = -(tab[i, j] + tab[j, i])
    v = v[:-1, :-1]
    diffs = diffs[:-1]
    try:
        stat = diffs.T @ np.linalg.inv(v) @ diffs
        p = 1 - chi2.cdf(stat, df=k - 1)
    except np.linalg.LinAlgError:
        p = np.nan

    return {"p-value": p}


def bowker_test(df, pre, post, covariates=None):
    x = df[pre].astype(str)
    y = df[post].astype(str)
    categories = sorted(set(x) | set(y))
    tab = (
        pd.crosstab(x, y)
        .reindex(index=categories, columns=categories, fill_value=0)
        .values
    )

    stat = 0
    for i in range(len(categories)):
        for j in range(i + 1, len(categories)):
            nij = tab[i, j]
            nji = tab[j, i]
            if nij + nji > 0:
                stat += (nij - nji) ** 2 / (nij + nji)
    df_ = (len(categories) * (len(categories) - 1)) // 2
    p = 1 - chi2.cdf(stat, df_)
    return {"p-value": p}


def run_all_tests_multinomial_prepost(df, pre, post, convariates=None):
    return {
        "Stuart-Maxwell test": safe_execute(
            stuart_maxwell_test, df, pre, post, convariates
        ),
        "Bowker test of symmetry": safe_execute(
            bowker_test, df, pre, post, convariates
        ),
    }


# Case 4: Ordinal IV vs Ordinal DV
def sign_test(df, pre, post, covariates=None):
    diff = (df[post] - df[pre]).dropna()
    diff = diff[diff != 0]  # Drop ties
    n_pos = np.sum(diff > 0)
    n = len(diff)
    res = binomtest(n_pos, n, p=0.5, alternative="two-sided")
    return {"p-value": res.pvalue}


def run_all_tests_ordinal_prepost(df, pre, post, convariates=None):
    return {
        "Wilcoxon signed-rank test": safe_execute(
            wilcoxon_signed_rank_test, df, pre, post, convariates
        ),
        "Sign test": safe_execute(sign_test, df, pre, post, convariates),
    }
