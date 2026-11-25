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


# Binary dependent_variable tests
def chi_square_test(df, case_control_col, dependent_variable_col):
    """Chi-square test of dependence: case_control x categorical dependent_variable"""
    table = pd.crosstab(df[case_control_col], df[dependent_variable_col])
    chi2, pval, dof, exp = chi2_contingency(table)
    return pval


def logistic_regression(df, case_control_col, dependent_variable_col, covariates=None):
    """Logistic regression (binary dependent_variable): dependent_variable ~ case_control [+ covariates]"""
    if len(df[dependent_variable_col].unique()) != 2:
        raise ValueError("Logistic regression requires a binary dependent_variable")

    cov_str = ""
    if covariates:
        cov_str = " + " + " + ".join(
            [
                f"C({c})" if str(df[c].dtype) in ["object", "category"] else c
                for c in covariates
            ]
        )
    formula = f"{case_control_col} ~ {dependent_variable_col}{cov_str}"
    model = smf.logit(formula, data=df).fit(disp=False)
    pval = model.pvalues[dependent_variable_col]
    return {"p-value": pval}


def fisher_test(df, case_control_col, dependent_variable_col):
    """Fisher's exact test (2x2 only)."""
    contingency = pd.crosstab(df[case_control_col], df[dependent_variable_col])
    if contingency.shape != (2, 2):
        raise ValueError("Fisher's exact test is only valid for 2x2 tables.")

    oddsratio, pval = fisher_exact(contingency)
    return {"p-value": pval}


def case_control_binary_dependent_variable_test(
    df, case_control_col, dependent_variable_col, covariates=None
):
    results_dictionary = {}
    chi_square_test_result = safe_execute(
        chi_square_test, df, case_control_col, dependent_variable_col
    )
    results_dictionary["Chi-square test of dependence"] = chi_square_test_result

    logistic_regression_result = safe_execute(
        logistic_regression, df, case_control_col, dependent_variable_col, covariates
    )
    results_dictionary["Logistic regression"] = logistic_regression_result

    fisher_test_result = safe_execute(
        fisher_test, df, case_control_col, dependent_variable_col
    )
    results_dictionary["Fisher's exact test (2x2)"] = fisher_test_result

    return results_dictionary


# multinomial dependent_variable tests
def fisher_exact_general(df, case_control_col, dependent_variable_col):
    """
    Generalized Fisher's exact test for r x c tables.
    Falls back to scipy fisher_exact if 2x2.
    """
    contingency = pd.crosstab(df[case_control_col], df[dependent_variable_col])

    if contingency.shape == (2, 2):
        _, pval = fisher_exact(contingency)
        return {"p-value": pval}
    else:
        table = Table(contingency)
        res = table.test_nominal_association()
        return {"p-value": res.pvalue}


def logistic_regression_categorical(
    df, case_control_col, dependent_variable_col, covariates=None
):
    """Logistic regression: dependent_variable ~ case_control + covariates"""

    df[dependent_variable_col] = df[dependent_variable_col].astype("category").cat.codes
    if covariates is None:
        covariates = []
    for i in covariates:
        if str(df[i].dtype) in ["object", "category"]:
            df[i] = df[i].astype("category").cat.codes
    cov_str = ""
    if covariates:
        cov_str = " + " + " + ".join(
            [
                f"C({c})" if str(df[c].dtype) in ["object", "category"] else c
                for c in covariates
            ]
        )
    formula = f"{case_control_col} ~ {dependent_variable_col}{cov_str}"
    model = smf.logit(formula, data=df).fit(disp=False)
    pval = model.pvalues[1:].max()  # max p-value for case_control terms
    return {"p-value": pval}


def chi_square_test(df, case_control_col, dependent_variable_col):
    """Chi-square test of dependence: case_control x categorical dependent_variable"""
    table = pd.crosstab(df[case_control_col], df[dependent_variable_col])
    chi2, pval, dof, exp = chi2_contingency(table)
    return {"p-value": pval}


def case_control_multinomial_dependent_variable_test(
    df, case_control_col, dependent_variable_col, covariates=None
):
    results_dictionary = {}
    chi_square_test_result = safe_execute(
        chi_square_test, df, case_control_col, dependent_variable_col
    )
    results_dictionary["Chi-square test of dependence"] = chi_square_test_result

    logistic_regression_result = safe_execute(
        logistic_regression_categorical,
        df,
        case_control_col,
        dependent_variable_col,
        covariates,
    )
    results_dictionary["Logistic regression"] = logistic_regression_result

    fisher_test_result = safe_execute(
        fisher_exact_general, df, case_control_col, dependent_variable_col
    )
    results_dictionary["Fisher's exact test (2x2)"] = fisher_test_result

    return results_dictionary


# Continuous dependent_variable tests
def t_test(df, case_control_col, dependent_variable_col):
    """Two-sample t-test: case_control x continuous dependent_variable"""
    case_controls = df[case_control_col].unique()
    if len(case_controls) != 2:
        raise ValueError("t-test requires exactly two case_control case_controls")

    case_control1 = df[df[case_control_col] == case_controls[0]][dependent_variable_col]
    case_control2 = df[df[case_control_col] == case_controls[1]][dependent_variable_col]

    stat, pval = stats.ttest_ind(case_control1, case_control2, nan_policy="omit")
    return {"p-value": pval}


def anova_test(df, case_control_col, dependent_variable_col):
    """ANOVA: case_control x continuous dependent_variable"""
    case_controls = df[case_control_col].unique()
    if len(case_controls) < 2:
        raise ValueError("ANOVA requires at least two case_control case_controls")

    case_control_data = [
        df[df[case_control_col] == g][dependent_variable_col] for g in case_controls
    ]
    stat, pval = f_oneway(*case_control_data)
    return {"p-value": pval}


# anova with covariates
def anova_with_covariates(df, case_control_col, dependent_variable_col, covariates):
    """ANOVA with covariates: dependent_variable ~ case_control + covariates"""
    if len(df[case_control_col].unique()) < 2:
        raise ValueError("ANOVA requires at least two case_control case_controls")
    cov = ""
    if covariates:
        cov = " + " + " + ".join(
            [
                f"C({c})" if str(df[c].dtype) in ["object", "category"] else c
                for c in covariates
            ]
        )
    formula = f"{dependent_variable_col} ~ C({case_control_col}){cov}"
    model = smf.ols(formula, data=df).fit()
    anova_results = anova_lm(model)
    pval = anova_results["PR(>F)"][0]  # p-value for case_control
    return {"p-value": pval}


def logistic_regression_continuous(
    df, case_control_col, dependent_variable_col, covariates=None
):
    """Linear regression (continuous dependent_variable): dependent_variable ~ case_control [+ covariates]"""
    cov_str = ""
    if covariates:
        cov_str = " + " + " + ".join(
            [
                f"C({c})" if str(df[c].dtype) in ["object", "category"] else c
                for c in covariates
            ]
        )
    formula = f"{case_control_col} ~ {dependent_variable_col}{cov_str}"
    model = smf.ols(formula, data=df).fit()
    pval = model.pvalues[dependent_variable_col]
    return {"p-value": pval}


def mann_whitney_u_test(df, case_control_col, dependent_variable_col):
    """Mann-Whitney U test: case_control x continuous dependent_variable"""
    case_controls = df[case_control_col].unique()
    if len(case_controls) != 2:
        raise ValueError(
            "Mann-Whitney U test requires exactly two case_control case_controls"
        )

    case_control1 = df[df[case_control_col] == case_controls[0]][dependent_variable_col]
    case_control2 = df[df[case_control_col] == case_controls[1]][dependent_variable_col]

    stat, pval = mannwhitneyu(case_control1, case_control2, alternative="two-sided")
    return {"p-value": pval}


def case_control_continous_dependent_variable_test(
    df, case_control_col, dependent_variable_col, covariates=None
):
    results_dictionary = {}
    t_test_result = safe_execute(t_test, df, case_control_col, dependent_variable_col)
    results_dictionary["Two-sample t-test"] = t_test_result
    anova_test_result = safe_execute(
        anova_test, df, case_control_col, dependent_variable_col
    )
    results_dictionary["ANOVA"] = anova_test_result
    anova_with_covariates_result = safe_execute(
        anova_with_covariates, df, case_control_col, dependent_variable_col, covariates
    )
    results_dictionary["ANOVA with covariates"] = anova_with_covariates_result
    logistic_regression_result = safe_execute(
        logistic_regression_continuous,
        df,
        case_control_col,
        dependent_variable_col,
        covariates,
    )
    results_dictionary["Logistic regression"] = logistic_regression_result
    mann_whitney_u_test_result = safe_execute(
        mann_whitney_u_test, df, case_control_col, dependent_variable_col
    )
    results_dictionary["Mann-Whitney U test"] = mann_whitney_u_test_result
    return results_dictionary


# Ordinal dependent_variable tests are the same as continuous and categorical tests but with ordinal considerations


# functions to generate case-control study designs
def generate_cont(
    n=1000,
    seed=42,
):
    """Generate synthetic case-control study data with continuous dependent_variable"""
    np.random.seed(seed)
    # Simulate case_control assignment (0 or 1)
    case_control = np.random.binomial(1, 0.5, n)

    # Simulate continuous dependent_variable with different means for case_control case_controls
    dependent_variable = np.random.normal(loc=5 + 2 * case_control, scale=1, size=n)

    # Create DataFrame
    df = pd.DataFrame(
        {
            "case_control": case_control,
            "dependent_variable": dependent_variable,
            "age": np.random.randint(20, 70, n),
            "sex": np.random.choice(["M", "F"], n),
        }
    )

    return df


def generate_binary(
    n=1000,
    seed=42,
):
    """Generate synthetic case-control study data with binary dependent_variable"""
    np.random.seed(seed)
    # Simulate case_control assignment (0 or 1)
    case_control = np.random.binomial(1, 0.5, n)

    # Simulate binary dependent_variable with different probabilities for case_control case_controls
    dependent_variable = np.random.binomial(1, 0.3 + 0.4 * case_control, n)

    # Create DataFrame
    df = pd.DataFrame(
        {
            "case_control": case_control,
            "dependent_variable": dependent_variable,
            "age": np.random.randint(20, 70, n),
            "sex": np.random.choice(["M", "F"], n),
        }
    )
    return df


def generate_categorical(
    n=1000,
    seed=42,
):
    """Generate synthetic case-control study data with categorical dependent_variable"""
    np.random.seed(seed)
    # Simulate case_control assignment (0 or 1)
    case_control = np.random.binomial(1, 0.5, n)

    # Simulate categorical dependent_variable with different distributions for case_control case_controls
    dependent_variable = np.random.choice(["A", "B", "C"], n)

    # Create DataFrame
    df = pd.DataFrame(
        {
            "case_control": case_control,
            "dependent_variable": dependent_variable,
            "age": np.random.randint(20, 70, n),
            "sex": np.random.choice(["M", "F"], n),
        }
    )

    return df


def dependent_variable_binary():
    """Case-control study with binary dependent_variable"""
    df = generate_binary()

    case_control_col = "case_control"
    dependent_variable_col = "dependent_variable"
    covariates = ["age", "sex"]
    print(chi_square_test(df, case_control_col, dependent_variable_col))
    print(logistic_regression(df, case_control_col, dependent_variable_col, covariates))
    print(fisher_test(df, case_control_col, dependent_variable_col))


def dependent_variable_continuous():
    """Case-control study with continuous dependent_variable"""
    df = generate_cont()
    case_control_col = "case_control"
    dependent_variable_col = "dependent_variable"
    covariates = ["age", "sex"]

    print(t_test(df, case_control_col, dependent_variable_col))
    print(anova_test(df, case_control_col, dependent_variable_col))
    if covariates:
        print(
            anova_with_covariates(
                df, case_control_col, dependent_variable_col, covariates
            )
        )
        print(
            logistic_regression_continuous(
                df, case_control_col, dependent_variable_col, covariates
            )
        )
    print(mann_whitney_u_test(df, case_control_col, dependent_variable_col))


def dependent_variable_categorical():
    """Case-control study with categorical dependent_variable"""
    df = generate_categorical()
    case_control_col = "case_control"
    dependent_variable_col = "dependent_variable"
    covariates = ["age", "sex"]

    print(chi_square_test(df, case_control_col, dependent_variable_col))
    print(
        logistic_regression_categorical(
            df, case_control_col, dependent_variable_col, covariates
        )
    )
    print(fisher_exact_general(df, case_control_col, dependent_variable_col))


def dependent_variable_ordinal():
    """Case-control study with ordinal dependent_variable"""
    df = generate_categorical()
    case_control_col = "case_control"
    dependent_variable_col = "dependent_variable"
    covariates = ["age", "sex"]

    print(chi_square_test(df, case_control_col, dependent_variable_col))
    print(
        logistic_regression_categorical(
            df, case_control_col, dependent_variable_col, covariates
        )
    )
    print(fisher_exact_general(df, case_control_col, dependent_variable_col))

    # convert dependent_variable to ordinal numeric for continuous tests
    df["dependent_variable_ord"] = df[dependent_variable_col].map({0: 1, 1: 2, 2: 3})
    print(t_test(df, case_control_col, "dependent_variable_ord"))
    print(anova_test(df, case_control_col, "dependent_variable_ord"))
    if covariates:
        print(
            anova_with_covariates(
                df, case_control_col, "dependent_variable_ord", covariates
            )
        )
        print(
            logistic_regression_continuous(
                df, case_control_col, "dependent_variable_ord", covariates
            )
        )
    print(mann_whitney_u_test(df, case_control_col, "dependent_variable_ord"))


if __name__ == "__main__":
    print("Case-Control Study Designs and Tests")
    print("-----------------------------------")
    print("Binary dependent_variable Tests:")
    print("-----------------------------------\n")
    dependent_variable_binary()
    print("Continuous dependent_variable Tests:")
    print("-----------------------------------\n")
    dependent_variable_continuous()
    print("Categorical dependent_variable Tests:")
    print("-----------------------------------\n")
    dependent_variable_categorical()
    print("Ordinal dependent_variable Tests:")
    print("-----------------------------------\n")
    dependent_variable_ordinal()
