import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import MNLogit, Probit
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.mixture import GaussianMixture
from scipy import stats
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.regression.quantile_regression import QuantReg
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.gam.api import GLMGam, BSplines
from patsy import dmatrix
from .utils_general_functions import safe_execute

# continuous IV vs multinomial DV
# continuous IV vs continuous DV
# continuous IV vs binary DV
# binary IV vs multinomial DV
# binary IV vs continuous DV
# binary IV vs binary DV
# multinomial IV vs multinomial DV
# multinomial IV vs continuous DV
# multinomial IV vs binary DV


# CASE: independent variable is continuous, dependent variable is multinomial
def multinomial_logistic_regression(df, iv, dv, covariates=None):
    """
    Multinomial logistic regression: dv ~ iv + covariates
    """
    for i in [iv] + (covariates if covariates else []):
        if df[i].dtype.name == "category" or df[i].dtype == object:
            df[i] = df[i].astype("category").cat.codes
    X = df[[iv] + (covariates if covariates else [])]
    X = sm.add_constant(X)
    y = df[dv].astype("category").cat.codes

    model = MNLogit(y, X).fit(disp=False)
    return {"p_values": model.pvalues.to_dict()}


def multinomial_probit(df, iv, dv, covariates=None):
    """
    Multinomial probit (approx via Probit for binary vs each category baseline).
    """
    y = df[dv].astype("category")
    categories = y.cat.categories
    results = {}

    for cat in categories[1:]:
        # binary outcome: current category vs baseline
        y_bin = (y == cat).astype(int)
        for i in [iv] + (covariates if covariates else []):
            if df[i].dtype.name == "category" or df[i].dtype == object:
                df[i] = df[i].astype("category").cat.codes
        X = df[[iv] + (covariates if covariates else [])]
        X = sm.add_constant(X)
        model = Probit(y_bin, X).fit(disp=False)
        results[f"{cat} vs {categories[0]}"] = {"p_values": model.pvalues.to_dict()}

    return results


def lda_test(df, iv, dv):
    """
    Linear Discriminant Analysis: classification of dv using continuous iv.
    """
    X = df[[iv]].values
    y = LabelEncoder().fit_transform(df[dv])
    lda = LinearDiscriminantAnalysis()
    lda.fit(X, y)

    # Wilks’ lambda (approx significance test for discriminant functions)
    # Here simplified: perform ANOVA F-test on group means
    groups = [X[y == i].ravel() for i in np.unique(y)]
    f_val, p_val = stats.f_oneway(*groups)
    return {"F-test": p_val}


def qda_test(df, iv, dv):
    """
    Quadratic Discriminant Analysis: classification of dv using continuous iv.
    """
    X = df[[iv]].values
    y = LabelEncoder().fit_transform(df[dv])
    qda = QuadraticDiscriminantAnalysis()
    qda.fit(X, y)

    # Same approach: group mean differences as proxy p-value
    groups = [X[y == i].ravel() for i in np.unique(y)]
    f_val, p_val = stats.f_oneway(*groups)
    return {"p-value": p_val, "f-value": f_val}


def latent_class_analysis(df, iv, dv, n_classes=3):
    """
    Latent class analysis via Gaussian Mixture Model.
    """
    X = df[[iv]].values
    gmm = GaussianMixture(n_components=n_classes, random_state=42)
    gmm.fit(X)

    # Likelihood ratio test vs 1-class model
    gmm1 = GaussianMixture(n_components=1, random_state=42).fit(X)
    lr_stat = 2 * (gmm.score(X) - gmm1.score(X)) * len(X)
    p_val = stats.chi2.sf(lr_stat, (n_classes - 1))
    return {"p-value": p_val}


def generate_cont_indep_multinomial_dep_data(n=100, n_categories=3, random_state=42):
    """
    Generate synthetic dataset with a continuous independent variable and a multinomial dependent variable.
    """
    np.random.seed(random_state)
    X = np.random.normal(size=n)
    # Create a multinomial dependent variable based on the continuous independent variable
    logits = np.array([0.5 * X - 0.2, -0.3 * X + 0.1, 0.1 * X + 0.3]).T
    exp_logits = np.exp(logits)
    probabilities = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    y = [np.random.choice(n_categories, p=prob) for prob in probabilities]

    df = pd.DataFrame(
        {
            "independent_variable": X,
            "dependent_variable": pd.Categorical(y),
            "Age": np.random.randint(18, 65, size=n),
            "Gender": np.random.choice(["Male", "Female"], size=n),
        }
    )
    return df


def test_cont_indep_multinomial_dep(
    df, independent_variable, dependent_variable, covariates=None
):
    """
    Test independence between a continuous independent variable and a multinomial dependent variable.
    Uses multinomial logistic regression and likelihood ratio test.
    """
    multinomial_logistic_result = safe_execute(
        multinomial_logistic_regression,
        df,
        independent_variable,
        dependent_variable,
        covariates=covariates,
    )
    multinomial_probit_result = safe_execute(
        multinomial_probit,
        df,
        independent_variable,
        dependent_variable,
        covariates=covariates,
    )
    lda_result = safe_execute(lda_test, df, independent_variable, dependent_variable)
    qda_result = safe_execute(qda_test, df, independent_variable, dependent_variable)
    lca_result = safe_execute(
        latent_class_analysis,
        df,
        independent_variable,
        dependent_variable,
        n_classes=len(df[dependent_variable].unique()),
    )

    return {
        "Multinomial Logistic Regression": multinomial_logistic_result,
        "Multinomial Probit Model": multinomial_probit_result,
        "Linear Discriminant Analysis": lda_result,
        "Quadratic Discriminant Analysis": qda_result,
        "Latent Class Analysis (GMM approx)": lca_result,
    }


# ==============================================================
# CASE: Continuous IV, Continuous DV
def pearson_correlation(df, iv, dv, covariates=None):
    r, p = stats.pearsonr(df[iv], df[dv])
    return {"r": r, "p": p}


def spearman_correlation(df, iv, dv, covariates=None):
    rho, p = stats.spearmanr(df[iv], df[dv])
    return {"rho": rho, "p": p}


def kendall_tau(df, iv, dv, covariates=None):
    tau, p = stats.kendalltau(df[iv], df[dv])
    return {"tau": tau, "p": p}


def linear_regression(df, iv, dv, covariates=None):
    df_copy = df.copy()
    for i in [iv] + (covariates if covariates else []):
        if df_copy[i].dtype.name == "category" or df_copy[i].dtype == object:
            df_copy[i] = df_copy[i].astype("category").cat.codes
    predictors = [iv] + (covariates if covariates else [])
    X = sm.add_constant(df_copy[predictors])
    y = df_copy[dv]
    model = sm.OLS(y, X).fit()
    return {"p_values": model.pvalues.to_dict()}


def partial_correlation(df, iv, dv, covariates):
    # regress iv and dv on covariates, then correlate residuals
    X = sm.add_constant(df[covariates])
    iv_resid = sm.OLS(df[iv], X).fit().resid
    dv_resid = sm.OLS(df[dv], X).fit().resid
    r, p = stats.pearsonr(iv_resid, dv_resid)
    return {"r": r, "p": p}


def quantile_regression(df, iv, dv, covariates=None):
    for i in [iv] + (covariates if covariates else []):
        if df[i].dtype.name == "category" or df[i].dtype == object:
            df[i] = df[i].astype("category").cat.codes
    predictors = [iv] + (covariates if covariates else [])
    formula = f"{dv} ~ {' + '.join(predictors)}"
    model = QuantReg.from_formula(formula, df)
    res = model.fit(q=0.5)  # median regression
    return {"p_values": res.pvalues.to_dict()}


def robust_regression(df, iv, dv, covariates=None):
    for i in [iv] + (covariates if covariates else []):
        if df[i].dtype.name == "category" or df[i].dtype == object:
            df[i] = df[i].astype("category").cat.codes
    predictors = [iv] + (covariates if covariates else [])
    X = sm.add_constant(df[predictors])
    y = df[dv]
    model = sm.RLM(y, X, M=sm.robust.norms.HuberT()).fit()
    return {"p_values": model.pvalues.to_dict()}


def generate_cont_indep_cont_dep_data(n=100, n_categories=3, random_state=42):
    """
    Generate synthetic dataset with a continuous independent variable and  a continuous dependent variable.
    """
    np.random.seed(random_state)
    X = np.random.normal(size=n)
    # Create a continuous dependent variable influenced by the independent variable
    y = 2 * X + np.random.normal(size=n)

    df = pd.DataFrame(
        {
            "independent_variable": X,
            "dependent_variable": y,
            "Age": np.random.randint(18, 65, size=n),
            "Gender": np.random.choice(["Male", "Female"], size=n),
        }
    )

    return df


def test_cont_indep_cont_dep(
    df, independent_variable, dependent_variable, covariates=None
):
    """
    Test independence between a continuous independent variable and a continuous dependent variable.
    Uses correlation tests and regression models.
    """
    pearson_result = safe_execute(
        pearson_correlation,
        df,
        independent_variable,
        dependent_variable,
        covariates=covariates,
    )
    spearman_result = safe_execute(
        spearman_correlation,
        df,
        independent_variable,
        dependent_variable,
        covariates=covariates,
    )
    kendall_result = safe_execute(
        kendall_tau, df, independent_variable, dependent_variable, covariates=covariates
    )
    linear_regression_result = safe_execute(
        linear_regression,
        df,
        independent_variable,
        dependent_variable,
        covariates=covariates,
    )
    if covariates:
        partial_corr_result = safe_execute(
            partial_correlation,
            df,
            independent_variable,
            dependent_variable,
            covariates=covariates,
        )
    else:
        partial_corr_result = ("Partial correlation", "N/A (no covariates provided)")
    quantile_regression_result = safe_execute(
        quantile_regression,
        df,
        independent_variable,
        dependent_variable,
        covariates=covariates,
    )
    robust_regression_result = safe_execute(
        robust_regression,
        df,
        independent_variable,
        dependent_variable,
        covariates=covariates,
    )

    return {
        "Pearson Correlation": pearson_result,
        "Spearman Correlation": spearman_result,
        "Kendall’s Tau": kendall_result,
        "Linear Regression (OLS)": linear_regression_result,
        "Partial Correlation": partial_corr_result,
        "Quantile Regression": quantile_regression_result,
        "Robust Regression": robust_regression_result,
    }


# ==============================================================
# CASE: independent variable is continuous, dependent variable is binary


def logistic_regression_binary(df, iv, dv, covariates=None):
    """
    Logistic regression: binary dv ~ continuous iv + covariates
    """
    for i in [iv] + (covariates if covariates else []):
        if df[i].dtype.name == "category" or df[i].dtype == object:
            df[i] = df[i].astype("category").cat.codes

    X = df[[iv] + (covariates if covariates else [])]
    X = sm.add_constant(X)
    y = df[dv].astype(int)

    model = sm.Logit(y, X).fit(disp=False)
    return {"p_values": model.pvalues.to_dict()}


def probit_regression_binary(df, iv, dv, covariates=None):
    """
    Probit regression: binary dv ~ continuous iv + covariates
    """
    for i in [iv] + (covariates if covariates else []):
        if df[i].dtype.name == "category" or df[i].dtype == object:
            df[i] = df[i].astype("category").cat.codes

    X = df[[iv] + (covariates if covariates else [])]
    X = sm.add_constant(X)
    y = df[dv].astype(int)

    model = Probit(y, X).fit(disp=False)
    return {"p_values": model.pvalues.to_dict()}


def point_biserial_correlation(df, iv, dv, covariates=None):
    """
    Point-biserial correlation between continuous iv and binary dv.
    This is equivalent to Pearson correlation when one variable is binary.
    """
    # Ensure binary variable is numeric
    y = df[dv].astype(int)
    x = df[iv]

    r, p = stats.pearsonr(x, y)
    return {"r": r, "p": p}


def mann_whitney_binary(df, iv, dv, covariates=None):
    """
    Mann-Whitney U test: Compare continuous iv distributions across binary dv groups.
    Non-parametric alternative to t-test.
    """
    y = df[dv].astype(int)
    group0 = df[y == 0][iv]
    group1 = df[y == 1][iv]

    statistic, p = stats.mannwhitneyu(group0, group1, alternative="two-sided")
    return {"statistic": statistic, "p": p}


def generate_cont_indep_binary_dep_data(n=100, random_state=42):
    """
    Generate synthetic dataset with a continuous independent variable and a binary dependent variable.
    """
    np.random.seed(random_state)
    X = np.random.normal(size=n)

    # Create a binary dependent variable based on the continuous independent variable
    # Using logistic function to generate probabilities
    logits = 0.8 * X + 0.2
    probabilities = 1 / (1 + np.exp(-logits))
    y = np.random.binomial(1, probabilities)

    df = pd.DataFrame(
        {
            "independent_variable": X,
            "dependent_variable": y,
            "Age": np.random.randint(18, 65, size=n),
            "Gender": np.random.choice(["Male", "Female"], size=n),
        }
    )

    return df


def test_cont_indep_binary_dep(
    df, independent_variable, dependent_variable, covariates=None
):
    """
    Test relationship between a continuous independent variable and a binary dependent variable.
    Uses logistic regression, probit regression, point-biserial correlation, and Mann-Whitney U test.
    """
    logistic_result = safe_execute(
        logistic_regression_binary,
        df,
        independent_variable,
        dependent_variable,
        covariates=covariates,
    )
    probit_result = safe_execute(
        probit_regression_binary,
        df,
        independent_variable,
        dependent_variable,
        covariates=covariates,
    )
    point_biserial_result = safe_execute(
        point_biserial_correlation,
        df,
        independent_variable,
        dependent_variable,
        covariates=covariates,
    )
    mann_whitney_result = safe_execute(
        mann_whitney_binary,
        df,
        independent_variable,
        dependent_variable,
        covariates=covariates,
    )

    return {
        "Logistic Regression": logistic_result,
        "Probit Regression": probit_result,
        "Point-Biserial Correlation": point_biserial_result,
        "Mann-Whitney U Test": mann_whitney_result,
    }


################################################
# Case: Binary IV vs Binary DV


def chi_square_test(df, iv, dv):
    """
    Chi-square test of independence for two binary variables.
    """
    contingency_table = pd.crosstab(df[iv], df[dv])
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    return {"chi2": chi2, "p": p}


def fishers_exact_test(df, iv, dv):
    """
    Fisher's Exact Test for 2x2 contingency tables.
    """
    contingency_table = pd.crosstab(df[iv], df[dv])
    oddsratio, p = stats.fisher_exact(contingency_table)
    return {"odds_ratio": oddsratio, "p": p}


def phi_coefficient(df, iv, dv):
    """
    Phi coefficient for 2x2 tables.
    """
    contingency_table = pd.crosstab(df[iv], df[dv])
    chi2, _, _, _ = stats.chi2_contingency(contingency_table)
    n = contingency_table.sum().sum()
    phi = np.sqrt(chi2 / n)
    return {"phi": phi}


def risk_ratio(df, iv, dv):
    """
    Calculates Risk Ratio (Relative Risk) assuming iv=1 is exposed.
    """
    # Assuming 0 and 1 coding
    try:
        tbl = pd.crosstab(df[iv], df[dv])
        # This assumes rows are IV (0, 1) and cols are DV (0, 1)
        # Risk = P(DV=1 | IV=1) / P(DV=1 | IV=0)
        # Valid only if index/columns are appropriate
        if 1 in tbl.index and 1 in tbl.columns:
            risk_exposed = tbl.loc[1, 1] / tbl.loc[1].sum()
            risk_unexposed = tbl.loc[0, 1] / tbl.loc[0].sum()
            rr = risk_exposed / risk_unexposed
            return {"RR": rr}
    except Exception:
        pass
    return "N/A or Error in calculation"


def generate_binary_indep_binary_dep_data(n=200, random_state=42):
    np.random.seed(random_state)
    # IV: 0 or 1
    iv = np.random.binomial(1, 0.5, size=n)
    # DV depends on IV
    # P(DV=1|IV=0) = 0.3, P(DV=1|IV=1) = 0.6
    probs = np.where(iv == 1, 0.6, 0.3)
    dv = np.random.binomial(1, probs)

    df = pd.DataFrame(
        {
            "independent_variable": iv,
            "dependent_variable": dv,
            "Age": np.random.randint(20, 70, size=n),
            "Gender": np.random.choice(["Male", "Female"], size=n),
        }
    )
    return df


def test_binary_indep_binary_dep(df, iv, dv, covariates=None):
    chi2_res = safe_execute(chi_square_test, df, iv, dv)
    fisher_res = safe_execute(fishers_exact_test, df, iv, dv)
    phi_res = safe_execute(phi_coefficient, df, iv, dv)
    rr_res = safe_execute(risk_ratio, df, iv, dv)
    # Re-use binary logistic regression from Case 3
    logit_res = safe_execute(logistic_regression_binary, df, iv, dv, covariates)

    return {
        "Chi-Square Test": chi2_res,
        "Fisher's Exact Test": fisher_res,
        "Phi Coefficient": phi_res,
        "Risk Ratio": rr_res,
        "Logistic Regression": logit_res,
    }


###############################################
# Case: Binary IV vs Continuous DV


def t_test_binary_iv(df, iv, dv):
    """
    Independent samples t-test.
    """
    groups = df[iv].unique()
    if len(groups) != 2:
        return "Error: IV must have exactly 2 levels"

    g1 = df[df[iv] == groups[0]][dv]
    g2 = df[df[iv] == groups[1]][dv]

    t_stat, p = stats.ttest_ind(g1, g2)
    return {"t": t_stat, "p": p}


def mann_whitney_binary_iv(df, iv, dv):
    """
    Mann-Whitney U test for binary IV.
    """
    groups = df[iv].unique()
    if len(groups) != 2:
        return "Error: IV must have exactly 2 levels"

    g1 = df[df[iv] == groups[0]][dv]
    g2 = df[df[iv] == groups[1]][dv]

    stat, p = stats.mannwhitneyu(g1, g2)
    return {"U": stat, "p": p}


def point_biserial_binary_iv(df, iv, dv):
    """
    Point-biserial correlation.
    """
    # Requires IV to be numeric 0/1 for the function usually,
    # but scipy pearsonr works if it's just numbers.
    # Point biserial is essentially pearsonr
    r, p = stats.pearsonr(df[iv], df[dv])
    return {"r": r, "p": p}


def cohens_d(df, iv, dv):
    """
    Calculate Cohen's d for effect size.
    """
    groups = df[iv].unique()
    if len(groups) != 2:
        return "Error: IV must have exactly 2 levels"

    g1 = df[df[iv] == groups[0]][dv]
    g2 = df[df[iv] == groups[1]][dv]

    n1, n2 = len(g1), len(g2)
    s1, s2 = np.var(g1, ddof=1), np.var(g2, ddof=1)
    s_pooled = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    d = (np.mean(g1) - np.mean(g2)) / s_pooled
    return {"d": d}


def generate_binary_indep_cont_dep_data(n=200, random_state=42):
    np.random.seed(random_state)
    iv = np.random.binomial(1, 0.5, size=n)
    # Continuous DV, mean shifted by IV
    dv = np.random.normal(loc=0, scale=1, size=n) + (iv * 1.5)

    df = pd.DataFrame(
        {
            "independent_variable": iv,
            "dependent_variable": dv,
            "Age": np.random.randint(20, 70, size=n),
            "Gender": np.random.choice(["Male", "Female"], size=n),
        }
    )
    return df


def test_binary_indep_cont_dep(df, iv, dv, covariates=None):
    ttest_res = safe_execute(t_test_binary_iv, df, iv, dv)
    mw_res = safe_execute(mann_whitney_binary_iv, df, iv, dv)
    pb_res = safe_execute(point_biserial_binary_iv, df, iv, dv)
    d_res = safe_execute(cohens_d, df, iv, dv)
    # Re-use Linear Regression from Case 2
    lin_reg_res = safe_execute(linear_regression, df, iv, dv, covariates)

    return {
        "T-test": ttest_res,
        "Mann-Whitney U": mw_res,
        "Point-biserial Correlation": pb_res,
        "Cohen's d": d_res,
        "Linear Regression": lin_reg_res,
    }


###############################################
# Case: Binary IV vs Multinomial DV


def cramers_v(df, iv, dv):
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


def generate_binary_indep_multinomial_dep_data(n=200, n_classes=3, random_state=42):
    np.random.seed(random_state)
    iv = np.random.binomial(1, 0.5, size=n)

    # Probabilities for classes 0, 1, 2 given IV
    # If IV=0 -> slightly favored class 0
    # If IV=1 -> slightly favored class 2
    probs_0 = [0.5, 0.3, 0.2]
    probs_1 = [0.2, 0.3, 0.5]

    dv = []
    for val in iv:
        p = probs_1 if val == 1 else probs_0
        dv.append(np.random.choice(range(n_classes), p=p))

    df = pd.DataFrame(
        {
            "independent_variable": iv,
            "dependent_variable": dv,
            "Age": np.random.randint(20, 70, size=n),
            "Gender": np.random.choice(["Male", "Female"], size=n),
        }
    )
    return df


def test_binary_indep_multinomial_dep(df, iv, dv, covariates=None):
    # Reuse generic chi-square
    chi2_res = safe_execute(chi_square_test, df, iv, dv)
    v_res = safe_execute(cramers_v, df, iv, dv)
    # Reuse Multinomial Logistic Regression from Case 1
    mnlogit_res = safe_execute(multinomial_logistic_regression, df, iv, dv, covariates)

    return {
        "Chi-Square Test": chi2_res,
        "Cramer's V": v_res,
        "Multinomial Logistic Regression": mnlogit_res,
    }


###############################################
# Case: Multinomial IV vs Binary DV


def generate_multinomial_indep_binary_dep_data(n=200, n_categories=3, random_state=42):
    np.random.seed(random_state)
    iv = np.random.choice(range(n_categories), size=n)

    # Probability of DV=1 depends on IV category
    # e.g., Cat 0: 0.2, Cat 1: 0.5, Cat 2: 0.8
    probs_map = {0: 0.2, 1: 0.5, 2: 0.8}
    probs = [probs_map.get(x, 0.5) for x in iv]

    dv = np.random.binomial(1, probs)

    df = pd.DataFrame(
        {
            "independent_variable": iv,
            "dependent_variable": dv,
            "Age": np.random.randint(20, 70, size=n),
            "Gender": np.random.choice(["Male", "Female"], size=n),
        }
    )
    return df


def test_multinomial_indep_binary_dep(df, iv, dv, covariates=None):
    # Reuse Chi-square
    chi2_res = safe_execute(chi_square_test, df, iv, dv)
    # Reuse Cramer's V
    v_res = safe_execute(cramers_v, df, iv, dv)
    # Logistic Regression with categorical IV
    # We need to treat IV as categorical implicitly or explicitly
    # logistic_regression_binary handles object/category conversion
    logit_res = safe_execute(logistic_regression_binary, df, iv, dv, covariates)

    return {
        "Chi-Square Test": chi2_res,
        "Cramer's V": v_res,
        "Logistic Regression": logit_res,
    }


###############################################
# Case: Multinomial IV vs Continuous DV


def anova_test_multinomial_iv(df, iv, dv):
    """
    One-way ANOVA with degrees of freedom.
    """
    clean_df = df[[iv, dv]].dropna()
    groups = [clean_df[clean_df[iv] == i][dv].values for i in clean_df[iv].unique()]
    f_stat, p = stats.f_oneway(*groups)
    k = len(groups)
    n = sum(len(g) for g in groups)
    df_between = max(k - 1, 1)
    df_within = max(n - k, 1)
    return {
        "F": float(f_stat),
        "p": float(p),
        "df_between": int(df_between),
        "df_within": int(df_within),
    }


def kruskal_wallis_test(df, iv, dv):
    """
    Kruskal-Wallis H-test (non-parametric ANOVA).
    """
    clean_df = df[[iv, dv]].dropna()
    groups = [clean_df[clean_df[iv] == i][dv].values for i in clean_df[iv].unique()]
    h_stat, p = stats.kruskal(*groups)
    k = len(groups)
    return {"H": float(h_stat), "p": float(p), "df": int(max(k - 1, 1))}


def tukey_hsd_test(df, iv, dv):
    """
    Tukey HSD post-hoc pairwise comparisons.
    """
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    clean_df = df[[iv, dv]].dropna()
    if clean_df[iv].nunique() < 2 or len(clean_df) <= clean_df[iv].nunique():
        return {}
    tukey = pairwise_tukeyhsd(clean_df[dv], clean_df[iv])
    summary = {}
    for row in tukey.summary().data[1:]:
        g1, g2, diff, p_adj, lower, upper, reject = row
        summary[f"{g1}_vs_{g2}"] = {
            "diff": float(diff),
            "p_adj": float(p_adj),
            "lower": float(lower),
            "upper": float(upper),
            "reject": bool(reject),
        }
    return summary


def eta_squared_anova(df, iv, dv):
    """
    Calculate Eta Squared for ANOVA.
    """
    # Using statsmodels OLS for convenience to get sum of squares
    model = sm.OLS.from_formula(f"{dv} ~ C({iv})", data=df).fit()
    aov_table = sm.stats.anova_lm(model, typ=2)
    # sum_sq for the factor (IV) / total sum_sq
    ss_factor = aov_table["sum_sq"][0]
    ss_total = aov_table["sum_sq"].sum()
    eta_sq = ss_factor / ss_total
    return {"eta_sq": eta_sq}


def generate_multinomial_indep_cont_dep_data(n=200, n_categories=3, random_state=42):
    np.random.seed(random_state)
    iv = np.random.choice(range(n_categories), size=n)

    # DV distribution depends on IV
    # Cat 0: N(0, 1), Cat 1: N(2, 1), Cat 2: N(-2, 1)
    means = {0: 0, 1: 2, 2: -2}
    dv = np.array([np.random.normal(loc=means.get(x, 0), scale=1) for x in iv])

    df = pd.DataFrame(
        {
            "independent_variable": iv,
            "dependent_variable": dv,
            "Age": np.random.randint(20, 70, size=n),
            "Gender": np.random.choice(["Male", "Female"], size=n),
        }
    )
    return df


def test_multinomial_indep_cont_dep(df, iv, dv, covariates=None):
    anova_res = safe_execute(anova_test_multinomial_iv, df, iv, dv)
    kw_res = safe_execute(kruskal_wallis_test, df, iv, dv)
    eta_res = safe_execute(eta_squared_anova, df, iv, dv)
    lin_reg_res = safe_execute(linear_regression, df, iv, dv, covariates)
    tukey_res = safe_execute(tukey_hsd_test, df, iv, dv)

    out = {
        "One-way ANOVA": anova_res,
        "Kruskal-Wallis H": kw_res,
        "Eta Squared": eta_res,
        "Linear Regression": lin_reg_res,
    }
    if tukey_res and isinstance(tukey_res, dict) and "error" not in tukey_res and len(tukey_res) > 0:
        out["Tukey HSD"] = tukey_res
    return out


###############################################
# Case: Multinomial IV vs Multinomial DV


def generate_multinomial_indep_multinomial_dep_data(
    n=200, n_cat_iv=3, n_cat_dv=3, random_state=42
):
    np.random.seed(random_state)
    iv = np.random.choice(range(n_cat_iv), size=n)

    # Probability matrix P(DV=j | IV=i)
    # This is a bit arbitrary for synthetic data
    # Let's say:
    # IV=0 favors DV=0
    # IV=1 favors DV=1
    # IV=2 favors DV=2
    probs_map = {0: [0.6, 0.2, 0.2], 1: [0.2, 0.6, 0.2], 2: [0.2, 0.2, 0.6]}

    dv = []
    for val in iv:
        p = probs_map.get(val, [1 / n_cat_dv] * n_cat_dv)
        dv.append(np.random.choice(range(n_cat_dv), p=p))

    df = pd.DataFrame(
        {
            "independent_variable": iv,
            "dependent_variable": dv,
            "Age": np.random.randint(20, 70, size=n),
            "Gender": np.random.choice(["Male", "Female"], size=n),
        }
    )
    return df


def test_multinomial_indep_multinomial_dep(df, iv, dv, covariates=None):
    chi2_res = safe_execute(chi_square_test, df, iv, dv)
    v_res = safe_execute(cramers_v, df, iv, dv)
    mnlogit_res = safe_execute(multinomial_logistic_regression, df, iv, dv, covariates)

    return {
        "Chi-Square Test": chi2_res,
        "Cramer's V": v_res,
        "Multinomial Logistic Regression": mnlogit_res,
    }


if __name__ == "__main__":
    # Case 1: Continuous IV, Multinomial DV
    df_nomial = generate_cont_indep_multinomial_dep_data(n=200, n_categories=3)
    results1 = test_cont_indep_multinomial_dep(
        df_nomial,
        "independent_variable",
        "dependent_variable",
        covariates=["Age", "Gender"],
    )

    # Case 2: Continuous IV, Continuous DV
    df_cont = generate_cont_indep_cont_dep_data(n=200)
    results2 = test_cont_indep_cont_dep(
        df_cont,
        "independent_variable",
        "dependent_variable",
        covariates=["Age", "Gender"],
    )

    # Case 3: Continuous IV, Binary DV
    df_binary = generate_cont_indep_binary_dep_data(n=200)
    results3 = test_cont_indep_binary_dep(
        df_binary,
        "independent_variable",
        "dependent_variable",
        covariates=["Age", "Gender"],
    )

    # Case 4: Binary IV, Binary DV
    df_cv4 = generate_binary_indep_binary_dep_data(n=200)
    results4 = test_binary_indep_binary_dep(
        df_cv4,
        "independent_variable",
        "dependent_variable",
        covariates=["Age", "Gender"],
    )

    # Case 5: Binary IV, Continuous DV
    df_cv5 = generate_binary_indep_cont_dep_data(n=200)
    results5 = test_binary_indep_cont_dep(
        df_cv5,
        "independent_variable",
        "dependent_variable",
        covariates=["Age", "Gender"],
    )

    # Case 6: Binary IV, Multinomial DV
    df_cv6 = generate_binary_indep_multinomial_dep_data(n=200)
    results6 = test_binary_indep_multinomial_dep(
        df_cv6,
        "independent_variable",
        "dependent_variable",
        covariates=["Age", "Gender"],
    )

    # Case 7: Multinomial IV, Binary DV
    df_cv7 = generate_multinomial_indep_binary_dep_data(n=200)
    results7 = test_multinomial_indep_binary_dep(
        df_cv7,
        "independent_variable",
        "dependent_variable",
        covariates=["Age", "Gender"],
    )

    # Case 8: Multinomial IV, Continuous DV
    df_cv8 = generate_multinomial_indep_cont_dep_data(n=200)
    results8 = test_multinomial_indep_cont_dep(
        df_cv8,
        "independent_variable",
        "dependent_variable",
        covariates=["Age", "Gender"],
    )

    # Case 9: Multinomial IV, Multinomial DV
    df_cv9 = generate_multinomial_indep_multinomial_dep_data(n=200)
    results9 = test_multinomial_indep_multinomial_dep(
        df_cv9,
        "independent_variable",
        "dependent_variable",
        covariates=["Age", "Gender"],
    )

    print("===== Case 1: Continuous IV, Multinomial DV =====")
    print(results1)
    print("\n===== Case 2: Continuous IV, Continuous DV =====")
    print(results2)
    print("\n===== Case 3: Continuous IV, Binary DV =====")
    print(results3)
    print("\n===== Case 4: Binary IV, Binary DV =====")
    print(results4)
    print("\n===== Case 5: Binary IV, Continuous DV =====")
    print(results5)
    print("\n===== Case 6: Binary IV, Multinomial DV =====")
    print(results6)
    print("\n===== Case 7: Multinomial IV, Binary DV =====")
    print(results7)
    print("\n===== Case 8: Multinomial IV, Continuous DV =====")
    print(results8)
    print("\n===== Case 9: Multinomial IV, Multinomial DV =====")
    print(results9)
