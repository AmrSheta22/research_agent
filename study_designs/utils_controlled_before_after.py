import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.stats.contingency_tables import StratifiedTable
from statsmodels.discrete.discrete_model import MNLogit
from scipy import stats
from scipy.stats import chi2
from .utils_general_functions import safe_execute
from statsmodels.stats.anova import AnovaRM


# ==============================================================
# CASE 1: Binary IV, Continuous DV
# ==============================================================


def difference_in_differences(
    df, dv, treatment, time, covariates=None, subject_id=None
):
    """OLS DiD: DV ~ treatment * time (+ covariates)"""
    formula = f"{dv} ~ {treatment}*{time}"
    if covariates:
        formula += " + " + " + ".join(covariates)

    # Fit model
    if subject_id:
        # Cluster standard errors by subject_id
        model = smf.ols(formula, data=df).fit(
            cov_type="cluster", cov_kwds={"groups": df[subject_id]}
        )
    else:
        # Standard OLS
        model = smf.ols(formula, data=df).fit()
    # Find interaction term
    interaction_term1 = f"{treatment}:{time}"
    interaction_term2 = f"{time}:{treatment}"
    p_val = model.pvalues.get(
        interaction_term1, model.pvalues.get(interaction_term2, None)
    )
    # fallback if colon order differs or other issues? Statsmodels is usually consistent.

    return {"p_values": model.pvalues.to_dict()}


def two_way_mixed_anova(df, dv, treatment, time, subject_id="subject"):
    """Two-way mixed ANOVA: time (within) × treatment (between)."""

    aovrm = AnovaRM(
        df, depvar=dv, subject=subject_id, within=[time], between=[treatment]
    )
    res = aovrm.fit()
    return res.anova_table["Pr > F"].to_dict()



def ancova(df, dv, treatment, time, covariates=None):
    """ANCOVA: Posttest DV ~ treatment + Pretest DV (+ covariates)"""
    df["dv_post"] = df[df[time] == 1][dv]
    df["dv_pre"] = df[df[time] == 0][dv]
    formula = f"dv_post ~ {treatment} + dv_pre"
    if covariates:
        formula += " + " + " + ".join(covariates)
    model = smf.ols(formula, data=df).fit()
    return {"p_values": model.pvalues.to_dict()}


def linear_mixed_model(df, dv, treatment, time, covariates=None, subject_id="subject"):
    """
    Linear mixed model: DV ~ treatment * time (+ covariates) + (1|subject)
    """
    formula = f"{dv} ~ {treatment}*{time}"
    if covariates:
        formula += " + " + " + ".join(covariates)
    model = smf.mixedlm(formula, data=df, groups=df[subject_id]).fit()
    return {"p_values": model.pvalues.to_dict()}


def change_score_ttest(df, dv, treatment, time, subject_id="subject"):
    """t-test on (post-pre) change between groups."""
    wide = df.pivot(index=subject_id, columns=time, values=dv)
    wide["change"] = wide[1] - wide[0]
    groups = []
    for trt in df[treatment].unique():
        subj_ids = df[df[treatment] == trt][subject_id].unique()
        groups.append(wide.loc[subj_ids, "change"].dropna())
    tstat, pval = stats.ttest_ind(groups[0], groups[1], equal_var=False)
    return {"t": tstat, "p": pval}


def controlled_before_after_binary_iv_continuous_dv(
    df, dv, treatment, time, covariates=None, subject_id=None
):
    if subject_id:
        did_output = safe_execute(
            difference_in_differences, df, dv, treatment, time, covariates, subject_id
        )
        two_way_mixed_anova_output = safe_execute(
            two_way_mixed_anova, df, dv, treatment, time, subject_id
        )
        linear_mixed_model_output = safe_execute(
            linear_mixed_model, df, dv, treatment, time, covariates, subject_id
        )
        change_score_ttest_output = safe_execute(
            change_score_ttest, df, dv, treatment, time, subject_id
        )
        return {
            "Difference-in-Differences": did_output,
            "Two-way Mixed ANOVA": two_way_mixed_anova_output,
            "Linear Mixed-Effects Model": linear_mixed_model_output,
            "Change-score t-test": change_score_ttest_output,
        }
    else:
        did_output = safe_execute(
            difference_in_differences, df, dv, treatment, time, covariates
        )
        # ancova_output = ancova(df, dv, treatment, time, covariates)
        return {
            "Difference-in-Differences": did_output,
        }


# ==============================================================
# CASE 2: Binary IV, Binary DV
# ==============================================================


def difference_in_differences_logistic(
    df, dv, treatment, time, covariates=None, subject_id=None
):
    """Logit DiD: DV ~ treatment*time"""
    formula = f"{dv} ~ {treatment}*{time}"
    if covariates:
        formula += " + " + " + ".join(covariates)
    if subject_id:
        model = smf.logit(formula, data=df).fit(
            cov_type="cluster", cov_kwds={"groups": df[subject_id]}
        )
    model = smf.logit(formula, data=df).fit()
    return {"p_values": model.pvalues.to_dict()}


def gee_logistic(df, dv, treatment, time, subject_id="id", covariates=None):
    """GEE logistic with treatment × time."""
    formula = f"{dv} ~ {treatment}*{time}"
    if covariates:
        formula += " + " + " + ".join(covariates)
    fam = sm.families.Binomial()
    ind = Exchangeable()
    model = GEE.from_formula(
        formula, groups=df[subject_id], data=df, family=fam, cov_struct=ind
    ).fit()
    return {"p_values": model.pvalues.to_dict()}


def modified_poisson(df, dv, treatment, time, covariates=None):
    """Modified Poisson regression with robust SEs."""
    formula = f"{dv} ~ {treatment}*{time}"
    if covariates:
        formula += " + " + " + ".join(covariates)
    model = smf.glm(formula, data=df, family=sm.families.Poisson()).fit(cov_type="HC0")
    return {"p_values": model.pvalues.to_dict()}


def controlled_before_after_binary_iv_binary_dv_with_time(
    df, dv, treatment, time, covariates=None, subject_id=None
):
    if subject_id:
        did_output = safe_execute(
            difference_in_differences_logistic,
            df,
            dv,
            treatment,
            time,
            covariates,
            subject_id,
        )
        gee_logistic_output = safe_execute(
            gee_logistic, df, dv, treatment, time, subject_id, covariates
        )
    else:
        did_output = safe_execute(
            difference_in_differences_logistic, df, dv, treatment, time, covariates
        )
        gee_logistic_output = safe_execute(
            gee_logistic, df, dv, treatment, time, covariates
        )
    return {
        "Difference-in-Differences (Logit)": did_output,
        "GEE Logistic": gee_logistic_output,
    }


# ==============================================================
# CASE 3: Binary IV, Multinomial DV
# ==============================================================


def multinomial_logit_did(df, dv, treatment, time, covariates=None):
    """Multinomial logit DiD: DV ~ treatment*time."""
    df = df.copy()
    df["interaction"] = df[treatment] * df[time]
    X_vars = [treatment, time, "interaction"]
    if covariates:
        X_vars += covariates
    X = sm.add_constant(df[X_vars])
    y = pd.Categorical(df[dv]).codes
    model = MNLogit(y, X).fit(method="newton", maxiter=100, disp=False)
    return  model.pvalues.to_dict()


def lr_test_interaction(df, dv, treatment, time, covariates=None):
    """LR test for interaction in multinomial logit."""
    df = df.copy()
    df["interaction"] = df[treatment] * df[time]
    X_full = [treatment, time, "interaction"]
    X_reduced = [treatment, time]
    if covariates:
        X_full += covariates
        X_reduced += covariates
    y = pd.Categorical(df[dv]).codes
    model_full = MNLogit(y, sm.add_constant(df[X_full])).fit(disp=False)
    model_reduced = MNLogit(y, sm.add_constant(df[X_reduced])).fit(disp=False)
    lr_stat = 2 * (model_full.llf - model_reduced.llf)
    df_diff = model_full.df_model - model_reduced.df_model
    p_value = chi2.sf(lr_stat, df_diff)
    return {"p-value": p_value}


def controlled_before_after_binary_iv_multinomial_dv_with_time(
    df, dv, treatment, time, covariates=None, subject_id=None
):
    if subject_id:
        return {"Support": "Subject clustering is not supported for this test."}
    else:
        return {
            "Multinomial logit DID": safe_execute(
                multinomial_logit_did, df, dv, treatment, time, covariates
            ),
            "LR test for interaction": safe_execute(
                lr_test_interaction, df, dv, treatment, time, covariates
            ),
        }


# ==============================================================
# CASE 4: Binary IV, Ordinal DV
# ==============================================================
# Strategy: treat ordinal DV as continuous (→ linear models) or categorical (→ ordered logit).


def ordinal_as_continuous_did(df, dv, treatment, time, covariates=None):
    """Treat ordinal DV as continuous in DiD OLS."""
    return difference_in_differences(df, dv, treatment, time, covariates)


def ordinal_as_ordered_logit(df, dv, treatment, time, covariates=None):
    """Ordered logit with interaction."""
    from statsmodels.miscmodels.ordinal_model import OrderedModel

    df = df.copy()
    df["interaction"] = df[treatment] * df[time]
    exog_vars = [treatment, time, "interaction"]
    if covariates:
        exog_vars += covariates
    exog = df[exog_vars]
    y = pd.Categorical(df[dv], ordered=True)
    model = OrderedModel(y.codes, exog, distr="logit")
    res = model.fit(method="bfgs", disp=False)
    return {"p_values": res.pvalues.to_dict()}


# ==============================================================
# Example test datasets
# ==============================================================

if __name__ == "__main__":
    # Binary IV, Continuous DV
    subject_id = np.repeat(np.arange(1, 21), 2)
    time = np.tile([0, 1], 20)
    treatment = np.repeat([1, 0], 20)
    outcome = np.random.normal(size=40)
    df = pd.DataFrame(
        {
            "subject": subject_id,
            "time": time,
            "treatment": treatment,
            "outcome": outcome,
        }
    )
    print(difference_in_differences(df, "outcome", "treatment", "time"))
    #print(ancova(df, "outcome", "outcome", "treatment"))
    print(
        linear_mixed_model(
            df, dv="outcome", treatment="treatment", time="time", subject_id="subject"
        )
    )
    print(change_score_ttest(df, "outcome", "treatment", "time", "subject"))

    # Binary IV, Binary DV
    outcome = np.random.binomial(1, 0.5, size=40)
    df["outcome"] = outcome
    print(difference_in_differences_logistic(df, "outcome", "treatment", "time"))
    print(gee_logistic(df, "outcome", "treatment", "time", subject_id="subject"))
    print(modified_poisson(df, "outcome", "treatment", "time"))

    # Binary IV, Multinomial DV
    outcome = np.random.choice(["A", "B", "C"], size=40)
    df["outcome"] = outcome
    print(multinomial_logit_did(df, "outcome", "treatment", "time"))
    print(lr_test_interaction(df, "outcome", "treatment", "time"))

    # Binary IV, Ordinal DV
    outcome = np.random.choice([1, 2, 3, 4], size=40, replace=True)
    df["outcome"] = outcome
    print(ordinal_as_continuous_did(df, "outcome", "treatment", "time"))
    print(ordinal_as_ordered_logit(df, "outcome", "treatment", "time"))
