"""Deterministic prose that reports computed quantities without inventing findings."""

from typing import Any

from research_agent.lats.evaluator import LATSEvaluator


def factor_summary(facts: Any) -> str:
    effects = facts.lmm_fixed_effects if facts.lmm_executed else facts.regression_coefficients
    effects = [c for c in effects if c.parameter.startswith(f"{facts.factor_name}:")]
    if not effects:
        return "No fitted factor contrast estimates are available; adjusted effects cannot be concluded."
    model = "mixed-effects" if facts.lmm_executed else "OLS"
    estimates = [
        f"{c.parameter}: coefficient {c.coef:+.4g}, 95% CI [{c.ci_lower:.4g}, {c.ci_upper:.4g}], p={c.p_value:.4g}"
        for c in effects
    ]
    return f"Computed {model} factor contrasts: " + "; ".join(estimates) + "."


def results_narrative(context: dict, results: dict, facts: Any = None) -> str:
    if facts is None:
        pairs = LATSEvaluator.extract_p_values(results or {})
        if not pairs:
            return "No valid inferential p-values are available. Statistical conclusions require a successful analysis."
        return ("The executed analyses returned the following p-values: "
                + "; ".join(f"{name}: {LATSEvaluator.format_p_value(p)}" for name, p in pairs)
                + ". These exploratory results must be interpreted with the study design, effect estimates, and multiplicity in mind.")
    parts = [f"The analytical sample included N = {facts.sample_size_total} observations."]
    parts.append("Group sample sizes: " + ", ".join(f"{k} (n = {v})" for k, v in facts.level_sample_sizes.items()) + ".")
    parts.append(
        f"Unadjusted one-way ANOVA: F({facts.anova_df_between}, {facts.anova_df_within}) = {facts.anova_f:.4g}, "
        f"p={facts.anova_p:.4g}, eta-squared={facts.eta_squared:.4g}. "
        f"Kruskal-Wallis: H({facts.kruskal_df}) = {facts.kruskal_h:.4g}, p={facts.kruskal_p:.4g}."
    )
    if facts.lmm_executed:
        parts.append(f"A mixed-effects model used {facts.num_clusters_total} clusters in {facts.cluster_variable}; ICC={facts.lmm_icc}.")
    parts.append(factor_summary(facts))
    if facts.ols_robust_se:
        parts.append("HC3 covariance estimates were computed for OLS: " + "; ".join(
            f"{name}: SE={v['std_err_hc3']:.4g}, p={v['p_value']:.4g}"
            for name, v in facts.ols_robust_se.items()) + ".")
    for contrast in facts.pairwise_contrasts:
        parts.append(f"Tukey comparison {contrast.group2} minus {contrast.group1}: difference={contrast.diff:+.4g}, adjusted p={contrast.p_value:.4g}.")
    return "\n\n".join(parts)


def methodology_narrative(context: dict, results: dict, facts: Any = None) -> str:
    design = context.get("research_type") or "unspecified"
    if facts is None:
        tests = ", ".join(k for k in (results or {}) if not k.startswith("_")) or context.get("test_names") or "not supplied"
        return (f"The supplied research design is {design}. The selected predictor is {context.get('independent_variable') or 'unspecified'} "
                f"and the outcome is {context.get('dependent_variable') or 'unspecified'}. "
                f"Recorded analyses: {tests}. A complete validated study-facts contract is unavailable for this design. "
                "Sampling procedures, preprocessing, and unreported diagnostic tests must be documented by the investigators.")
    text = (f"This {design} analysis used N = {facts.sample_size_total} complete observations. "
            f"The factor {facts.factor_name} has levels {', '.join(facts.factor_levels)}; the numeric outcome is {facts.dependent_variable}. "
            "Unadjusted group comparisons used one-way ANOVA, Kruskal-Wallis, and Tukey HSD. ")
    if facts.regression_coefficients:
        text += f"OLS used reference category {facts.reference_level} and covariates {', '.join(facts.covariates) or 'none'}. "
    if facts.ols_robust_se:
        text += "HC3 heteroscedasticity-robust standard errors were also computed. "
    if facts.lmm_executed:
        text += f"A converged random-intercept mixed model grouped observations by {facts.cluster_variable}. "
    if facts.manifest:
        m = facts.manifest
        method = m.outlier_method
        if not method or str(method).casefold() in {"none","no_outlier_removal"}:
            method = "no outlier removal"
        text += (f"The cleaning audit records {m.raw_sample_size} input rows, {m.missing_rows_removed} exclusions for missing or invalid values, "
                 f"and {m.outliers_removed} outlier exclusions using {method}. ")
        if m.variable_transformations:
            text += "Recorded transformations: " + str(m.variable_transformations) + ". "
    diagnostics = []
    if facts.shapiro_wilk_p is not None:
        diagnostics.append("Shapiro-Wilk residual normality")
    if facts.breusch_pagan_p is not None:
        diagnostics.append("Breusch-Pagan heteroscedasticity")
    if facts.vif_max is not None:
        diagnostics.append("variance inflation factors")
    if diagnostics:
        text += "Computed diagnostics: " + ", ".join(diagnostics) + "."
    return text


def conclusion_narrative(context: dict, results: dict, facts: Any = None) -> str:
    if facts is None:
        return "A validated conclusion is unavailable. Review the executed statistical results and study design before interpreting the research hypothesis."
    return (factor_summary(facts) + " These estimates describe associations in the analyzed sample. "
            "A non-significant contrast alone establishes neither absence of an effect nor practical equivalence. "
            "Causal and population-level conclusions require support from the sampling and study design.")
