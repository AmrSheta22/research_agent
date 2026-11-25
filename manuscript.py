"""Shared manuscript contracts: compact sections and structured estimate tables."""
import json
from research_agent.latex_templates.formatter import latex_escape


def p_text(value):
    return "< 0.0001" if value < .0001 else f"= {value:.4f}"


def coefficient_table(facts):
    import ast
    def label(value):
        if value.startswith("Q(") and value.endswith(")"):
            try:
                return str(ast.literal_eval(value[2:-1]))
            except (ValueError,SyntaxError):
                pass
        return value
    def probability(value):
        return "< 0.0001" if value < .0001 else f"{value:.4f}"
    rows = []
    for model, coefs in [("OLS", facts.regression_coefficients), ("LMM", facts.lmm_fixed_effects)]:
        for c in coefs:
            rows.append(f"{model} & {latex_escape(label(c.parameter))} & {c.coef:.4g} & {c.std_err:.4g} & [{c.ci_lower:.4g}, {c.ci_upper:.4g}] & ${probability(c.p_value)}$ " + r"\\")
    for model, estimates, se_key in [("HC3", facts.ols_robust_se, "std_err_hc3"), ("Cluster robust", facts.cluster_robust_ols, "std_err")]:
        for name, v in estimates.items():
            rows.append(f"{model} & {latex_escape(label(name))} & {v['coef']:.4g} & {v[se_key]:.4g} & [{v['ci_lower']:.4g}, {v['ci_upper']:.4g}] & ${probability(v['p_value'])}$ " + r"\\")
    if not rows:
        return ""
    return (r"\begin{table}[htbp]\centering\scriptsize\caption{Model estimates and 95\% confidence intervals. Outcome units: " + latex_escape(facts.variable_units.get(facts.dependent_variable,"not supplied")) + r". HC3 addresses heteroscedasticity; cluster-robust estimates use the same grouping variable as the LMM. OLS, HC3 and cluster-robust rows share OLS point estimates and differ in their covariance estimates.}\label{tab:estimates}" + "\n" +
            r"\begin{tabularx}{\linewidth}{lXrrrr}\toprule Model & Parameter & Estimate & SE & 95\% CI & $p$ \\ \midrule" + "\n" +
            "\n".join(rows) + "\n" + r"\bottomrule\end{tabularx}\end{table}")


def abstract_text(f):
    effects = [c for c in (f.lmm_fixed_effects if f.lmm_executed else f.regression_coefficients) if c.parameter.startswith(f.factor_name+":")]
    ranges = "; ".join(f"{c.parameter.split(':',1)[1].strip()}: {c.coef:+.2f} (95% CI {c.ci_lower:+.2f} to {c.ci_upper:+.2f})" for c in effects)
    mdes = ", ".join(f"{k}: {v:.2f}" for k,v in f.mdes_by_factor_level.items())
    outcome=f.dependent_variable.removeprefix("mean_").replace("_", " ")
    units=f.variable_units.get(f.dependent_variable,"outcome units")
    model="random-intercept mixed-model" if f.lmm_executed else "covariate-adjusted OLS"
    return (f"This observational study estimates mean {outcome} offsets ({units}) across {f.factor_name} categories, "
            f"adjusting for {', '.join(f.covariates)}. The analytical extract included {f.sample_size_total} records "
            f"from {f.num_clusters_total or 'unreported'} clusters. Unadjusted group comparisons were complemented by "
            "covariate-adjusted regression and, where supported, a random-intercept model for clustered observations. "
            f"The {model} contrasts were {ranges}. Approximate minimum detectable offsets at 80% power were {mdes}, "
            f"in {units}. These planning estimates depend on fitted standard errors and do not explain why any "
            "particular contrast is non-significant. Estimates describe conditional associations within the analyzed "
            "extract; their confidence intervals characterize sampling uncertainty under the fitted model. "
            "Unadjusted omnibus significance does not establish an independent categorical effect after covariate "
            "and cluster adjustment. Limited subgroup representation, unmeasured environmental controls, and incomplete "
            "upstream selection metadata restrict interpretation. A non-significant contrast establishes neither zero "
            "effect nor equivalence. The analysis provides estimates and precision for subsequent validation rather "
            "than a causal test of substrate control.")


def methods_text(f, extra=None):
    from research_agent.reporting import methodology_narrative
    text = methodology_narrative({"research_type":"cross-sectional"}, {}, f)
    text += " OLS cave-cluster-robust covariance uses finite-sample correction and t-based intervals. The mixed model uses REML and asymptotic fixed-effect intervals. "
    text += "Sensitivity analysis applies a single fixed-rule global 1.5-IQR outcome fence to the retained sample; it does not select an analysis by significance. "
    text += "Mixed-model residuals and random-intercept estimates were assessed separately. Empirical random-effect normality tests are descriptive because fitted random effects are shrunken estimates. "
    text += "Targeted leave-one-cluster-out fits exclude clusters from each non-reference factor level, use the same cluster variable, and report convergence counts and coefficient ranges. "
    return text


def results_text(f, extra=None):
    text = (f"The analytical sample included N = {f.sample_size_total} observations from {f.num_clusters_total} clusters. "
            f"Unadjusted ANOVA: F({f.anova_df_between}, {f.anova_df_within}) = {f.anova_f:.3f}, p {p_text(f.anova_p)}, "
            f"eta-squared = {f.eta_squared:.4f}. Kruskal-Wallis H({f.kruskal_df}) = {f.kruskal_h:.3f}, p {p_text(f.kruskal_p)}. "
            "All adjusted estimates and uncertainty intervals appear in Table 1. OLS and HC3 are sensitivity specifications; "
            "cluster-robust OLS and the mixed model account for cave dependence. ")
    text += "Group sizes were " + ", ".join(f"{k}: {v}" for k,v in f.level_sample_sizes.items()) + ". "
    if f.lmm_executed:
        text += f"LMM ICC = {f.lmm_icc}; design effect = {f.design_effect}, approximate effective sample size = {f.effective_sample_size}; marginal and conditional R-squared = {f.lmm_marginal_r2} and {f.lmm_conditional_r2}. "
    if f.shapiro_wilk_p is not None:
        text += f"OLS residual Shapiro-Wilk p {p_text(f.shapiro_wilk_p)}. "
    if f.breusch_pagan_p is not None:
        text += f"OLS Breusch-Pagan p {p_text(f.breusch_pagan_p)}. "
    d=f.additional_diagnostics
    for label in ["lmm_residuals","lmm_random_intercepts"]:
        v=d.get(label)
        if v:
            text += f"{label.replace('_',' ')}: Shapiro-Wilk W = {v['shapiro_w']:.4f}, p {p_text(v['p_value'])}. "
    v=d.get("entities_per_cluster")
    if v:
        text += f"Entities per cave ranged from {v['min']} to {v['max']}, median {v['median']}; {v['singletons']} caves had one entity. "
    v=d.get("extreme_value_sensitivity",{})
    if v:
        text += f"The global 1.5-IQR sensitivity fence [{v['lower']:.3f}, {v['upper']:.3f}] retained {v['n']} records and excluded {v['excluded']}. "
        if "p_value" in v:
            text += f"Sensitivity ANOVA F = {v['anova_f']:.3f}, p {p_text(v['p_value'])}. "
        text += f"Sensitivity mixed-model convergence: {v.get('converged','unavailable')}. Full sensitivity estimates are in the analysis audit. "
    if extra:
        sensitivity=extra.get("sensitivity_analysis",{}).get("sensitivity_by_level",{})
        for level,v in sensitivity.items():
            text += f"Targeted non-reference cave exclusion for {level}: {v['successful_fits']}/{v['clusters_tested']} successful fits; coefficient range {v['jackknife_coefficient_range']}. "
        if extra.get("data_provenance",{}).get("unavailable"):
            text += "The upstream selection ledger, original lithology coding rules, and missing environmental controls cannot be reconstructed from this extract. Repository searches are documented separately and do not establish local exclusions. "
    return text


def enforce_budget(skill, text, maximum, fallback):
    if len(text.split()) <= maximum:
        return text
    from langchain_core.messages import HumanMessage
    try:
        revised=skill._invoke_with_fallback([HumanMessage(content=f"Edit this section to at most {maximum} words. Preserve uncertainty and citations. Refer to the estimate table instead of repeating numbers. Return prose only.\n"+text)],temperature=0)
        if 10 <= len(revised.split()) <= maximum:
            return revised
    except Exception:
        pass
    return fallback


def conclusion_text(facts):
    primary=[c for c in (facts.lmm_fixed_effects if facts.lmm_executed else facts.regression_coefficients) if c.parameter.startswith(facts.factor_name+":")]
    uncertain=bool(primary) and all(c.p_value>=.05 for c in primary)
    alternative=any(v["p_value"]<.05 for k,v in facts.cluster_robust_ols.items() if k.startswith(facts.factor_name+":"))
    model="primary random-intercept mixed model" if facts.lmm_executed else "covariate-adjusted OLS specification"
    if not primary:
        return "Validated group-offset estimates are unavailable; substantive conclusions require a successful model fit."
    text=(f"The {model} yielded imprecise group-offset estimates whose intervals included zero. " if uncertain else
          f"The {model} provides model-conditional group-offset estimates and uncertainty intervals (Table 1). ")
    if uncertain and alternative:
        text+="Cluster-robust OLS supported at least one contrast, so inference depends on model specification. "
    text+="These observational associations establish neither causal factor effects nor equivalence. Sampling, subgroup representation, relevant controls and provenance require verification before broader generalization."
    return text


def discussion_text(facts, extra=None):
    text=conclusion_text(facts)
    text+="\n\nUnadjusted omnibus comparisons and conditional group offsets answer different questions. A detectable omnibus difference does not establish an independent effect under every adjustment model. Differences between OLS and mixed-model estimates should be treated as specification sensitivity, rather than proof of confounding. Table 1 reports the uncertainty for each specification."
    diagnostics=facts.additional_diagnostics
    if any(diagnostics.get(k,{}).get("p_value",1)<.05 for k in ("lmm_residuals","lmm_random_intercepts")):
        text+="\n\nMixed-model diagnostics detected departures from Gaussian assumptions. Its intervals are conditional on those assumptions; the cluster-robust OLS comparison is an important sensitivity analysis. Tests of fitted random intercepts are descriptive because those estimates are shrunken, rather than direct observations of the random-effect distribution."
    if extra and extra.get("sensitivity_analysis",{}).get("sensitivity_by_level"):
        levels=extra["sensitivity_analysis"]["sensitivity_by_level"]
        failures=sum(len(v.get("failed_fits",[])) for v in levels.values())
        text+="\n\nCave-exclusion ranges describe successful fits for the targeted non-reference groups. "
        text+=(f"There were {failures} failed exclusion fits, limiting claims of stability. " if failures else "All attempted exclusion fits converged in this run. ")
        text+="The fixed-rule outcome-fence comparison is a sensitivity analysis, not a replacement selected for its p-value."
    unavailable=(extra or {}).get("data_provenance",{}).get("unavailable",[])
    if unavailable:
        text+="\n\nUnresolved metadata in this extract include "+", ".join(unavailable)+". Mineral phase and chronology must be verified from appropriate measurements, not inferred from formation-type labels. These limitations require additional metadata before broader claims."
    return text
