"""
Centralized prompt templates for every Research Agent skill.

All prompt text is extracted from the legacy code
(llm_generation/utils_prediction.py, literature/utils_literature_generation.py)
and exposed as class-level constants or classmethods that return
fully-assembled prompt strings.

Convention
----------
* Constants are named ``<SKILL>_<SECTION>`` in UPPER_SNAKE_CASE.
* Builder classmethods are named ``build_<skill>_prompt(...)`` and accept the
  minimum context needed to fill in the template.
"""

from __future__ import annotations

import json
from typing import Any, Optional


class PromptTemplates:
    """Single source of truth for all LLM prompts."""

    # ==================================================================
    # HYPOTHESIS
    # ==================================================================
    HYPOTHESIS_SYSTEM = (
        "As a research assistant, your task is to formulate a clear and "
        "testable hypothesis for a research study based on the provided "
        "information."
    )

    HYPOTHESIS_SUFFIX = (
        "\nFormulate a specific, measurable, relevant, and time-bound hypothesis."
        "\nReturn ONLY the hypothesis text."
        "\nDo not use a title like 'Hypothesis:'."
    )

    @classmethod
    def build_hypothesis_prompt(
        cls,
        *,
        name: str | None = None,
        problem_statement: str | None = None,
        research_type: str | None = None,
        indep_var: Any = None,
        dep_var: str | None = None,
        covar: Any = None,
        independent_var_type: str | None = None,
        dependent_var_type: str | None = None,
        indep_summary: str | None = None,
        dep_summary: str | None = None,
        hypothesis_draft: str | None = None,
    ) -> str:
        parts: list[str] = [cls.HYPOTHESIS_SYSTEM]

        if name:
            parts.append(f"Name: {name}")
        if problem_statement:
            parts.append(f"Problem Statement: {problem_statement}")
        if research_type:
            parts.append(f"Research Type: {research_type}")

        if indep_var:
            formatted = (
                ", ".join([f"'{v}'" for v in indep_var])
                if isinstance(indep_var, list)
                else f"'{indep_var}'"
            )
            parts.append(
                f"Independent Variable: {formatted} and its type is "
                f"{independent_var_type}. Here is a summary of the "
                f"independent variable: {indep_summary}"
            )

        if dep_var:
            parts.append(
                f"Dependent Variable: '{dep_var}' and its type is "
                f"{dependent_var_type}. Here is a summary of the "
                f"dependent variable: {dep_summary}"
            )

        if covar:
            formatted_covars = (
                ", ".join([f"'{v}'" for v in covar])
                if isinstance(covar, list)
                else f"'{covar}'"
            )
            parts.append(f"Covariates/Confounding Variables: {formatted_covars}")

        if hypothesis_draft:
            parts.append(f"Current Draft/Context for Hypothesis:\n{hypothesis_draft}")

        parts.append(cls.HYPOTHESIS_SUFFIX)
        return "\n\n".join(parts)

    # ==================================================================
    # ABSTRACT
    # ==================================================================
    ABSTRACT_SYSTEM = (
        "As a research assistant, your task is to write a comprehensive "
        "abstract for a research study based on the provided information. "
        "Regard this study already completed"
    )

    ABSTRACT_JSON_INSTRUCTION = (
        "\n CRITICAL: Write the abstract in the following JSON format with no "
        "extra text before or after the JSON object:\n"
        "The entire abstract MUST be concise and strictly 180 to 220 words total across all 4 keys combined. "
        "The JSON object must have the following keys:\n"
        "{\n"
        '    "Problem Statement": "(1-2 sentences, ~40 words: articulates the central research question and scientific rationale)",\n'
        '    "Methods": "(2 sentences, ~50 words: defines sample size N, grouping factors, continuous covariates, study design, and statistical modeling frameworks used)",\n'
        '    "Results": "(2-3 sentences, ~70 words: summarizes primary omnibus and regression/multilevel findings, reporting key statistics, effect sizes, and confidence intervals)",\n'
        '    "Conclusion": "(1-2 sentences, ~40 words: delivers a grounded scientific conclusion directly reflecting the empirical evidence, statistical precision, and practical implications)",\n'
        "}\n"
        "CRITICAL RULES:\n"
        "- DO NOT paste long raw results narratives or subheadings like 'Results: Empirical Results and Statistical Analysis'.\n"
        "- DO NOT write empty filler phrases like 'validated evidence addressing the primary research objectives'.\n"
        "- Base all numbers, sample sizes, and statistics strictly on the provided study facts.\n"
        "- Ensure total word count is strictly between 180 and 220 words."
    )

    @classmethod
    def build_abstract_prompt(
        cls,
        *,
        hypothesis: str | None = None,
        test_interpretation: str | None = None,
        sample_size: int | None = None,
        research_type: str | None = None,
        indep_var: str | None = None,
        dep_var: str | None = None,
        indep_summary: str | None = None,
        dep_summary: str | None = None,
        problem_statement: str | None = None,
        participants: str | None = None,
        inclusion_criteria: str | None = None,
        study_facts: Any = None,
    ) -> str:
        parts: list[str] = [cls.ABSTRACT_SYSTEM]

        if study_facts is not None:
            try:
                from research_agent.study_facts import generate_detailed_state_narrative
                facts_txt = generate_detailed_state_narrative(study_facts)
            except Exception:
                facts_txt = study_facts.to_prompt_context() if hasattr(study_facts, "to_prompt_context") else str(study_facts)
            parts.append(facts_txt)
            if hasattr(study_facts, "sample_size_total"):
                sample_size = study_facts.sample_size_total

        if problem_statement:
            parts.append(f"Problem Statement:\n{problem_statement}")
        if sample_size:
            parts.append(f"Sample Size:\n{sample_size}")
        if research_type:
            parts.append(f"Research Type:\n{research_type}")
        if indep_var:
            parts.append(
                f"Independent Variable:\n{indep_var}, with the summary {indep_summary}"
            )
        if participants:
            parts.append(f"Participants:\n{participants}")
        if inclusion_criteria:
            parts.append(f"Inclusion Criteria:\n{inclusion_criteria}")
        if hypothesis:
            parts.append(f"Hypothesis:\n{hypothesis}")
        if test_interpretation:
            parts.append(f"Test Interpretation:\n{test_interpretation}")
        if dep_var:
            parts.append(
                f"Dependent Variable:\n{dep_var}, with the summary {dep_summary}"
            )

        parts.append(cls.ABSTRACT_JSON_INSTRUCTION)
        return "\n\n".join(parts)

    # ==================================================================
    # METHODOLOGY
    # ==================================================================
    METHODOLOGY_SYSTEM = (
        "You are an expert scientific researcher and academic author. Your task is to write a rigorous, "
        "extensive, publication-grade Methodology (Materials and Methods) section for a top-tier peer-reviewed journal. "
        "The methodology must be technically detailed, mathematically rigorous, and organized into four clear subsections:\n"
        "1. Dataset Compilation, Provenance, and Inclusion Screening (sample size N, data archives, inclusion/exclusion criteria, outlier detection)\n"
        "2. Variable Operationalization and Measurement Protocols (exact units, variable definitions, operationalization protocols, covariates)\n"
        "3. Statistical Modeling Architecture and Formal Equation Specifications (ANOVA, effect size eta-squared, Kruskal-Wallis, multivariate OLS regression equation, and Linear Mixed Model for site clustering)\n"
        "4. Model Assumptions, Diagnostics, and Computational Implementation (VIF collinearity, normality, homoscedasticity, alpha threshold, software)\n\n"
        "CRITICAL RULES:\n"
        "- ZERO CONVERSATIONAL LEAKS: Output ONLY pure academic prose. NEVER begin with 'Here is a publication-grade Methodology section' or end with 'Note: All statistical outputs adhere strictly...'.\n"
        "- HIERARCHICAL STRUCTURE: Explicitly define the unit of observation and cluster structure (N_clusters, cluster groupings, and effective sample size N_eff). If hierarchical clustering or substantial ICC is present, specify the Linear Mixed Model (MixedLM) with random cluster intercepts fitted to account for intra-cluster correlation.\n"
        "- ACCURATE DIAGNOSTICS: Note that Shapiro-Wilk tested raw marginal data non-normality (motivating non-parametric alternatives) vs OLS residual diagnostics separately. Do NOT fabricate laboratory or procedural steps not present in the empirical dataset provenance.\n"
        "- Do NOT write a title like 'Methodology' at the top. Begin directly with the first subsection."
    )

    METHODOLOGY_STYLE_INSTRUCTION = (
        "\nWrite an extensive and mathematically rigorous methodology section formatted with clear markdown subheadings (###). "
        "CRITICAL: Do NOT write a top-level section title (e.g. '# Methods' or 'Methodology') at the top. "
        "The output should be a publication-ready block of text ready to be placed directly into a research paper."
    )

    METHODOLOGY_JSON_INSTRUCTION = (
        "CRITICAL: You may output either rich markdown with the 4 subsections above or write in JSON format with keys "
        "'Dataset Compilation', 'Variable Operationalization', 'Statistical Architecture', and 'Model Diagnostics'. "
        "Ensure each section contains substantive, thorough academic explanations."
    )

    @classmethod
    def build_methodology_prompt(
        cls,
        *,
        research_type: str | None = None,
        sample_size: int | None = None,
        participants: str | None = None,
        inclusion_criteria: str | None = None,
        collection_location: str | None = None,
        collection_time: str | None = None,
        indep_var: str | None = None,
        dep_var: str | None = None,
        covar: Any = None,
        indep_summary: str | None = None,
        dep_summary: str | None = None,
        statistical_analysis: str | None = None,
        signif_level: float = 0.05,
        software: str = "Python 3.10 and Significan Platform",
        test_names: list[str] | None = None,
        study_facts: Any = None,
    ) -> str:
        parts: list[str] = [cls.METHODOLOGY_SYSTEM]

        if study_facts is not None:
            try:
                from research_agent.study_facts import generate_detailed_state_narrative
                facts_txt = generate_detailed_state_narrative(study_facts)
            except Exception:
                facts_txt = study_facts.to_prompt_context() if hasattr(study_facts, "to_prompt_context") else str(study_facts)
            parts.append(facts_txt)
            if hasattr(study_facts, "sample_size_total"):
                sample_size = study_facts.sample_size_total

        if research_type:
            parts.append(f"Study Design Type: {research_type}")
        if sample_size:
            parts.append(f"Sample Size: {sample_size}")
        if participants:
            parts.append(f"Participants: {participants}")
        if inclusion_criteria:
            parts.append(f"Inclusion Criteria: {inclusion_criteria}")
        if collection_location:
            parts.append(f"Collection Location: {collection_location}")
        if collection_time:
            parts.append(f"Collection Time: {collection_time}")
        if indep_var:
            parts.append(
                f"Independent Variable: {indep_var}. Here is a summary of "
                f"the independent variable: {indep_summary}"
            )
        if dep_var:
            parts.append(
                f"Dependent Variable: {dep_var}. Here is a summary of "
                f"the dependent variable: {dep_summary}"
            )
        if covar:
            parts.append(f"Covariates: {covar}")
        if statistical_analysis:
            parts.append(f"Statistical Analysis: {statistical_analysis}")
        if test_names:
            parts.append(f"Test Names: {test_names}")

        parts.append(f"Significance Level: {signif_level}")
        parts.append(f"Software: {software}")
        parts.append(cls.METHODOLOGY_STYLE_INSTRUCTION)
        parts.append(cls.METHODOLOGY_JSON_INSTRUCTION)
        return "\n\n".join(parts)

    # ==================================================================
    # INTRODUCTION
    # ==================================================================
    INTRODUCTION_SYSTEM = (
        "You are an expert academic writer and lead scientific researcher. Your task is to write an "
        "extensive, publication-grade academic introduction for a top-tier peer-reviewed journal "
        "using the specific details provided below. Regard this study already completed."
    )

    INTRODUCTION_CONTEXT = (
        "The introduction must comprehensively establish the broader scientific and environmental "
        "context, provide deep theoretical foundations, clearly state the research problem, "
        "critique prior empirical methodologies, identify specific unconstrained parameters or gaps, "
        "and establish the concrete objectives and hypotheses of the study."
    )

    INTRODUCTION_JSON_INSTRUCTION = (
        "CRITICAL: Write the introduction in the following JSON format with "
        "no extra text before or after the JSON object:\n"
        "Each section must be detailed, comprehensive, and thoroughly developed (120-200 words each):\n"
        "{\n"
        '    "Problem statement": "(Establishes global context, foundational scientific mechanisms, environmental significance, and core definitions)",\n'
        '    "Literature review": "(Synthesizes theoretical background, prior empirical discoveries, and established scientific consensus using ONLY verified citations from the provided literature list)",\n'
        '    "Research Gap": "(Explicitly articulates unaddressed questions, empirical conflicts, or unconstrained parameters in current literature)",\n'
        '    "Gap importance": "(Explains the critical scientific and practical importance of resolving this gap)",\n'
        '    "Study objectives": "(Details specific investigation goals, tested empirical hypotheses, sample scope, and article organization)"\n'
        "}\n"
        "CRITICAL RULES FOR INTRODUCTION:\n"
        "1. ZERO RESULTS LEAKAGE: Do NOT report or quote empirical test results (e.g. eta^2, F-statistics, p-values, or ICC) in the Introduction! The Introduction sets the stage before results are known.\n"
        "2. VERIFIED CITATIONS ONLY: Cite ONLY papers present in the verified literature review list. Include at least 3-5 authentic in-text citations across the sections. NEVER invent phantom citations. Use standard in-text citations (e.g. Author et al., Year).\n"
        "3. PROVENANCE & REPOSITORY: Explicitly name and cite the data repository, database, or archive based on the provided study context and citations.\n"
        "4. OBJECTIVE HYPOTHESIS: Frame the hypothesis neutrally as estimating potential offsets with exact confidence intervals and uncertainty, rather than assuming a pre-confirmed significant difference. Note that null findings will be interpreted as statistically inconclusive under bounded statistical power.\n"
        "5. NO TANGENTIAL CLAIMS: Avoid unrelated practical tangents; maintain focus strictly on the foundational theoretical principles and substantive mechanisms of the empirical variables.\n"
        "6. ACCURATE VARIABLE DEFINITIONS: Refer consistently to the verified factor levels, dependent variables, and covariates designated in the study context.\n"
        "7. METHODOLOGICAL MOTIVATION: Ground the analytical framework (e.g. robust standard errors, non-parametric procedures, or multilevel clustering) in the empirical data structure.\n"
        "8. NO CONVERSATIONAL FILLER: Return strictly the requested JSON object."
    )

    @classmethod
    def build_introduction_prompt(
        cls,
        *,
        hypothesis: str | None = None,
        problem_statement: str | None = None,
        literature_review: str | None = None,
        indep_var: str | None = None,
        dep_var: str | None = None,
        study_facts: Any = None,
    ) -> str:
        parts: list[str] = [cls.INTRODUCTION_SYSTEM, cls.INTRODUCTION_CONTEXT]

        if study_facts is not None:
            try:
                from research_agent.study_facts import generate_detailed_state_narrative
                facts_txt = generate_detailed_state_narrative(study_facts)
            except Exception:
                facts_txt = study_facts.to_prompt_context() if hasattr(study_facts, "to_prompt_context") else str(study_facts)
            parts.append(facts_txt)

        if problem_statement:
            parts.append(f"Problem Statement:\n{problem_statement}")
        if indep_var and dep_var:
            parts.append(
                f"The central focus of this research is to investigate the "
                f"relationship in the hypothesis {hypothesis}. This is done by "
                f"examining the relationship between the dependent variable, "
                f"'{dep_var}', and the following independent variable(s): {indep_var}"
            )
        if hypothesis:
            parts.append(
                f'The hypothesis of the study is as follows: "{hypothesis}"'
            )
        if literature_review:
            parts.append(
                f"The following academic literature has been reviewed to inform "
                f"the introduction. The papers that were retrieved include:\n"
                f"{literature_review}\n\n Only use the information from these "
                f"papers to inform the introduction."
            )

        parts.append(cls.INTRODUCTION_JSON_INSTRUCTION)
        return "\n\n".join(parts)

    # ==================================================================
    # STRUCTURAL METADATA
    # ==================================================================
    METADATA_SYSTEM = (
        "As a research assistant, analyze the provided information to "
        "identify the following structural components of the research study:"
    )

    METADATA_RESEARCH_TYPES_DESCRIPTION = (
        "Here is a brief of the research types you are choosing from:\n"
        "Individual Randomized Controlled Trial (RCT):\n"
        "Participants are randomly assigned to intervention groups. The "
        "analysis focuses on the relationship between the independent "
        "variable (treatment) and the dependent variable (outcome).\n\n"
        "Crossover RCT:\n"
        "Each participant receives multiple interventions in different "
        "periods. In addition to the independent and dependent variables, "
        "a subject ID is required to account for repeated measurements "
        "within the same individual.\n\n"
        "Pre–Post Study:\n"
        "Outcomes are measured before and after an intervention in the same "
        "group. The design includes the independent variable, dependent "
        "variable, and time to capture change.\n\n"
        "Controlled Before–After Study:\n"
        "Outcomes are measured over time in both intervention and control "
        "groups. Analysis includes the independent variable, dependent "
        "variable, time, and subject ID to control for individual-level "
        "variation.\n\n"
        "Cross-Sectional Study:\n"
        "Exposure and outcome are measured at a single point in time. The "
        "design examines associations between independent and dependent "
        "variables without a time component.\n\n"
        "Case–Control Study:\n"
        "Participants are selected based on outcome status (case or control), "
        "and prior exposure is assessed. The analysis focuses on the "
        "relationship between independent and dependent variables.\n\n"
        "Retrospective Cohort Study:\n"
        "Groups are defined by past exposure, and outcomes are assessed using "
        "historical data. The design analyzes the association between "
        "independent and dependent variables over an observed follow-up period."
    )

    METADATA_JSON_INSTRUCTION = (
        "\nReturn the output as a strictly valid JSON object with keys: "
        "'independent_var', 'dependent_var', 'research_type', "
        "'time_variable', 'subject_id'.\n"
        "If a value is unknown or not applicable, use null.\n"
        "Do not include any explanation or markdown formatting "
        "(like ```json), just the raw JSON string."
    )

    METADATA_CRITICAL_SUFFIX = (
        "\n CRITICAL: Mainly use the user specified variables to predict "
        "remaining structural metadata if available. If not available, "
        "use your best guess."
    )

    @classmethod
    def build_structural_metadata_prompt(
        cls,
        *,
        data_columns: list[str],
        file_data_str: str | None = None,
        problem_statement: str | None = None,
        name: str | None = None,
        hypothesis: str | None = None,
        research_type: str | None = None,
        indep_var: str | None = None,
        dep_var: str | None = None,
    ) -> str:
        parts: list[str] = [cls.METADATA_SYSTEM]

        parts.append(
            f"1. Independent Variable (String) (choose one of {data_columns})"
        )
        parts.append(
            f"2. Dependent Variable (String) (choose one of {data_columns})"
        )
        parts.append(
            "3. Research Type (String, choose from: Independent RCT, "
            "Crossover RCT, Pre-post study, Controlled before-after, "
            "Cross-sectional, Case-control, Retrospective cohort)"
        )
        parts.append(
            f"4. Time Variable (String, column name representing time "
            f"periods. only retrieve for 'Controlled before-after', "
            f"'Crossover RCT', 'Pre-post study'. Else None), "
            f"(choose one of {data_columns})"
        )
        parts.append(
            f"5. Subject ID (String, column name representing unique "
            f"subject identifiers. only retrieve for 'Crossover RCT', "
            f"'Controlled before-after'. Else None), "
            f"(choose one of {data_columns})"
        )
        parts.append(cls.METADATA_JSON_INSTRUCTION)

        if file_data_str:
            parts.append(
                f"Dataset Preview (Header + First 5 rows):\n{file_data_str}\n"
                "If variables are present in the dataset, use the exact "
                "column names."
            )
            parts.append(
                "Read these row carefully and determine the required "
                "variables based on the available columns and the context."
            )
            parts.append(cls.METADATA_RESEARCH_TYPES_DESCRIPTION)

        if hypothesis:
            parts.append(f"User specified Hypothesis :\n{hypothesis}")
        if problem_statement:
            parts.append(f"User specified problem_statement :\n{problem_statement}")
        if name:
            parts.append(f"User specified Name:\n{name}")
        if research_type:
            parts.append(f"User specified Research Type: {research_type}")
        if indep_var:
            parts.append(f"User specified Independent Variables: {indep_var}")
        if dep_var:
            parts.append(f"User specified Dependent Variable: {dep_var}")

        parts.append(cls.METADATA_CRITICAL_SUFFIX)
        return "\n\n".join(parts)

    # ==================================================================
    # TEST INTERPRETATION
    # ==================================================================
    TEST_INTERPRETATION_TEMPLATE = (
        "You are a lead quantitative researcher and biometrician writing the Empirical Results and Statistical Analysis "
        "section of a top-tier peer-reviewed scientific paper.\n\n"
        "Study Context:\n"
        "- Research Type: {research_type}\n"
        "- Independent Variable: {indep_var}\n"
        "- Dependent Variable: {dep_var}\n"
        "- Dataset Preview:\n{dataset_preview}\n\n"
        "Statistical Test Results (JSON format):\n"
        "{test_results_json}\n\n"
        "Your task is to provide an in-depth, rigorous, and comprehensive Results section composed of 4 structured academic paragraphs:\n"
        "1. Overview & Sample Characteristics: Document total sample size N, data audit / outlier exclusions, cluster structure if hierarchical grouping is present, and factor level counts strictly matching the provided Study Facts.\n"
        "2. Inferential Omnibus Testing: Thoroughly interpret primary parametric tests (e.g. One-Way ANOVA F-statistic, degrees of freedom, p-value, effect size eta-squared) and non-parametric tests (e.g. Kruskal-Wallis H). Explicitly attribute omnibus effect size metrics to the omnibus test (not to regression models).\n"
        "3. Pairwise Contrasts & Epistemic Honesty: Report exact post-hoc pairwise contrasts with adjusted p-values, mean differences, and 95% confidence intervals. "
        "   CRITICAL RULE: Adhere strictly to empirical significance thresholds: do not claim non-significant contrasts (p >= 0.05) are distinct or significant. Report confidence intervals accurately with proper signs.\n"
        "4. Covariate Controls, Hierarchical Linear Mixed Model, & Epistemic Outcome: Report multiple regression alongside Linear Mixed Model fixed effects if hierarchical clustering is present.\n"
        "   - Report Intra-Class Correlation (ICC) and variance explained (marginal R^2_m and conditional R^2_c) if hierarchical models were estimated.\n"
        "   - Describe any directional shifts or attenuation in parameter estimates before and after covariate adjustment.\n"
        "   - If confidence intervals are wide or primary fixed effects are non-significant (p >= 0.05), emphasize that the empirical data are STATISTICALLY INCONCLUSIVE regarding independent factor effects after controlling for covariates and clustering, rather than claiming a proven zero effect.\n"
        "   - NEVER claim 'provides robust empirical confirmation of the research hypothesis' when primary fixed effects are non-significant.\n"
        "   - Refer to tables as 'Table 1' or 'Table 2'; NEVER write 'in the study facts' or 'presented in the study facts'.\n\n"
        "CRITICAL: Output ONLY pure academic text. Do NOT include conversational filler ('Here is the Results section', 'Note: ...')."
    )

    @classmethod
    def build_test_interpretation_prompt(
        cls,
        *,
        results: Any,
        research_type: str | None = None,
        indep_var: str | None = None,
        dep_var: str | None = None,
        dataset_preview: str | None = None,
        study_facts: Any = None,
    ) -> str:
        prompt_body = cls.TEST_INTERPRETATION_TEMPLATE.format(
            research_type=research_type,
            indep_var=indep_var,
            dep_var=dep_var,
            dataset_preview=dataset_preview,
            test_results_json=json.dumps(results, indent=2),
        )
        if study_facts is not None:
            facts_txt = study_facts.to_prompt_context() if hasattr(study_facts, "to_prompt_context") else str(study_facts)
            return f"{facts_txt}\n\n{prompt_body}"
        return prompt_body

    # ==================================================================
    # DISCUSSION & CONCLUSION
    # ==================================================================
    DISCUSSION_TEMPLATE = (
        "You are an expert scientist writing the Discussion section of a top-tier peer-reviewed academic paper.\n\n"
        "Study Details:\n"
        "- Title / Topic: {name}\n"
        "- Problem Statement: {problem_statement}\n"
        "- Hypothesis: {hypothesis}\n"
        "- Independent Variable: {indep_var}\n"
        "- Dependent Variable: {dep_var}\n"
        "- Covariates: {covars}\n"
        "- Statistical Results:\n{test_results_json}\n\n"
        "Write a thorough, multi-paragraph academic Discussion section (400-600 words) covering:\n"
        "1. Theoretical & Empirical Working Mechanisms:\n"
        "   - Ground the observed relationships in established scientific principles of the domain.\n"
        "   - Synthesize the physical, biological, or chemical pathways connecting {indep_var} and {dep_var}, maintaining complete consistency with the Introduction's theoretical framework.\n"
        "2. Epistemic Calibration & Statistical Synthesis:\n"
        "   - CRITICAL OPENING: If adjusted hierarchical model fixed effects are non-significant (p >= 0.05), you MUST open the Discussion by explicitly stating that after conditioning on continuous covariates and hierarchical clustering, the independent effect of {indep_var} on {dep_var} is statistically inconclusive across the empirical sample. NEVER open by claiming that {indep_var} exerts an independent influence.\n"
        "   - BANNED WORDS: NEVER use the word 'dissolve', 'vanish', or 'disappear'. Use 'become statistically inconclusive' or 'attenuate'.\n"
        "   - BANNED JARGON: NEVER use empty buzzwords such as 'preservation of full observational ecological validity'.\n"
        "   - Report exact empirical metrics from the Empirical Data Audit Narrative: unadjusted ANOVA (F, p, eta^2), post-hoc contrasts (group differences, 95% CIs, directional polarity), and multivariable/hierarchical model outputs.\n"
        "   - Maintain uniform contrast polarity (Non-baseline minus Baseline). Explicitly label all Tukey post-hoc tests as unadjusted bivariate comparisons on raw group means.\n"
        "   - If hierarchical clustering was present, discuss the Intraclass Correlation Coefficient (ICC), cluster structure, and variance partitioning (marginal R^2_m vs conditional R^2_c).\n"
        "   - If adjusted fixed effects are non-significant (p > 0.05) and confidence intervals are wide, explicitly emphasize that the findings are STATISTICALLY INCONCLUSIVE rather than proving absence of an effect.\n"
        "3. Statistical Power & Minimum Detectable Effect Sizes (MDES):\n"
        "   - Evaluate statistical power and detectable effect size thresholds given group sample sizes and standard errors.\n"
        "   - Explain that non-significance reflects bounded statistical power (MDES threshold) rather than proof of zero effect.\n"
        "4. Methodological Diagnostics & Robust Covariance:\n"
        "   - Clarify that residual heteroscedasticity (Breusch-Pagan) and non-normality (Shapiro-Wilk) motivate Huber-White HC3 robust standard errors on OLS; they do NOT validate the Linear Mixed Model. The LMM is motivated strictly by hierarchical clustering and high Intraclass Correlation (ICC).\n"
        "5. Integration with Verified Literature:\n"
        "   - Cite ONLY authentic peer-reviewed literature provided in the reference list. NEVER fabricate phantom citations or unverified author names.\n"
        "6. Methodological Strengths & Key Study Limitations:\n"
        "   - Transparently articulate core limitations based on the study facts: observational constraints, potential unmeasured confounders, sampling boundary constraints, and sample size imbalances across factor levels.\n"
        "   - Prioritize actionable recommendations for future research, prospective balanced sampling designs, and mechanistic process modeling across matched conditions.\n\n"
        "CRITICAL: Do NOT write a section title/heading (such as '# Discussion' or '**Discussion**' or 'Discussion') at the top. Output ONLY academic prose."
    )

    @classmethod
    def build_discussion_prompt(
        cls,
        *,
        name: str | None = None,
        problem_statement: str | None = None,
        hypothesis: str | None = None,
        indep_var: str | None = None,
        dep_var: str | None = None,
        covars: Any = None,
        results: Any = None,
        study_facts: Any = None,
    ) -> str:
        prompt_body = cls.DISCUSSION_TEMPLATE.format(
            name=name or "Empirical Scientific Study",
            problem_statement=problem_statement or "",
            hypothesis=hypothesis or "",
            indep_var=indep_var or "",
            dep_var=dep_var or "",
            covars=str(covars) if covars else "None",
            test_results_json=json.dumps(results or {}, indent=2),
        )
        if study_facts is not None:
            try:
                from research_agent.study_facts import generate_detailed_state_narrative
                facts_txt = generate_detailed_state_narrative(study_facts)
            except Exception:
                facts_txt = study_facts.to_prompt_context() if hasattr(study_facts, "to_prompt_context") else str(study_facts)
            return f"{facts_txt}\n\n{prompt_body}"
        return prompt_body

    CONCLUSION_TEMPLATE = (
        "You are an expert scientist writing the Conclusion section of a top-tier peer-reviewed academic paper.\n\n"
        "Study Details:\n"
        "- Title / Topic: {name}\n"
        "- Hypothesis: {hypothesis}\n"
        "- Independent Variable: {indep_var}\n"
        "- Dependent Variable: {dep_var}\n"
        "- Key Results Interpretation:\n{test_interpretation}\n\n"
        "Write a comprehensive, publication-ready Conclusion section (250-400 words) covering:\n"
        "1. Definitive Synthesis: Directly answer the central research question with empirical accuracy:\n"
        "   - Synthesize primary empirical findings, contrasting unadjusted associations with multivariate/hierarchical model outcomes.\n"
        "   - State clearly that after controlling for hierarchical clustering and continuous covariates, primary factor fixed effects did not attain statistical significance (p >= 0.05).\n"
        "   - Label Tukey post-hoc tests as unadjusted raw comparisons on sample means; NEVER claim they 'became non-significant under adjustment' (since Tukey was not re-run in an adjusted model).\n"
        "   - BANNED WORD: NEVER use the word 'dissolve', 'vanish', or 'disappear'. State that differences 'become statistically inconclusive' or 'attenuate'.\n"
        "   - Because estimated 95% confidence intervals remain wide and Minimum Detectable Effect Sizes bound detection power, state clearly that empirical findings are STATISTICALLY INCONCLUSIVE rather than proving zero effect.\n"
        "2. Scientific Significance & Methodological Implications: Detail practical recommendations for researchers in this field based on empirical findings, emphasizing the critical necessity of accounting for hierarchical clustering to prevent pseudo-replication.\n"
        "3. Future Empirical Research Directions: Propose 3 specific, actionable avenues for future empirical research: (1) high-resolution prospective or in situ continuous monitoring to track dynamic response processes; (2) expanded balanced sampling across underrepresented factor levels; and (3) incorporating forward mechanistic process models.\n\n"
        "CRITICAL: Do NOT write a section title/heading (such as '# Conclusion' or '**Conclusion**' or 'Conclusion') at the top. Output ONLY academic prose."
    )

    @classmethod
    def build_conclusion_prompt(
        cls,
        *,
        name: str | None = None,
        hypothesis: str | None = None,
        indep_var: str | None = None,
        dep_var: str | None = None,
        test_interpretation: str | None = None,
        study_facts: Any = None,
    ) -> str:
        prompt_body = cls.CONCLUSION_TEMPLATE.format(
            name=name or "Empirical Scientific Study",
            hypothesis=hypothesis or "",
            indep_var=indep_var or "",
            dep_var=dep_var or "",
            test_interpretation=test_interpretation or "",
        )
        if study_facts is not None:
            try:
                from research_agent.study_facts import generate_detailed_state_narrative
                facts_txt = generate_detailed_state_narrative(study_facts)
            except Exception:
                facts_txt = study_facts.to_prompt_context() if hasattr(study_facts, "to_prompt_context") else str(study_facts)
            return f"{facts_txt}\n\n{prompt_body}"
        return prompt_body

    # ==================================================================
    # CONFOUNDING VARIABLES
    # ==================================================================
    CONFOUNDING_AGENT_SYSTEM = (
        "You are a helpful assistant. Make sure to use the duckduckgo_search "
        "tool for information."
    )

    CONFOUNDING_PROMPT_TEMPLATE = (
        "Given that the dependent variable is {dep_var_str}, and the "
        "independent variables are [{indep_vars_str}], and knowing that the "
        "research type is '{research_type}', use your web search tool to find "
        "academic papers and studies that explore these relationships.\n"
        "\n"
        "From your search results, identify and list all possible confounding "
        "variables that could influence the relationship between the dependent "
        "and independent variables.\n"
        "\n"
        "Your final answer must be a single line containing only a "
        "comma-separated list of the confounding variables you found.\n"
        "For example: Age, Socioeconomic Status, Previous Illnesses\n"
        "\n"
        "Do not include any explanations, references, or introductory text "
        'like "Here are the confounding variables:".\n'
        "If you cannot find any relevant confounding variables after "
        'searching, your final answer must be the single word "None".'
    )

    @classmethod
    def build_confounding_prompt(
        cls,
        *,
        dependent_variable: str | None = None,
        dep_type: str | None = None,
        independent_variables: Any = None,
        independent_type: Any = None,
        research_type: str | None = None,
    ) -> str:
        # Format independent variables
        if independent_variables and independent_type:
            formatted_indeps = ", ".join(
                f"'{var}' (type: {t})"
                for var, t in zip(independent_variables, independent_type)
            )
        elif independent_variables:
            if isinstance(independent_variables, list):
                formatted_indeps = ", ".join(f"'{v}'" for v in independent_variables)
            else:
                formatted_indeps = f"'{independent_variables}'"
        else:
            formatted_indeps = "[]"

        # Format dependent variable
        if dependent_variable and dep_type:
            dep_var_str = f"'{dependent_variable}' (type: {dep_type})"
        elif dependent_variable:
            dep_var_str = f"'{dependent_variable}'"
        else:
            dep_var_str = "None"

        return cls.CONFOUNDING_PROMPT_TEMPLATE.format(
            dep_var_str=dep_var_str,
            indep_vars_str=formatted_indeps,
            research_type=research_type,
        )

    # ==================================================================
    # LITERATURE REVIEW
    # ==================================================================
    LITERATURE_AGENT_SYSTEM = (
        "You are an expert academic research assistant writing a comprehensive literature review. "
        "Synthesize high-impact, peer-reviewed academic literature directly relevant to the study variables and hypotheses. "
        "For each paper, compile a complete scholarly citation in standard APA format (starting with the author surnames, publication year, title, journal, volume, and pages/DOI), followed by an in-depth analytical review paragraph. "
        "\n\n"
        "CRITICAL: Your final response must be a SINGLE valid JSON LIST of objects with the following structure:\n"
        "[\n"
        "    {\n"
        '        "title": "Exact Article Title",\n'
        '        "citation": "Author, A. B., & Colleague, C. D. (2020). Exact Article Title. Journal of Scientific Research, 125(4), 101-120. https://doi.org/10.1016/example",\n'
        '        "review": "In-depth analytical review detailing the methodological approaches, empirical findings, and relevance to the research hypotheses."\n'
        "    }\n"
        "]\n"
        "CRITICAL: The 'citation' field must ALWAYS contain a real, authentic APA citation string starting directly with author names. NEVER output 'Citation not retrieved' or placeholder text. "
        "Do NOT include any markdown formatting (like ```json) in the final output. Just the raw JSON string."
    )

    LITERATURE_USER_TEMPLATE = (
        "Generate a literature review based on the following context:\n"
        "{research_context}\n"
        "Find at least 5 relevant papers. Ensure you get citations for them "
        "using the tool."
    )

    @classmethod
    def build_literature_review_context(
        cls,
        *,
        independent_variable: str | None = None,
        dependent_variable: str | None = None,
        research_type: str | None = None,
        hypothesis: str | None = None,
        name: str | None = None,
        problem_statement: str | None = None,
    ) -> str:
        """Build the research-context block and the full user message."""
        context_parts = [
            f"Independent Variable: {independent_variable}",
            f"Dependent Variable: {dependent_variable}",
            f"Research Type: {research_type}",
            f"Hypothesis: {hypothesis}",
            f"Name: {name}",
        ]
        if problem_statement:
            context_parts.append(f"Problem Statement: {problem_statement}")

        research_context = "\n".join(context_parts)
        return cls.LITERATURE_USER_TEMPLATE.format(
            research_context=research_context
        )
