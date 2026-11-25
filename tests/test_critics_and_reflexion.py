"""
Unit tests for Paired Critics, Reflexion engine, and Cross-Section Auditing.
"""

import pytest
from research_agent.config import AgentConfig
from research_agent.schemas import PairwiseContrast, RegressionCoefficient, StudyFacts
from research_agent.skills.critic import (
    GlobalPaperCritic,
    IntroConclusionCritic,
    MethodsResultsCritic,
    ResultsDiscussionCritic,
)
from research_agent.validation.paper_validator import PaperValidator


@pytest.fixture
def mock_study_facts():
    return StudyFacts(
        sample_size_total=542,
        factor_name="treatment_group",
        factor_levels=["control", "group_b", "group_c", "group_d"],
        reference_level="control",
        level_sample_sizes={"control": 417, "group_b": 58, "group_c": 47, "group_d": 20},
        dependent_variable="score",
        covariates=["covariate_1", "covariate_2"],
        anova_f=6.8754,
        anova_df_between=3,
        anova_df_within=538,
        anova_p=0.00015,
        eta_squared=0.0369,
        kruskal_h=18.1661,
        kruskal_df=3,
        kruskal_p=0.00041,
        lmm_executed=True,
        lmm_icc=0.8945,
        cluster_variable="site_id",
        num_clusters_total=245,
        effective_sample_size=260.5,
        pairwise_contrasts=[
            PairwiseContrast(
                group1="control",
                group2="group_d",
                mean1=9.30,
                mean2=8.30,
                diff=1.0,
                p_value=0.50,
                significant=False,
                direction="control is not significantly different from group_d",
            )
        ],
        regression_coefficients=[
            RegressionCoefficient(parameter="const", coef=8.5, std_err=0.4, t_value=21.2, p_value=0.0001, ci_lower=7.7, ci_upper=9.3),
            RegressionCoefficient(parameter="covariate_1", coef=-1.8147, std_err=0.2, t_value=-9.07, p_value=0.00001, ci_lower=-2.2, ci_upper=-1.4),
            RegressionCoefficient(parameter="covariate_2", coef=-0.0393, std_err=0.01, t_value=-3.93, p_value=0.00007, ci_lower=-0.06, ci_upper=-0.02),
        ],
    )


def test_methods_results_critic_sample_mismatch(mock_study_facts):
    critic = MethodsResultsCritic(AgentConfig())
    methods_text = "A cross-sectional study was conducted on N = 1842 empirical records across 12 study sites."
    results_text = "The sample consisted of 542 empirical records. ANOVA revealed F(3, 538) = 6.88, p = 0.00015."
    
    eval_res = critic.evaluate(methodology=methods_text, results_text=results_text, study_facts=mock_study_facts)
    assert not eval_res.passed
    assert any("Sample size mismatch" in issue for issue in eval_res.issues)


def test_results_discussion_critic_epistemic_honesty(mock_study_facts):
    critic = ResultsDiscussionCritic(AgentConfig())
    results_text = "Group_d did not differ significantly from control (p = 0.50)."
    discussion_text = "Our findings unequivocally confirm that group_d is distinct from all other groups with a highly significant depletion effect."
    
    eval_res = critic.evaluate(results_text=results_text, discussion_text=discussion_text, study_facts=mock_study_facts)
    assert not eval_res.passed
    assert any("significant" in issue.lower() for issue in eval_res.issues)


def test_intro_conclusion_critic_unkept_promises(mock_study_facts):
    critic = IntroConclusionCritic(AgentConfig())
    intro_text = "This paper proposes to implement bayesian hierarchical calibration and non-linear interaction terms to predict outcome values."
    conclusion_text = "We demonstrated that treatment group accounts for baseline offsets."
    
    eval_res = critic.evaluate(introduction=intro_text, conclusion=conclusion_text, study_facts=mock_study_facts)
    assert not eval_res.passed
    assert any("unkept promise" in issue for issue in eval_res.issues)
    assert any("proposal" in issue.lower() for issue in eval_res.issues)


def test_global_paper_critic_ai_leak_purge(mock_study_facts):
    critic = GlobalPaperCritic(AgentConfig())
    abstract = "Here is a publication-grade abstract for the research paper on treatment groups."
    intro = "Outcome score serves as a key proxy."
    methods = "Hierarchical mixed-effects models were fitted across 245 sites."
    results = "Results confirmed F = 6.88, p = 0.00015."
    discussion = "Treatment effects modulate observed variance."
    conclusion = "Note: All statistical outputs adhere strictly to the empirical dataset."

    eval_res = critic.evaluate(
        abstract=abstract,
        introduction=intro,
        methodology=methods,
        results_text=results,
        discussion=discussion,
        conclusion=conclusion,
        study_facts=mock_study_facts,
    )
    assert not eval_res.passed
    assert any("conversational" in issue.lower() for issue in eval_res.issues)


def test_paper_validator_auto_sanitization(mock_study_facts):
    validator = PaperValidator(mock_study_facts)
    dirty_intro = "Here is the introduction: This paper would provide an evaluation that proposes to use cross-validation."
    clean_intro = validator.correct_section_discrepancies("Introduction", dirty_intro)
    
    assert "Here is" not in clean_intro
    assert "would provide" not in clean_intro
    assert "proposes to" not in clean_intro
    assert "provides" in clean_intro
