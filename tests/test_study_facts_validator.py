"""
Unit tests for StudyFacts and PaperValidator integrity engine.
"""

import pandas as pd
import pytest
from unittest.mock import patch

from research_agent.schemas import ResearchContext, StudyFacts
from research_agent.study_facts import build_study_facts
from research_agent.validation.paper_validator import PaperValidator


@pytest.fixture
def speleothem_df():
    try:
        return pd.read_csv("research_agent/data/noaa_sisal_speleothem_lithology.csv")
    except Exception:
        # Synthetic fallback if file not in path
        return pd.DataFrame({
            "geology": ["limestone"] * 40 + ["dolomite"] * 30 + ["marble"] * 20 + ["mixed"] * 10,
            "mean_d18O": [-6.7] * 40 + [-6.4] * 30 + [-7.6] * 20 + [-5.6] * 10,
            "elevation": [500.0] * 100,
            "latitude": [35.0] * 100,
            "site_name": [f"cave_{i % 10}" for i in range(100)],
        })


def test_build_study_facts(speleothem_df):
    ctx = ResearchContext(
        independent_variable="geology",
        dependent_variable="mean_d18O",
        covariates=["elevation", "latitude"],
    )
    facts = build_study_facts(speleothem_df, ctx)

    assert isinstance(facts, StudyFacts)
    assert facts.sample_size_total == len(speleothem_df)
    assert facts.factor_name == "geology"
    assert facts.dependent_variable == "mean_d18O"
    assert "limestone" in facts.factor_levels
    assert facts.reference_level == "limestone"
    assert facts.anova_df_between == len(facts.factor_levels) - 1
    assert facts.anova_df_within == len(speleothem_df) - len(facts.factor_levels)
    assert facts.regression_model_type.startswith("Ordinary Least Squares")
    assert len(facts.regression_coefficients) >= 3

    # Check prompt context generation
    ctx_text = facts.to_prompt_context()
    assert "IMMUTABLE STUDY FACTS" in ctx_text
    assert f"Total Sample Size (N): {facts.sample_size_total}" in ctx_text
    assert f"F({facts.anova_df_between}, {facts.anova_df_within})" in ctx_text


def test_paper_validator_catches_discrepancies(speleothem_df):
    ctx = ResearchContext(
        independent_variable="geology",
        dependent_variable="mean_d18O",
        covariates=["elevation", "latitude"],
    )
    facts = build_study_facts(speleothem_df, ctx)
    validator = PaperValidator(facts)

    hallucinated_text = (
        "In our evaluation of N = 1,842 observations across 12 tropical Andean sites, "
        "we found F(4, 1837) = 6.88. Furthermore, volcanic tuff was assessed. "
        "limestone was less negative than dolomite, and rock lattice supplied lighter oxygen to the water."
    )

    issues = validator.validate_section("Results", hallucinated_text)
    assert len(issues) >= 3

    rules = [i.rule for i in issues]
    assert "sample_size_match" in rules
    assert "anova_df_match" in rules
    assert "directionality_inversion" in rules

    # Test Actor model native revision based on validator issues
    from research_agent.skills.actor import PaperActorReviser
    from research_agent.skills.base import BaseSkill
    actor = PaperActorReviser(None)
    with patch.object(
        BaseSkill,
        "_invoke_with_fallback",
        return_value=f"In our evaluation of N = {facts.sample_size_total} observations, we found F({facts.anova_df_between}, {facts.anova_df_within}) = 6.88.",
    ):
        revised = actor.revise_section("Results", hallucinated_text, [i.message for i in issues], facts)
    assert f"N = {facts.sample_size_total}" in revised
    assert f"F({facts.anova_df_between}, {facts.anova_df_within})" in revised
    assert "1,842" not in revised


def test_subgroup_and_effective_sample_sizes_preserved(speleothem_df):
    ctx = ResearchContext(
        independent_variable="geology",
        dependent_variable="mean_d18O",
        covariates=["elevation", "latitude"],
    )
    facts = build_study_facts(speleothem_df, ctx)
    validator = PaperValidator(facts)

    sub_strs = [f"{lvl} (n = {lvl_n})" for lvl, lvl_n in facts.level_sample_sizes.items()]
    correct_narrative = (
        f"The total cohort comprises N = {facts.sample_size_total} records with effective N = {facts.effective_sample_size:.1f}. "
        f"Subgroups included {', '.join(sub_strs)}."
    )

    issues = validator.validate_section("Results", correct_narrative)
    # None of the subgroup sample sizes or effective sample size should be flagged as errors!
    err_rules = [i.rule for i in issues if i.severity == "ERROR"]
    assert "sample_size_match" not in err_rules
    assert "subgroup_sample_size_match" not in err_rules

    # And auto-correction must NOT overwrite subgroup sizes with total N
    corrected = validator.correct_section_discrepancies("Results", correct_narrative)
    for lvl, lvl_n in facts.level_sample_sizes.items():
        assert f"{lvl} (n = {lvl_n})" in corrected
    assert f"effective N = {facts.effective_sample_size:.1f}" in corrected
    assert f"total cohort comprises N = {facts.sample_size_total}" in corrected
