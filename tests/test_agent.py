"""Unit tests for ResearchAgent orchestrator."""

from unittest.mock import patch
import pandas as pd
import numpy as np
import pytest

from research_agent.agent import ResearchAgent
from research_agent.config import AgentConfig
from research_agent.schemas import ResearchContext
from research_agent.skills.base import BaseSkill


@pytest.fixture
def agent():
    return ResearchAgent(AgentConfig())


def test_agent_initialization(agent):
    assert len(agent.tools) >= 9
    assert len(agent.skills) >= 8
    assert "read_file" in agent.tools
    assert "conduct_tests" in agent.tools
    assert "create_latex_paper" in agent.tools
    assert "hypothesis" in agent.skills
    assert "abstract" in agent.skills


def test_agent_run_statistical_tests(agent):
    np.random.seed(42)
    df = pd.DataFrame({
        "group": np.random.choice(["A", "B"], 80),
        "measurement": np.random.normal(100, 15, 80),
    })

    context = ResearchContext(
        research_type="cross_sectional",
        independent_variable="group",
        dependent_variable="measurement",
        independent_variable_type="binary",
        dependent_variable_type="numeric",
    )

    with patch.object(
        BaseSkill,
        "_invoke_with_fallback",
        return_value="The two-sample test demonstrated a statistically significant difference between groups."
    ):
        results, interpretation = agent.run_statistical_tests(context, df=df)

    assert isinstance(results, dict)
    assert "error" not in results
    assert isinstance(interpretation, str)
    assert "statistically significant" in interpretation


def test_agent_generate_hypothesis(agent):
    context = ResearchContext(
        independent_variable="treatment",
        dependent_variable="recovery",
        research_type="cross_sectional",
    )
    with patch.object(
        BaseSkill,
        "_invoke_with_fallback",
        return_value="Higher treatment leads to faster recovery.",
    ):
        hypo = agent.generate_hypothesis(context)
    assert isinstance(hypo, str)
    assert len(hypo) > 0


def test_agent_run_full_analysis(agent):
    context = ResearchContext(
        independent_variable="treatment",
        dependent_variable="recovery",
        research_type="cross_sectional",
    )
    df = pd.DataFrame({
        "treatment": ["A", "B", "A", "B"],
        "recovery": [10.0, 20.0, 12.0, 22.0],
    })
    with patch.object(
        BaseSkill,
        "_invoke_with_fallback",
        return_value="Mocked LLM generation response.",
    ):
        output = agent.run_full_analysis(context=context, df=df, tasks={"tests", "abstract"})
    assert output is not None
    assert output.abstract != ""
