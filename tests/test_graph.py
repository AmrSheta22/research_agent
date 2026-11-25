"""Unit tests for Research Agent StateGraph workflow."""

from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import pytest

from research_agent.agent import ResearchAgent
from research_agent.config import AgentConfig
from research_agent.graph import build_research_graph, ResearchGraphState
from research_agent.schemas import LiteratureItem, ResearchContext, VisualizationSpec
from research_agent.skills.base import BaseSkill
from research_agent.skills.literature_review import LiteratureReviewSkill
from research_agent.tools.visualization_tools import PlanVisualizationsTool


@pytest.fixture
def agent():
    return ResearchAgent(AgentConfig())


def test_stategraph_structure(agent):
    """Test that all required states are registered in the StateGraph."""
    graph = agent.graph
    node_keys = list(graph.nodes.keys())
    
    # Must mirror original workflow states
    expected_states = [
        "ingest_and_clean",
        "metadata_prediction",
        "conduct_tests",
        "literature_review",
        "introduction",
        "methodology",
        "abstract",
        "visualization",
        "playground",
        "create_latex_paper",
    ]
    for state_name in expected_states:
        assert state_name in node_keys, f"Expected state '{state_name}' missing from StateGraph"


def test_stategraph_execution_flow(agent):
    """Test traveling from state to state through the StateGraph."""
    np.random.seed(42)
    df = pd.DataFrame({
        "treatment": np.random.choice(["Drug", "Placebo"], 60),
        "recovery_time": np.random.normal(14, 3, 60),
        "age": np.random.normal(45, 10, 60),
    })

    context = ResearchContext(
        name="Clinical Trial Test",
        research_type="cross_sectional",
        independent_variable="treatment",
        dependent_variable="recovery_time",
        independent_variable_type="binary",
        dependent_variable_type="numeric",
        hypothesis="Drug reduces recovery time compared to placebo.",
    )

    mock_lit_items = [
        LiteratureItem(
            title="Efficacy of Clinical Drug",
            citation="Smith et al. (2024)",
            review="Demonstrated notable recovery acceleration.",
        )
    ]

    mock_viz_specs = [
        {
            "title": "BoxPlot of treatment vs recovery_time",
            "mark": "BoxPlot",
            "x_axis": "treatment",
            "y_axis": "recovery_time",
            "color": None,
        }
    ]

    with patch.object(
        BaseSkill,
        "_invoke_with_fallback",
        return_value='{"Problem Statement": "Examining drug effect.", "Methods": "Cross-sectional analysis.", "Results": "Significant reduction.", "Conclusion": "Treatment is effective."}'
    ), patch.object(
        LiteratureReviewSkill,
        "run",
        return_value=mock_lit_items
    ), patch.object(
        PlanVisualizationsTool,
        "execute",
        return_value=MagicMock(success=True, data=mock_viz_specs)
    ):
        final_state: ResearchGraphState = agent.run_graph(context=context, df=df, compile_pdf=False)

    # Verify state transitions and status updates
    assert final_state["test_status"] == "completed"
    assert "test_results" in final_state
    assert final_state["literature_review_status"] == "completed"
    assert len(final_state["literature_review"]) > 0
    assert final_state["introduction_status"] == "completed"
    assert final_state["methodology_status"] == "completed"
    assert final_state["abstract_status"] == "completed"
    assert final_state["visualization_status"] == "completed"
    assert final_state["playground_status"] == "completed"
    assert final_state["latex_paper_status"] == "completed"
    assert "latex_paper_bundle" in final_state
    assert final_state.get("latex_paper_path") is not None
    assert len(final_state["tool_history"]) > 0


def test_non_significance_does_not_trigger_retry(agent):
    """Null findings must not trigger repeated cleaning or hypothesis changes."""
    df = pd.DataFrame({
        "treatment": ["Drug"] * 30 + ["Placebo"] * 30,
        "recovery_time": [10.0] * 60,
    })
    context = ResearchContext(
        name="LATS Loop Test",
        research_type="cross_sectional",
        independent_variable="treatment",
        dependent_variable="recovery_time",
        independent_variable_type="categorical",
        dependent_variable_type="numeric",
        hypothesis="Drug impacts recovery time.",
    )

    call_count = 0
    def mock_conduct_tests(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        # Call 1: non-significant p = 0.35, Call 2: significant p = 0.01
        p_val = 0.35 if call_count == 1 else 0.01
        return MagicMock(
            success=True,
            data={
                "t_test": {
                    "statistic": 2.1,
                    "p_value": p_val,
                }
            },
        )

    from research_agent.tools.statistics_tools import ConductTestsTool
    from research_agent.skills.literature_review import LiteratureReviewSkill
    from research_agent.tools.visualization_tools import PlanVisualizationsTool

    with patch.object(
        ConductTestsTool,
        "execute",
        side_effect=mock_conduct_tests,
    ), patch.object(
        LiteratureReviewSkill,
        "run",
        return_value=[],
    ), patch.object(
        PlanVisualizationsTool,
        "execute",
        return_value=MagicMock(success=True, data=[]),
    ), patch.object(
        BaseSkill,
        "_invoke_with_fallback",
        return_value="Detailed academic text satisfying all quality criteria.",
    ):
        final_state = agent.run_graph(context=context, df=df, compile_pdf=False)

    assert call_count == 1
    assert final_state.get("lats_retry_count", 0) == 0
    assert final_state.get("latex_paper_status") == "completed"


def test_quality_gate_refinement_loop(agent):
    """Test that a deficient draft triggers the Quality Gate loop for refinement."""
    df = pd.DataFrame({
        "group": ["A"] * 20 + ["B"] * 20,
        "score": list(range(40)),
    })
    context = ResearchContext(
        name="Quality Gate Test",
        research_type="cross_sectional",
        independent_variable="group",
        dependent_variable="score",
        independent_variable_type="categorical",
        dependent_variable_type="numeric",
        hypothesis="Group influences test score.",
    )

    methodology_calls = 0
    def mock_methodology_run(*args, **kwargs):
        nonlocal methodology_calls
        methodology_calls += 1
        if methodology_calls == 1:
            return "Too brief."  # Deficient: will fail quality gate
        return (
            "Study Design and Setting: This cross-sectional investigation systematically evaluated empirical score variations between group cohorts. "
            "Data Collection and Protocol: A standardized observational dataset of forty subjects was curated under controlled data integrity conditions. "
            "Variables of Interest: The categorical predictor was group membership, and the primary quantitative outcome was test score. "
            "Statistical Analysis: Formal two-sample inferential hypothesis testing was conducted with significance established at alpha = 0.05."
        )

    from research_agent.tools.statistics_tools import ConductTestsTool
    from research_agent.skills.methodology import MethodologySkill
    from research_agent.skills.literature_review import LiteratureReviewSkill
    from research_agent.tools.visualization_tools import PlanVisualizationsTool

    with patch.object(
        ConductTestsTool,
        "execute",
        return_value=MagicMock(success=True, data={"t_test": {"p_value": 0.01}}),
    ), patch.object(
        MethodologySkill,
        "run",
        side_effect=mock_methodology_run,
    ), patch.object(
        LiteratureReviewSkill,
        "run",
        return_value=[],
    ), patch.object(
        PlanVisualizationsTool,
        "execute",
        return_value=MagicMock(success=True, data=[]),
    ), patch.object(
        BaseSkill,
        "_invoke_with_fallback",
        return_value="Valid detailed academic text meeting all necessary quality criteria.",
    ):
        final_state = agent.run_graph(context=context, df=df, compile_pdf=False)

    assert methodology_calls >= 1
    assert "Too brief." not in final_state.get("methodology", "")
    assert len(final_state.get("methodology", "").split()) >= 20
    assert final_state.get("quality_status") == "passed"
    assert final_state.get("latex_paper_status") == "completed"
