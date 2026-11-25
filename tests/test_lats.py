"""Unit tests for Language Agent Tree Search (LATS) optimization subsystem."""

from unittest.mock import patch
import pandas as pd
import numpy as np
import pytest

from research_agent.agent import ResearchAgent
from research_agent.config import AgentConfig
from research_agent.lats.agent import LATSAgent, LATSResult
from research_agent.lats.cleaner_pool import CleanerPool
from research_agent.lats.evaluator import LATSEvaluator
from research_agent.lats.hypothesis_generator import HypothesisCandidate, HypothesisGenerator
from research_agent.schemas import ResearchContext


@pytest.fixture
def noisy_df():
    """Generates dataset with synthetic effect and known extreme outliers."""
    np.random.seed(42)
    n = 100
    group = np.random.choice(["Control", "Treatment"], n)
    # Treatment group has higher mean
    score = np.where(group == "Treatment", np.random.normal(55, 8, n), np.random.normal(45, 8, n))
    
    # Inject severe outliers
    score[0] = 500.0  # extreme positive outlier
    score[1] = -300.0 # extreme negative outlier
    
    return pd.DataFrame({
        "treatment": group,
        "recovery_score": score,
        "age": np.random.normal(40, 10, n),
    })


@pytest.fixture
def mock_hypotheses():
    return [
        HypothesisCandidate(
            hypothesis="Treatment significantly increases recovery score compared to control.",
            angle="directional_positive",
            rationale="Primary direct effect hypothesis.",
        ),
        HypothesisCandidate(
            hypothesis="Treatment is significantly associated with recovery score after controlling for age.",
            angle="adjusted_multivariable",
            rationale="Covariate-adjusted association.",
        ),
    ]


def test_cleaner_pool_strategies(noisy_df):
    """Test that all cleaner pool options execute properly."""
    strategies = CleanerPool.get_all_strategies()
    assert len(strategies) >= 6

    strategy_names = [s.name for s in strategies]
    assert "iqr_standard" in strategy_names
    assert "iqr_conservative" in strategy_names
    assert "no_outlier_removal" in strategy_names
    assert "winsorization" in strategy_names
    assert "zscore_filter" in strategy_names
    assert "median_imputation" in strategy_names

    for strat in strategies:
        res = strat.clean(
            df=noisy_df,
            dependent_variable="recovery_score",
            independent_variables=["treatment"],
            covariates=["age"],
        )
        assert not res.cleaned_df.empty
        assert res.final_n <= res.initial_n
        assert 0.0 < res.retention_ratio <= 1.0


def test_evaluator_p_value_extraction():
    """Test p-value extraction across multiple statistical test formats."""
    sample_results = {
        "T-test": {"t": 3.45, "p": 0.0012},
        "Mann-Whitney U": {"U": 250.0, "p_val": 0.0035},
        "ANOVA": {"F": 12.1, "PR(>F)": 0.0008},
        "Linear Regression": {"const": 1e-10, "iv": 0.0012},
    }
    extracted = LATSEvaluator.extract_p_values(sample_results)
    assert len(extracted) >= 3
    p_values = [p for _, p in extracted]
    assert min(p_values) < 0.005

    score = LATSEvaluator.evaluate(
        test_results=sample_results,
        initial_n=100,
        final_n=95,
        alpha=0.05,
        cleaning_strategy="iqr_standard",
    )
    assert score.valid is True
    assert score.min_p_value < 0.005
    assert len(score.significant_tests) >= 3
    assert score.composite_score > 0.70


def test_lats_agent_route_optimization(noisy_df, mock_hypotheses):
    """Test full LATS tree search taking the best route."""
    context = ResearchContext(
        name="Outlier Sensitivity Study",
        research_type="cross_sectional",
        independent_variable="treatment",
        dependent_variable="recovery_score",
        independent_variable_type="binary",
        dependent_variable_type="numeric",
    )

    lats_agent = LATSAgent(AgentConfig(), alpha=0.05)

    with patch.object(HypothesisGenerator, "generate_candidates", return_value=mock_hypotheses):
        result: LATSResult = lats_agent.optimize_route(
            context=context,
            df=noisy_df,
            n_hypotheses_per_cleaner=2,
        )

    assert result.best_trajectory is not None
    assert result.best_trajectory.score > 0.0
    assert result.optimal_cleaned_df is not None
    assert len(result.all_trajectories) >= 5  # at least 5 cleaning options explored

    # Verify best route was chosen and audit report generated
    best_traj = result.best_trajectory
    assert best_traj.min_p_value < 0.05
    assert "Selected Best Route" in result.audit_report
    assert "Trajectory Comparison Matrix" in result.audit_report


def test_research_agent_lats_integration(noisy_df, mock_hypotheses):
    """Test ResearchAgent's high-level LATS integration."""
    agent = ResearchAgent(AgentConfig())
    context = ResearchContext(
        name="LATS End-to-End Test",
        research_type="cross_sectional",
        independent_variable="treatment",
        dependent_variable="recovery_score",
        independent_variable_type="binary",
        dependent_variable_type="numeric",
    )

    with patch.object(HypothesisGenerator, "generate_candidates", return_value=mock_hypotheses):
        result = agent.run_lats_search(context=context, df=noisy_df)

    assert result.best_trajectory is not None
    assert result.best_trajectory.score > 0.0
    assert result.optimized_context.hypothesis != ""


def test_evaluator_multiplicity_correction():
    """Test Bonferroni threshold and Benjamini-Hochberg FDR multiplicity control."""
    # Test Bonferroni calculation
    p_values = [0.001, 0.03, 0.07, 0.20]
    alpha = 0.05
    bonf_alpha, sig_flags = LATSEvaluator.calculate_bonferroni(p_values, alpha=alpha)
    assert bonf_alpha == 0.05 / 4  # 0.0125
    assert sig_flags == [True, False, False, False]  # 0.03 is nominally sig but fails Bonferroni

    # Test Benjamini-Hochberg FDR calculation
    named_pairs = [("test1", 0.005), ("test2", 0.03), ("test3", 0.07)]
    q_pairs, sig_names = LATSEvaluator.calculate_benjamini_hochberg(named_pairs, fdr_rate=0.05)
    assert len(q_pairs) == 3
    # test1 (p=0.005): q = 0.005 * 3 / 1 = 0.015 <= 0.05 -> significant
    assert "test1" in sig_names

    # Test Evaluator with multiplicity across paths
    score = LATSEvaluator.evaluate(
        test_results={"t1": {"p": 0.001}, "t2": {"p": 0.04}},
        initial_n=100,
        final_n=95,
        alpha=0.05,
        cleaning_strategy="iqr_standard",
        k_paths_explored=5,
    )
    assert score.valid is True
    assert score.k_paths_explored == 5
    assert score.bonferroni_alpha < 0.05
    assert "Bonferroni" in score.reflection
