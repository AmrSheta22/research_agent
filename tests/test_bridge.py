"""Unit tests for backward-compatibility bridge."""

from unittest.mock import patch
import pandas as pd
import pytest

from research_agent.bridge import (
    get_agent,
    predict_methodology,
    predict_methodlogy,
    conduct_tests,
    read_file_safely,
    clean_dataframe,
    determine_type_of_variable,
)
from research_agent.skills.base import BaseSkill


def test_bridge_lazy_get_agent():
    agent1 = get_agent()
    agent2 = get_agent()
    assert agent1 is agent2
    assert hasattr(agent1, "tools")
    assert hasattr(agent1, "skills")


def test_bridge_methodology_alias():
    assert predict_methodology == predict_methodlogy

    with patch.object(
        BaseSkill,
        "_invoke_with_fallback",
        return_value="Detailed methodology description.",
    ):
        res = predict_methodology(
            independent_variable="treatment",
            dependent_variable="score",
            test_names=["two_sample_t_test"],
        )
    assert "methodology" in res.lower() or len(res) > 0


def test_bridge_shims_execution():
    df = pd.DataFrame({
        "group": ["A", "B", "A", "B"],
        "val": [10.0, 20.0, 15.0, 25.0],
    })
    var_type = determine_type_of_variable("group", df)
    assert var_type in ("binary", "categorical")

    results = conduct_tests(
        df=df,
        research_design="cross_sectional",
        independent_variable_type="binary",
        dependent_variable_type="numeric",
        independent_variable="group",
        dependent_variable="val",
    )
    assert isinstance(results, dict)
    assert "error" not in results
