"""Unit tests for research_agent tools."""

import pandas as pd
import numpy as np
import pytest

from research_agent.tools.data_tools import (
    CleanDataFrameTool,
    DetectVariableTypeTool,
    ProduceVariableSummaryTool,
    ReadFileTool,
    read_file_safely,
)
from research_agent.tools.statistics_tools import ConductTestsTool
from research_agent.tools.visualization_tools import PlanVisualizationsTool


@pytest.fixture
def sample_df():
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "age": np.random.normal(35, 10, n),
        "treatment": np.random.choice(["control", "treated"], n),
        "score": np.random.normal(50, 15, n),
        "category": np.random.choice(["A", "B", "C"], n),
    })


def test_detect_variable_type(sample_df):
    tool = DetectVariableTypeTool()
    
    # Numeric check
    res = tool.execute(column_name="age", df=sample_df)
    assert res.success is True
    assert res.data == "numeric"

    # Binary check
    res = tool.execute(column_name="treatment", df=sample_df)
    assert res.success is True
    assert res.data == "binary"


def test_produce_variable_summary(sample_df):
    tool = ProduceVariableSummaryTool()
    res = tool.execute(column_name="age", column_type="numeric", df=sample_df)
    assert res.success is True
    assert "mean" in res.data


def test_clean_dataframe(sample_df):
    tool = CleanDataFrameTool()
    res = tool.execute(
        df=sample_df,
        dependent_variable="score",
        independent_variables=["treatment"],
        covariates=["age"],
    )
    assert res.success is True
    assert not res.data.empty
    assert len(res.data) <= len(sample_df)


def test_conduct_tests_tool(sample_df):
    tool = ConductTestsTool()
    res = tool.execute(
        df=sample_df,
        research_design="cross_sectional",
        iv_type="binary",
        dv_type="numeric",
        iv="treatment",
        dv="score",
    )
    assert res.success is True
    assert isinstance(res.data, dict)


def test_read_file_tool_and_size_guard(tmp_path):
    # Test valid read
    csv_file = tmp_path / "valid.csv"
    csv_file.write_text("col_a,col_b\n1,10\n2,20\n", encoding="utf-8")
    
    tool = ReadFileTool()
    res = tool.execute(file_path=str(csv_file))
    assert res.success is True
    assert list(res.data.columns) == ["col_a", "col_b"]
    assert len(res.data) == 2

    # Test size guard triggers when file exceeds max_file_size_mb
    small_df = read_file_safely(str(csv_file), max_file_size_mb=0)  # 0 MB threshold
    assert small_df is None


def test_plan_visualizations_tool_initialization():
    tool = PlanVisualizationsTool()
    assert tool.name == "plan_visualizations"
    assert tool.config is not None
