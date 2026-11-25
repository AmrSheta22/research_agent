"""
Visualization tools: LLM-driven visualization planning via tool-calling.

Ported from ``visualization/custom_visualization.py``.  All logic is
self-contained — nothing is imported from legacy modules.
"""

from __future__ import annotations

import logging
import os
import re
import warnings
from typing import Any, Dict, List, Literal, Optional, Type

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from research_agent.config import AgentConfig
from research_agent.tools.base import BaseTool, ToolResult
from research_agent.tools.data_tools import determine_type_of_variable as _determine_type_of_variable

logger = logging.getLogger(__name__)


# ============================================================================
# LLM helpers
# ============================================================================


def _get_llm(llm: Any = None, config: Optional[AgentConfig] = None) -> Any:
    """Return the provided LLM or resolve one via AgentConfig fallback chain."""
    if llm is not None:
        return llm
    cfg = config or AgentConfig()
    return cfg.get_chat_model(temperature=0.2)


def _get_column_types_info(df: pd.DataFrame, columns: list[str]) -> tuple[str, dict[str, str]]:
    """Return (formatted type string, column→type dict)."""
    type_info: list[str] = []
    col_types: dict[str, str] = {}
    for col in columns:
        if col not in df.columns:
            continue
        col_type = _determine_type_of_variable(col, df)
        if col_type is None:
            dtype = str(df[col].dtype)
            if "int" in dtype or "float" in dtype:
                col_type = "numeric"
            elif "object" in dtype or "category" in dtype:
                col_type = "categorical"
            else:
                col_type = "unknown"
        col_types[col] = col_type
        type_info.append(f"- {col}: {col_type}")
    return "\n".join(type_info), col_types


def _create_visualization_tools(column_types: dict[str, str]) -> list[type[BaseModel]]:
    """Build Pydantic models (tool schemas) dynamically bound to column names."""
    numeric_cols = [c for c, t in column_types.items() if t == "numeric"]
    categorical_cols = [c for c, t in column_types.items() if t != "numeric"]

    if not numeric_cols:
        numeric_cols = ["_no_numeric_cols_"]
    if not categorical_cols:
        categorical_cols = ["_no_categorical_cols_"]

    NumericColEnum = Literal[tuple(numeric_cols)]  # type: ignore[valid-type]
    CategoricalColEnum = Literal[tuple(categorical_cols)]  # type: ignore[valid-type]

    class ScatterRegPlot(BaseModel):
        """Use for ScatterPlot or RegPlot. Requires Two Numeric Variables."""
        title: str = Field(description="Title of the plot")
        mark: Literal["ScatterPlot", "RegPlot"] = Field(description="Plot type")
        x_axis: NumericColEnum = Field(description="Numeric column for X axis")  # type: ignore
        y_axis: NumericColEnum = Field(description="Numeric column for Y axis")  # type: ignore
        color: Optional[CategoricalColEnum] = Field(None, description="Optional color column")  # type: ignore

    class HistogramPlot(BaseModel):
        """Use for Histogram. Requires One Numeric Variable."""
        title: str = Field(description="Title of the plot")
        mark: Literal["Histogram"] = Field(default="Histogram")
        x_axis: NumericColEnum = Field(description="Numeric column for distribution")  # type: ignore

    class CategoricalNumericPlot(BaseModel):
        """Use for BarPlot, BoxPlot, ViolinPlot, StripPlot, SwarmPlot."""
        title: str = Field(description="Title of the plot")
        mark: Literal["BarPlot", "BoxPlot", "ViolinPlot", "StripPlot", "SwarmPlot"] = Field(description="Plot type")
        x_axis: CategoricalColEnum = Field(description="Categorical column (X)")  # type: ignore
        y_axis: NumericColEnum = Field(description="Numeric column (Y)")  # type: ignore
        color: Optional[CategoricalColEnum] = Field(None, description="Optional color grouping")  # type: ignore

    class CompositionPlot(BaseModel):
        """Use for StackedBarPlot or GroupedBarPlot."""
        title: str = Field(description="Title of the plot")
        mark: Literal["StackedBarPlot", "GroupedBarPlot"] = Field(description="Plot type")
        x_axis: CategoricalColEnum = Field(description="Categorical column (X)")  # type: ignore
        y_axis: NumericColEnum = Field(description="Numeric column (Y)")  # type: ignore
        color: CategoricalColEnum = Field(description="Categorical column for stacking (REQUIRED)")  # type: ignore

    class MosaicPlotSchema(BaseModel):
        """Use for MosaicPlot. Requires Two Categorical Variables."""
        title: str = Field(description="Title of the plot")
        mark: Literal["MosaicPlot"] = Field(default="MosaicPlot")
        x_axis: CategoricalColEnum = Field(description="Categorical column 1")  # type: ignore
        y_axis: CategoricalColEnum = Field(description="Categorical column 2")  # type: ignore

    return [ScatterRegPlot, HistogramPlot, CategoricalNumericPlot, CompositionPlot, MosaicPlotSchema]


def _generate_visualizations(
    df: pd.DataFrame,
    hypothesis: str,
    columns_to_use: Optional[list[str]] = None,
    llm: Any = None,
    config: Optional[AgentConfig] = None,
) -> list[dict[str, Any]]:
    """Core logic — plan & execute visualization specs via LLM tool-calling."""
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    llm = _get_llm(llm, config=config)

    # 1. Filter columns
    rows_limit = 5
    if columns_to_use:
        curr_cols = [c for c in columns_to_use if c in df.columns]
        df_subset = df[curr_cols]
    else:
        df_subset = df
        curr_cols = df.columns.tolist()

    column_types_str, column_types_dict = _get_column_types_info(df, curr_cols)
    dataset_head_md = df_subset.head(rows_limit).to_markdown()

    # 2. Plan
    planner_prompt = ChatPromptTemplate.from_template(
        """You are a data visualization expert.

Dataset Head (First 5 rows):
{dataset_head}

Column Types:
{column_types}

Hypothesis/Goal: {hypothesis}

Generate a list of 3-5 distinct visualization ideas that would help explore or validate this hypothesis.
Available Plots: ScatterPlot, RegPlot, ViolinPlot, Histogram, BoxPlot, StripPlot, BarPlot, StackedBarPlot, GroupedBarPlot, MosaicPlot, SwarmPlot.

Return ONLY a numbered list of natural language statements. Do not include introductory text.
"""
    )
    planner_chain = planner_prompt | llm | StrOutputParser()
    plan_output = planner_chain.invoke({
        "dataset_head": dataset_head_md,
        "column_types": column_types_str,
        "hypothesis": hypothesis,
    })

    # 3. Execute via tool calling
    tools = _create_visualization_tools(column_types_dict)
    llm_with_tools = llm.bind_tools(tools)

    exec_prompt = ChatPromptTemplate.from_template(
        """You are a Precise Data Visualization Executor.

CONTEXT:
- Hypothesis: {hypothesis}
- Column Types:
{column_types}

PLAN (Visualizations to create):
{plan}

INSTRUCTIONS:
- Use the provided tools to create specific visualizations for the plan.
- You MUST select the tool that matches the variable types.
- If a plan item asks for an impossible plot, skip it or choose the correct tool.
- Create as many plots as referenced in the plan.
"""
    )
    chain = exec_prompt | llm_with_tools

    try:
        response = chain.invoke({
            "hypothesis": hypothesis,
            "column_types": column_types_str,
            "plan": plan_output,
        })
    except Exception as exc:
        logger.exception("Error in visualization tool generation")
        return []

    # 4. Parse tool calls
    final_specs: list[dict[str, Any]] = []
    if hasattr(response, "tool_calls") and response.tool_calls:
        for tool_call in response.tool_calls:
            args = tool_call["args"]
            spec = {
                "title": args.get("title") or (
                    f"{args.get('mark', '')} of {args.get('x_axis', '')} vs {args.get('y_axis', '')}"
                ),
                "mark": args.get("mark", ""),
                "x_axis": args.get("x_axis", ""),
                "y_axis": args.get("y_axis", ""),
                "color": args.get("color", ""),
            }
            final_specs.append(spec)
    else:
        logger.info("No visualization tool calls generated by LLM.")

    return final_specs


# ============================================================================
# Tool class
# ============================================================================


class _PlanVisualizationsInput(BaseModel):
    hypothesis: str = Field(description="Research hypothesis or goal to visualise.")
    columns_to_use: Optional[List[str]] = Field(
        default=None,
        description="Subset of columns to consider. If omitted, all columns are used.",
    )


class PlanVisualizationsTool(BaseTool):
    """Plan a suite of visualizations for a dataset and hypothesis using an LLM."""

    name = "plan_visualizations"
    description = (
        "Uses an LLM to plan and specify 3-5 visualizations that explore or "
        "validate a hypothesis against a DataFrame. Returns a list of "
        "VisualizationSpec dicts (title, mark, x_axis, y_axis, color)."
    )
    input_schema: type[BaseModel] = _PlanVisualizationsInput

    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()

    def execute(self, **kwargs: Any) -> ToolResult:
        df: pd.DataFrame = kwargs["df"]
        hypothesis: str = kwargs["hypothesis"]
        columns_to_use: Optional[list[str]] = kwargs.get("columns_to_use")
        llm = kwargs.get("llm")

        if not isinstance(df, pd.DataFrame):
            return ToolResult.fail("Expected a pandas DataFrame for 'df'.")
        if df.empty:
            return ToolResult.fail("DataFrame is empty — nothing to visualise.")

        try:
            specs = _generate_visualizations(
                df, hypothesis, columns_to_use, llm=llm, config=self.config
            )
            return ToolResult.ok(specs)
        except Exception as exc:
            logger.exception("Visualization planning failed")
            return ToolResult.fail(f"Visualization planning error: {exc}")
