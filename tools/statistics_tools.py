"""
Statistics tools: dispatch and run the correct statistical tests
based on research design and variable types.

Delegates execution to research_agent.study_designs.dispatcher to ensure
single source of truth for all statistical test mappings and dispatch logic.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Type

import pandas as pd
from pydantic import BaseModel, Field

from research_agent.study_designs.dispatcher import (
    clean_for_json,
    dispatch_tests,
    normalize_type,
    resolve_design_key,
)
from research_agent.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


# ============================================================================
# Tool class
# ============================================================================


class _ConductTestsInput(BaseModel):
    research_design: str = Field(
        description="Study design (e.g. 'cross sectional', 'pre-post', 'individual RCT')."
    )
    iv_type: str = Field(description="Type of the independent variable (e.g. 'numeric', 'binary').")
    dv_type: str = Field(description="Type of the dependent variable.")
    iv: str = Field(description="Column name of the independent variable.")
    dv: str = Field(description="Column name of the dependent variable.")
    covariates: Optional[List[str]] = Field(
        default=None, description="Optional list of covariate column names."
    )


class ConductTestsTool(BaseTool):
    """Run the appropriate battery of statistical tests for a research design."""

    name = "conduct_tests"
    description = (
        "Selects and runs the correct statistical tests based on the study "
        "design, independent-variable type, and dependent-variable type. "
        "Returns a dict of test results."
    )
    input_schema: type[BaseModel] = _ConductTestsInput

    def execute(self, **kwargs: Any) -> ToolResult:
        df: Optional[pd.DataFrame] = kwargs.get("df") if "df" in kwargs else kwargs.get("data")
        if df is None:
            return ToolResult.fail("No DataFrame provided.")

        research_design: str = kwargs.get("research_design") or kwargs.get("research_type", "")
        iv_type_raw: str = kwargs.get("iv_type") or kwargs.get("independent_variable_type", "")
        dv_type_raw: str = kwargs.get("dv_type") or kwargs.get("dependent_variable_type", "")
        independent_variable: str = kwargs.get("iv") or kwargs.get("independent_variable", "")
        dependent_variable: str = kwargs.get("dv") or kwargs.get("dependent_variable", "")
        covariates: Optional[list] = kwargs.get("covariates")

        if not research_design or not independent_variable or not dependent_variable:
            return ToolResult.fail("Missing required parameters (research_design, iv, dv).")

        extra_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in (
                "df", "data", "research_design", "research_type",
                "iv_type", "independent_variable_type",
                "dv_type", "dependent_variable_type",
                "iv", "independent_variable",
                "dv", "dependent_variable",
                "covariates",
            )
        }

        try:
            result = dispatch_tests(
                df=df,
                research_design=research_design,
                independent_variable_type=iv_type_raw,
                dependent_variable_type=dv_type_raw,
                independent_variable=independent_variable,
                dependent_variable=dependent_variable,
                covariates=covariates,
                **extra_kwargs,
            )

            if isinstance(result, dict) and "error" in result and len(result) == 1:
                return ToolResult.fail(result["error"])

            return ToolResult.ok(result)
        except Exception as exc:
            logger.exception("Statistical test execution failed")
            return ToolResult.fail(f"Test execution error: {exc}")
