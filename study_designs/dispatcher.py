"""
Test dispatch logic for statistical study designs.

Dispatches study design tests based on research design, IV type, and DV type,
and cleans results for JSON serialization.
"""

from __future__ import annotations

import inspect
import logging
import math
from typing import Any, Optional

import pandas as pd

from research_agent.study_designs import utils_conduct_tests

logger = logging.getLogger(__name__)


def normalize_type(t: str) -> str:
    """Map human-friendly variable type labels to canonical short keys."""
    t = t.lower().replace("-", " ").strip()
    if any(kw in t for kw in ("continuous", "numeric", "interval", "ratio")):
        return "cont"
    if any(kw in t for kw in ("binary", "dichotomous")):
        return "binary"
    if any(kw in t for kw in ("multiple", "multinomial", "categorical", "nominal")):
        return "multinomial"
    if "ordinal" in t:
        return "ordinal"
    return t


def clean_for_json(obj: Any) -> Any:
    """Recursively replace NaN / Infinity with None for JSON safety."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        cleaned = [clean_for_json(v) for v in obj]
        return tuple(cleaned) if isinstance(obj, tuple) else cleaned
    return obj


def resolve_design_key(rd: str) -> str:
    """Normalise the research-design string to a dispatch key."""
    rd_clean = rd.lower().replace("-", " ").strip()
    if "controlled" in rd_clean and "before" in rd_clean and "after" in rd_clean:
        return "controlled_before_after"
    if "before" in rd_clean and "after" in rd_clean:
        return "pre_post"
    if "pre" in rd_clean and "post" in rd_clean:
        return "pre_post"
    if "cross" in rd_clean and "sectional" in rd_clean:
        return "cross_sectional"
    if "case" in rd_clean and "control" in rd_clean:
        return "case_control"
    if "crossover" in rd_clean:
        return "crossover_rct"
    if "individual" in rd_clean and "rct" in rd_clean:
        return "individual_rct"
    if "rct" in rd_clean or "experimental" in rd_clean or "trial" in rd_clean:
        return "individual_rct"
    if "retrospective" in rd_clean:
        return "retrospective"
    return rd_clean.replace(" ", "_")


_DISPATCH: dict[str, dict[tuple[str, str], str]] = {
    "cross_sectional": {
        ("cont", "multinomial"): "conduct_cross_sectional_cont_iv_multinomial_dv",
        ("cont", "cont"): "conduct_cross_sectional_cont_iv_cont_dv",
        ("cont", "binary"): "conduct_cross_sectional_cont_iv_binary_dv",
        ("binary", "binary"): "conduct_cross_sectional_binary_iv_binary_dv",
        ("binary", "cont"): "conduct_cross_sectional_binary_iv_cont_dv",
        ("binary", "multinomial"): "conduct_cross_sectional_binary_iv_multinomial_dv",
        ("multinomial", "binary"): "conduct_cross_sectional_multinomial_iv_binary_dv",
        ("ordinal", "binary"): "conduct_cross_sectional_multinomial_iv_binary_dv",
        ("multinomial", "cont"): "conduct_cross_sectional_multinomial_iv_cont_dv",
        ("ordinal", "cont"): "conduct_cross_sectional_multinomial_iv_cont_dv",
        ("multinomial", "multinomial"): "conduct_cross_sectional_multinomial_iv_multinomial_dv",
        ("ordinal", "multinomial"): "conduct_cross_sectional_multinomial_iv_multinomial_dv",
    },
    "case_control": {
        ("*", "binary"): "conduct_case_control_binary_dv",
        ("*", "multinomial"): "conduct_case_control_multinomial_dv",
        ("*", "cont"): "conduct_case_control_cont_dv",
        ("*", "ordinal"): "conduct_case_control_multinomial_dv",
    },
    "controlled_before_after": {
        ("binary", "cont"): "conduct_controlled_before_after_binary_iv_cont_dv",
        ("binary", "binary"): "conduct_controlled_before_after_binary_iv_binary_dv",
        ("binary", "multinomial"): "conduct_controlled_before_after_binary_iv_multinomial_dv",
    },
    "crossover_rct": {
        ("binary", "cont"): "conduct_crossover_rct_binary_iv_cont_dv",
        ("binary", "binary"): "conduct_crossover_rct_binary_iv_binary_dv",
        ("binary", "multinomial"): "conduct_crossover_rct_binary_iv_multinomial_dv",
        ("multinomial", "cont"): "conduct_crossover_rct_categorical_iv_cont_dv",
        ("multinomial", "binary"): "conduct_crossover_rct_multinomial_iv_binary_dv",
        ("multinomial", "multinomial"): "conduct_crossover_rct_multinomial_iv_multinomial_dv",
        ("ordinal", "cont"): "conduct_crossover_rct_ordinal_iv_cont_dv",
        ("ordinal", "binary"): "conduct_crossover_rct_ordinal_iv_binary_dv",
        ("ordinal", "multinomial"): "conduct_crossover_rct_ordinal_iv_multinomial_dv",
    },
    "individual_rct": {
        ("binary", "cont"): "conduct_individual_rct_binary_iv_cont_dv",
        ("binary", "binary"): "conduct_individual_rct_binary_iv_binary_dv",
        ("binary", "multinomial"): "conduct_individual_rct_binary_iv_multinomial_dv",
        ("multinomial", "cont"): "conduct_individual_rct_multinomial_iv_cont_dv",
        ("multinomial", "binary"): "conduct_individual_rct_multinomial_iv_binary_dv",
        ("multinomial", "multinomial"): "conduct_individual_rct_multinomial_iv_multinomial_dv",
        ("ordinal", "cont"): "conduct_individual_rct_ordinal_iv_cont_dv",
        ("ordinal", "binary"): "conduct_individual_rct_ordinal_iv_binary_dv",
        ("ordinal", "multinomial"): "conduct_individual_rct_ordinal_iv_multinomial_dv",
    },
    "pre_post": {
        ("*", "cont"): "conduct_pre_post_cont_dv",
        ("*", "binary"): "conduct_pre_post_binary_dv",
        ("*", "multinomial"): "conduct_pre_post_multinomial_dv",
        ("*", "ordinal"): "conduct_pre_post_ordinal_dv",
    },
    "retrospective": {
        ("cont", "cont"): "conduct_retrospective_cont_iv_cont_dv",
        ("cont", "binary"): "conduct_retrospective_cont_iv_binary_dv",
        ("cont", "multinomial"): "conduct_retrospective_cont_iv_multinomial_dv",
        ("binary", "cont"): "conduct_retrospective_binary_iv_cont_dv",
        ("binary", "binary"): "conduct_retrospective_binary_iv_binary_dv",
        ("binary", "multinomial"): "conduct_retrospective_binary_iv_multinomial_dv",
        ("multinomial", "cont"): "conduct_retrospective_multinomial_iv_cont_dv",
        ("ordinal", "cont"): "conduct_retrospective_multinomial_iv_cont_dv",
        ("multinomial", "binary"): "conduct_retrospective_multinomial_iv_binary_dv",
        ("ordinal", "binary"): "conduct_retrospective_multinomial_iv_binary_dv",
        ("multinomial", "multinomial"): "conduct_retrospective_multinomial_iv_multinomial_dv",
        ("ordinal", "multinomial"): "conduct_retrospective_multinomial_iv_multinomial_dv",
    },
}


def lookup_function_name(design_key: str, iv_type: str, dv_type: str) -> Optional[str]:
    """Find the test function name for given design key and variable types."""
    design_table = _DISPATCH.get(design_key)
    if design_table is None:
        return None
    fn_name = design_table.get((iv_type, dv_type))
    if fn_name:
        return fn_name
    return design_table.get(("*", dv_type))


def invoke_test_function(
    fn: Any,
    df: pd.DataFrame,
    independent_variable: Any,
    dependent_variable: Any,
    covariates: Optional[list] = None,
    **kwargs: Any,
) -> Any:
    """Introspect fn's signature and call it with matching arguments."""
    sig = inspect.signature(fn)
    args_to_pass: dict[str, Any] = {}

    for param_name in sig.parameters:
        if param_name == "df":
            args_to_pass[param_name] = df
        elif param_name in ("iv", "treatment", "case_control_col", "pre", "independent_variable"):
            args_to_pass[param_name] = independent_variable
        elif param_name in ("dv", "outcome", "post", "dependent_variable"):
            args_to_pass[param_name] = dependent_variable
        elif param_name == "covariates":
            args_to_pass[param_name] = covariates
        elif param_name in ("time", "time_col"):
            args_to_pass[param_name] = kwargs.get("time") or kwargs.get("time_col")
        elif param_name in ("subject_id", "id_col"):
            args_to_pass[param_name] = (
                kwargs.get("subject_id") or kwargs.get("id_col") or kwargs.get("subject_group")
            )
        else:
            if param_name in kwargs:
                args_to_pass[param_name] = kwargs[param_name]
            elif sig.parameters[param_name].default == inspect.Parameter.empty:
                pass

    args_to_pass["df"] = df.copy(deep=True)
    result = fn(**args_to_pass)
    return clean_for_json(result)


def dispatch_tests(
    df: pd.DataFrame,
    research_design: str,
    independent_variable_type: str,
    dependent_variable_type: str,
    independent_variable: str,
    dependent_variable: str,
    covariates: Optional[list] = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Conducts statistical tests based on research design and variable types.
    Returns cleaned dict of test results.
    """
    iv_type = normalize_type(independent_variable_type)
    dv_type = normalize_type(dependent_variable_type)
    design_key = resolve_design_key(research_design)

    fn_name = lookup_function_name(design_key, iv_type, dv_type)
    if fn_name is None:
        return {
            "error": f"No matching test found for design '{research_design}', "
                     f"IV type '{independent_variable_type}', DV type '{dependent_variable_type}'."
        }

    fn = getattr(utils_conduct_tests, fn_name, None)
    if fn is None:
        return {"error": f"Function '{fn_name}' not found in study_designs.utils_conduct_tests."}

    try:
        return invoke_test_function(
            fn, df, independent_variable, dependent_variable, covariates, **kwargs
        )
    except Exception as exc:
        logger.exception("Error executing statistical test %s", fn_name)
        return {"error": f"Execution error in {fn_name}: {exc}"}
