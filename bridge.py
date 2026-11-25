"""
Backward-compatibility bridge between the backend and the new research_agent package.

Allows legacy routers and background tasks to call agent functionality with
the original function signatures while routing everything through the new
modular Tool and Skill architecture and continuous learning memory.
"""

from __future__ import annotations

from typing import Any, List, Optional
import pandas as pd

from research_agent.agent import ResearchAgent
from research_agent.schemas import ResearchContext
from research_agent.skills.introduction import format_literature_items
from research_agent.tools.data_tools import (
    clean_dataframe as _clean_dataframe,
    determine_type_of_variable as _determine_type_of_variable,
    ensure_correct_data_type as _ensure_correct_data_type,
    get_file_header_and_first_five_rows as _get_file_header_and_first_five_rows,
    produce_variable_summary as _produce_variable_summary,
    read_file_safely as _read_file_safely,
)
from research_agent.study_designs.dispatcher import dispatch_tests as _dispatch_tests
from research_agent.tools.visualization_tools import PlanVisualizationsTool

# Singleton agent instance for backend usage (lazy initialized)
_agent: Optional[ResearchAgent] = None


def get_agent() -> ResearchAgent:
    """Return the global ResearchAgent instance (lazy initialized)."""
    global _agent
    if _agent is None:
        _agent = ResearchAgent()
    return _agent


# ---------------------------------------------------------------------------
# Data utilities shims
# ---------------------------------------------------------------------------
def read_file_safely(file_path: str, delimiter: Optional[str] = None, encoding: Optional[str] = None) -> Optional[pd.DataFrame]:
    return _read_file_safely(file_path, delimiter, encoding)


def clean_dataframe(
    df: pd.DataFrame,
    dependent_variable: str,
    independent_variables: list[str],
    covariates: Optional[list[str]] = None,
) -> pd.DataFrame:
    return _clean_dataframe(df, dependent_variable, independent_variables, covariates)


def determine_type_of_variable(col_name: str, data: pd.DataFrame) -> Optional[str]:
    return _determine_type_of_variable(col_name, data)


def ensure_correct_data_type(data_type: str, series: pd.Series) -> bool:
    return _ensure_correct_data_type(data_type, series)


def produce_variable_summary(column_name: str, column_type: str, df: pd.DataFrame) -> str:
    return _produce_variable_summary(column_name, column_type, df)


def get_file_header_and_first_five_rows(file_path: str) -> str:
    return _get_file_header_and_first_five_rows(file_path)


# ---------------------------------------------------------------------------
# Statistical tests shim
# ---------------------------------------------------------------------------
def conduct_tests(
    df: pd.DataFrame,
    research_design: str,
    independent_variable_type: str,
    dependent_variable_type: str,
    independent_variable: str,
    dependent_variable: str,
    covariates: Optional[list] = None,
    **kwargs: Any,
) -> dict[str, Any]:
    return _dispatch_tests(
        df=df,
        research_design=research_design,
        independent_variable_type=independent_variable_type,
        dependent_variable_type=dependent_variable_type,
        independent_variable=independent_variable,
        dependent_variable=dependent_variable,
        covariates=covariates,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# LLM generation shims (routing via Skills & Agent)
# ---------------------------------------------------------------------------
def predict_all_missing(
    file_path: str,
    problem_statement: Optional[str] = None,
    name: Optional[str] = None,
    hypothesis: Optional[str] = None,
    research_type: Optional[str] = None,
    indep_var: Optional[str] = None,
    dep_var: Optional[str] = None,
    covar: Optional[list] = None,
    time: Optional[str] = None,
    subject_group: Optional[str] = None,
    indep_var_type: Optional[str] = None,
    dep_var_type: Optional[str] = None,
):
    """Bridge for routers/analysis.py:predict_missing_fields."""
    ctx = ResearchContext(
        file_path=file_path,
        problem_statement=problem_statement,
        name=name,
        hypothesis=hypothesis,
        research_type=research_type,
        independent_variable=indep_var,
        dependent_variable=dep_var,
        covariates=covar or [],
        time_variable=time,
        subject_group=subject_group,
        independent_variable_type=indep_var_type,
        dependent_variable_type=dep_var_type,
    )
    agent = get_agent()
    res = agent.predict_missing_fields(ctx)
    return (
        res.hypothesis,
        res.research_type,
        res.independent_variable,
        res.independent_variable_type,
        res.dependent_variable,
        res.dependent_variable_type,
        res.covariates,
        res.time_variable,
        res.subject_group,
    )


def predict_hypothesis(**kwargs) -> str:
    ctx = ResearchContext(**{k: v for k, v in kwargs.items() if k in ResearchContext.model_fields})
    return get_agent().generate_hypothesis(ctx)


def predict_abstract(**kwargs) -> str:
    test_interp = kwargs.get("test_interpretation")
    ctx = ResearchContext(**{k: v for k, v in kwargs.items() if k in ResearchContext.model_fields and v is not None})
    return get_agent().generate_abstract(ctx, test_interpretation=test_interp)


def predict_methodology(**kwargs) -> str:
    test_names = kwargs.get("test_names")
    if isinstance(test_names, str):
        test_names = [t.strip() for t in test_names.split(",")]
    ctx = ResearchContext(**{k: v for k, v in kwargs.items() if k in ResearchContext.model_fields and v is not None})
    return get_agent().generate_methodology(ctx, test_names=test_names)


# Backward-compatibility alias for legacy router typo
predict_methodlogy = predict_methodology


def predict_introduction(**kwargs) -> str:
    lit = kwargs.get("literature_review")
    ctx = ResearchContext(**{k: v for k, v in kwargs.items() if k in ResearchContext.model_fields and v is not None})
    return get_agent().generate_introduction(ctx, literature=lit)


def interpret_test_results(results: Any, **kwargs) -> str:
    agent = get_agent()
    ctx = ResearchContext(**{k: v for k, v in kwargs.items() if k in ResearchContext.model_fields and v is not None})
    skill = agent.get_skill("test_interpretation")
    payload = ctx.model_dump()
    payload["results"] = results
    return skill.run(payload, memory=agent.memory)


def format_literature(literature: Any) -> str:
    return format_literature_items(literature)


def generate_literature_review(**kwargs) -> list[dict[str, str]]:
    ctx = ResearchContext(
        independent_variable=kwargs.get("independent_variable"),
        dependent_variable=kwargs.get("dependent_variable"),
        research_type=kwargs.get("research_type"),
        problem_statement=kwargs.get("active_learning_query"),
        hypothesis=kwargs.get("hypothesis"),
        name=kwargs.get("name"),
    )
    items = get_agent().generate_literature_review(ctx)
    return [i.model_dump() for i in items]


def generate_custom_visualizations(df: pd.DataFrame, hypothesis: str, columns_to_use: Optional[list[str]] = None) -> list[dict[str, str]]:
    tool = get_agent().get_tool("plan_visualizations")
    res = tool.execute(df=df, hypothesis=hypothesis, columns_to_use=columns_to_use)
    return res.data if res.success else []


# ---------------------------------------------------------------------------
# LATS optimization shim
# ---------------------------------------------------------------------------
def run_lats_optimization(
    file_path: str,
    research_type: str,
    independent_variable: str,
    dependent_variable: str,
    independent_variable_type: Optional[str] = None,
    dependent_variable_type: Optional[str] = None,
    hypothesis: Optional[str] = None,
    problem_statement: Optional[str] = None,
    name: Optional[str] = None,
    covariates: Optional[list[str]] = None,
    time: Optional[str] = None,
    subject_group: Optional[str] = None,
    n_hypotheses_per_cleaner: int = 2,
):
    """
    Executes Language Agent Tree Search to evaluate cleaning strategies and
    hypothesis variants, extracting p-values from statistical tests and selecting
    the optimal route.
    """
    ctx = ResearchContext(
        file_path=file_path,
        name=name,
        problem_statement=problem_statement,
        hypothesis=hypothesis,
        research_type=research_type,
        independent_variable=independent_variable,
        dependent_variable=dependent_variable,
        independent_variable_type=independent_variable_type,
        dependent_variable_type=dependent_variable_type,
        covariates=covariates or [],
        time_variable=time,
        subject_group=subject_group,
    )
    return get_agent().run_lats_search(ctx, n_hypotheses_per_cleaner=n_hypotheses_per_cleaner)


# ---------------------------------------------------------------------------
# LaTeX paper generation shim
# ---------------------------------------------------------------------------
def generate_latex_paper(
    template: str = "generic_arxiv",
    title: Optional[str] = None,
    abstract: Optional[str] = None,
    introduction: Optional[str] = None,
    literature_review: Optional[list] = None,
    methodology: Optional[str] = None,
    results_text: Optional[str] = None,
    test_results: Optional[dict] = None,
    visualizations: Optional[list] = None,
    output_dir: Optional[str] = None,
    compile_pdf: bool = True,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Creates a publication-ready LaTeX paper using famous journal templates.
    """
    tool = get_agent().get_tool("create_latex_paper")
    res = tool.execute(
        template=template,
        title=title,
        abstract=abstract,
        introduction=introduction,
        literature_review=literature_review,
        methodology=methodology,
        results_text=results_text,
        test_results=test_results,
        visualizations=visualizations,
        output_dir=output_dir,
        compile_pdf=compile_pdf,
        **kwargs,
    )
    return res.data if res.success else {"error": res.error}


