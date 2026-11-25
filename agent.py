"""
ResearchAgent - Modular, continuous-learning research assistant orchestrator.

Coordinates Tools, Skills, and Memory to automate end-to-end scientific research workflows:
  1. Structural metadata & missing field prediction
  2. Dataset cleaning and variable summary generation
  3. Statistical test selection, execution, and interpretation
  4. Hypothesis, Abstract, Methodology, and Introduction generation
  5. Literature review with academic citations
  6. Visualization planning with schema-enforced plotting specs
  7. Continuous learning from session feedback and test outcomes
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Set

import pandas as pd

from research_agent.config import AgentConfig
from research_agent.memory.context import SessionContext
from research_agent.memory.store import MemoryStore
from research_agent.schemas import (
    AgentOutput,
    LiteratureItem,
    ResearchContext,
    StructuralMetadata,
    TestResult,
    VisualizationSpec,
)
from research_agent.graph.state import ResearchGraphState
from research_agent.graph.graph import build_research_graph
from research_agent.lats.agent import LATSAgent, LATSResult
from research_agent.skills.abstract import AbstractSkill
from research_agent.skills.base import BaseSkill
from research_agent.skills.confounding_variables import ConfoundingVariablesSkill
from research_agent.skills.hypothesis import HypothesisSkill
from research_agent.skills.introduction import IntroductionSkill
from research_agent.skills.literature_review import LiteratureReviewSkill
from research_agent.skills.metadata_prediction import MetadataPredictionSkill
from research_agent.skills.methodology import MethodologySkill
from research_agent.skills.test_interpretation import TestInterpretationSkill
from research_agent.tools.base import BaseTool
from research_agent.tools.analysis_tools import AutonomousAnalysisTool
from research_agent.tools.data_tools import (
    CleanDataFrameTool,
    DetectVariableTypeTool,
    ProduceVariableSummaryTool,
    ReadFileTool,
)
from research_agent.tools.rag_tool import StatisticalRAGTool
from research_agent.tools.search_tools import CitationLookupTool, WebSearchTool
from research_agent.tools.statistics_tools import ConductTestsTool
from research_agent.tools.visualization_tools import PlanVisualizationsTool
from research_agent.tools.latex_paper_tool import CreateLatexPaperTool

logger = logging.getLogger(__name__)


class ResearchAgent:
    """The continuous-learning research assistant agent."""

    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        self.memory = MemoryStore(
            backend=self.config.memory_backend,
            path=self.config.memory_path,
        )

        # 1. Register Core Tools
        self.tools: dict[str, BaseTool] = {}
        for tool in [
            ReadFileTool(),
            CleanDataFrameTool(),
            DetectVariableTypeTool(),
            ProduceVariableSummaryTool(),
            AutonomousAnalysisTool(),
            StatisticalRAGTool(),
            WebSearchTool(),
            CitationLookupTool(),
            ConductTestsTool(),
            PlanVisualizationsTool(config=self.config),
            CreateLatexPaperTool(config=self.config),
        ]:
            self.register_tool(tool)

        # 2. Register Core Skills
        self.skills: dict[str, BaseSkill] = {}
        for skill_cls in [
            HypothesisSkill,
            AbstractSkill,
            MethodologySkill,
            IntroductionSkill,
            LiteratureReviewSkill,
            MetadataPredictionSkill,
            TestInterpretationSkill,
            ConfoundingVariablesSkill,
        ]:
            self.register_skill(skill_cls(self.config))

        # 3. Compile StateGraph with shared tool and skill registries
        self.graph = build_research_graph(
            config=self.config,
            memory=self.memory,
            tools=self.tools,
            skills=self.skills,
        )

        # 4. Initialize LATS Tree Search Agent
        self.lats = LATSAgent(config=self.config, memory=self.memory)

    # ------------------------------------------------------------------
    # Registry management
    # ------------------------------------------------------------------
    def register_tool(self, tool: BaseTool):
        """Register a new tool."""
        self.tools[tool.name] = tool

    def get_tool(self, name: str) -> BaseTool:
        """Lookup a tool by name."""
        if name not in self.tools:
            raise KeyError(f"Tool '{name}' is not registered.")
        return self.tools[name]

    def register_skill(self, skill: BaseSkill):
        """Register a new skill."""
        self.skills[skill.name] = skill

    def get_skill(self, name: str) -> BaseSkill:
        """Lookup a skill by name."""
        if name not in self.skills:
            raise KeyError(f"Skill '{name}' is not registered.")
        return self.skills[name]

    def create_session(self, context: Optional[ResearchContext] = None) -> SessionContext:
        """Start a new working memory session."""
        return SessionContext(research_context=context or ResearchContext())

    # ------------------------------------------------------------------
    # High-level Agent Actions
    # ------------------------------------------------------------------
    def read_and_clean_data(
        self,
        file_path: str,
        context: ResearchContext,
    ) -> Optional[pd.DataFrame]:
        """Reads dataset, detects types, and cleans NA/outliers."""
        read_tool = self.get_tool("read_file")
        read_res = read_tool.execute(file_path=file_path)
        if not read_res.success or read_res.data is None:
            logger.warning("Failed to read file: %s", read_res.error)
            return None

        df = read_res.data
        if not context.dependent_variable:
            return df

        iv_list = [context.independent_variable] if context.independent_variable else []
        extra_vars = list(context.covariates or [])
        if context.time_variable:
            extra_vars.append(context.time_variable)
        if context.subject_group:
            extra_vars.append(context.subject_group)

        clean_tool = self.get_tool("clean_dataframe")
        clean_res = clean_tool.execute(
            df=df,
            dependent_variable=context.dependent_variable,
            independent_variables=iv_list,
            covariates=extra_vars,
        )
        if clean_res.success:
            return clean_res.data
        return df

    def predict_missing_fields(
        self,
        context: ResearchContext,
        session: Optional[SessionContext] = None,
    ) -> ResearchContext:
        """Predicts missing structural fields (IV, DV, research type, hypothesis)."""
        sess = session or self.create_session(context)
        df = None
        if context.file_path and os.path.exists(context.file_path):
            read_res = self.get_tool("read_file").execute(file_path=context.file_path)
            if read_res.success and read_res.data is not None:
                df = read_res.data
                context.dataset_columns = df.columns.tolist()
                context.dataset_preview = df.head(5).to_string()
                context.sample_size = len(df)

        # Detect types for available columns
        detect_tool = self.get_tool("detect_variable_type")
        sure_columns = []
        if df is not None:
            for col in context.dataset_columns:
                learned_type = self.memory.get_learned_variable_type(col)
                if learned_type:
                    sure_columns.append(col)
                else:
                    t_res = detect_tool.execute(column_name=col, df=df)
                    if t_res.success and t_res.data:
                        sure_columns.append(col)

        # Predict structural metadata if needed
        needs_structural = (
            not context.independent_variable
            or not context.dependent_variable
            or not context.research_type
        )
        if needs_structural:
            meta_skill = self.get_skill("metadata_prediction")
            skill_input = context.model_dump()
            skill_input["data_columns"] = sure_columns if sure_columns else context.dataset_columns
            metadata: StructuralMetadata = meta_skill.run(skill_input, memory=self.memory)

            if not context.independent_variable and metadata.independent_variable:
                context.independent_variable = metadata.independent_variable
            if not context.dependent_variable and metadata.dependent_variable:
                context.dependent_variable = metadata.dependent_variable
            if not context.research_type and metadata.research_type:
                context.research_type = metadata.research_type
            if not context.time_variable and metadata.time_variable:
                context.time_variable = metadata.time_variable
            if not context.subject_group and metadata.subject_group:
                context.subject_group = metadata.subject_group

        # Determine variable types and compute summaries
        summary_tool = self.get_tool("produce_variable_summary")
        if df is not None:
            if context.independent_variable and not context.independent_variable_type:
                t_res = detect_tool.execute(column_name=context.independent_variable, df=df)
                context.independent_variable_type = t_res.data if t_res.success else "numeric"
            if context.dependent_variable and not context.dependent_variable_type:
                t_res = detect_tool.execute(column_name=context.dependent_variable, df=df)
                context.dependent_variable_type = t_res.data if t_res.success else "numeric"

            if context.independent_variable and context.independent_variable_type:
                s_res = summary_tool.execute(
                    column_name=context.independent_variable,
                    column_type=context.independent_variable_type,
                    df=df,
                )
                context.independent_variable_summary = s_res.data if s_res.success else ""

            if context.dependent_variable and context.dependent_variable_type:
                s_res = summary_tool.execute(
                    column_name=context.dependent_variable,
                    column_type=context.dependent_variable_type,
                    df=df,
                )
                context.dependent_variable_summary = s_res.data if s_res.success else ""

            context.dataset_summary = df.describe().to_string()

        # Predict hypothesis if missing
        if not context.hypothesis:
            hypo_skill = self.get_skill("hypothesis")
            context.hypothesis = hypo_skill.run(context.model_dump(), memory=self.memory)

        sess.update_context(**context.model_dump())
        return context

    def generate_hypothesis(
        self,
        context: ResearchContext,
        session: Optional[SessionContext] = None,
    ) -> str:
        """Generates a testable scientific hypothesis."""
        sess = session or self.create_session(context)
        skill = self.get_skill("hypothesis")
        hypothesis = skill.run(context.model_dump(), memory=self.memory)
        sess.set_skill_output("hypothesis", hypothesis)
        return hypothesis

    def run_statistical_tests(
        self,
        context: ResearchContext,
        df: Optional[pd.DataFrame] = None,
        session: Optional[SessionContext] = None,
    ) -> tuple[dict[str, Any], str]:
        """Runs the statistical tests and produces a natural-language interpretation."""
        sess = session or self.create_session(context)
        target_df = df
        if target_df is None:
            file_to_read = context.processed_file_path or context.file_path
            if file_to_read and os.path.exists(file_to_read):
                read_res = self.get_tool("read_file").execute(file_path=file_to_read)
                if read_res.success:
                    target_df = read_res.data

        if target_df is None:
            return {"error": "Dataset not available for statistical testing."}, "No data available."

        test_tool = self.get_tool("conduct_tests")
        tool_args = {
            "df": target_df,
            "research_design": context.research_type or "cross_sectional",
            "iv_type": context.independent_variable_type or "numeric",
            "dv_type": context.dependent_variable_type or "numeric",
            "iv": context.independent_variable,
            "dv": context.dependent_variable,
            "covariates": context.covariates,
            "time": context.time_variable,
            "subject_id": context.subject_group,
        }
        res = test_tool.execute(**tool_args)
        sess.record_tool_call("conduct_tests", tool_args, res)

        results = res.data if res.success else {"error": res.error}

        # Interpret results via skill
        interp_skill = self.get_skill("test_interpretation")
        interp_input = context.model_dump()
        interp_input["results"] = results
        interpretation = interp_skill.run(interp_input, memory=self.memory)

        sess.set_skill_output("test_results", results)
        sess.set_skill_output("test_interpretation", interpretation)

        return results, interpretation

    def generate_abstract(
        self,
        context: ResearchContext,
        test_interpretation: Optional[str] = None,
        session: Optional[SessionContext] = None,
    ) -> str:
        """Generates the paper abstract."""
        sess = session or self.create_session(context)
        skill = self.get_skill("abstract")
        data = context.model_dump()
        if test_interpretation:
            data["test_interpretation"] = test_interpretation
        abstract = skill.run(data, memory=self.memory)
        sess.set_skill_output("abstract", abstract)
        return abstract

    def generate_methodology(
        self,
        context: ResearchContext,
        test_names: Optional[list[str]] = None,
        session: Optional[SessionContext] = None,
    ) -> str:
        """Generates the methodology section."""
        sess = session or self.create_session(context)
        skill = self.get_skill("methodology")
        data = context.model_dump()
        if test_names:
            data["test_names"] = ", ".join(test_names)
        methodology = skill.run(data, memory=self.memory)
        sess.set_skill_output("methodology", methodology)
        return methodology

    def generate_literature_review(
        self,
        context: ResearchContext,
        session: Optional[SessionContext] = None,
    ) -> list[LiteratureItem]:
        """Conducts a literature review with citations."""
        sess = session or self.create_session(context)
        skill = self.get_skill("literature_review")
        items = skill.run(context.model_dump(), memory=self.memory)
        sess.set_skill_output("literature_review", items)
        return items

    def generate_introduction(
        self,
        context: ResearchContext,
        literature: Optional[list[LiteratureItem]] = None,
        session: Optional[SessionContext] = None,
    ) -> str:
        """Generates the paper introduction."""
        sess = session or self.create_session(context)
        skill = self.get_skill("introduction")
        data = context.model_dump()
        if literature is not None:
            data["literature_review"] = literature
        intro = skill.run(data, memory=self.memory)
        sess.set_skill_output("introduction", intro)
        return intro

    def plan_visualizations(
        self,
        context: ResearchContext,
        df: Optional[pd.DataFrame] = None,
        session: Optional[SessionContext] = None,
    ) -> list[VisualizationSpec]:
        """Plans custom visualizations tailored to dataset columns and hypothesis."""
        sess = session or self.create_session(context)
        target_df = df
        if target_df is None:
            file_to_read = context.processed_file_path or context.file_path
            if file_to_read and os.path.exists(file_to_read):
                read_res = self.get_tool("read_file").execute(file_path=file_to_read)
                if read_res.success:
                    target_df = read_res.data

        if target_df is None:
            return []

        tool = self.get_tool("plan_visualizations")
        res = tool.execute(df=target_df, hypothesis=context.hypothesis or "")
        sess.record_tool_call("plan_visualizations", {"hypothesis": context.hypothesis}, res)

        specs: list[VisualizationSpec] = []
        if res.success and isinstance(res.data, list):
            for item in res.data:
                specs.append(VisualizationSpec(
                    title=item.get("title", ""),
                    mark=item.get("mark", ""),
                    x_axis=item.get("x_axis"),
                    y_axis=item.get("y_axis"),
                    color=item.get("color"),
                ))
        sess.set_skill_output("visualizations", specs)
        return specs

    def find_confounding_variables(
        self,
        context: ResearchContext,
        session: Optional[SessionContext] = None,
    ) -> list[str]:
        """Discovers potential confounding variables via web search + reasoning."""
        sess = session or self.create_session(context)
        skill = self.get_skill("confounding_variables")
        confounders = skill.run(context.model_dump(), memory=self.memory)
        sess.set_skill_output("confounding_variables", confounders)
        return confounders

    # ------------------------------------------------------------------
    # End-to-end Pipeline Execution
    # ------------------------------------------------------------------
    def run_full_analysis(
        self,
        context: ResearchContext,
        df: Optional[pd.DataFrame] = None,
        tasks: Optional[Set[str]] = None,
        paper_template: Optional[str] = None,
        output_dir: Optional[str] = None,
        compile_pdf: bool = True,
        **kwargs: Any,
    ) -> AgentOutput:
        """
        Executes the full research pipeline (or a selected subset of tasks)
        by delegating to the canonical LangGraph StateGraph engine.
        Converts the final graph state into an AgentOutput contract.
        """
        task_overrides: Dict[str, Any] = {}
        if tasks is not None:
            task_overrides["max_lats_retries"] = 0
            task_overrides["max_quality_retries"] = 0
            task_overrides["max_peer_review_retries"] = 0
            if "tests" not in tasks and "abstract" not in tasks and "methodology" not in tasks:
                task_overrides["test_status"] = "skipped"
            if "literature_review" not in tasks and "introduction" not in tasks:
                task_overrides["literature_review_status"] = "skipped"
            if "introduction" not in tasks:
                task_overrides["introduction_status"] = "skipped"
            if "methodology" not in tasks:
                task_overrides["methodology_status"] = "skipped"
            if "abstract" not in tasks:
                task_overrides["abstract_status"] = "skipped"
            if "discussion" not in tasks and "paper" not in tasks and "latex_paper" not in tasks:
                task_overrides["discussion_status"] = "skipped"
            if "conclusion" not in tasks and "paper" not in tasks and "latex_paper" not in tasks:
                task_overrides["conclusion_status"] = "skipped"
            if "visualizations" not in tasks:
                task_overrides["visualization_status"] = "skipped"
            if "paper" not in tasks and "latex_paper" not in tasks:
                task_overrides["latex_paper_status"] = "skipped"

        graph_state = self.run_graph(
            context=context,
            df=df,
            tasks=task_overrides if task_overrides else None,
            paper_template=paper_template,
            output_dir=output_dir,
            compile_pdf=compile_pdf,
            **kwargs,
        )
        out_context = graph_state.get("context", context)

        interp = graph_state.get("test_interpretation", "")
        output = AgentOutput(
            hypothesis=out_context.hypothesis,
            test_interpretation=interp,
            abstract=graph_state.get("abstract", "") or "",
            introduction=graph_state.get("introduction", "") or "",
            methodology=graph_state.get("methodology", "") or "",
            discussion=graph_state.get("discussion", "") or "",
            background=graph_state.get("background", "") or "",
            conclusion=graph_state.get("conclusion", "") or "",
            errors=graph_state.get("errors", []),
            statuses={k: v for k, v in graph_state.items() if k.endswith("_status")},
            autonomous_analysis=graph_state.get("autonomous_analysis", {}),
            study_facts=graph_state.get("study_facts"),
        )

        test_results = graph_state.get("test_results", {})
        if isinstance(test_results, dict):
            for t_name, t_data in test_results.items():
                if t_name != "error" and not t_name.startswith("_") and isinstance(t_data, dict):
                    output.test_results.append(
                        TestResult(
                            test_name=t_name,
                            results=t_data,
                            interpretation=interp,
                        )
                    )

        lit = graph_state.get("literature_review", [])
        if isinstance(lit, list):
            output.literature_review = [
                i if isinstance(i, LiteratureItem) else LiteratureItem(**i) for i in lit
            ]

        vis = graph_state.get("visualizations", [])
        if isinstance(vis, list):
            output.visualizations = [
                v if isinstance(v, VisualizationSpec) else VisualizationSpec(**v) for v in vis
            ]

        if tasks is None or "confounders" in tasks:
            output.confounding_variables = self.find_confounding_variables(out_context)

        # LaTeX Paper outputs from StateGraph
        if graph_state.get("latex_paper_bundle"):
            bundle = graph_state["latex_paper_bundle"]
            output.paper_bundle = bundle
            output.paper_latex = bundle.get("latex_source") or bundle.get("tex_source")
            output.latex_paper_path = bundle.get("tex_file") or bundle.get("tex_path")
            output.latex_pdf_path = bundle.get("pdf_file") or bundle.get("pdf_path")
        elif graph_state.get("latex_paper_status") not in ("failed", "blocked") and tasks is not None and ("paper" in tasks or "latex_paper" in tasks):
            paper_res = self.create_paper(
                out_context,
                agent_output=output,
                template=paper_template or "generic_arxiv",
                output_dir=output_dir,
                compile_pdf=compile_pdf,
            )
            if isinstance(paper_res, dict) and ("latex_source" in paper_res or "tex_source" in paper_res):
                output.paper_latex = paper_res.get("latex_source") or paper_res.get("tex_source")
                output.paper_bundle = paper_res
                output.latex_paper_path = paper_res.get("tex_file") or paper_res.get("tex_path")
                output.latex_pdf_path = paper_res.get("pdf_file") or paper_res.get("pdf_path")

        return output

    def run_graph(
        self,
        context: ResearchContext,
        df: Optional[pd.DataFrame] = None,
        tasks: Optional[Dict[str, Any]] = None,
        paper_template: Optional[str] = None,
        output_dir: Optional[str] = None,
        compile_pdf: bool = True,
        **kwargs: Any,
    ) -> ResearchGraphState:
        """
        Executes the agent workflow via the compiled LangGraph StateGraph.
        Travels from state to state, with each state executing its bound tools and skills.
        The final state compiles all research findings into a publication-ready LaTeX paper.
        """
        initial_state: ResearchGraphState = {
            "context": context.model_copy(deep=True),
            "df": df,
            "test_status": "not_started",
            "literature_review_status": "not_started",
            "abstract_status": "not_started",
            "introduction_status": "not_started",
            "methodology_status": "not_started",
            "visualization_status": "not_started",
            "playground_status": "not_started",
            "latex_paper_status": "not_started",
            "tool_history": [],
            "errors": [],
        }
        if paper_template:
            initial_state["paper_template"] = paper_template
        if output_dir:
            initial_state["output_dir"] = output_dir
        if compile_pdf is not None:
            initial_state["compile_pdf"] = compile_pdf
        for k, v in kwargs.items():
            initial_state[k] = v

        if tasks:
            initial_state.update(tasks)
        for toggle, status in {
            "do_tests": "test_status", "do_literature_review": "literature_review_status",
            "do_abstract": "abstract_status", "do_introduction": "introduction_status",
            "do_methodology": "methodology_status", "do_visualization": "visualization_status",
            "do_playground": "playground_status",
        }.items():
            if initial_state.get(toggle) is False:
                initial_state[status] = "skipped"
        return self.graph.invoke(initial_state)

    def create_latex_paper(
        self,
        context: ResearchContext,
        agent_output: Optional[AgentOutput] = None,
        template: str = "generic_arxiv",
        output_dir: Optional[str] = None,
        compile_pdf: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Creates a publication-ready academic LaTeX paper from research context and outputs."""
        tool = self.get_tool("create_latex_paper")
        res = tool.execute(
            context=context,
            agent_output=agent_output,
            template=template,
            output_dir=output_dir,
            compile_pdf=compile_pdf,
            **kwargs,
        )
        if res.success and isinstance(res.data, dict):
            return res.data
        raise RuntimeError(f"LaTeX paper creation failed: {res.error}")

    # Alias for convenience
    create_paper = create_latex_paper

    # ------------------------------------------------------------------
    # LATS (Language Agent Tree Search) Optimization
    # ------------------------------------------------------------------
    def run_lats_search(
        self,
        context: ResearchContext,
        df: Optional[pd.DataFrame] = None,
        n_hypotheses_per_cleaner: int = 2,
    ) -> LATSResult:
        """
        Executes LATS across different cleaning options and hypothesis candidates,
        evaluates empirical p-values from statistical tests, and selects the best route.
        """
        return self.lats.optimize_route(
            context=context,
            df=df,
            n_hypotheses_per_cleaner=n_hypotheses_per_cleaner,
        )

    def run_lats_pipeline(
        self,
        context: ResearchContext,
        df: Optional[pd.DataFrame] = None,
    ) -> AgentOutput:
        """
        End-to-end pipeline:
          1. Uses LATS tree search to find the optimal data cleaning strategy and hypothesis.
          2. Carries forward the optimal trajectory to generate abstract, methodology,
             literature review, and visualizations.
        """
        lats_result = self.run_lats_search(context=context, df=df)
        opt_context = lats_result.optimized_context

        # Use winning trajectory's test results & cleaned df
        session = self.create_session(opt_context)
        output = AgentOutput(
            hypothesis=opt_context.hypothesis,
            test_interpretation=lats_result.best_trajectory.reflection,
        )

        for t_name, t_data in lats_result.best_trajectory.test_results.items():
            if t_name != "error" and isinstance(t_data, dict):
                output.test_results.append(
                    TestResult(
                        test_name=t_name,
                        results=t_data,
                        interpretation=lats_result.best_trajectory.reflection,
                    )
                )

        output.abstract = self.generate_abstract(
            opt_context,
            test_interpretation=lats_result.best_trajectory.reflection,
            session=session,
        )
        test_names = [t.test_name for t in output.test_results]
        output.methodology = self.generate_methodology(opt_context, test_names=test_names, session=session)
        output.visualizations = self.plan_visualizations(opt_context, df=lats_result.optimal_cleaned_df, session=session)

        # Generate final academic LaTeX paper directly from dataset & winning trajectory
        try:
            paper_res = self.create_latex_paper(opt_context, agent_output=output)
            output.paper_bundle = paper_res
            output.paper_latex = paper_res.get("latex_source") or paper_res.get("tex_source")
            output.latex_paper_path = paper_res.get("tex_file") or paper_res.get("tex_path")
            output.latex_pdf_path = paper_res.get("pdf_file") or paper_res.get("pdf_path")
        except Exception as exc:
            logger.warning("Auto-generation of LaTeX paper in LATS pipeline failed: %s", exc)

        return output
