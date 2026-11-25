"""
Graph Node implementations for the Research Agent StateGraph.

Each node represents a distinct state in the research workflow and has
its own dedicated tools and skills bound to that state.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any, Dict, List, Optional
import pandas as pd

from research_agent.config import AgentConfig
from research_agent.graph.state import ResearchGraphState
from research_agent.memory.store import MemoryStore
from research_agent.schemas import LiteratureItem, ResearchContext, VisualizationSpec
from research_agent.skills.abstract import AbstractSkill
from research_agent.skills.actor import PaperActorReviser
from research_agent.skills.conclusion import ConclusionSkill
from research_agent.skills.critic import (
    CitationGroundingCritic,
    CrossSectionConsistencyCritic,
    FinalPeerReviewCritic,
    GlobalPaperCritic,
    IntroConclusionCritic,
    MethodsResultsCritic,
    ResultsDiscussionCritic,
    WebGroundedCriticSkill,
)
from research_agent.skills.discussion import DiscussionSkill
from research_agent.skills.hypothesis import HypothesisSkill
from research_agent.skills.introduction import IntroductionSkill
from research_agent.skills.literature_review import LiteratureReviewSkill
from research_agent.skills.metadata_prediction import MetadataPredictionSkill
from research_agent.skills.methodology import MethodologySkill
from research_agent.skills.test_interpretation import TestInterpretationSkill
from research_agent.tools.data_tools import (
    CleanDataFrameTool,
    DetectVariableTypeTool,
    ProduceVariableSummaryTool,
    ReadFileTool,
)
from research_agent.tools.search_tools import CitationLookupTool, WebSearchTool
from research_agent.tools.statistics_tools import ConductTestsTool
from research_agent.tools.analysis_tools import AutonomousAnalysisTool
from research_agent.tools.visualization_tools import PlanVisualizationsTool
from research_agent.tools.latex_paper_tool import CreateLatexPaperTool
from research_agent.lats.cleaner_pool import CleanerPool
from research_agent.lats.evaluator import LATSEvaluator
from research_agent.lats.hypothesis_generator import HypothesisGenerator

logger = logging.getLogger(__name__)


class ResearchGraphNodes:
    """Encapsulates node execution logic and state-tool bindings."""

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        memory: Optional[MemoryStore] = None,
        tools: Optional[Dict[str, Any]] = None,
        skills: Optional[Dict[str, Any]] = None,
    ):
        self.config = config or AgentConfig()
        self.memory = memory or MemoryStore(
            backend=self.config.memory_backend,
            path=self.config.memory_path,
        )
        tools = tools or {}
        skills = skills or {}

        # -------------------------------------------------------------
        # State-to-Tool Bindings (reuses shared tool/skill instances if provided)
        # -------------------------------------------------------------
        # 1. Ingestion / Data Cleaning State Tools
        self.ingestion_tools = {
            "read_file": tools.get("read_file") or ReadFileTool(),
            "clean_dataframe": tools.get("clean_dataframe") or CleanDataFrameTool(),
            "detect_variable_type": tools.get("detect_variable_type") or DetectVariableTypeTool(),
            "produce_variable_summary": tools.get("produce_variable_summary") or ProduceVariableSummaryTool(),
        }

        # 2. Metadata Prediction State Tools & Skills
        self.metadata_skill = skills.get("metadata_prediction") or MetadataPredictionSkill(self.config)
        self.hypothesis_skill = skills.get("hypothesis") or HypothesisSkill(self.config)

        # 3. Statistical Testing State Tools & Skills
        self.testing_tools = {
            "conduct_tests": tools.get("conduct_tests") or ConductTestsTool(),
        }
        self.test_interp_skill = skills.get("test_interpretation") or TestInterpretationSkill(self.config)

        # 3b. Autonomous Analysis Tool
        self.autonomous_analysis_tool = tools.get("autonomous_data_analysis") or AutonomousAnalysisTool()

        # 4. Literature Review State Tools & Skills
        self.lit_review_tools = {
            "web_search": tools.get("web_search") or WebSearchTool(),
            "citation_lookup": tools.get("citation_lookup") or CitationLookupTool(),
        }
        self.lit_review_skill = skills.get("literature_review") or LiteratureReviewSkill(self.config)

        # 5. Writing State Skills
        self.introduction_skill = skills.get("introduction") or IntroductionSkill(self.config)
        self.methodology_skill = skills.get("methodology") or MethodologySkill(self.config)
        self.abstract_skill = skills.get("abstract") or AbstractSkill(self.config)
        self.discussion_skill = skills.get("discussion") or DiscussionSkill(self.config)
        self.conclusion_skill = skills.get("conclusion") or ConclusionSkill(self.config)

        # 5b. Paired Reflexion & Cross-Section Critic Skills
        self.methods_results_critic = skills.get("methods_results_critic") or MethodsResultsCritic(self.config)
        self.results_discussion_critic = skills.get("results_discussion_critic") or ResultsDiscussionCritic(self.config)
        self.intro_conclusion_critic = skills.get("intro_conclusion_critic") or IntroConclusionCritic(self.config)
        self.web_grounded_critic = skills.get("web_grounded_critic") or WebGroundedCriticSkill(self.config)
        self.global_paper_critic = skills.get("global_paper_critic") or GlobalPaperCritic(self.config)
        self.cross_section_critic = skills.get("cross_section_consistency_critic") or CrossSectionConsistencyCritic(self.config)
        self.citation_critic = skills.get("citation_grounding_critic") or CitationGroundingCritic(self.config)
        self.final_peer_review_critic = skills.get("final_peer_review_critic") or FinalPeerReviewCritic(self.config)
        self.paper_actor_reviser = skills.get("paper_actor_reviser") or PaperActorReviser(self.config)

        # 6. Visualization State Tools
        self.visualization_tools = {
            "plan_visualizations": tools.get("plan_visualizations") or PlanVisualizationsTool(),
        }

        # 7. Paper Generation State Tools
        self.paper_tools = {
            "create_latex_paper": tools.get("create_latex_paper") or CreateLatexPaperTool(config=self.config),
        }

        # 8. LATS Optimization & Evaluation Components
        self.cleaner_pool = CleanerPool()
        self.hypothesis_generator = HypothesisGenerator(self.config)
        self.lats_evaluator = LATSEvaluator()

    # =========================================================================
    # 1. State: DATA_INGESTION_AND_CLEANING
    # =========================================================================
    def ingest_and_clean_state(self, state: ResearchGraphState) -> Dict[str, Any]:
        """State for loading, inspecting, type-casting, and cleaning the dataset."""
        logger.info("Entering State: DATA_INGESTION_AND_CLEANING")
        context: ResearchContext = state.get("context", ResearchContext())
        node_history: List[Dict[str, Any]] = []
        node_errors: List[str] = []

        raw_df = state.get("raw_df")
        df = state.get("df")
        file_to_read = context.processed_file_path or context.file_path

        # Tool 1: read_file if dataframe not yet loaded
        if df is None and file_to_read and os.path.exists(file_to_read):
            read_res = self.ingestion_tools["read_file"].execute(file_path=file_to_read)
            node_history.append({"state": "ingest_and_clean", "tool": "read_file", "success": read_res.success})
            if read_res.success:
                df = read_res.data
            else:
                node_errors.append(f"Failed to read file: {read_res.error}")

        # Cache original uncleaned raw dataframe for safe cyclic LATS retries
        if df is not None and raw_df is None:
            raw_df = df.copy()

        source_df = raw_df if raw_df is not None else df
        cleaning_strategy = state.get("cleaning_strategy") or "no_outlier_removal"
        retention = 1.0

        if source_df is not None:
            context.dataset_columns = source_df.columns.tolist()
            initial_n = len(source_df)
            cleaned_df = source_df

            # Tool 2: clean_dataframe / cleaner_pool with chosen LATS strategy
            if context.dependent_variable:
                iv_list = [context.independent_variable] if context.independent_variable else []
                covs = list(context.covariates or [])
                # Identifiers and time coordinates must never be outlier-filtered as covariates.
                protected = {context.time_variable, context.subject_group} - {None}
                covs = [col for col in covs if col not in protected]

                try:
                    clean_res = self.cleaner_pool.apply_strategy(
                        df=source_df,
                        strategy_name=cleaning_strategy,
                        dependent_variable=context.dependent_variable,
                        independent_variables=iv_list,
                        covariates=covs,
                    )
                    cleaned_df = clean_res.cleaned_df
                    retention = clean_res.retention_ratio
                    node_history.append({
                        "state": "ingest_and_clean",
                        "tool": f"cleaner_pool({cleaning_strategy})",
                        "success": True,
                        "initial_n": clean_res.initial_n,
                        "final_n": clean_res.final_n,
                        "retention_ratio": retention,
                    })
                except Exception as exc:
                    logger.warning("Cleaner pool strategy %s failed, falling back to clean_dataframe: %s", cleaning_strategy, exc)
                    fallback_res = self.ingestion_tools["clean_dataframe"].execute(
                        df=source_df,
                        dependent_variable=context.dependent_variable,
                        independent_variables=iv_list,
                        covariates=covs,
                    )
                    node_history.append({"state": "ingest_and_clean", "tool": "clean_dataframe", "success": fallback_res.success})
                    if fallback_res.success and fallback_res.data is not None and not fallback_res.data.empty:
                        cleaned_df = fallback_res.data
                        retention = (len(cleaned_df) / initial_n) if initial_n > 0 else 1.0

            df = cleaned_df
            context.sample_size = len(df) if df is not None else 0
            context.dataset_preview = df.head(5).to_string() if df is not None else ""

            # Tool 3 & 4: detect_variable_type and produce_variable_summary
            detect_tool = self.ingestion_tools["detect_variable_type"]
            summary_tool = self.ingestion_tools["produce_variable_summary"]

            if context.independent_variable and not context.independent_variable_type:
                t_res = detect_tool.execute(column_name=context.independent_variable, df=df)
                if t_res.success and t_res.data:
                    context.independent_variable_type = t_res.data

            if context.dependent_variable and not context.dependent_variable_type:
                t_res = detect_tool.execute(column_name=context.dependent_variable, df=df)
                if t_res.success and t_res.data:
                    context.dependent_variable_type = t_res.data

            if context.independent_variable and context.independent_variable_type:
                s_res = summary_tool.execute(
                    column_name=context.independent_variable,
                    column_type=context.independent_variable_type,
                    df=df,
                )
                if s_res.success:
                    context.independent_variable_summary = s_res.data

            if context.dependent_variable and context.dependent_variable_type:
                s_res = summary_tool.execute(
                    column_name=context.dependent_variable,
                    column_type=context.dependent_variable_type,
                    df=df,
                )
                if s_res.success:
                    context.dependent_variable_summary = s_res.data

            context.dataset_summary = df.describe().to_string() if df is not None else ""

        return {
            "context": context,
            "df": df,
            "raw_df": raw_df,
            "cleaning_strategy": cleaning_strategy,
            "cleaning_history": [cleaning_strategy],
            "dataset_preview": context.dataset_preview,
            "sample_size": context.sample_size,
            "retention_ratio": retention,
            "tool_history": node_history,
            "errors": node_errors,
        }

    # =========================================================================
    # 2. State: METADATA_PREDICTION
    # =========================================================================
    def metadata_prediction_state(self, state: ResearchGraphState) -> Dict[str, Any]:
        """State for inferring study design, missing variables, and hypothesis."""
        logger.info("Entering State: METADATA_PREDICTION")
        context: ResearchContext = state.get("context", ResearchContext())
        df = state.get("df")
        node_history: List[Dict[str, Any]] = []

        needs_structural = (
            not context.independent_variable
            or not context.dependent_variable
            or not context.research_type
        )

        if needs_structural:
            columns = context.dataset_columns or (df.columns.tolist() if df is not None else [])
            payload = context.model_dump()
            payload["data_columns"] = columns
            metadata = self.metadata_skill.run(payload, memory=self.memory)

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

        preparation = {}
        if needs_structural and context.independent_variable and context.dependent_variable and df is not None:
            preparation = self.ingest_and_clean_state({**state, "context": context})
            context = preparation["context"]

        # Hypothesise or pivot hypothesis if routed via LATS retry
        is_hypothesis_retry = state.get("lats_route_decision") == "retry_hypothesis"
        if not context.hypothesis or is_hypothesis_retry:
            if is_hypothesis_retry:
                prev_hypo = context.hypothesis
                try:
                    candidates = self.hypothesis_generator.generate_candidates(context.model_dump(), n_candidates=3)
                    new_hypo = candidates[0].hypothesis
                    for c in candidates:
                        if c.hypothesis != prev_hypo:
                            new_hypo = c.hypothesis
                            break
                    context.hypothesis = new_hypo
                    logger.info("LATS hypothesis pivoted to: %s", new_hypo)
                    node_history.append({"state": "metadata_prediction", "action": "pivot_hypothesis", "hypothesis": new_hypo})
                except Exception as exc:
                    logger.warning("Hypothesis generator failed on retry: %s", exc)
                    context.hypothesis = f"{context.hypothesis} (adjusted for confounding variables)"
            else:
                hypo = self.hypothesis_skill.run(context.model_dump(), memory=self.memory)
                context.hypothesis = hypo
                node_history.append({"state": "metadata_prediction", "action": "generate_hypothesis", "hypothesis": hypo})

        return {
            "context": context,
            **preparation,
            "hypothesis_history": [context.hypothesis] if context.hypothesis else [],
            "tool_history": preparation.get("tool_history", []) + node_history,
        }

    # =========================================================================
    # 3. State: TESTS (Statistical Testing State)
    # =========================================================================
    def conduct_tests_state(self, state: ResearchGraphState) -> Dict[str, Any]:
        """State corresponding to original test_status workflow with LATS evaluation."""
        logger.info("Entering State: TESTS")
        context: ResearchContext = state.get("context", ResearchContext())
        df = state.get("df")
        raw_df = state.get("raw_df")

        if state.get("test_status") == "skipped":
            return {"test_status": "skipped", "lats_route_decision": "proceed"}

        if df is None:
            return {
                "test_status": "failed",
                "errors": ["No data available for testing"],
                "lats_route_decision": "proceed",
            }

        # Tool: conduct_tests
        test_tool = self.testing_tools["conduct_tests"]
        tool_args = {
            "df": df,
            "research_design": context.research_type or "cross_sectional",
            "iv_type": context.independent_variable_type or "numeric",
            "dv_type": context.dependent_variable_type or "numeric",
            "iv": context.independent_variable,
            "dv": context.dependent_variable,
            "covariates": context.covariates,
            "time": context.time_variable,
            "subject_id": context.subject_group,
        }
        test_res = test_tool.execute(**tool_args)
        node_history = [{"state": "tests", "tool": "conduct_tests", "success": test_res.success}]

        if not test_res.success or not isinstance(test_res.data, dict) or "error" in test_res.data:
            err_msg = test_res.error or (test_res.data.get("error", "Unknown test error") if isinstance(test_res.data, dict) else "Invalid statistical result")
            return {
                "test_status": "failed",
                "test_results": {"error": err_msg},
                "test_interpretation": "",
                "tool_history": node_history,
                "errors": [err_msg],
                "lats_route_decision": "proceed",
            }

        results = test_res.data
        test_names = [k for k in results.keys() if k != "error" and not k.startswith("_") and isinstance(results[k], dict)]

        # Extract immutable StudyFacts single source of truth
        study_facts = None
        try:
            from research_agent.study_facts import build_study_facts
            study_facts = build_study_facts(df, context, test_results=results, raw_df=raw_df)
        except Exception as exc:
            logger.warning("Could not build StudyFacts: %s", exc)

        # Skill: test_interpretation
        interp_payload = context.model_dump()
        interp_payload["results"] = results
        if study_facts is not None:
            interp_payload["study_facts"] = study_facts
        interp = self.test_interp_skill.run(interp_payload, memory=self.memory)

        # LATS Evaluation: Extract empirical p-values & evaluate retention
        p_pairs = LATSEvaluator.extract_p_values(results)
        p_vals = [p for _, p in p_pairs if not math.isnan(p)]
        min_p = min(p_vals) if p_vals else 1.0

        raw_n = len(raw_df) if raw_df is not None else len(df)
        final_n = len(df)
        retention = (final_n / raw_n) if raw_n > 0 else 1.0

        # Data preparation is specified independently of observed significance.
        # Null findings are valid; never change cleaning or hypothesis to chase p < alpha.
        lats_retries = state.get("lats_retry_count", 0)
        max_lats_retries = state.get("max_lats_retries", 3)
        route_decision = "proceed"
        reflection = (
            f"Analysis completed with minimum reported p={min_p:.4g} and "
            f"{retention:.1%} sample retention. No significance-driven retries were performed."
        )

        logger.info(
            "LATS Evaluation: decision=%s, min_p=%.4f, retention=%.1f%% (iteration %d/%d)",
            route_decision, min_p, retention * 100, lats_retries, max_lats_retries,
        )

        return {
            "test_status": "completed",
            "test_results": results,
            "test_names": test_names,
            "test_interpretation": interp,
            "study_facts": study_facts,
            "min_p_value": min_p,
            "retention_ratio": retention,
            "lats_route_decision": route_decision,
            "lats_reflection": reflection,
            "tool_history": node_history,
        }

    def handle_lats_retry_state(self, state: ResearchGraphState) -> Dict[str, Any]:
        """Prepares state for the next LATS iteration by selecting new cleaning strategy or hypothesis."""
        logger.info("Entering State: HANDLE_LATS_RETRY")
        lats_retries = state.get("lats_retry_count", 0) + 1
        decision = state.get("lats_route_decision", "retry_cleaning")

        strategies = ["iqr_standard", "iqr_conservative", "winsorization", "zscore_filter", "no_outlier_removal"]
        history = state.get("cleaning_history", [])

        next_strategy = state.get("cleaning_strategy", "iqr_standard")
        if decision == "retry_cleaning":
            if state.get("retention_ratio", 1.0) < 0.75:
                for candidate in ["winsorization", "iqr_conservative", "no_outlier_removal"]:
                    if candidate not in history:
                        next_strategy = candidate
                        break
            else:
                for candidate in strategies:
                    if candidate not in history:
                        next_strategy = candidate
                        break

        logger.info("LATS Retry #%d: next_strategy=%s, route=%s", lats_retries, next_strategy, decision)
        return {
            "lats_retry_count": lats_retries,
            "cleaning_strategy": next_strategy,
            "lats_route_decision": decision,
        }

    # =========================================================================
    # 4. State: LITERATURE_REVIEW (Literature Search & Synthesis State)
    # =========================================================================
    def literature_review_state(self, state: ResearchGraphState) -> Dict[str, Any]:
        """State corresponding to original literature_review_status workflow."""
        logger.info("Entering State: LITERATURE_REVIEW")
        if state.get("literature_review_status") == "skipped":
            return {"literature_review_status": "skipped"}
        context: ResearchContext = state.get("context", ResearchContext())

        try:
            items = self.lit_review_skill.run(context.model_dump(), memory=self.memory)
            if not items:
                from research_agent.latex_templates.manager import LaTeXPaperManager
                mgr = LaTeXPaperManager(config=self.config)
                items = mgr._synthesize_literature_items(context.model_dump())

            return {
                "literature_review_status": "completed",
                "literature_review": items,
                "background": getattr(self.lit_review_skill,"last_background", ""),
            }
        except Exception as exc:
            logger.exception("Literature review failed: %s", exc)
            return {
                "literature_review_status": "failed",
                "literature_review": [],
                "errors": [str(exc)],
            }

    # =========================================================================
    # 5. State: INTRODUCTION (Introduction Writing State)
    # =========================================================================
    def introduction_state(self, state: ResearchGraphState) -> Dict[str, Any]:
        """State corresponding to original introduction_status workflow (depends on literature review)."""
        logger.info("Entering State: INTRODUCTION")
        if state.get("introduction_status") == "skipped":
            return {"introduction_status": "skipped"}
        context: ResearchContext = state.get("context", ResearchContext())
        lit_items = state.get("literature_review", [])

        try:
            payload = context.model_dump()
            payload["literature_review"] = lit_items
            if state.get("study_facts"):
                payload["study_facts"] = state.get("study_facts")
            intro = self.introduction_skill.run(payload, memory=self.memory)
            return {
                "introduction_status": "completed",
                "introduction": intro,
            }
        except Exception as exc:
            logger.exception("Introduction generation failed: %s", exc)
            return {
                "introduction_status": "failed",
                "introduction": "",
                "errors": [str(exc)],
            }

    # =========================================================================
    # 6. State: METHODOLOGY (Methodology Writing State)
    # =========================================================================
    def methodology_state(self, state: ResearchGraphState) -> Dict[str, Any]:
        """State corresponding to original methodology_status workflow (depends on tests)."""
        logger.info("Entering State: METHODOLOGY")
        if state.get("methodology_status") == "skipped":
            return {"methodology_status": "skipped"}
        context: ResearchContext = state.get("context", ResearchContext())
        test_names = state.get("test_names", [])

        try:
            payload = context.model_dump()
            if test_names:
                payload["test_names"] = ", ".join(test_names)
            if state.get("study_facts"):
                payload["study_facts"] = state.get("study_facts")
            critique = state.get("quality_critique")
            if critique and state.get("quality_target") == "methodology":
                payload["critique"] = critique
                payload["revision_instruction"] = f"Address quality critique: {critique}"
            methodology = self.methodology_skill.run(payload, memory=self.memory)
            return {
                "methodology_status": "completed",
                "methodology": methodology,
            }
        except Exception as exc:
            logger.exception("Methodology generation failed: %s", exc)
            return {
                "methodology_status": "failed",
                "methodology": "",
                "errors": [str(exc)],
            }

    # =========================================================================
    # 7. State: ABSTRACT (Abstract Writing State)
    # =========================================================================
    def abstract_state(self, state: ResearchGraphState) -> Dict[str, Any]:
        """State corresponding to original abstract_status workflow (depends on test interpretation)."""
        logger.info("Entering State: ABSTRACT")
        if state.get("abstract_status") == "skipped":
            return {"abstract_status": "skipped"}
        context: ResearchContext = state.get("context", ResearchContext())
        test_interpretation = state.get("test_interpretation", "")

        try:
            payload = context.model_dump()
            payload["test_interpretation"] = test_interpretation
            if state.get("study_facts"):
                payload["study_facts"] = state.get("study_facts")
            critique = state.get("quality_critique")
            if critique and state.get("quality_target") == "abstract":
                payload["critique"] = critique
                payload["revision_instruction"] = f"Address quality critique: {critique}"
            abstract = self.abstract_skill.run(payload, memory=self.memory)
            return {
                "abstract_status": "completed",
                "abstract": abstract,
            }
        except Exception as exc:
            logger.exception("Abstract generation failed: %s", exc)
            return {
                "abstract_status": "failed",
                "abstract": "",
                "errors": [str(exc)],
            }

    # =========================================================================
    # 8. State: VISUALIZATION (Visualization Planning State)
    # =========================================================================
    def visualization_state(self, state: ResearchGraphState) -> Dict[str, Any]:
        """State corresponding to original visualization_status workflow."""
        logger.info("Entering State: VISUALIZATION")
        if state.get("visualization_status") == "skipped":
            return {"visualization_status": "skipped"}
        context: ResearchContext = state.get("context", ResearchContext())
        df = state.get("df")

        if df is None or df.empty:
            return {
                "visualization_status": "failed",
                "visualizations": [],
            }

        # Tool: plan_visualizations
        plan_tool = self.visualization_tools["plan_visualizations"]
        viz_res = plan_tool.execute(df=df, hypothesis=context.hypothesis or "")
        node_history = [{"state": "visualization", "tool": "plan_visualizations", "success": viz_res.success}]

        specs: List[VisualizationSpec] = []
        if viz_res.success and isinstance(viz_res.data, list):
            for item in viz_res.data:
                specs.append(VisualizationSpec(
                    title=item.get("title", ""),
                    mark=item.get("mark", ""),
                    x_axis=item.get("x_axis"),
                    y_axis=item.get("y_axis"),
                    color=item.get("color"),
                ))
            return {
                "visualization_status": "completed",
                "visualizations": specs,
                "tool_history": node_history,
            }
        else:
            return {
                "visualization_status": "failed",
                "visualizations": [],
                "tool_history": node_history,
            }

    # =========================================================================
    # 9. State: QUALITY_GATE (Publication Quality Gate Validation)
    # =========================================================================
    def quality_gate_state(self, state: ResearchGraphState) -> Dict[str, Any]:
        """Validates academic abstract and methodology against publication quality criteria."""
        logger.info("Entering State: QUALITY_GATE")
        abstract = state.get("abstract", "") or ""
        methodology = state.get("methodology", "") or ""
        context = state.get("context", ResearchContext())

        # 1. Methodology validation
        method_critiques: List[str] = []
        if state.get("methodology_status") == "skipped":
            method_ok = True
        else:
            if len(methodology.split()) < 20 or len(methodology) < 120:
                method_critiques.append("Methodology is too brief. Expand with comprehensive description of research design, sample selection, and statistical tests.")
            for err_kw in ["failed to generate", "model failed", "error calling model", "placeholder", "todo"]:
                if err_kw in methodology.lower():
                    method_critiques.append(f"Contains error/placeholder snippet: '{err_kw}'")
            method_ok = len(method_critiques) == 0

        # 2. Abstract validation
        abstract_critiques: List[str] = []
        if state.get("abstract_status") == "skipped":
            abstract_ok = True
        else:
            if len(abstract.split()) < 20 or len(abstract) < 120:
                abstract_critiques.append("Abstract is too brief. Provide a complete summary of background, methods, key findings, and conclusion.")
            for err_kw in ["failed to generate", "model failed", "error calling model", "placeholder", "todo"]:
                if err_kw in abstract.lower():
                    abstract_critiques.append(f"Contains error/placeholder snippet: '{err_kw}'")
            abstract_ok = len(abstract_critiques) == 0

        q_retries = state.get("quality_retry_count", 0)
        max_q_retries = state.get("max_quality_retries", 2)

        if method_ok and abstract_ok:
            status = "passed"
            critique = "All academic quality checks passed."
            target = None
        elif q_retries >= max_q_retries:
            status = "exhausted"
            critique = f"Max quality retries ({max_q_retries}) reached. Advancing to consolidation."
            target = None
        elif not method_ok:
            status = "needs_refinement"
            critique = " | ".join(method_critiques)
            target = "methodology"
        else:
            status = "needs_refinement"
            critique = " | ".join(abstract_critiques)
            target = "abstract"

        logger.info("Quality Gate: status=%s, target=%s (iteration %d/%d)", status, target, q_retries, max_q_retries)

        return {
            "abstract_quality_passed": abstract_ok,
            "methodology_quality_passed": method_ok,
            "quality_status": status,
            "quality_critique": critique,
            "quality_target": target,
        }

    def prepare_quality_refinement_state(self, state: ResearchGraphState) -> Dict[str, Any]:
        """Increments quality retry counter before routing back to section generation."""
        q_retries = state.get("quality_retry_count", 0) + 1
        logger.info("Preparing quality refinement iteration #%d for %s", q_retries, state.get("quality_target"))
        return {
            "quality_retry_count": q_retries,
        }

    # =========================================================================
    # Routing Condition Helpers
    # =========================================================================
    @staticmethod
    def route_after_tests(state: ResearchGraphState) -> str:
        """Routes to handle_lats_retry if non-significant or poor retention, else proceeds."""
        if state.get("test_status") == "skipped":
            return "proceed"
        decision = state.get("lats_route_decision", "proceed")
        if decision in ("retry_cleaning", "retry_hypothesis"):
            return decision
        return "proceed"

    @staticmethod
    def route_after_lats_retry(state: ResearchGraphState) -> str:
        """Determines target node after LATS retry configuration."""
        decision = state.get("lats_route_decision", "retry_cleaning")
        if decision == "retry_hypothesis":
            return "metadata_prediction"
        return "ingest_and_clean"

    @staticmethod
    def route_quality_gate(state: ResearchGraphState) -> str:
        """Routes from quality gate to intro_conclusion_critic or refinement preparation."""
        status = state.get("quality_status", "passed")
        if status in ("passed", "exhausted"):
            return "passed"
        return "prepare_quality_refinement"

    @staticmethod
    def route_quality_target(state: ResearchGraphState) -> str:
        """Routes to either methodology or abstract for quality refinement."""
        target = state.get("quality_target")
        if target == "methodology":
            return "methodology"
        return "abstract"

    # =========================================================================
    # 9. State: PLAYGROUND (Consolidation & Continuous Learning State)
    # =========================================================================
    def playground_state(self, state: ResearchGraphState) -> Dict[str, Any]:
        """Consolidation state corresponding to original playground_status, recording learnings."""
        logger.info("Entering State: PLAYGROUND (Final Consolidation)")
        context: ResearchContext = state.get("context", ResearchContext())
        if state.get("playground_status") == "skipped":
            return {"playground_status": "skipped"}
        test_results = state.get("test_results", {})
        test_interp = state.get("test_interpretation", "")

        # Continuous Learning: Record outcomes in MemoryStore
        if test_results and context.research_type:
            is_sig = bool(LATSEvaluator.get_significant_tests(test_results, self.config.default_significance_level))
            self.memory.record_test_outcome(
                design=context.research_type,
                iv_type=context.independent_variable_type or "numeric",
                dv_type=context.dependent_variable_type or "numeric",
                test_used=str(list(test_results.keys())),
                significant=is_sig,
            )

        return {
            "playground_status": "completed",
        }

    # =========================================================================
    # Paired Reflexion & Cross-Section Critic States
    # =========================================================================
    def methods_results_critic_state(self, state: ResearchGraphState) -> Dict[str, Any]:
        """Critic state: audits alignment between Methodology and Results."""
        logger.info("Entering State: METHODS_RESULTS_CRITIC")
        if state.get("methodology_status") == "skipped" or state.get("test_status") == "skipped":
            return {"methods_results_critique": {"passed": True, "score": 1.0, "issues": []}}

        methodology = state.get("methodology", "")
        results_text = state.get("test_interpretation", "")
        study_facts = state.get("study_facts")
        df = state.get("df")
        references = state.get("literature_review")

        eval_res = self.methods_results_critic.evaluate(
            methodology=methodology,
            results_text=results_text,
            study_facts=study_facts,
        )

        refined_methodology = methodology
        if study_facts and not eval_res.passed:
            logger.info("Methods-Results Critic flagged %d issues. Invoking PaperActorReviser.", len(eval_res.issues))
            refined_methodology = self.paper_actor_reviser.revise_section(
                section_name="methodology",
                current_text=methodology,
                issues=eval_res.issues,
                study_facts=study_facts,
                df=df,
                references=references,
            )

        return {
            "methodology": refined_methodology,
            "methods_results_critique": eval_res.model_dump(),
            "reflexion_history": [{
                "critic": "methods_results",
                "passed": eval_res.passed,
                "score": eval_res.score,
                "issues": eval_res.issues,
            }],
        }

    def discussion_state(self, state: ResearchGraphState) -> Dict[str, Any]:
        """Generates the Discussion section with physical mechanisms and effect size calibration."""
        logger.info("Entering State: DISCUSSION")
        if state.get("discussion_status") == "skipped":
            return {"discussion_status": "skipped", "discussion": ""}

        context: ResearchContext = state.get("context", ResearchContext())
        ctx_payload = {
            "name": context.name,
            "problem_statement": context.problem_statement,
            "hypothesis": context.hypothesis,
            "independent_variable": context.independent_variable,
            "dependent_variable": context.dependent_variable,
            "covariates": context.covariates,
            "test_results": state.get("test_results", {}),
            "results_text": state.get("test_interpretation", ""),
            "study_facts": state.get("study_facts"),
            "autonomous_analysis": state.get("autonomous_analysis", {}),
            "reviewed_sources": [x.model_dump() if hasattr(x,"model_dump") else x for x in state.get("literature_review",[])],
        }

        try:
            discussion_text = self.discussion_skill.run(ctx_payload)
        except Exception as exc:
            logger.warning("DiscussionSkill failed (%s), using structured fallback.", exc)
            discussion_text = ""

        if not discussion_text and state.get("study_facts"):
            try:
                from research_agent.latex_templates.manager import LaTeXPaperManager
                mgr = LaTeXPaperManager(config=self.config)
                discussion_text = mgr._synthesize_discussion(ctx_payload, state.get("test_results", {}), study_facts=state["study_facts"])
            except Exception as inner_exc:
                logger.debug("Structured discussion synthesis fallback failed: %s", inner_exc)

        if state.get("study_facts"):
            from research_agent.validation.paper_validator import PaperValidator
            validator = PaperValidator(state["study_facts"])
            discussion_text = validator.correct_section_discrepancies("Discussion", discussion_text)

        return {"discussion": discussion_text}

    def results_discussion_critic_state(self, state: ResearchGraphState) -> Dict[str, Any]:
        """Critic state: audits alignment between Results and Discussion."""
        logger.info("Entering State: RESULTS_DISCUSSION_CRITIC")
        if state.get("discussion_status") == "skipped" or state.get("test_status") == "skipped":
            return {"results_discussion_critique": {"passed": True, "score": 1.0, "issues": []}}

        results_text = state.get("test_interpretation", "")
        discussion = state.get("discussion", "")
        study_facts = state.get("study_facts")
        df = state.get("df")
        references = state.get("literature_review")

        eval_res = self.results_discussion_critic.evaluate(
            results_text=results_text,
            discussion_text=discussion,
            study_facts=study_facts,
        )

        cross_res = self.cross_section_critic.evaluate_section_pair(
            sec1_name="results",
            sec1_text=results_text,
            sec2_name="discussion",
            sec2_text=discussion,
            study_facts=study_facts,
        )

        all_issues = list(eval_res.issues)
        for iss in cross_res.issues:
            if iss not in all_issues:
                all_issues.append(iss)

        refined_discussion = discussion
        if all_issues and study_facts:
            logger.info("Results-Discussion Critic flagged %d issues. Invoking PaperActorReviser.", len(all_issues))
            refined_discussion = self.paper_actor_reviser.revise_section(
                section_name="discussion",
                current_text=discussion,
                issues=all_issues,
                study_facts=study_facts,
                df=df,
                references=references,
            )

        passed = len(all_issues) == 0
        score = min(eval_res.score, cross_res.score)
        return {
            "discussion": refined_discussion,
            "results_discussion_critique": {
                "passed": passed,
                "score": score,
                "issues": all_issues,
            },
            "reflexion_history": [{
                "critic": "results_discussion",
                "passed": passed,
                "score": score,
                "issues": all_issues,
            }],
        }

    def conclusion_state(self, state: ResearchGraphState) -> Dict[str, Any]:
        """Generates the Conclusion section answering the research hypothesis."""
        logger.info("Entering State: CONCLUSION")
        if state.get("conclusion_status") == "skipped":
            return {"conclusion_status": "skipped", "conclusion": ""}

        context: ResearchContext = state.get("context", ResearchContext())
        ctx_payload = {
            "name": context.name,
            "problem_statement": context.problem_statement,
            "hypothesis": context.hypothesis,
            "independent_variable": context.independent_variable,
            "dependent_variable": context.dependent_variable,
            "test_results": state.get("test_results", {}),
            "test_interpretation": state.get("test_interpretation", ""),
            "study_facts": state.get("study_facts"),
            "autonomous_analysis": state.get("autonomous_analysis", {}),
            "reviewed_sources": [x.model_dump() if hasattr(x,"model_dump") else x for x in state.get("literature_review",[])],
        }

        try:
            conclusion_text = self.conclusion_skill.run(ctx_payload)
        except Exception as exc:
            logger.warning("ConclusionSkill failed (%s), using structured fallback.", exc)
            conclusion_text = ""

        if not conclusion_text and state.get("study_facts"):
            try:
                from research_agent.latex_templates.manager import LaTeXPaperManager
                mgr = LaTeXPaperManager(config=self.config)
                conclusion_text = mgr._synthesize_conclusion(ctx_payload, state.get("test_results", {}), study_facts=state["study_facts"])
            except Exception as inner_exc:
                logger.debug("Structured conclusion synthesis fallback failed: %s", inner_exc)

        if state.get("study_facts"):
            from research_agent.validation.paper_validator import PaperValidator
            validator = PaperValidator(state["study_facts"])
            conclusion_text = validator.correct_section_discrepancies("Conclusion", conclusion_text)

        return {"conclusion": conclusion_text}

    def intro_conclusion_critic_state(self, state: ResearchGraphState) -> Dict[str, Any]:
        """Critic state: audits alignment between Introduction and Conclusion."""
        logger.info("Entering State: INTRO_CONCLUSION_CRITIC")
        if state.get("introduction_status") == "skipped" or state.get("conclusion_status") == "skipped":
            return {"intro_conclusion_critique": {"passed": True, "score": 1.0, "issues": []}}

        intro = state.get("introduction", "")
        conclusion = state.get("conclusion", "")
        discussion = state.get("discussion", "")
        study_facts = state.get("study_facts")
        df = state.get("df")
        references = state.get("literature_review")

        eval_res = self.intro_conclusion_critic.evaluate(
            introduction=intro,
            conclusion=conclusion,
            study_facts=study_facts,
        )

        cross_res = self.cross_section_critic.evaluate_section_pair(
            sec1_name="introduction",
            sec1_text=intro,
            sec2_name="discussion",
            sec2_text=discussion,
            study_facts=study_facts,
        )

        all_issues = list(eval_res.issues)
        for iss in cross_res.issues:
            if iss not in all_issues:
                all_issues.append(iss)

        refined_intro = intro
        refined_conclusion = conclusion
        refined_discussion = discussion

        if all_issues and study_facts:
            logger.info("Intro-Conclusion / Cross-Section Critic flagged %d issues. Invoking PaperActorReviser.", len(all_issues))
            intro_issues = [i for i in all_issues if "intro" in i.lower()]
            conc_issues = [i for i in all_issues if "conclusion" in i.lower()]
            disc_issues = [i for i in all_issues if "discussion" in i.lower()]

            if intro_issues:
                refined_intro = self.paper_actor_reviser.revise_section(
                    section_name="introduction",
                    current_text=intro,
                    issues=intro_issues,
                    study_facts=study_facts,
                    df=df,
                    references=references,
                )
            if conc_issues:
                refined_conclusion = self.paper_actor_reviser.revise_section(
                    section_name="conclusion",
                    current_text=conclusion,
                    issues=conc_issues,
                    study_facts=study_facts,
                    df=df,
                    references=references,
                )
            if disc_issues and discussion:
                refined_discussion = self.paper_actor_reviser.revise_section(
                    section_name="discussion",
                    current_text=discussion,
                    issues=disc_issues,
                    study_facts=study_facts,
                    df=df,
                    references=references,
                )

        passed = len(all_issues) == 0
        score = min(eval_res.score, cross_res.score)
        return {
            "introduction": refined_intro,
            "conclusion": refined_conclusion,
            "discussion": refined_discussion,
            "intro_conclusion_critique": {
                "passed": passed,
                "score": score,
                "issues": all_issues,
            },
            "reflexion_history": [{
                "critic": "intro_conclusion",
                "passed": passed,
                "score": score,
                "issues": all_issues,
            }],
        }

    def global_paper_critic_state(self, state: ResearchGraphState) -> Dict[str, Any]:
        """Global Critic state: executes holistic multi-point audit across all sections."""
        logger.info("Entering State: GLOBAL_PAPER_CRITIC")
        study_facts = state.get("study_facts")

        abstract = state.get("abstract", "")
        intro = state.get("introduction", "")
        methodology = state.get("methodology", "")
        results_text = state.get("test_interpretation", "")
        discussion = state.get("discussion", "")
        conclusion = state.get("conclusion", "")

        eval_res = self.global_paper_critic.evaluate(
            abstract=abstract,
            introduction=intro,
            methodology=methodology,
            results_text=results_text,
            discussion=discussion,
            conclusion=conclusion,
            study_facts=study_facts,
        )

        if study_facts:
            from research_agent.validation.paper_validator import PaperValidator
            validator = PaperValidator(study_facts)
            abstract = validator.correct_section_discrepancies("Abstract", abstract)
            intro = validator.correct_section_discrepancies("Introduction", intro)
            methodology = validator.correct_section_discrepancies("Methods", methodology)
            results_text = validator.correct_section_discrepancies("Results", results_text)
            discussion = validator.correct_section_discrepancies("Discussion", discussion)
            conclusion = validator.correct_section_discrepancies("Conclusion", conclusion)

        return {
            "abstract": abstract,
            "introduction": intro,
            "methodology": methodology,
            "test_interpretation": results_text,
            "discussion": discussion,
            "conclusion": conclusion,
            "global_paper_critique": eval_res.model_dump(),
            "reflexion_history": [{
                "critic": "global_paper",
                "passed": eval_res.passed,
                "score": eval_res.score,
                "issues": eval_res.issues,
            }],
        }

    # =========================================================================
    # 10. State: CREATE_LATEX_PAPER (Final Publication-Ready Paper Generation)
    # =========================================================================
    def create_latex_paper_state(self, state: ResearchGraphState) -> Dict[str, Any]:
        """State for transforming all generated sections and outputs into a publication-ready LaTeX paper directly from the dataset."""
        logger.info("Entering State: CREATE_LATEX_PAPER (Final Step)")
        context: ResearchContext = state.get("context", ResearchContext())
        node_history: List[Dict[str, Any]] = []

        if state.get("latex_paper_status") == "skipped":
            return {"latex_paper_status": "skipped"}

        if state.get("test_status") == "failed":
            return {"latex_paper_status": "blocked", "errors": ["Paper generation blocked because statistical testing failed."]}

        paper_tool = self.paper_tools["create_latex_paper"]
        template = state.get("paper_template") or "generic_arxiv"
        output_dir = state.get("output_dir")
        compile_pdf = state.get("compile_pdf", True)

        tool_args = {
            "template": template,
            "title": context.name or context.problem_statement,
            "abstract": state.get("abstract") or context.hypothesis,
            "introduction": state.get("introduction"),
            "literature_review": state.get("literature_review"),
            "background_text": state.get("background"),
            "methodology": state.get("methodology"),
            "results_text": state.get("test_interpretation"),
            "test_results": state.get("test_results"),
            "discussion": state.get("discussion"),
            "conclusion": state.get("conclusion"),
            "visualizations": state.get("visualizations"),
            "context": context,
            "study_facts": state.get("study_facts"),
            "output_dir": output_dir,
            "compile_pdf": compile_pdf,
            "authors": state.get("authors"),
            "keywords": state.get("keywords"),
            "review_gate_passed": bool(state.get("quality_status") == "passed" and state.get("peer_review_status") == "passed" and not state.get("errors")),
        }

        res = paper_tool.execute(**tool_args)
        node_history.append({"state": "create_latex_paper", "tool": "create_latex_paper", "success": res.success})

        if res.success and isinstance(res.data, dict):
            bundle = res.data
            bundle["publication_ready"] = bool(
                bundle.get("publication_ready") and state.get("quality_status") == "passed"
                and state.get("peer_review_status") == "passed" and not state.get("errors")
            )
            bundle["quality_status"] = state.get("quality_status", "unverified")
            bundle["peer_review_status"] = state.get("peer_review_status", "unverified")
            tex_path = bundle.get("tex_file") or bundle.get("tex_path")
            pdf_path = bundle.get("pdf_file") or bundle.get("pdf_path")
            source = bundle.get("latex_source") or bundle.get("tex_source")
            return {
                "latex_paper_status": "completed",
                "latex_paper_bundle": bundle,
                "publication_ready": bundle["publication_ready"],
                "validation_report": bundle.get("validation_report"),
                "latex_paper_path": tex_path,
                "latex_pdf_path": pdf_path,
                "latex_source": source,
                "tool_history": node_history,
            }
        else:
            return {
                "latex_paper_status": "failed",
                "errors": [f"LaTeX paper generation failed: {res.error}"],
                "tool_history": node_history,
            }

    def autonomous_analysis_state(self, state: ResearchGraphState) -> Dict[str, Any]:
        """Execute autonomous statistical analysis tools (power/MDES, variance partitioning, sensitivity)."""
        logger.info("Entering State: AUTONOMOUS_ANALYSIS")
        df = state.get("df")
        context: ResearchContext = state.get("context", ResearchContext())
        study_facts = state.get("study_facts")
        results = state.get("test_results") or {}

        autonomous_data = {}
        if state.get("test_status") != "skipped" and study_facts and self.autonomous_analysis_tool and df is not None:
            cluster_var = study_facts.cluster_variable or context.subject_group or next((col for col in ["cluster_id", "cluster", "site_id", "site_name", "site", "group_id", "subject_id", "location_id"] if col in df.columns), None)
            res = self.autonomous_analysis_tool.execute(
                df=df,
                iv=context.independent_variable,
                dv=context.dependent_variable,
                cluster_var=cluster_var,
                covars=context.covariates or [],
                analyses=["power_and_mdes", "variance_partitioning", "sensitivity_analysis", "cohort_and_covariates", "equivalence_testing"],
            )
            if res.success and isinstance(res.data, dict):
                autonomous_data = res.data

        from research_agent.provenance import investigate
        provenance_report = investigate(self.introduction_skill, context, df) if df is not None else {}
        autonomous_data["data_provenance"] = provenance_report
        if study_facts:
            from research_agent.manuscript import results_text
            interpretation = results_text(study_facts, autonomous_data)
        else:
            interpretation = state.get("test_interpretation", "")
        return {
            "test_interpretation": interpretation,
            "autonomous_analysis": autonomous_data,
            "tool_history": [{"state": "autonomous_analysis", "tool": "autonomous_analysis", "success": bool(autonomous_data)}],
        }

    def web_grounded_critic_state(self, state: ResearchGraphState) -> Dict[str, Any]:
        """Critic state: executes CitationGroundingCritic to audit literature grounding, results leakage, and mechanisms."""
        logger.info("Entering State: WEB_GROUNDED_CRITIC")
        if state.get("web_grounded_critic_status") == "skipped":
            return {"web_grounded_critique": {"passed": True, "score": 1.0, "issues": []}}

        intro = state.get("introduction", "")
        disc = state.get("discussion", "")
        lit_items = state.get("literature_review", [])
        study_facts = state.get("study_facts")
        df = state.get("df")

        eval_res = self.citation_critic.evaluate(
            introduction=intro,
            discussion=disc,
            literature_items=lit_items,
            study_facts=study_facts,
        )

        refined_intro = intro
        refined_disc = disc
        updated_lit = list(lit_items)

        if not eval_res.passed and study_facts:
            logger.info("CitationGroundingCritic flagged %d issues. Invoking PaperActorReviser.", len(eval_res.issues))
            if eval_res.target_section == "introduction" or any("introduction" in i.lower() for i in eval_res.issues):
                refined_intro = self.paper_actor_reviser.revise_section(
                    section_name="introduction",
                    current_text=intro,
                    issues=eval_res.issues,
                    study_facts=study_facts,
                    df=df,
                    references=updated_lit,
                )
            if eval_res.target_section == "discussion" or any("discussion" in i.lower() for i in eval_res.issues):
                refined_disc = self.paper_actor_reviser.revise_section(
                    section_name="discussion",
                    current_text=disc,
                    issues=eval_res.issues,
                    study_facts=study_facts,
                    df=df,
                    references=updated_lit,
                )

        return {
            "introduction": refined_intro,
            "discussion": refined_disc,
            "literature_review": updated_lit,
            "web_grounded_critique": eval_res.model_dump(),
            "reflexion_history": [{
                "critic": "citation_grounding",
                "passed": eval_res.passed,
                "score": eval_res.score,
                "issues": eval_res.issues,
            }],
        }

    # =========================================================================
    # 11. Final Peer Review & Actor Revision Loop
    # =========================================================================
    def final_peer_review_state(self, state: ResearchGraphState) -> Dict[str, Any]:
        """Fresh-context Final Peer Review Critic: audits complete manuscript against immutable StudyFacts."""
        logger.info("Entering State: FINAL_PEER_REVIEW")
        if state.get("peer_review_status") == "skipped":
            return {"peer_review_status": "skipped"}

        study_facts = state.get("study_facts")
        if not study_facts:
            return {
                "peer_review_status": "unverified",
                "peer_review_critique": {"passed": False, "score": 0.0, "issues": ["StudyFacts unavailable; manuscript cannot be verified."]},
            }

        manuscript_sections = {
            "abstract": state.get("abstract", "") or "",
            "introduction": state.get("introduction", "") or "",
            "methodology": state.get("methodology", "") or "",
            "results": state.get("test_interpretation", "") or "",
            "discussion": state.get("discussion", "") or "",
            "conclusion": state.get("conclusion", "") or "",
        }

        from research_agent.latex_templates.manager import TemplateManager
        manager = TemplateManager(config=self.config)
        payload = manager.prepare_payload(
            title=state["context"].name, authors=state.get("authors"), context=state["context"],
            literature_review=state.get("literature_review"), study_facts=study_facts, background_text=state.get("background"),
            abstract=manuscript_sections["abstract"], introduction=manuscript_sections["introduction"],
            methodology=manuscript_sections["methodology"], results_text=manuscript_sections["results"],
            discussion=manuscript_sections["discussion"], conclusion=manuscript_sections["conclusion"],
            test_results=state.get("test_results"), visualizations=state.get("visualizations"), keywords=state.get("keywords"),
        )
        rendered = manager.render_paper(state.get("paper_template") or "generic_arxiv",payload)
        # Review the actual assembled source, including all tables, Background and declarations.
        eval_res = self.final_peer_review_critic.evaluate_manuscript(
            manuscript_sections={"assembled_manuscript":rendered},
            study_facts=study_facts, references_bib=payload["bib_content"],
            source_evidence=[r.model_dump() if hasattr(r,"model_dump") else r for r in state.get("literature_review",[])],
            supplemental_audit=state.get("autonomous_analysis",{}),
        )

        retries = state.get("peer_review_retry_count", 0)
        max_retries = state.get("max_peer_review_retries", 2)

        if eval_res.passed:
            status = "passed"
            target = None
        elif eval_res.target_sections and not any(t in {"abstract","introduction","methodology","methods","results","discussion","conclusion","background"} for t in eval_res.target_sections):
            status = "requires_investigator_input"
            target = None
        elif retries >= max_retries:
            status = "exhausted"
            target = None
        else:
            status = "needs_revision"
            target = eval_res.target_section or "discussion"

        logger.info(
            "Final Peer Review: status=%s, target=%s, score=%.2f, retries=%d/%d",
            status, target, eval_res.score, retries, max_retries,
        )

        return {
            "peer_review_status": status,
            "peer_review_target": target,
            "review_coverage": getattr(self.final_peer_review_critic,"last_review_coverage",[]),
            "peer_review_critique": eval_res.model_dump(),
            "reflexion_history": [{
                "critic": "final_peer_review",
                "passed": eval_res.passed,
                "score": eval_res.score,
                "issues": eval_res.issues,
            }],
        }

    def actor_revision_state(self, state: ResearchGraphState) -> Dict[str, Any]:
        """Actor Model Revision State: natively rewrites flawed section based on Peer Review feedback."""
        logger.info("Entering State: ACTOR_REVISION")
        critique = state.get("peer_review_critique") or {}
        targets = critique.get("target_sections") or [state.get("peer_review_target") or "discussion"]
        updates = {}
        for target in targets:
            target = {"methods":"methodology"}.get(target,target)
            if target == "background":
                relevant=critique.get("section_issues",{}).get("background",[])
                if relevant:
                    updates["background"]=self.lit_review_skill.synthesize_background(state["context"].model_dump(),state.get("literature_review",[]),relevant)
                continue
            if target not in {"abstract","introduction","methodology","results","discussion","conclusion"}:
                continue
            key = "test_interpretation" if target=="results" else target
            scoped=critique.get("section_issues")
            relevant=scoped.get(target,[]) if scoped is not None else critique.get("issues",[])
            if not relevant:
                continue
            updates[key] = self.paper_actor_reviser.revise_section(
                section_name=target,current_text=state.get(key, ""),issues=relevant,
                study_facts=state.get("study_facts"),df=state.get("df"),references=state.get("literature_review"))
        updates["peer_review_retry_count"] = state.get("peer_review_retry_count",0)+1
        return updates

    @staticmethod
    def route_peer_review(state: ResearchGraphState) -> str:
        """Determines whether to proceed to LaTeX compilation or route to Actor revision."""
        if state.get("peer_review_status") == "skipped":
            return "proceed"
        status = state.get("peer_review_status", "passed")
        retries = state.get("peer_review_retry_count", 0)
        max_retries = state.get("max_peer_review_retries", 2)
        if status in {"passed", "unverified", "requires_investigator_input"} or retries >= max_retries or status == "exhausted":
            return "proceed"
        return "revise"
