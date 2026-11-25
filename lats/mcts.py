"""
Monte Carlo Tree Search (MCTS) engine for Language Agent Tree Search.

Executes tree search over data cleaning strategies and hypothesis formulations,
runs statistical tests, extracts p-values, backpropagates rewards,
and selects the best possible trajectory.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
import pandas as pd

from research_agent.config import AgentConfig
from research_agent.lats.cleaner_pool import CleanerPool, CleaningResult, CleaningStrategy
from research_agent.lats.evaluator import EvaluationScore, LATSEvaluator
from research_agent.lats.hypothesis_generator import HypothesisCandidate, HypothesisGenerator
from research_agent.lats.tree import DecisionLevel, LATSNode, SearchTree, Trajectory
from research_agent.schemas import ResearchContext
from research_agent.study_designs.dispatcher import dispatch_tests

logger = logging.getLogger(__name__)


class LATSSearchEngine:
    """Core search engine implementing tree exploration across cleaning and hypotheses."""

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        cleaner_pool: Optional[List[CleaningStrategy]] = None,
        hypothesis_generator: Optional[HypothesisGenerator] = None,
        alpha: float = 0.05,
    ):
        self.config = config or AgentConfig()
        self.cleaner_pool: List[CleaningStrategy] = cleaner_pool or CleanerPool.get_all_strategies()
        self.hypothesis_gen: HypothesisGenerator = hypothesis_generator or HypothesisGenerator(self.config)
        self.alpha: float = alpha

    def search(
        self,
        raw_df: pd.DataFrame,
        context: ResearchContext,
        n_hypotheses_per_cleaner: int = 2,
    ) -> SearchTree:
        """
        Executes the LATS tree search:
          1. Level 1: Expands cleaning strategies.
          2. Level 2: Expands hypothesis candidates for each cleaned dataset.
          3. Level 3: Evaluates statistical test results and p-values.
          4. Backpropagates scores and identifies the optimal trajectory.
        """
        tree = SearchTree(root_name=f"Study:{context.name or 'Research'}")
        root = tree.root

        iv = context.independent_variable
        dv = context.dependent_variable
        covs = context.covariates or []
        design = context.research_type or "cross_sectional"
        iv_type = context.independent_variable_type or "numeric"
        dv_type = context.dependent_variable_type or "numeric"

        if not iv or not dv or raw_df.empty:
            logger.warning("Insufficient variables or empty dataset for LATS tree search.")
            return tree

        logger.info(
            "Starting LATS search across %d cleaning options for IV='%s', DV='%s'",
            len(self.cleaner_pool), iv, dv
        )

        # Generate candidate hypothesis formulations once for the study context
        # to avoid redundant LLM round-trips across cleaner branches
        hypo_context = context.model_dump()
        hypo_context["sample_size"] = len(raw_df)
        # Wording changes do not specify a different statistical estimand. Run the
        # prespecified association once per cleaning sensitivity, without inventing
        # directional/interaction claims from identical model fits.
        base_hypotheses = [HypothesisCandidate(
            hypothesis=context.hypothesis or f"Association between {iv} and {dv}.",
            angle="prespecified_association",
            rationale="The same analysis specification is held fixed across cleaning sensitivities.",
        )]

        # -----------------------------------------------------------------
        # Level 1: Cleaning Options Expansion
        # -----------------------------------------------------------------
        for cleaner in self.cleaner_pool:
            clean_res: CleaningResult = cleaner.clean(
                df=raw_df,
                dependent_variable=dv,
                independent_variables=[iv],
                covariates=covs,
            )

            if clean_res.cleaned_df.empty:
                logger.warning("Cleaner '%s' yielded empty dataframe, skipping branch.", cleaner.name)
                continue

            cleaning_node = LATSNode(
                level=DecisionLevel.CLEANING,
                name=f"Cleaning:{cleaner.name}",
                action=cleaner,
                state_data={
                    "clean_result": clean_res,
                    "cleaned_df": clean_res.cleaned_df,
                    "retention_ratio": clean_res.retention_ratio,
                    "final_n": clean_res.final_n,
                    "initial_n": clean_res.initial_n,
                },
            )
            root.add_child(cleaning_node)

            # -----------------------------------------------------------------
            # Level 2: Hypothesis Candidates Expansion
            # -----------------------------------------------------------------
            for hypo_cand in base_hypotheses:
                hypo_node = LATSNode(
                    level=DecisionLevel.HYPOTHESIS,
                    name=f"Hypothesis:{hypo_cand.angle}",
                    action=hypo_cand,
                    state_data={"hypothesis_candidate": hypo_cand},
                )
                cleaning_node.add_child(hypo_node)

                # -----------------------------------------------------------------
                # Level 3: Execution & Environment Feedback (Statistical Tests)
                # -----------------------------------------------------------------
                test_kwargs = {}
                if context.time_variable:
                    test_kwargs["time"] = context.time_variable
                if context.subject_group:
                    test_kwargs["subject_id"] = context.subject_group

                test_results = dispatch_tests(
                    df=clean_res.cleaned_df,
                    research_design=design,
                    independent_variable_type=iv_type,
                    dependent_variable_type=dv_type,
                    independent_variable=iv,
                    dependent_variable=dv,
                    covariates=covs,
                    **test_kwargs,
                )

                # Evaluate p-values and compute composite score
                eval_score: EvaluationScore = LATSEvaluator.evaluate(
                    test_results=test_results,
                    initial_n=clean_res.initial_n,
                    final_n=clean_res.final_n,
                    alpha=self.alpha,
                    cleaning_strategy=cleaner.name,
                    hypothesis=hypo_cand.hypothesis,
                )

                # Create trajectory and evaluation leaf
                eval_node = LATSNode(
                    level=DecisionLevel.EVALUATION,
                    name=f"Eval:min_p={eval_score.min_p_value:.4g}",
                    action=eval_score,
                    state_data={"evaluation": eval_score},
                )
                eval_node.reward = eval_score.composite_score
                eval_node.reflection = eval_score.reflection
                eval_node.is_terminal = True
                hypo_node.add_child(eval_node)

                traj = Trajectory(
                    cleaning_strategy=cleaner.name,
                    cleaning_params={"description": cleaner.description},
                    initial_sample_size=clean_res.initial_n,
                    retained_sample_size=clean_res.final_n,
                    retention_ratio=clean_res.retention_ratio,
                    hypothesis=hypo_cand.hypothesis,
                    hypothesis_angle=hypo_cand.angle,
                    test_results=test_results,
                    test_names=[k for k in test_results.keys() if k != "error" and isinstance(test_results[k], dict)],
                    min_p_value=eval_score.min_p_value,
                    significant_tests=eval_score.significant_tests,
                    score=eval_score.composite_score,
                    reflection=eval_score.reflection,
                    cleaned_df=clean_res.cleaned_df,
                )
                tree.register_trajectory(traj)

                # -----------------------------------------------------------------
                # Level 4: Backpropagation
                # -----------------------------------------------------------------
                eval_node.backpropagate(eval_score.composite_score)

        # Apply family-wise Bonferroni and Benjamini-Hochberg FDR multiplicity corrections
        LATSEvaluator.apply_multiplicity_correction(tree.trajectories, alpha=self.alpha)

        return tree
