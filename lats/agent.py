"""
LATSAgent - Language Agent Tree Search orchestrator for automated scientific optimization.

Tests different data cleaning options and hypothesis candidates, evaluates the
extracted p-values from statistical tests, and autonomously takes the best route.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import pandas as pd

from research_agent.config import AgentConfig
from research_agent.lats.mcts import LATSSearchEngine
from research_agent.lats.tree import SearchTree, Trajectory
from research_agent.memory.store import MemoryStore
from research_agent.schemas import ResearchContext
from research_agent.tools.data_tools import ReadFileTool

logger = logging.getLogger(__name__)


@dataclass
class LATSResult:
    """The outcome of a Language Agent Tree Search run."""
    best_trajectory: Trajectory
    all_trajectories: List[Trajectory]
    optimized_context: ResearchContext
    optimal_cleaned_df: pd.DataFrame
    audit_report: str
    search_tree: SearchTree


class LATSAgent:
    """
    Autonomous Language Agent Tree Search agent.

    Explores a combinatorial search space of:
      • Data Cleaning Options (Standard IQR, Conservative IQR, No Outliers, Winsorization, Z-Score)
      • Scientific Hypothesis Angles (Directional, Adjusted, Contrast, Interaction)
    Evaluates real environment feedback (statistical test p-values and sample retention),
    and selects the optimal route.
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        memory: Optional[MemoryStore] = None,
        alpha: float = 0.05,
    ):
        self.config = config or AgentConfig()
        self.memory = memory or MemoryStore(
            backend=self.config.memory_backend,
            path=self.config.memory_path,
        )
        self.alpha: float = alpha
        self.search_engine = LATSSearchEngine(config=self.config, alpha=self.alpha)
        self.read_file_tool = ReadFileTool()

    def optimize_route(
        self,
        context: ResearchContext,
        df: Optional[pd.DataFrame] = None,
        n_hypotheses_per_cleaner: int = 2,
    ) -> LATSResult:
        """
        Executes tree search across cleaning strategies and hypothesis variations,
        extracts p-values, and takes the best possible route.
        """
        raw_df = df
        if raw_df is None and context.file_path and os.path.exists(context.file_path):
            read_res = self.read_file_tool.execute(file_path=context.file_path)
            if read_res.success:
                raw_df = read_res.data

        if raw_df is None or raw_df.empty:
            raise ValueError("No valid dataset available to perform LATS optimization.")

        logger.info("Starting LATS search for '%s'...", context.name or "Untitled Study")

        # 1. Execute Tree Search
        tree = self.search_engine.search(
            raw_df=raw_df,
            context=context,
            n_hypotheses_per_cleaner=n_hypotheses_per_cleaner,
        )

        best_traj = tree.get_best_trajectory()
        if best_traj is None:
            raise RuntimeError("LATS search yielded no valid trajectories.")

        ranked = tree.get_ranked_trajectories()

        # 2. Continuous Learning: Record winning route
        self.memory.save_learning("lats_optimizations", f"{context.name or 'study'}", {
            "best_cleaning": best_traj.cleaning_strategy,
            "best_hypothesis": best_traj.hypothesis,
            "min_p_value": best_traj.min_p_value,
            "retention_ratio": best_traj.retention_ratio,
            "score": best_traj.score,
        })

        # 3. Update context with optimal selections
        optimized_context = context.model_copy(deep=True)
        optimized_context.hypothesis = best_traj.hypothesis
        optimized_context.sample_size = best_traj.retained_sample_size

        # 4. Generate Comprehensive Audit Report
        bonf_status = (
            "✅ Survives Bonferroni adjustment"
            if best_traj.is_significant_after_bonferroni
            else ("⚠️ Fails Bonferroni (nominal only)" if best_traj.min_p_value < self.alpha else "❌ Not significant")
        )
        fdr_status = (
            "✅ Survives Benjamini-Hochberg FDR control"
            if best_traj.is_significant_after_fdr
            else ("⚠️ Fails FDR control" if best_traj.min_p_value < self.alpha else "❌ Not significant")
        )

        report_lines = [
            f"# LATS Optimization Report for {context.name or 'Research Study'}",
            "",
            "## Selected Best Route",
            f"- **Cleaning Strategy**: `{best_traj.cleaning_strategy}`",
            f"- **Sample Retention**: {best_traj.retained_sample_size}/{best_traj.initial_sample_size} rows ({best_traj.retention_ratio*100:.1f}%)",
            f"- **Optimal Hypothesis**: *\"{best_traj.hypothesis}\"* ({best_traj.hypothesis_angle})",
            f"- **Minimum p-value Achieved**: `{best_traj.min_p_value:.4g}`",
            f"- **Significant Tests**: {', '.join(best_traj.significant_tests) if best_traj.significant_tests else 'None'}",
            f"- **Composite Value Score**: `{best_traj.score:.4f}`",
            f"- **Multiplicity & Forking Paths Control** ({len(ranked)} routes explored):",
            f"  - Family-wise Bonferroni Threshold (α/K): `{best_traj.bonferroni_alpha:.4g}` -> {bonf_status}",
            f"  - Benjamini-Hochberg FDR (q={best_traj.fdr_adjusted_min_p:.4g}): {fdr_status}",
            f"- **Rationale**: {best_traj.reflection}",
            "",
            "## Trajectory Comparison Matrix",
            "| Rank | Cleaning Option | N (Retained) | Min p-value | FDR q-value | Sig (Bonf) | Score | Hypothesis Angle |",
            "|---|---|---|---|---|---|---|---|",
        ]

        for rank, t in enumerate(ranked, 1):
            p_str = f"{t.min_p_value:.4g}" if t.min_p_value < 1.0 else ">=1.0"
            q_str = f"{t.fdr_adjusted_min_p:.4g}" if t.fdr_adjusted_min_p < 1.0 else "1.0"
            bonf_mark = "Yes" if t.is_significant_after_bonferroni else "No"
            report_lines.append(
                f"| #{rank} | `{t.cleaning_strategy}` | {t.retained_sample_size} ({t.retention_ratio*100:.0f}%) | "
                f"`{p_str}` | `{q_str}` | {bonf_mark} | **{t.score:.3f}** | {t.hypothesis_angle} |"
            )

        audit_report = "\n".join(report_lines)

        return LATSResult(
            best_trajectory=best_traj,
            all_trajectories=ranked,
            optimized_context=optimized_context,
            optimal_cleaned_df=best_traj.cleaned_df if best_traj.cleaned_df is not None else raw_df,
            audit_report=audit_report,
            search_tree=tree,
        )
