"""
Test interpretation skill.

Analyzes statistical test output and generates a clear, concise
natural-language interpretation explaining the significance and implications.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from langchain_core.messages import HumanMessage

from research_agent.prompts.templates import PromptTemplates
from research_agent.skills.base import BaseSkill

logger = logging.getLogger(__name__)


class TestInterpretationSkill(BaseSkill):
    """Interprets the results of statistical tests performed for a research study."""

    name = "test_interpretation"
    description = (
        "Analyzes statistical test results (p-values, statistics), assesses significance, "
        "and produces a concise natural-language interpretation of the findings."
    )

    def run(self, context: dict[str, Any], memory: Optional[Any] = None) -> str:
        results = context.get("results") or context.get("test_results") or {}
        study_facts = context.get("study_facts")
        if study_facts:
            from research_agent.manuscript import results_text
            return results_text(study_facts)
        prompt_text = PromptTemplates.build_test_interpretation_prompt(
            results=results,
            research_type=context.get("research_type"),
            indep_var=context.get("independent_variable"),
            dep_var=context.get("dependent_variable"),
            dataset_preview=context.get("dataset_preview"),
            study_facts=study_facts,
        )

        try:
            interpretation = self._invoke_with_fallback(
                [HumanMessage(content=prompt_text)],
                temperature=0.3,
            )
            cleaned = interpretation.strip()

            if memory and hasattr(memory, "record_test_outcome"):
                from research_agent.lats.evaluator import LATSEvaluator
                is_sig = bool(LATSEvaluator.get_significant_tests(results, self.config.default_significance_level))
                memory.record_test_outcome(
                    design=str(context.get("research_type", "")),
                    iv_type=str(context.get("independent_variable_type", "")),
                    dv_type=str(context.get("dependent_variable_type", "")),
                    test_used=str(list(results.keys()) if isinstance(results, dict) else ""),
                    significant=is_sig,
                )

            return cleaned
        except Exception as exc:
            logger.warning("LLM test interpretation failed (%s), constructing empirical synthesis.", exc)
            if study_facts:
                try:
                    from research_agent.latex_templates.manager import LaTeXPaperManager
                    mgr = LaTeXPaperManager(config=self.config)
                    return mgr._synthesize_results_narrative(context, results, study_facts=study_facts)
                except Exception as inner_exc:
                    logger.debug("LaTeXPaperManager results synthesis failed: %s", inner_exc)

            iv = context.get("independent_variable", "independent variable")
            dv = context.get("dependent_variable", "dependent variable")

            import math
            from research_agent.lats.evaluator import LATSEvaluator
            p_pairs = LATSEvaluator.extract_p_values(results)
            p_vals = [p for _, p in p_pairs if not math.isnan(p)]
            min_p = min(p_vals) if p_vals else 1.0

            if min_p < 0.05:
                sig_text = f"demonstrated statistically significant variations (minimum p = {min_p:.4f} < 0.05)"
            else:
                sig_text = f"did not reach statistical significance at alpha = 0.05 (minimum p = {min_p:.4f})"

            test_names = [k for k in results.keys() if k != "error"] if isinstance(results, dict) else ["inferential tests"]
            test_str = ", ".join(test_names)

            return (
                f"Statistical evaluation conducted using {test_str} {sig_text} across {iv} relative to {dv}. "
                f"These quantitative outcomes provide empirical evidence directly addressing the investigated study hypothesis."
            )
