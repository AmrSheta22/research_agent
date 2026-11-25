"""
Discussion section generation skill.

Generates an authoritative, publication-ready academic Discussion section
grounded in empirical findings, substantive domain mechanisms,
literature comparison, and explicit epistemic calibration.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

from langchain_core.messages import HumanMessage

from research_agent.prompts.templates import PromptTemplates
from research_agent.skills.base import BaseSkill

logger = logging.getLogger(__name__)


class DiscussionSkill(BaseSkill):
    """Generates an academic Discussion section for a research study."""

    name = "discussion"
    description = (
        "Writes an in-depth academic Discussion section comparing empirical findings "
        "with published literature, explaining substantive domain mechanisms, and "
        "evaluating methodological limitations without inflating effect sizes."
    )

    def run(self, context: dict[str, Any], memory: Optional[Any] = None) -> str:
        study_facts = context.get("study_facts")
        if study_facts:
            from research_agent.manuscript import discussion_text
            return discussion_text(study_facts, context.get("autonomous_analysis"))
        results = context.get("test_results") or {}
        critique = context.get("critique")

        prompt_text = PromptTemplates.build_discussion_prompt(
            name=context.get("name"),
            problem_statement=context.get("problem_statement"),
            hypothesis=context.get("hypothesis"),
            indep_var=context.get("independent_variable"),
            dep_var=context.get("dependent_variable"),
            covars=context.get("covariates"),
            results=results,
            study_facts=study_facts,
        )

        if critique:
            prompt_text += (
                f"\n\nCRITICAL REFLEXION / REVISION INSTRUCTION:\n"
                f"You previously generated a draft that received this specific critique:\n{critique}\n"
                f"Address every point of this critique directly. Do not repeat previous errors."
            )

        import json
        prompt_text += "\nEDITORIAL CONTRACT: Maximum 350 words. Interpret findings without repeating coefficient tables, test statistics, or methods paragraphs. Cite only the supplied evidence-backed bibliography. Do not invent mechanisms or claim novelty from missing search results. Refer to Table 1 for estimates. No numbered headings."
        prompt_text += "\nReviewed source evidence: " + json.dumps(context.get("reviewed_sources",[]),default=str)
        prompt_text += "\nAdditional executed analyses: " + json.dumps(context.get("autonomous_analysis",{}),default=str)
        try:
            response_text = self._invoke_with_fallback(
                [HumanMessage(content=prompt_text)],
                temperature=0.4,
            )
            # Clean heading & conversational leaks
            cleaned = self._clean_prose(response_text)

            if memory and hasattr(memory, "record_successful_prompt"):
                memory.record_successful_prompt(
                    skill_name=self.name,
                    context_summary=f"discussion for {context.get('name')}",
                    result_quality=1.0,
                )
            from research_agent.manuscript import enforce_budget
            return enforce_budget(self,cleaned,350,"The adjusted estimates and their confidence intervals are reported in Table 1. They describe conditional associations in this extract. Unmeasured environmental controls and uncertain upstream selection limit interpretation; non-significance establishes neither zero effect nor equivalence.")

        except Exception as exc:
            logger.exception("Discussion generation failed: %s", exc)
            if study_facts:
                try:
                    from research_agent.latex_templates.manager import LaTeXPaperManager
                    mgr = LaTeXPaperManager(config=self.config)
                    return mgr._synthesize_discussion(context, results, study_facts=study_facts)
                except Exception as inner:
                    logger.debug("Structured discussion fallback failed: %s", inner)
            return ""

    @staticmethod
    def _clean_prose(text: str) -> str:
        """Strip conversational AI preambles, postambles, and markdown headings."""
        if not text:
            return ""
        cleaned = text.strip()
        # Remove markdown heading
        cleaned = re.sub(r"^(?:#+\s*|\*\*\s*|\b(?:\d+\.?\s*)?)Discussion(?:\*\*)?[:\s-]*\n*", "", cleaned, flags=re.IGNORECASE).strip()
        # Remove conversational leaks
        cleaned = re.sub(r"^(?:Here is(?: a| the)?[\w\s-]+?Discussion[\w\s-]*?[:\.\n]+)", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"(?:Note:[\s\S]*?(?:study facts|provided data|ground truth)[\s\S]*?$)", "", cleaned, flags=re.IGNORECASE).strip()
        return cleaned
