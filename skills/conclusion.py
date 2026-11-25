"""
Conclusion section generation skill.

Generates a definitive, publication-ready academic Conclusion section
synthesizing the empirical findings, practical methodological implications,
and future research avenues.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

from langchain_core.messages import HumanMessage

from research_agent.prompts.templates import PromptTemplates
from research_agent.skills.base import BaseSkill

logger = logging.getLogger(__name__)


class ConclusionSkill(BaseSkill):
    """Generates an academic Conclusion section for a research study."""

    name = "conclusion"
    description = (
        "Writes an academic Conclusion section providing a definitive empirical answer "
        "to the research question, outlining practical implications, and defining "
        "actionable avenues for future empirical research."
    )

    def run(self, context: dict[str, Any], memory: Optional[Any] = None) -> str:
        study_facts = context.get("study_facts")
        if study_facts:
            from research_agent.manuscript import conclusion_text
            return conclusion_text(study_facts)
        critique = context.get("critique")

        prompt_text = PromptTemplates.build_conclusion_prompt(
            name=context.get("name"),
            hypothesis=context.get("hypothesis"),
            indep_var=context.get("independent_variable"),
            dep_var=context.get("dependent_variable"),
            test_interpretation=context.get("test_interpretation"),
            study_facts=study_facts,
        )

        if critique:
            prompt_text += (
                f"\n\nCRITICAL REFLEXION / REVISION INSTRUCTION:\n"
                f"You previously generated a draft that received this specific critique:\n{critique}\n"
                f"Address every point of this critique directly. Do not repeat previous errors."
            )

        import json
        prompt_text += "\nEDITORIAL CONTRACT: Maximum 120 words. Interpret findings without repeating coefficient tables, test statistics, or methods paragraphs. Cite only the supplied evidence-backed bibliography. Do not invent mechanisms or claim novelty from missing search results. Refer to Table 1 for estimates. No numbered headings."
        prompt_text += "\nReviewed source evidence: " + json.dumps(context.get("reviewed_sources",[]),default=str)
        prompt_text += "\nAdditional executed analyses: " + json.dumps(context.get("autonomous_analysis",{}),default=str)
        try:
            response_text = self._invoke_with_fallback(
                [HumanMessage(content=prompt_text)],
                temperature=0.4,
            )
            cleaned = self._clean_prose(response_text)

            if memory and hasattr(memory, "record_successful_prompt"):
                memory.record_successful_prompt(
                    skill_name=self.name,
                    context_summary=f"conclusion for {context.get('name')}",
                    result_quality=1.0,
                )
            from research_agent.manuscript import enforce_budget
            return enforce_budget(self,cleaned,120,"The adjusted estimates and their confidence intervals are reported in Table 1. They describe conditional associations in this extract. Unmeasured environmental controls and uncertain upstream selection limit interpretation; non-significance establishes neither zero effect nor equivalence.")

        except Exception as exc:
            logger.exception("Conclusion generation failed: %s", exc)
            if study_facts:
                try:
                    from research_agent.latex_templates.manager import LaTeXPaperManager
                    mgr = LaTeXPaperManager(config=self.config)
                    return mgr._synthesize_conclusion(context, {}, study_facts=study_facts)
                except Exception as inner:
                    logger.debug("Structured conclusion fallback failed: %s", inner)
            return ""

    @staticmethod
    def _clean_prose(text: str) -> str:
        """Strip conversational AI preambles, postambles, and markdown headings."""
        if not text:
            return ""
        cleaned = text.strip()
        cleaned = re.sub(r"^(?:#+\s*|\*\*\s*|\b(?:\d+\.?\s*)?)Conclusion(?:\*\*)?[:\s-]*\n*", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"^(?:Here is(?: a| the)?[\w\s-]+?Conclusion[\w\s-]*?[:\.\n]+)", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"(?:Note:[\s\S]*?(?:study facts|provided data|ground truth)[\s\S]*?$)", "", cleaned, flags=re.IGNORECASE).strip()
        return cleaned
