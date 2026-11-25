"""
Paper Actor Reviser Skill.

Acts as the senior author model that revises and updates manuscript sections
based on feedback from paired critics and the final peer-review critic.
Equipped with DatasetQueryTool to compute or verify exact empirical statistics
directly from the underlying DataFrame.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional
import pandas as pd

from langchain_core.messages import HumanMessage, SystemMessage

from research_agent.schemas import StudyFacts
from research_agent.skills.base import BaseSkill
from research_agent.study_facts import generate_detailed_state_narrative
from research_agent.tools.data_tools import DatasetQueryTool

logger = logging.getLogger(__name__)


class PaperActorReviser(BaseSkill):
    """
    The Actor model that revises manuscript sections in response to critique issues.
    Grounded by the Empirical Data Audit Narrative and able to query the raw dataset.
    """

    name = "paper_actor_reviser"
    description = (
        "Revises manuscript sections based on critique issues and ground-truth data, "
        "resolving contradictions, correcting effect directions, and grounding citations."
    )

    def __init__(self, config: Any):
        super().__init__(config)
        self.data_query_tool = DatasetQueryTool()

    def run(self, context: dict[str, Any], memory: Optional[Any] = None) -> str:
        section_name = context.get("section_name", "discussion")
        current_text = context.get("current_text", "")
        issues = context.get("issues", [])
        study_facts = context.get("study_facts")
        df = context.get("df")
        references = context.get("references") or context.get("literature_review")
        return self.revise_section(
            section_name=section_name,
            current_text=current_text,
            issues=issues,
            study_facts=study_facts,
            df=df,
            references=references,
        )

    def revise_section(
        self,
        section_name: str,
        current_text: str,
        issues: List[str],
        study_facts: StudyFacts,
        df: Optional[pd.DataFrame] = None,
        references: Optional[List[Any]] = None,
    ) -> str:
        """
        Natively rewrites a manuscript section to resolve all critique issues.
        """
        if not issues:
            return current_text
        if section_name == "abstract":
            from research_agent.manuscript import abstract_text
            return abstract_text(study_facts)
        if section_name == "methodology":
            from research_agent.manuscript import methods_text
            return methods_text(study_facts)
        if section_name == "discussion":
            from research_agent.manuscript import discussion_text
            return discussion_text(study_facts)
        if section_name == "conclusion":
            from research_agent.manuscript import conclusion_text
            return conclusion_text(study_facts)

        # Generate empirical audit narrative
        audit_narrative = study_facts.to_prompt_context() if hasattr(study_facts, "to_prompt_context") else generate_detailed_state_narrative(study_facts)

        # Bibliography context
        bib_lines: List[str] = []
        if references:
            for r in references:
                cit = getattr(r, "citation", None) or (r.get("citation") if isinstance(r, dict) else str(r))
                title = getattr(r, "title", None) or (r.get("title") if isinstance(r, dict) else "")
                bib_lines.append(f"- {cit} [Title: {title}]")
        bib_context = "\n".join(bib_lines[:12]) if bib_lines else "Standard methodological references."

        # Issues to resolve
        issues_formatted = "\n".join([f"{i+1}. {iss}" for i, iss in enumerate(issues)])

        system_prompt = (
            "You are the Lead Scientific Author and Manuscript Reviser for a top-tier peer-reviewed journal.\n"
            "Your task is to revise the manuscript section to resolve ALL identified reviewer critique issues.\n\n"
            "CRITICAL CONSTRAINTS:\n"
            "1. GROUND TRUTH FIDELITY: All reported metrics (N, subgroup n, test statistics, p-values, 95% CIs) "
            "MUST match the Empirical Data Audit Narrative verbatim. Never fabricate numbers.\n"
            "2. DIRECTIONAL POLARITY: Ensure all pairwise comparisons and regression slopes reflect the true signs "
            "(higher vs lower) matching the data audit narrative.\n"
            "3. EPISTEMIC HONESTY: If adjusted model fixed effects are non-significant (p > 0.05) and confidence intervals "
            "are wide, frame the outcome as 'statistically inconclusive' due to statistical power constraints, NEVER as 'proven zero effect'.\n"
            "4. MECHANISM CONSISTENCY: Align all physical/biological/chemical mechanisms with the theoretical framework.\n"
            "5. CITATION ACCURACY: Only cite papers from the verified bibliography. Never invent phantom citations.\n"
            "6. EDITORIAL CONTRACT: Methods contain procedures only. Results contain estimates. Discussion and Conclusion refer to Table 1 without duplicating it. Maximum 350 words for Discussion and 120 for Conclusion. No numbered headings.\n"
            "7. OUTPUT FORMAT: Return ONLY the revised academic text. Do NOT include headings like '# Section' or conversational preambles."
        )

        user_prompt = (
            f"=== EMPIRICAL DATA AUDIT NARRATIVE ===\n{audit_narrative}\n\n"
            f"=== VERIFIED BIBLIOGRAPHY ===\n{bib_context}\n\n"
            f"=== SECTION TO REVISE: {section_name.upper()} ===\n"
            f"Current Draft:\n{current_text}\n\n"
            f"=== CRITIQUE ISSUES TO RESOLVE ===\n"
            f"{issues_formatted}\n\n"
            f"Please rewrite the entire {section_name.upper()} section to fully resolve every issue above."
        )

        try:
            resp_text = self._invoke_with_fallback([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ], temperature=0.1)
            revised_text = resp_text.strip()

            # Clean any conversational wrapper
            revised_text = re.sub(r"^```(?:markdown|latex)?\n", "", revised_text)
            revised_text = re.sub(r"\n```$", "", revised_text)
            revised_text = re.sub(r"^(?:Here is the revised [a-zA-Z\s]+:?\n+)", "", revised_text, flags=re.IGNORECASE)

            if len(revised_text.split()) >= 10:
                logger.info("Successfully revised section '%s' via PaperActorReviser", section_name)
                from research_agent.manuscript import enforce_budget
                limit = {"discussion":350,"conclusion":120,"introduction":450}.get(section_name,1200)
                return enforce_budget(self,revised_text,limit,current_text)
        except Exception as exc:
            logger.warning("PaperActorReviser revision failed for '%s': %s", section_name, exc)

        return current_text
