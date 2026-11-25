"""
Abstract generation skill.

Generates a structured research abstract (Problem Statement, Methods,
Results, Conclusion) and converts it to coherent prose.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

from langchain_core.messages import HumanMessage

from research_agent.prompts.templates import PromptTemplates
from research_agent.skills.base import BaseSkill

logger = logging.getLogger(__name__)


class AbstractSkill(BaseSkill):
    """Generates a comprehensive scientific abstract for a research study."""

    name = "abstract"
    description = (
        "Writes a comprehensive research abstract divided into Problem Statement, "
        "Methods, Results, and Conclusion sections, formatted as coherent academic text."
    )

    def run(self, context: dict[str, Any], memory: Optional[Any] = None) -> str:
        study_facts = context.get("study_facts")
        if study_facts:
            from research_agent.manuscript import abstract_text
            return abstract_text(study_facts)
        prompt_text = PromptTemplates.build_abstract_prompt(
            hypothesis=context.get("hypothesis"),
            test_interpretation=context.get("test_interpretation"),
            sample_size=context.get("sample_size"),
            research_type=context.get("research_type"),
            indep_var=context.get("independent_variable"),
            dep_var=context.get("dependent_variable"),
            indep_summary=context.get("independent_variable_summary"),
            dep_summary=context.get("dependent_variable_summary"),
            problem_statement=context.get("problem_statement"),
            participants=context.get("subjects_or_participants"),
            inclusion_criteria=context.get("inclusion_criteria"),
            study_facts=study_facts,
        )

        if context.get("critique"):
            prompt_text += "\nRevision requirements: " + str(context["critique"])

        try:
            response_text = self._invoke_with_fallback(
                [HumanMessage(content=prompt_text)],
                temperature=0.3,
            )
            parsed_json = self._extract_json(response_text)
            if isinstance(parsed_json, dict) and all(k in parsed_json for k in ["Problem Statement", "Methods", "Results", "Conclusion"]):
                # Clean any stray subheadings or leaked markdown from json values
                clean_parts = []
                for k in ["Problem Statement", "Methods", "Results", "Conclusion"]:
                    val = str(parsed_json.get(k, "")).strip()
                    val = re.sub(r"^\*{1,3}[^*]+\*{1,3}\s*", "", val)
                    val = re.sub(r"^(?:Problem Statement|Methods|Results|Conclusion)[:\s-]*", "", val, flags=re.IGNORECASE)
                    if val:
                        clean_parts.append(val)
                prose = " ".join(clean_parts).strip()
                words = prose.split()
                if len(words) >= 30 and "Empirical Results and Statistical Analysis" not in prose:
                    return prose

            prose = self._format_json_as_prose(parsed_json)
            words = prose.split()
            if len(words) >= 30 and "Empirical Results and Statistical Analysis" not in prose:
                return prose
        except Exception as exc:
            logger.warning("LLM abstract generation failed (%s), constructing structured academic synthesis.", exc)

        from research_agent.reporting import results_narrative
        design = context.get("research_type") or "unspecified"
        summary = results_narrative(context, context.get("results") or {}, study_facts)
        return (
            f"This study examines {context.get('independent_variable') or getattr(study_facts, 'factor_name', 'the supplied predictor')} "
            f"in relation to {context.get('dependent_variable') or getattr(study_facts, 'dependent_variable', 'the supplied outcome')} "
            f"using the supplied {design} design. {summary} "
            "Interpretation is limited to the recorded analyses and requires consideration of sampling, model assumptions, and uncertainty."
        )
