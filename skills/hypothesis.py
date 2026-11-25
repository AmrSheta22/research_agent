"""
Hypothesis generation skill.

Formulates a specific, measurable, testable research hypothesis
from the research context and variable summaries.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from langchain_core.messages import HumanMessage

from research_agent.prompts.templates import PromptTemplates
from research_agent.skills.base import BaseSkill

logger = logging.getLogger(__name__)


class HypothesisSkill(BaseSkill):
    """Generates a research hypothesis based on study context and variables."""

    name = "hypothesis"
    description = (
        "Formulates a specific, measurable, relevant, and testable research hypothesis "
        "given the research type, variables, problem statement, and data summaries."
    )

    def run(self, context: dict[str, Any], memory: Optional[Any] = None) -> str:
        prompt_text = PromptTemplates.build_hypothesis_prompt(
            name=context.get("name"),
            problem_statement=context.get("problem_statement"),
            research_type=context.get("research_type"),
            indep_var=context.get("independent_variable"),
            dep_var=context.get("dependent_variable"),
            covar=context.get("covariates"),
            independent_var_type=context.get("independent_variable_type"),
            dependent_var_type=context.get("dependent_variable_type"),
            indep_summary=context.get("independent_variable_summary"),
            dep_summary=context.get("dependent_variable_summary"),
            hypothesis_draft=context.get("hypothesis"),
        )

        try:
            result = self._invoke_with_fallback(
                [HumanMessage(content=prompt_text)],
                temperature=0.7,
            )
            hypothesis = result.strip().strip('"')

            if memory and hasattr(memory, "record_successful_prompt"):
                memory.record_successful_prompt(
                    skill_name=self.name,
                    context_summary=f"vars: {context.get('independent_variable')} -> {context.get('dependent_variable')}",
                    result_quality=1.0,
                )

            return hypothesis
        except Exception as exc:
            logger.exception("Failed to generate hypothesis: %s", exc)
            return "Error predicting hypothesis."
