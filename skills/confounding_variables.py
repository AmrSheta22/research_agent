"""
Confounding variables discovery skill.

Uses web search and LLM reasoning to identify potential confounding variables
and covariates that could influence the relationship between study variables.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from langchain_core.messages import HumanMessage

from research_agent.prompts.templates import PromptTemplates
from research_agent.skills.base import BaseSkill
from research_agent.tools.search_tools import WebSearchTool

logger = logging.getLogger(__name__)


class ConfoundingVariablesSkill(BaseSkill):
    """Identifies potential confounding variables for a research study."""

    name = "confounding_variables"
    description = (
        "Discovers and recommends potential confounding variables and covariates "
        "that may confound the relationship between the independent and dependent variables."
    )

    def run(self, context: dict[str, Any], memory: Optional[Any] = None) -> list[str]:
        indep_var = context.get("independent_variable") or context.get("independent_variables")
        dep_var = context.get("dependent_variable")

        # 1. Perform targeted search to enrich context
        search_tool = WebSearchTool()
        search_query = f"confounding variables {indep_var} {dep_var} {context.get('research_type', '')}"
        search_result = search_tool.execute(query=search_query)
        evidence = search_result.data if search_result.success else ""

        # 2. Build prompt
        prompt_text = PromptTemplates.build_confounding_prompt(
            dependent_variable=dep_var,
            dep_type=context.get("dependent_variable_type"),
            independent_variables=indep_var,
            independent_type=context.get("independent_variable_type"),
            research_type=context.get("research_type"),
        )
        if evidence:
            prompt_text += f"\n\nSearch Context:\n{evidence}"

        try:
            response_text = self._invoke_with_fallback(
                [HumanMessage(content=prompt_text)],
                temperature=0.3,
            )
            raw = response_text.strip()
            if raw.lower() == "none" or not raw:
                return []

            # Parse comma-separated list
            confounders = [c.strip() for c in raw.split(",") if c.strip() and c.strip().lower() != "none"]

            if memory and hasattr(memory, "save_learning"):
                memory.save_learning("confounding_patterns", f"{indep_var}->{dep_var}", confounders)

            return confounders
        except Exception as exc:
            logger.exception("Failed to discover confounding variables: %s", exc)
            return []
