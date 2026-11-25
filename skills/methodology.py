"""
Methodology generation skill.

Formulates a structured scientific methodology section detailing study design,
setting, variables, data collection, and statistical analysis.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from langchain_core.messages import HumanMessage

from research_agent.prompts.templates import PromptTemplates
from research_agent.skills.base import BaseSkill

logger = logging.getLogger(__name__)


class MethodologySkill(BaseSkill):
    """Generates a clear and structured methodology section."""

    name = "methodology"
    description = (
        "Formulates a clear and structured methodology section for a scientific study, "
        "covering study design, setting, variables, data collection, and statistical analysis."
    )

    def run(self, context: dict[str, Any], memory: Optional[Any] = None) -> str:
        study_facts = context.get("study_facts")
        if isinstance(study_facts, dict):
            try:
                from research_agent.schemas import StudyFacts
                study_facts = StudyFacts(**study_facts)
            except Exception:
                pass

        if study_facts:
            from research_agent.manuscript import methods_text
            return methods_text(study_facts, context.get("autonomous_analysis"))

        prompt_text = PromptTemplates.build_methodology_prompt(
            research_type=context.get("research_type"),
            sample_size=study_facts.sample_size_total if study_facts else context.get("sample_size"),
            participants=context.get("subjects_or_participants"),
            inclusion_criteria=context.get("inclusion_criteria"),
            collection_location=context.get("collection_location"),
            collection_time=context.get("data_collection_period"),
            indep_var=study_facts.factor_name if study_facts else context.get("independent_variable"),
            dep_var=study_facts.dependent_variable if study_facts else context.get("dependent_variable"),
            covar=study_facts.covariates if study_facts else context.get("covariates"),
            indep_summary=context.get("independent_variable_summary"),
            dep_summary=context.get("dependent_variable_summary"),
            statistical_analysis=context.get("statistical_analysis") or context.get("dataset_summary"),
            signif_level=context.get("signif_level", 0.05),
            software=context.get("software", "Python 3.10 and Statistical Analysis Library"),
            test_names=context.get("test_names"),
            study_facts=study_facts,
        )

        if context.get("critique"):
            prompt_text += "\nRevision requirements: " + str(context["critique"])

        try:
            response_text = self._invoke_with_fallback(
                [HumanMessage(content=prompt_text)],
                temperature=0.4,
            )
            # Validate LLM output:
            # 1. Must have at least 180 words and 3+ section headers
            # 2. Math blocks \[ ... \] and braces must be fully balanced
            # 3. Must not end abruptly mid-formula or mid-sentence
            # 4. Must contain diagnostic / assumption discussion
            math_balanced = response_text.count(r"\[") == response_text.count(r"\]")
            braces_balanced = response_text.count("{") == response_text.count("}")
            has_diagnostics = any(k in response_text.lower() for k in ["assumption", "diagnostic", "collinearity", "vif", "levene"])
            not_truncated = not response_text.rstrip().endswith(("+", "=", ",", "\\", r"\text{", r"\hat{", "_", "{"))

            if (
                len(response_text.split()) >= 180
                and response_text.count("###") >= 3
                and math_balanced
                and braces_balanced
                and has_diagnostics
                and not_truncated
            ):
                return response_text.strip()

            parsed_json = self._extract_json(response_text)
            if isinstance(parsed_json, dict) and len(parsed_json) >= 3:
                parts = []
                for k, v in parsed_json.items():
                    if v and str(v).strip():
                        parts.append(f"### {k}\n{str(v).strip()}")
                if parts:
                    candidate = "\n\n".join(parts)
                    if (
                        candidate.count(r"\[") == candidate.count(r"\]")
                        and candidate.count("{") == candidate.count("}")
                        and not candidate.rstrip().endswith(("+", "=", ",", "\\", "{"))
                    ):
                        return candidate

            prose = self._format_json_as_prose(parsed_json)
            if (
                len(prose.split()) >= 120
                and prose.count(r"\[") == prose.count(r"\]")
                and prose.count("{") == prose.count("}")
            ):
                return prose
        except Exception as exc:
            logger.warning("LLM methodology generation failed (%s), constructing structured academic synthesis.", exc)

        from research_agent.reporting import methodology_narrative
        return methodology_narrative(context, context.get("results") or {}, study_facts)
