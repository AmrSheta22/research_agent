"""
Diverse hypothesis candidate generator for LATS search.

Generates multiple distinct hypothesis formulations (angles) for a given
cleaned dataset and research problem statement.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage

from research_agent.config import AgentConfig
from research_agent.skills.base import BaseSkill

logger = logging.getLogger(__name__)


@dataclass
class HypothesisCandidate:
    hypothesis: str
    angle: str           # directional | adjusted | categorical_contrast | interaction
    rationale: str


class HypothesisGenerator(BaseSkill):
    """Generates a diverse set of testable hypothesis candidates."""

    name = "hypothesis_generator"
    description = "Generates diverse, plausible scientific hypothesis candidates for tree exploration."

    def __init__(self, config: Optional[AgentConfig] = None):
        super().__init__(config or AgentConfig())

    def generate_candidates(
        self,
        context: Dict[str, Any],
        n_candidates: int = 3,
    ) -> List[HypothesisCandidate]:
        """Generates diverse hypothesis candidates using LLM reasoning."""
        iv = context.get("independent_variable", "Independent Variable")
        dv = context.get("dependent_variable", "Dependent Variable")
        design = context.get("research_type", "study")
        problem = context.get("problem_statement", "")
        covars = context.get("covariates", [])
        covar_str = ", ".join(covars) if covars else "none"

        prompt = (
            f"You are a principal scientific investigator designing hypotheses for a research study.\n\n"
            f"Context:\n"
            f"- Study Design: {design}\n"
            f"- Independent Variable: {iv} (Type: {context.get('independent_variable_type', 'unknown')})\n"
            f"- Dependent Variable: {dv} (Type: {context.get('dependent_variable_type', 'unknown')})\n"
            f"- Covariates: {covar_str}\n"
            f"- Problem Statement: {problem}\n\n"
            f"Task:\n"
            f"Formulate {n_candidates} DISTINCT and methodologically rigorous scientific hypotheses for this dataset.\n"
            f"Provide diverse angles, such as:\n"
            f"1. Direct Directional (e.g. positive/negative effect of {iv} on {dv})\n"
            f"2. Confounder-Adjusted Association (e.g. relationship between {iv} and {dv} while controlling for {covar_str})\n"
            f"3. Threshold / Categorical Contrast (e.g. high exposure vs control group differences in {dv})\n\n"
            f"CRITICAL: Return ONLY a JSON list of {n_candidates} objects with keys 'hypothesis', 'angle', 'rationale'.\n"
            f"No markdown fences, no explanatory preamble."
        )

        try:
            response_text = self._invoke_with_fallback(
                [HumanMessage(content=prompt)],
                temperature=0.7,
            )
            raw_list = self._extract_json(response_text)
            candidates: List[HypothesisCandidate] = []
            if isinstance(raw_list, list):
                for item in raw_list:
                    if isinstance(item, dict) and item.get("hypothesis"):
                        candidates.append(
                            HypothesisCandidate(
                                hypothesis=str(item["hypothesis"]).strip(),
                                angle=str(item.get("angle", "general")),
                                rationale=str(item.get("rationale", "")),
                            )
                        )
            if candidates:
                return candidates[:n_candidates]
        except Exception as exc:
            logger.warning("LLM hypothesis generation failed (%s), using heuristic fallbacks.", exc)

        # Fallback deterministic diverse hypotheses
        return [
            HypothesisCandidate(
                hypothesis=f"{iv} has a statistically significant positive effect on {dv}.",
                angle="directional_positive",
                rationale="Standard positive primary effect.",
            ),
            HypothesisCandidate(
                hypothesis=f"{iv} is significantly associated with {dv}, controlling for {covar_str}.",
                angle="adjusted_multivariable",
                rationale="Controlling for confounders to assess independent association.",
            ),
            HypothesisCandidate(
                hypothesis=f"Variations in {iv} account for a statistically significant proportion of variance in {dv}.",
                angle="variance_explained",
                rationale="General omnibus association hypothesis.",
            ),
        ]

    def run(self, context: dict[str, Any], memory: Optional[Any] = None) -> List[HypothesisCandidate]:
        return self.generate_candidates(context)
