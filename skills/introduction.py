"""
Introduction generation skill.

Generates a professional academic introduction covering problem statement,
literature review summary, research gap, gap importance, and study objectives.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from langchain_core.messages import HumanMessage

from research_agent.prompts.templates import PromptTemplates
from research_agent.skills.base import BaseSkill

logger = logging.getLogger(__name__)


def format_literature_items(literature: Any) -> str:
    """Format list of literature items/dicts into a coherent reference paragraph."""
    if not literature:
        return ""
    if isinstance(literature, str):
        return literature

    formatted_paragraphs: list[str] = []
    if isinstance(literature, list):
        for idx, item in enumerate(literature):
            if hasattr(item, "review") and hasattr(item, "citation"):
                review = (item.review or "").strip()
                if review:
                    formatted_paragraphs.append(f"{review} [{idx + 1}]")
            elif isinstance(item, dict):
                review = (item.get("review") or "").strip()
                if review:
                    formatted_paragraphs.append(f"{review} [{idx + 1}]")

    return " ".join(formatted_paragraphs)


class IntroductionSkill(BaseSkill):
    """Generates an academic research introduction based on hypothesis and literature."""

    name = "introduction"
    description = (
        "Writes a professional academic introduction establishing problem context, "
        "literature review, research gap, significance, and study objectives."
    )

    def run(self, context: dict[str, Any], memory: Optional[Any] = None) -> str:
        lit_input = context.get("literature_review") or []
        evidence=[]
        for item in lit_input:
            record=item.model_dump() if hasattr(item,"model_dump") else item
            evidence.append({**{k:record.get(k) for k in ("title","doi","year","review","review_scope","evidence_quotes")},"authors":record.get("authors",[])[:3]})
        import json
        prompt_text = (
            "Write a concise 300-450 word introduction for this actual study. Use only supplied reviewed source evidence. "
            "Cite exact author surnames and years from metadata. Do not refer to supplied reviews, prompts or retrieved snippets as authorities; cite the actual source directly. Distinguish prior analyses of within-site variability "
            "from the present estimation of mean lithology offsets with covariates and clustering. Never assert no global study exists. "
            "Explain what prior work tested and the narrower contribution of this analysis. "
            "Frame the objective as estimating offsets and confidence intervals, not obtaining significance. "
            "Do not report our numerical results, invent mechanisms, or paste source text. No numbered headings. "
            "If evidence is insufficient, explicitly limit novelty claims. Return plain prose.\n" +
            json.dumps({"study":{k:context.get(k) for k in ["name","problem_statement","hypothesis"]},"reviewed_evidence":evidence},default=str)
        )
        study_facts = context.get("study_facts")
        try:
            response_text = self._invoke_with_fallback(
                [HumanMessage(content=prompt_text)],
                temperature=0.3,
            )
            try:
                parsed_json = self._extract_json(response_text)
                prose = self._format_json_as_prose(parsed_json)
            except Exception:
                # If LLM returned direct prose rather than JSON, use the prose directly
                if len(response_text.strip().split()) >= 80:
                    prose = response_text.strip()
                else:
                    raise

            if memory and hasattr(memory, "record_successful_prompt"):
                memory.record_successful_prompt(
                    skill_name=self.name,
                    context_summary="academic introduction",
                    result_quality=1.0,
                )

            from research_agent.manuscript import enforce_budget
            prose=re.sub(r"\bthe supplied review (?:indicates|suggests|states|reports) that\s+", "", prose, flags=re.I)
            return enforce_budget(self,prose,450,"This analysis estimates mean offsets with covariate and cave adjustment. Available reviewed evidence does not establish the absence of previous global studies. Its contribution is restricted to estimation in the supplied extract.")
        except Exception as exc:
            logger.warning("LLM introduction generation fallback (%s), constructing structured academic synthesis.", exc)
            return self._grounded_fallback(context,lit_input)

    @staticmethod
    def _grounded_fallback(context, sources):
        facts=context.get("study_facts")
        iv=context.get("independent_variable") or "the supplied grouping factor"
        dv=(context.get("dependent_variable") or "the supplied outcome").removeprefix("mean_")
        covars=", ".join(context.get("covariates") or [])
        citations=[]
        for obj in sources:
            r=obj.model_dump() if hasattr(obj,"model_dump") else obj
            if r.get("authors") and r.get("year") and r.get("metadata_verified"):
                citations.append(r["authors"][0]["family"]+(" et al." if len(r["authors"])>1 else "")+" ("+str(r["year"])+")")
        refs="; ".join(citations)
        text=f"This observational analysis estimates mean {dv} offsets across {iv} categories in the supplied analytical extract. "
        if refs:
            text+="The reviewed literature provides domain context and database documentation ("+refs+"). The Background summarizes the findings supporting that context. "
        text+="Within-cluster variability, replication among records, and conditional differences between category means are distinct questions. The present contribution is restricted to estimating the latter in this extract, rather than claiming that earlier global work is absent. "
        if facts:
            text+=f"The extract contains {facts.sample_size_total} observations and {facts.num_clusters_total or 'unreported'} clusters. "
        text+=(f"Adjustment uses {covars}. " if covars else "No additional covariate adjustment was specified. ")
        text+="The primary objective is to quantify offset direction, magnitude and confidence intervals. Unadjusted comparisons describe differences in raw sample means; adjusted models evaluate conditional associations, and cluster-aware specifications account for dependence among observations. Comparisons between specifications assess model sensitivity rather than select a preferred result by its p-value. "
        text+="A non-significant estimate does not establish zero effect or equivalence, while an observational association does not establish a causal mechanism. Interpretation depends on the sampled clusters, model assumptions and data provenance. The resulting estimates provide a transparent basis for subsequent validation with independently verified selection rules and relevant environmental controls."
        return text
