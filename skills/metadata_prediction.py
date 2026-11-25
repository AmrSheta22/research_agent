"""
Structural metadata prediction skill.

Predicts the structural components of a research study (independent variable,
dependent variable, research type, time column, subject ID) from dataset columns
and user-supplied context.
"""

from __future__ import annotations

import logging
import re
import string
from typing import Any, Optional

from langchain_core.messages import HumanMessage

from research_agent.prompts.templates import PromptTemplates
from research_agent.schemas import StructuralMetadata
from research_agent.skills.base import BaseSkill

logger = logging.getLogger(__name__)


def _normalize_col(name: str) -> str:
    """Normalize a column name for fuzzy matching."""
    name = name.strip()
    if re.fullmatch(r"\d+(\.\d+)?", name):
        return name
    name = re.sub(r"[\"'`\(\)\[\]\{\}]", "", name)
    name = re.sub(r"\d+", "", name)
    name = name.translate(str.maketrans("", "", string.punctuation))
    name = re.sub(r"\s+", " ", name).strip().lower()
    return name


def matching_column_name(column_name: Optional[str], df_columns: list[str]) -> Optional[str]:
    """Find the best matching column name in the dataset."""
    if not column_name or not isinstance(column_name, str):
        return column_name
    if column_name in df_columns:
        return column_name
    target = _normalize_col(column_name)
    for col in df_columns:
        if _normalize_col(col) == target:
            return col
    return column_name


class MetadataPredictionSkill(BaseSkill):
    """Predicts IV, DV, research type, time variable, and subject ID."""

    name = "metadata_prediction"
    description = (
        "Analyzes dataset preview and problem context to determine the research type, "
        "independent variable, dependent variable, time column, and subject ID."
    )

    def run(self, context: dict[str, Any], memory: Optional[Any] = None) -> StructuralMetadata:
        data_columns = context.get("data_columns") or context.get("dataset_columns") or []
        if not data_columns or len(data_columns) < 2:
            return StructuralMetadata()

        file_data_str = context.get("dataset_preview") or context.get("file_data_str")

        prompt_text = PromptTemplates.build_structural_metadata_prompt(
            data_columns=data_columns,
            file_data_str=file_data_str,
            problem_statement=context.get("problem_statement"),
            name=context.get("name"),
            hypothesis=context.get("hypothesis"),
            research_type=context.get("research_type"),
            indep_var=context.get("independent_variable"),
            dep_var=context.get("dependent_variable"),
        )

        try:
            response_text = self._invoke_with_fallback(
                [HumanMessage(content=prompt_text)],
                temperature=0.0,
            )
            data = self._extract_json(response_text)
            if not isinstance(data, dict):
                return StructuralMetadata()

            res_indep = data.get("independent_var")
            res_dep = data.get("dependent_var")
            res_type = data.get("research_type")
            res_time = data.get("time_variable")
            res_subj = data.get("subject_id")

            if res_time == "None" or res_time == "null":
                res_time = None
            if res_subj == "None" or res_subj == "null":
                res_subj = None

            # Fuzzy match to actual columns
            indep_matched = matching_column_name(res_indep, data_columns)
            dep_matched = matching_column_name(res_dep, data_columns)
            time_matched = matching_column_name(res_time, data_columns) if res_time else None
            subj_matched = matching_column_name(res_subj, data_columns) if res_subj else None

            if memory and hasattr(memory, "save_learning"):
                memory.save_learning("metadata_predictions", f"{context.get('name')}", {
                    "research_type": res_type,
                    "iv": indep_matched,
                    "dv": dep_matched,
                })

            return StructuralMetadata(
                independent_variable=indep_matched,
                dependent_variable=dep_matched,
                research_type=res_type,
                time_variable=time_matched,
                subject_group=subj_matched,
            )
        except Exception as exc:
            logger.exception("Failed to predict structural metadata: %s", exc)
            return StructuralMetadata()
