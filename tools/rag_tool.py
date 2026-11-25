"""
Statistical RAG Tool: enables retrieval of authoritative textbook guidelines
on hypothesis testing, ANOVA, linear mixed models, pseudo-replication, power analysis,
and equivalence testing.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Type

from pydantic import BaseModel, Field

from research_agent.rag.statistical_rag import StatisticalRAG
from research_agent.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class StatisticalRAGInput(BaseModel):
    query: str = Field(..., description="Query topic on statistical tests, e.g. 'power analysis and MDES in mixed models', 'interpreting non-significant p-values', 'variance partitioning Nakagawa'.")
    top_k: int = Field(default=3, description="Number of textbook sections to retrieve.")


class StatisticalRAGTool(BaseTool):
    """Retrieves authoritative guidance and formulas from statistical test textbooks."""

    name = "query_statistical_textbook"
    description = (
        "Queries an indexed library of authoritative statistical textbooks to retrieve formal definitions, "
        "model specifications, assumption checks, variance partitioning formulas, power analysis guidelines, "
        "and epistemic reporting rules."
    )
    input_schema: Optional[Type[BaseModel]] = StatisticalRAGInput

    def __init__(self, books_dir: Optional[str] = None):
        super().__init__()
        self.rag = StatisticalRAG(books_dir=books_dir) if books_dir else StatisticalRAG.get_instance()

    def execute(self, **kwargs) -> ToolResult:
        query = kwargs.get("query", "").strip()
        top_k = kwargs.get("top_k", 3)
        if not query:
            return ToolResult.fail("Query cannot be empty.")

        try:
            results = self.rag.query(query, top_k=top_k)
            formatted = self.rag.get_context_for_prompt(query, top_k=top_k)
            return ToolResult.ok({
                "query": query,
                "matches": results,
                "formatted_context": formatted,
            })
        except Exception as exc:
            logger.warning("Statistical RAG query failed: %s", exc)
            return ToolResult.fail(f"Statistical RAG query error: {exc}")
