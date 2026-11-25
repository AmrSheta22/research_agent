"""
Search tools: web search (DuckDuckGo) and citation lookup (Crossref / Habanero).

All logic is self-contained — nothing is imported from legacy modules.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional, Type

from pydantic import BaseModel, Field

from research_agent.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


# ============================================================================
# Tool 1: WebSearchTool
# ============================================================================


class _WebSearchInput(BaseModel):
    query: str = Field(description="The search query to execute.")


class WebSearchTool(BaseTool):
    """Search the web for academic papers, studies, and articles using DuckDuckGo."""

    name = "web_search"
    description = (
        "Searches the web using DuckDuckGo for academic papers, studies, and "
        "articles. Returns a text summary of search results."
    )
    input_schema: type[BaseModel] = _WebSearchInput

    def execute(self, **kwargs: Any) -> ToolResult:
        query: str = kwargs["query"]
        try:
            from langchain_community.tools import DuckDuckGoSearchRun

            search = DuckDuckGoSearchRun()
            results = search.run(query)
            return ToolResult.ok(results)
        except ImportError:
            return ToolResult.fail(
                "langchain-community is not installed. "
                "Install it with: pip install langchain-community duckduckgo-search"
            )
        except Exception as exc:
            logger.exception("Web search failed for query: %s", query)
            return ToolResult.fail(f"Web search error: {exc}")


# ============================================================================
# Tool 3: AcademicLiteratureSearchTool
# ============================================================================


class _CitationLookupInput(BaseModel):
    title: str = Field(description="Title of the paper to look up.")


class CitationLookupTool(BaseTool):
    """Look up an APA citation for a paper by its title using Crossref."""

    name = "citation_lookup"
    description = (
        "Fetches the APA-formatted citation for a given paper title via "
        "the Crossref API (using habanero). Returns the citation string."
    )
    input_schema: type[BaseModel] = _CitationLookupInput

    def execute(self, **kwargs: Any) -> ToolResult:
        title: str = kwargs.get("title", "")
        try:
            from habanero import Crossref

            cr = Crossref(mailto="research-agent@empirical-science.org")
            works = cr.works(query=title, limit=1)
            items = works.get("message", {}).get("items", [])
            from research_agent.evidence import canonical_record, title_matches
            for item in items:
                record = canonical_record(item)
                if record and title_matches(title,record["title"]):
                    return ToolResult.ok(record["citation"])
            return ToolResult.fail("No matching work found on Crossref.")
        except Exception as exc:
            logger.warning("Citation lookup failed for title: %s (%s)", title, exc)
            return ToolResult.fail(f"Citation lookup error: {exc}")


class _AcademicLiteratureSearchInput(BaseModel):
    query: str = Field(description="Search keywords, topic, author, or research domain.")
    limit: int = Field(default=5, description="Maximum number of scholarly records to retrieve (default 5).")


class AcademicLiteratureSearchTool(BaseTool):
    """
    Searches Crossref for candidate journal records; registration is not scientific validation.
    Returns real paper metadata, titles, DOIs, publication years, and APA citations.
    """

    name = "academic_literature_search"
    description = (
        "Searches Crossref for candidate journal articles requiring source and relevance review. "
        "Returns canonical identity metadata; registry presence alone does not establish quality or claim support."
    )
    input_schema: type[BaseModel] = _AcademicLiteratureSearchInput

    def execute(self, **kwargs: Any) -> ToolResult:
        query: str = kwargs.get("query", "").strip()
        limit: int = kwargs.get("limit", 5)

        if not query:
            return ToolResult.fail("Empty search query provided.")

        try:
            from habanero import Crossref

            cr = Crossref(mailto="research-agent@empirical-science.org")
            import requests
            response = requests.get("https://api.crossref.org/works", params={
                "query.title":query,"filter":"type:journal-article","rows":limit,"sort":"relevance"},timeout=25)
            response.raise_for_status()
            works = response.json()
            items = works.get("message", {}).get("items", [])

            from research_agent.evidence import canonical_record
            results = [r for item in items if (r := canonical_record(item))]
            return ToolResult.ok(results)
        except Exception as exc:
            logger.warning("AcademicLiteratureSearch failed for '%s': %s", query, exc)
            return ToolResult.fail(f"Academic search error: {exc}")

