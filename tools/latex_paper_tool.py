"""
LaTeX Paper Creation Tool for Research Agent.

Transforms research findings, statistical outputs, literature reviews, and section texts
produced by the agent into publication-ready academic papers formatted with official
journal templates (IEEE, Elsevier, Springer Nature, ACM, PLOS, arXiv).
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

from pydantic import BaseModel, Field

from research_agent.config import AgentConfig
from research_agent.latex_templates.downloader import TemplateDownloader
from research_agent.latex_templates.manager import TemplateManager
from research_agent.schemas import AgentOutput, LiteratureItem, ResearchContext, VisualizationSpec
from research_agent.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool Input Schema
# ---------------------------------------------------------------------------
class CreateLatexPaperInput(BaseModel):
    """Input parameters for creating an academic paper in LaTeX."""

    template: str = Field(
        default="generic_arxiv",
        description="Journal template name: 'ieee', 'elsevier', 'nature', 'acm', 'plos', or 'generic_arxiv'.",
    )
    title: Optional[str] = Field(
        default=None,
        description="Paper title. If omitted, inferred from research context or problem statement.",
    )
    authors: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="List of author objects with 'name', 'affiliation', and optional 'email'.",
    )
    abstract: Optional[str] = Field(
        default=None,
        description="Paper abstract text (e.g. from AbstractSkill).",
    )
    keywords: Optional[List[str]] = Field(
        default=None,
        description="List of academic keywords / index terms.",
    )
    introduction: Optional[str] = Field(
        default=None,
        description="Introduction section text (e.g. from IntroductionSkill).",
    )
    literature_review: Optional[List[Any]] = Field(
        default=None,
        description="List of literature items with citations, titles, and reviews.",
    )
    background_text: Optional[str] = None
    methodology: Optional[str] = Field(
        default=None,
        description="Methodology section text (e.g. from MethodologySkill).",
    )
    results_text: Optional[str] = Field(
        default=None,
        description="Results section interpretation prose (e.g. from TestInterpretationSkill).",
    )
    test_results: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Raw statistical test results dictionary for generating LaTeX tables.",
    )
    discussion: Optional[str] = Field(
        default=None,
        description="Discussion section text evaluating findings, limitations, and implications.",
    )
    conclusion: Optional[str] = Field(
        default=None,
        description="Conclusion and future work section text.",
    )
    study_facts: Optional[Any] = Field(
        default=None,
        description="Optional immutable StudyFacts telemetry contract ensuring 100% data integrity.",
    )
    visualizations: Optional[List[Any]] = Field(
        default=None,
        description="Planned visualization specifications or figures to embed.",
    )
    output_dir: Optional[str] = Field(
        default=None,
        description="Target directory to save .tex, .bib, and compiled files. Uses a temporary directory if omitted.",
    )
    review_gate_passed: bool = Field(default=False, description="Whether required upstream manuscript reviews passed.")
    compile_pdf: bool = Field(
        default=True,
        description="Whether to compile the .tex into a PDF if a LaTeX compiler is available on the system.",
    )
    download_template_url: Optional[str] = Field(
        default=None,
        description="Optional URL to download an external LaTeX template from prior to generation.",
    )


# ---------------------------------------------------------------------------
# Tool Implementation
# ---------------------------------------------------------------------------
class CreateLatexPaperTool(BaseTool):
    """
    Creates publication-ready academic research papers in LaTeX.

    Fills out journal templates (IEEE, Elsevier, Springer Nature, ACM, PLOS, arXiv)
    using the research content, statistical tests, literature reviews, and figures
    produced by the research agent.
    """

    name = "create_latex_paper"
    description = (
        "Creates a publication-ready academic LaTeX paper using famous journal templates "
        "(IEEE, Elsevier, Springer Nature, ACM, PLOS, arXiv). Automatically formats sections, "
        "converts literature reviews into BibTeX references, renders statistical test tables, "
        "embeds figures, and optionally compiles to PDF."
    )
    input_schema: Type[BaseModel] = CreateLatexPaperInput

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        templates_dir: Optional[str] = None,
    ):
        self.config = config or AgentConfig()
        self.manager = TemplateManager(templates_dir=templates_dir)
        self.downloader = TemplateDownloader(templates_dir=self.manager.templates_dir)

    def list_templates(self) -> List[Dict[str, Any]]:
        """List all available local journal templates and their metadata."""
        return self.manager.list_templates()

    def download_template(
        self,
        template_name_or_url: str,
        custom_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Download an external or catalog template from a remote repository."""
        return self.downloader.download_template(template_name_or_url, custom_name=custom_name)

    def download_all_famous_templates(self) -> Dict[str, Any]:
        """Download and verify all famous templates in the catalog."""
        return self.downloader.download_all_famous_templates()

    def execute(self, **kwargs: Any) -> ToolResult:
        """
        Execute paper generation.
        Accepts parameters matching CreateLatexPaperInput or an entire ResearchContext / AgentOutput.
        """
        try:
            # Check for download request first
            download_url = kwargs.get("download_template_url")
            if download_url:
                dl_res = self.download_template(download_url)
                if not dl_res.get("success"):
                    logger.warning("Template download warning: %s", dl_res.get("error"))

            template_name = kwargs.get("template", "generic_arxiv")

            # Extract fields with support for ResearchContext / AgentOutput passed in kwargs
            context = kwargs.get("context")
            agent_output = kwargs.get("agent_output")

            title = kwargs.get("title")
            authors = kwargs.get("authors")
            abstract = kwargs.get("abstract")
            keywords = kwargs.get("keywords")
            introduction = kwargs.get("introduction")
            literature_review = kwargs.get("literature_review")
            background_text = kwargs.get("background_text")
            methodology = kwargs.get("methodology")
            results_text = kwargs.get("results_text")
            test_results = kwargs.get("test_results")
            discussion = kwargs.get("discussion")
            conclusion = kwargs.get("conclusion")
            visualizations = kwargs.get("visualizations")
            study_facts = kwargs.get("study_facts")

            # Hydrate from AgentOutput if provided
            if agent_output is not None:
                if isinstance(agent_output, AgentOutput) or hasattr(agent_output, "model_dump"):
                    ao = agent_output.model_dump()
                elif isinstance(agent_output, dict):
                    ao = agent_output
                else:
                    ao = {}

                abstract = abstract or ao.get("abstract")
                introduction = introduction or ao.get("introduction")
                methodology = methodology or ao.get("methodology")
                discussion = discussion or ao.get("discussion")
                conclusion = conclusion or ao.get("conclusion")
                results_text = results_text or ao.get("test_interpretation")
                literature_review = literature_review or ao.get("literature_review")
                background_text = background_text or ao.get("background")
                visualizations = visualizations or ao.get("visualizations")
                study_facts = study_facts or ao.get("study_facts")
                if not test_results and ao.get("test_results"):
                    test_results = {t.get("test_name", f"Test {i}"): t.get("results", {})
                                    for i, t in enumerate(ao.get("test_results", []))}

            # Hydrate from ResearchContext if provided
            if context is not None:
                if isinstance(context, ResearchContext) or hasattr(context, "model_dump"):
                    c_dict = context.model_dump()
                elif isinstance(context, dict):
                    c_dict = context
                else:
                    c_dict = {}

                title = title or c_dict.get("name") or c_dict.get("problem_statement")
                abstract = abstract or c_dict.get("hypothesis")
                study_facts = study_facts or c_dict.get("study_facts")

            # Output directory
            output_dir = kwargs.get("output_dir")
            if not output_dir:
                output_dir = tempfile.mkdtemp(prefix="academic_paper_")
            os.makedirs(output_dir, exist_ok=True)

            compile_pdf = kwargs.get("compile_pdf", True)

            # Build LaTeX payload
            payload = self.manager.prepare_payload(
                title=title,
                authors=authors,
                abstract=abstract,
                keywords=keywords,
                introduction=introduction,
                literature_review=literature_review,
                background_text=background_text,
                methodology=methodology,
                results_text=results_text,
                test_results=test_results,
                discussion=discussion,
                conclusion=conclusion,
                visualizations=visualizations,
                context=context,
                study_facts=study_facts,
                review_gate_passed=kwargs.get("review_gate_passed", False),
            )

            # Generate paper files (.tex, .bib, and optional PDF)
            bundle = self.manager.generate_paper_bundle(
                template_name=template_name,
                output_dir=output_dir,
                payload=payload,
                compile_pdf=compile_pdf,
            )

            return ToolResult.ok(bundle)

        except Exception as exc:
            logger.exception("Failed to create LaTeX paper: %s", exc)
            return ToolResult.fail(f"Paper generation failed: {exc}")
