"""
LaTeX Templates Package for Academic Journals.

Provides multi-journal LaTeX templates (IEEE, Elsevier, Springer Nature, ACM, PLOS, arXiv),
template downloading, Jinja2 LaTeX-safe rendering, and paper generation.
"""

from research_agent.latex_templates.manager import TemplateManager, get_latex_jinja_env
from research_agent.latex_templates.downloader import TemplateDownloader
from research_agent.latex_templates.formatter import (
    latex_escape,
    markdown_to_latex,
    generate_bibtex,
    format_statistical_table,
    format_variables_table,
)

__all__ = [
    "TemplateManager",
    "TemplateDownloader",
    "get_latex_jinja_env",
    "latex_escape",
    "markdown_to_latex",
    "generate_bibtex",
    "format_statistical_table",
    "format_variables_table",
]
