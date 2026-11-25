"""
Template Manager for LaTeX academic paper generation.

Coordinates Jinja2 rendering with LaTeX-safe delimiters, payload preparation,
table and figure inclusion, and multi-file paper bundle export.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import jinja2
from jinja2.sandbox import SandboxedEnvironment

from research_agent.config import AgentConfig
from research_agent.latex_templates.formatter import (
    format_statistical_table,
    format_variables_table,
    format_variance_partitioning_table,
    generate_bibtex,
    latex_escape,
    link_citations_in_text,
    markdown_to_latex,
    normalize_scientific_unicode,
)

logger = logging.getLogger(__name__)


def get_latex_jinja_env(templates_dir: Union[str, Path]) -> jinja2.Environment:
    """
    Initialize a Jinja2 environment with LaTeX-safe delimiters to prevent
    conflicts with TeX curly braces and commands.
    """
    return SandboxedEnvironment(
        loader=jinja2.FileSystemLoader(str(templates_dir)),
        block_start_string=r"\BLOCK{",
        block_end_string="}",
        variable_start_string=r"\VAR{",
        variable_end_string="}",
        comment_start_string=r"\#{",
        comment_end_string="}",
        line_statement_prefix="%%",
        line_comment_prefix="%#",
        trim_blocks=True,
        lstrip_blocks=True,
        autoescape=False,
    )


class TemplateManager:
    """Manages LaTeX templates, payload normalization, and document rendering."""

    def __init__(
        self,
        templates_dir: Optional[Union[str, Path, AgentConfig]] = None,
        config: Optional[AgentConfig] = None,
    ):
        if isinstance(templates_dir, AgentConfig):
            config = templates_dir
            templates_dir = None
        if templates_dir:
            self.templates_dir = Path(templates_dir)
        else:
            self.templates_dir = Path(__file__).parent / "templates"
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        self.env = get_latex_jinja_env(self.templates_dir)
        self.config = config or AgentConfig()

    def list_templates(self) -> list[dict[str, Any]]:
        """List all discovered local templates with their metadata."""
        templates: list[dict[str, Any]] = []
        if not self.templates_dir.exists():
            return templates

        for entry in self.templates_dir.iterdir():
            if entry.is_dir():
                template_id = entry.name
                meta_file = entry / "template.json"
                tex_file = entry / "template.tex"

                meta: dict[str, Any] = {
                    "id": template_id,
                    "name": template_id.replace("_", " ").title(),
                    "has_tex": tex_file.exists(),
                    "path": str(entry),
                }
                if meta_file.exists():
                    try:
                        with open(meta_file, "r", encoding="utf-8") as f:
                            meta.update(json.load(f))
                    except Exception as exc:
                        logger.warning("Error reading template meta %s: %s", meta_file, exc)

                templates.append(meta)

        return sorted(templates, key=lambda x: x["id"])

    def get_template_path(self, template_name: str) -> Optional[Path]:
        """Find the template .tex file for a given template identifier."""
        norm_name = template_name.lower().strip()
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,99}", norm_name):
            raise ValueError("Invalid template identifier")
        candidate = self.templates_dir / norm_name / "template.tex"
        if candidate.exists():
            return candidate

        # Search recursively in template folder for any .tex file
        cand_dir = self.templates_dir / norm_name
        if cand_dir.exists() and cand_dir.is_dir():
            for f in cand_dir.glob("*.tex"):
                return f

        # Fallback to generic_arxiv
        fallback = self.templates_dir / "generic_arxiv" / "template.tex"
        if fallback.exists():
            return fallback

        return None

    def prepare_payload(
        self,
        title: Optional[str] = None,
        authors: Optional[list[dict[str, str]]] = None,
        abstract: Optional[str] = None,
        keywords: Optional[list[str]] = None,
        introduction: Optional[str] = None,
        literature_review: Optional[list[Any]] = None,
        methodology: Optional[str] = None,
        results_text: Optional[str] = None,
        test_results: Optional[dict[str, Any]] = None,
        discussion: Optional[str] = None,
        conclusion: Optional[str] = None,
        visualizations: Optional[list[Any]] = None,
        context: Optional[Any] = None,
        study_facts: Optional[Any] = None,
        review_gate_passed: bool = False,
        background_text: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Normalize agent inputs into a comprehensive LaTeX rendering payload.
        Handles markdown conversion, citation linking, BibTeX generation,
        table creation, and automated consistency validation.
        """
        # Context extraction if provided
        ctx_dict: dict[str, Any] = {}
        if context is not None:
            if hasattr(context, "model_dump"):
                ctx_dict = context.model_dump()
            elif isinstance(context, dict):
                ctx_dict = context

        # StudyFacts extraction
        if study_facts is None and ctx_dict.get("study_facts"):
            study_facts = ctx_dict.get("study_facts")
        if isinstance(study_facts, dict):
            try:
                from research_agent.schemas import StudyFacts
                study_facts = StudyFacts(**study_facts)
            except Exception as e:
                logger.debug("Failed to deserialize study_facts dict: %s", e)

        # 1. Title
        paper_title = title or ctx_dict.get("name") or ctx_dict.get("problem_statement") or "Scientific Investigation and Empirical Findings"
        paper_title = latex_escape(paper_title.strip())

        # 2. Authors (Credible scientific authorship)
        author_list = authors or []
        sanitized_authors: list[dict[str, str]] = []
        for auth in author_list:
            sanitized_authors.append({
                "name": latex_escape(auth.get("name", "Author")),
                "affiliation": latex_escape(auth.get("affiliation", "Academic Institution")),
                "email": latex_escape(auth.get("email", "")),
            })

        # 3. BibTeX & Literature Review
        lit_items = list(literature_review) if literature_review else []
        if not lit_items:
            default_items = self._synthesize_literature_items(ctx_dict)
            existing_titles = {getattr(it, "title", "").lower() if hasattr(it, "title") else it.get("title", "").lower() for it in lit_items}
            for d in default_items:
                d_title = getattr(d, "title", "").lower() if hasattr(d, "title") else d.get("title", "").lower()
                if d_title not in existing_titles:
                    lit_items.append(d)
                    existing_titles.add(d_title)
        else:
            domain_refs = self._get_default_domain_literature(ctx_dict)
            existing_titles = {getattr(it, "title", "").lower() if hasattr(it, "title") else it.get("title", "").lower() for it in lit_items}
            for d in domain_refs:
                d_title = getattr(d, "title", "").lower() if hasattr(d, "title") else d.get("title", "").lower()
                if d_title not in existing_titles:
                    lit_items.append(d)
                    existing_titles.add(d_title)
        bib_content, cite_map = generate_bibtex(lit_items)

        domain_parts: list[str] = []
        method_parts: list[str] = []
        for i, item in enumerate(lit_items, 1):
            if hasattr(item, "model_dump"):
                d = item.model_dump()
            elif isinstance(item, dict):
                d = item
            else:
                d = {
                    "title": getattr(item, "title", f"Work {i}"),
                    "citation": getattr(item, "citation", ""),
                    "review": getattr(item, "review", ""),
                }

            t = d.get("title", "")
            rev = re.sub(r"^Original synthesis:\s*", "", d.get("review", ""), flags=re.I)
            cit = d.get("citation", "")
            key = cite_map.get(t) or cite_map.get(cit) or f"ref_{i}"
            part = f"{rev} \\cite{{{key}}}"

            if any(m_word in t.lower() for m_word in ["mixed-effects", "r2", "false discovery", "power analysis", "multilevel", "hierarchical models", "multiple comparisons"]):
                method_parts.append(part)
            else:
                domain_parts.append(part)

        sections_list = []
        if domain_parts:
            sections_list.append("### Empirical Foundations and Domain Literature\n\n" + "\n\n".join(domain_parts))
        if method_parts:
            sections_list.append("### Methodological and Statistical Modeling Frameworks\n\n" + "\n\n".join(method_parts))

        literature_review_prose = markdown_to_latex("\n\n".join(sections_list)) if sections_list else ""
        if background_text:
            from research_agent.evidence import normalize_doi
            def replace_doi(match):
                key=cite_map.get(normalize_doi(match.group(1)))
                if not key:
                    raise ValueError("Background cites a DOI outside the vetted evidence set")
                return r"\cite{"+key+"}"
            literature_review_prose=markdown_to_latex(re.sub(r"\[@doi:([^\]]+)\]",replace_doi,background_text))

        # 4. Sections with Markdown Conversion & StudyFacts Integrity
        tests_dict = test_results or ctx_dict.get("test_results") or {}

        raw_abstract = abstract or "This study presents an empirical investigation into the hypothesized relationships."
        raw_intro = introduction or ctx_dict.get("problem_statement") or "The purpose of this research is to investigate the proposed empirical questions."
        raw_method = methodology
        if not raw_method:
            raw_method = self._synthesize_methodology(ctx_dict, tests_dict, study_facts=study_facts)

        raw_results = results_text or ctx_dict.get("test_interpretation") or ""
        if not raw_results:
            raw_results = self._synthesize_results_narrative(ctx_dict, tests_dict, raw_results, study_facts=study_facts)

        raw_discussion = discussion
        if not raw_discussion:
            raw_discussion = self._synthesize_discussion(ctx_dict, tests_dict, raw_results, study_facts=study_facts)

        raw_conclusion = conclusion
        if not raw_conclusion:
            raw_conclusion = self._synthesize_conclusion(ctx_dict, tests_dict, raw_results, study_facts=study_facts)

        # Automated Integrity Check and Auto-Correction Gate on RAW text before markdown_to_latex
        if study_facts is not None:
            try:
                from research_agent.validation.paper_validator import PaperValidator
                validator = PaperValidator(study_facts)
                raw_abstract = validator.correct_section_discrepancies("Abstract", raw_abstract)
                raw_intro = validator.correct_section_discrepancies("Introduction", raw_intro)
                raw_method = validator.correct_section_discrepancies("Methods", raw_method)
                raw_results = validator.correct_section_discrepancies("Results", raw_results)
                raw_discussion = validator.correct_section_discrepancies("Discussion", raw_discussion)
                raw_conclusion = validator.correct_section_discrepancies("Conclusion", raw_conclusion)
                ctx_dict["sample_size"] = study_facts.sample_size_total
            except Exception as e:
                logger.warning("PaperValidator auto-correction encountered error: %s", e)

        clean_abstract = markdown_to_latex(self._clean_section_heading(raw_abstract, "Abstract"))
        clean_intro = link_citations_in_text(markdown_to_latex(self._clean_section_heading(raw_intro, "Introduction")), cite_map)
        clean_method = markdown_to_latex(self._clean_section_heading(raw_method, "Methods"))
        clean_results = markdown_to_latex(self._clean_section_heading(raw_results, "Results"))
        clean_discussion = link_citations_in_text(markdown_to_latex(self._clean_section_heading(raw_discussion, "Discussion")), cite_map)
        clean_conclusion = markdown_to_latex(self._clean_section_heading(raw_conclusion, "Conclusion"))

        # 5. Tables
        tests_dict = test_results or ctx_dict.get("test_results") or {}
        from research_agent.manuscript import coefficient_table
        stat_table = coefficient_table(study_facts) if study_facts else format_statistical_table(tests_dict) if tests_dict else ""
        variables_context=dict(ctx_dict)
        if study_facts:
            variables_context.update(num_clusters_total=study_facts.num_clusters_total,level_cluster_counts=study_facts.level_cluster_counts,factor_levels=study_facts.factor_levels,reference_level=study_facts.reference_level)
        vars_table = format_variables_table(variables_context) if variables_context else ""

        # Plot plans remain workflow artifacts; they are not completed manuscript figures.
        figures_latex = ""

        # 7. Keywords
        kw_list = list(keywords) if keywords else ["empirical research", "data analysis", "statistical testing"]
        if ctx_dict.get("research_type"):
            kw_list.append(ctx_dict["research_type"].replace("_", " "))
        clean_keywords = [latex_escape(k) for k in kw_list]

        var_table = ""  # No uncomputed predictor-share decomposition.

        from research_agent.validation.paper_validator import PaperValidator
        report = None
        if study_facts is not None:
            report = PaperValidator(study_facts).validate_full_paper(
                clean_abstract, clean_intro, clean_method, clean_results,
                clean_discussion, clean_conclusion, bib_content,
            )
        validation = report.model_dump() if report else {
            "is_valid": False, "issues": [{"severity": "ERROR", "rule": "missing_study_facts",
            "message": "No supported study-facts contract is available for manuscript validation."}],
        }
        from research_agent.evidence import identity_issues
        for issue in identity_issues(lit_items):
            validation["is_valid"] = False
            validation["issues"].append({"severity":"ERROR", "rule":"source_identity", "message":issue})
        if not lit_items:
            validation["is_valid"] = False
            validation["issues"].append({"severity":"ERROR", "rule":"missing_reviewed_sources", "message":"No reviewed source evidence is available."})
        ready = bool(validation["is_valid"] and author_list and review_gate_passed)
        if not review_gate_passed:
            validation["is_valid"] = False
            validation["issues"].append({"severity": "ERROR", "rule": "review_gate", "message": "Required manuscript reviews have not passed."})
        if not author_list:
            validation["is_valid"] = False
            validation["issues"].append({"severity": "ERROR", "rule": "missing_authors", "message": "Author identities must be supplied by the investigators."})
        if not ready:
            paper_title = "[DRAFT - NOT VALIDATED] " + paper_title

        from research_agent.provenance import availability_statements
        data_availability,code_availability=availability_statements(ctx_dict,lit_items)
        return {
            "title": paper_title,
            "publication_ready": ready,
            "validation_report": validation,
            "authors": sanitized_authors,
            "abstract": self._clean_section_heading(clean_abstract, "Abstract"),
            "keywords": clean_keywords,
            "keywords_joined": ", ".join(clean_keywords),
            "introduction": self._clean_section_heading(clean_intro, "Introduction"),
            "literature_review": bool(literature_review_prose),
            "literature_review_text": self._clean_section_heading(literature_review_prose, "Background and Related Work"),
            "methodology": self._clean_section_heading(clean_method, "Methods"),
            "results_text": self._clean_section_heading(clean_results, "Results"),
            "discussion": self._clean_section_heading(clean_discussion, "Discussion"),
            "conclusion": self._clean_section_heading(clean_conclusion, "Conclusion"),
            "statistical_table": stat_table,
            "variance_table": var_table,
            "variables_table": vars_table,
            "figures_block": figures_latex,
            "has_bibliography": bool(bib_content),
            "bib_content": bib_content,
            "bib_filename": "references",
            "author_summary": clean_abstract,
            "data_availability_text": latex_escape(data_availability),
            "code_availability_text": latex_escape(code_availability),
        }

    @staticmethod
    def _clean_section_heading(text: str, section_name: str) -> str:
        """Strip redundant leading headings/titles such as '# Discussion', '4 Discussion', '\\section{Discussion}', '**Discussion**', etc."""
        if not text:
            return ""
        cleaned = text.strip()

        # Strip leading comments and blank lines
        while cleaned.startswith("%"):
            idx = cleaned.find("\n")
            if idx == -1:
                return ""
            cleaned = cleaned[idx + 1:].strip()

        section_titles = [
            section_name,
            "discussion",
            "results and findings",
            "empirical results and statistical analysis",
            "results",
            "materials and methods",
            "methods",
            "methodology",
            "introduction",
            "background and related work",
            "related work",
            "conclusion and future work",
            "conclusion",
            "abstract",
        ]
        titles_regex = "|".join(re.escape(t) for t in section_titles)

        while True:
            prev_len = len(cleaned)
            # 1. LaTeX \section / \subsection
            cleaned = re.sub(
                rf"^(?:\\(?:sub)*section\*?\{{(?:{titles_regex})\}}|\bSection\s*\d*[:\s-]*)\s*",
                "",
                cleaned,
                flags=re.IGNORECASE,
            ).strip()

            # 2. Markdown headings (#, ##, ###)
            cleaned = re.sub(
                rf"^#+\s*(?:\d+\.?\s*)?(?:{titles_regex})[:\s-]*\n*",
                "",
                cleaned,
                flags=re.IGNORECASE,
            ).strip()

            # 3. Bold headings (**Title**, \textbf{Title})
            cleaned = re.sub(
                rf"^(?:\*\*|\\textbf\{{)\s*(?:\d+\.?\s*)?(?:{titles_regex})[:\s-]*\}}?\**\s*",
                "",
                cleaned,
                flags=re.IGNORECASE,
            ).strip()

            # 4. Check first line explicitly
            lines = cleaned.splitlines()
            if lines:
                first_line = lines[0].strip()
                norm = re.sub(r"^(?:\\textbf\{|\*\*|#+|\d+\.?\s*|\bsection\s*\d*[:\s-]*)+", "", first_line, flags=re.IGNORECASE)
                norm = re.sub(r"[\}\*\:\.\-]+$", "", norm).strip().lower()
                if norm in [t.lower() for t in section_titles] or norm == section_name.lower():
                    cleaned = "\n".join(lines[1:]).strip()

            if len(cleaned) == prev_len:
                break

        return cleaned

    def _synthesize_methodology(self, ctx_dict, tests_dict, brief_interp="", study_facts=None):
        from research_agent.reporting import methodology_narrative
        return methodology_narrative(ctx_dict, tests_dict, study_facts)

    def _synthesize_literature_items(self, ctx_dict: dict[str, Any]) -> list[Any]:
        """Synthesize rich, peer-reviewed literature items with authentic APA citations."""
        return []  # Only the explicit evidence-review stage may add references.

    def _get_default_domain_literature(self, ctx_dict: Optional[dict[str, Any]] = None) -> list[dict[str, str]]:
        """Return authentic, verified peer-reviewed domain and methodological citations."""
        return []

    def _get_default_methodological_literature(self) -> list[dict[str, str]]:
        return self._get_default_domain_literature()

    def _synthesize_results_narrative(self, ctx_dict, tests_dict, brief_interp="", study_facts=None):
        from research_agent.reporting import results_narrative
        return results_narrative(ctx_dict, tests_dict, study_facts)

    def _synthesize_discussion(self, ctx_dict, tests_dict, brief_interp="", study_facts=None):
        from research_agent.reporting import conclusion_narrative
        return conclusion_narrative(ctx_dict, tests_dict, study_facts)

    def _synthesize_conclusion(self, ctx_dict, tests_dict, brief_interp="", study_facts=None):
        from research_agent.reporting import conclusion_narrative
        return conclusion_narrative(ctx_dict, tests_dict, study_facts)

    def render_paper(
        self,
        template_name: str,
        payload: dict[str, Any],
    ) -> str:
        """Render the chosen template with the prepared payload into a LaTeX string."""
        norm_template = template_name.lower().strip()
        template_rel_path = f"{norm_template}/template.tex"

        try:
            template = self.env.get_template(template_rel_path)
        except jinja2.TemplateNotFound:
            logger.warning(
                "Template '%s' not found. Falling back to 'generic_arxiv/template.tex'",
                template_rel_path,
            )
            template = self.env.get_template("generic_arxiv/template.tex")

        rendered = template.render(**payload)
        return rendered

    def generate_paper_bundle(
        self,
        template_name: str,
        output_dir: Union[str, Path],
        payload: dict[str, Any],
        compile_pdf: bool = True,
    ) -> dict[str, Any]:
        """
        Generate complete LaTeX paper package in output_dir:
        - main.tex
        - references.bib
        - PDF compilation if compile_pdf is True and a LaTeX engine is available.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # 1. Render LaTeX source
        tex_source = self.render_paper(template_name, payload)
        tex_file = out_path / "main.tex"
        with open(tex_file, "w", encoding="utf-8") as f:
            f.write(tex_source)

        # 2. Write BibTeX file
        bib_content = normalize_scientific_unicode(payload.get("bib_content", ""))
        bib_file = out_path / "references.bib"
        with open(bib_file, "w", encoding="utf-8") as f:
            f.write(bib_content)

        result: dict[str, Any] = {
            "template": template_name,
            "tex_file": str(tex_file),
            "bib_file": str(bib_file),
            "output_dir": str(out_path),
            "latex_source": tex_source,
            "pdf_file": None,
            "compiled": False,
            "publication_ready": bool(payload.get("publication_ready", False)),
            "validation_report": payload.get("validation_report"),
        }

        # 3. Compile PDF if requested and compiler available
        if compile_pdf:
            pdf_path = self.compile_to_pdf(tex_file)
            if pdf_path and pdf_path.exists():
                result["pdf_file"] = str(pdf_path)
                result["compiled"] = True
            else:
                result["compile_message"] = (
                    "LaTeX source generated. PDF compilation was unavailable or failed; inspect the compiler logs."
                )

        return result

    @staticmethod
    def compile_to_pdf(tex_file_path: Union[str, Path], timeout: int = 60) -> Optional[Path]:
        """
        Attempt to compile a .tex file to PDF using tectonic, pdflatex, or latexmk if installed.
        Returns the Path to the compiled PDF if successful, else None.
        """
        import sys

        tex_path = Path(tex_file_path).resolve()
        parent_dir = tex_path.parent
        pdf_path = parent_dir / f"{tex_path.stem}.pdf"

        # Check for compilers (prefer tectonic if available as it automatically fetches packages)
        venv_tectonic = Path(sys.executable).parent / ("tectonic.exe" if sys.platform == "win32" else "tectonic")
        tectonic_path = str(venv_tectonic) if venv_tectonic.exists() else shutil.which("tectonic")

        compiler = tectonic_path or shutil.which("latexmk") or shutil.which("pdflatex") or shutil.which("xelatex")
        if not compiler:
            logger.info("No LaTeX compiler found on system PATH. Skipping PDF compilation.")
            return None

        if "tectonic" in Path(compiler).stem.lower():
            cmd = [
                compiler,
                tex_path.name,
                "--outdir",
                str(parent_dir),
            ]
        elif "latexmk" in Path(compiler).stem.lower():
            cmd = [compiler, "-pdf", "-no-shell-escape", "-interaction=nonstopmode", "-halt-on-error", tex_path.name]
        else:
            cmd = [
                compiler,
                "-no-shell-escape",
                "-interaction=nonstopmode",
                "-halt-on-error",
                tex_path.name,
            ]

        try:
            logger.info("Compiling LaTeX paper via %s in %s", compiler, parent_dir)
            if pdf_path.exists():
                try:
                    pdf_path.unlink()
                except Exception:
                    pass

            proc = subprocess.run(
                cmd,
                cwd=str(parent_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                check=False,
            )
            if proc.returncode == 0 and Path(compiler).stem.lower() in {"pdflatex", "xelatex"}:
                aux_path = tex_path.with_suffix(".aux")
                if aux_path.exists() and "\\bibdata" in aux_path.read_text(errors="replace"):
                    bibtex = shutil.which("bibtex")
                    if not bibtex:
                        logger.warning("BibTeX is required to resolve this manuscript's references.")
                        return None
                    bib_proc = subprocess.run(
                        [bibtex, tex_path.stem], cwd=str(parent_dir),
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, check=False,
                    )
                    if bib_proc.returncode != 0:
                        logger.warning("BibTeX failed with code %s", bib_proc.returncode)
                        return None
                # Resolve references and page numbers after bibliography generation.
                for _ in range(2):
                    proc = subprocess.run(cmd, cwd=str(parent_dir), stdout=subprocess.PIPE,
                                          stderr=subprocess.PIPE, timeout=timeout, check=False)
                    if proc.returncode != 0:
                        break
            if proc.returncode == 0 and pdf_path.exists():
                return pdf_path
            else:
                out_msg = proc.stdout.decode("utf-8", errors="ignore")[-400:]
                err_msg = proc.stderr.decode("utf-8", errors="ignore")[-400:]
                logger.warning("LaTeX compilation failed (code %s): stdout: %s | stderr: %s", proc.returncode, out_msg, err_msg)
                return None
        except Exception as exc:
            logger.warning("Failed to compile LaTeX to PDF: %s", exc)
            return None


# Backward-compatible alias for nodes and skills importing LaTeXPaperManager
LaTeXPaperManager = TemplateManager

