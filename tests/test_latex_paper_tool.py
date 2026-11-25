"""
Unit and integration tests for the LaTeX paper generation tool and journal templates.
"""

import os
from pathlib import Path
import pytest

from research_agent.agent import ResearchAgent
from research_agent.latex_templates.downloader import TemplateDownloader
from research_agent.latex_templates.formatter import (
    format_statistical_table,
    format_variables_table,
    generate_bibtex,
    latex_escape,
    markdown_to_latex,
)
from research_agent.latex_templates.manager import TemplateManager
from research_agent.schemas import AgentOutput, LiteratureItem, ResearchContext, TestResult, VisualizationSpec
from research_agent.tools.latex_paper_tool import CreateLatexPaperTool


@pytest.fixture
def sample_research_context():
    return ResearchContext(
        name="The Impact of Sleep Deprivation on Cognitive Processing Speed",
        problem_statement="How does chronic sleep loss affect neural reaction times in adult subjects?",
        hypothesis="Chronic sleep deprivation significantly degrades cognitive processing speed.",
        research_type="cross_sectional",
        independent_variable="sleep_hours",
        independent_variable_type="numeric",
        dependent_variable="reaction_time_ms",
        dependent_variable_type="numeric",
        covariates=["age", "caffeine_intake"],
        sample_size=120,
    )


@pytest.fixture
def sample_agent_output():
    return AgentOutput(
        hypothesis="Sleep deprivation significantly increases reaction time.",
        abstract="**Background:** Sleep loss impairs executive function. **Methods:** Linear regression and t-tests were evaluated. **Results:** Significant latency increases ($p < 0.001$) were observed. **Conclusion:** Adequate sleep is critical for cognitive resilience.",
        introduction="Sleep loss is an escalating public health concern worldwide. Recent neuroimaging studies indicate substantial disruptions in prefrontal cortical activation under deprivation.",
        methodology="Participants were monitored under controlled sleep intervals. Reaction latency was assessed via psychomotor vigilance tasks. All data underwent automated outlier removal and normality verification.",
        test_interpretation="A statistically significant negative relationship was confirmed between sleep duration and response latency ($t = -4.82$, $p < 0.0001$). Effect size Cohen's $d = 0.88$ indicates a strong effect.",
        test_results=[
            TestResult(
                test_name="Independent Samples T-Test",
                results={"t": -4.821, "p": 0.000042, "df": 118},
                interpretation="Statistically significant difference.",
            ),
            TestResult(
                test_name="Linear Regression",
                results={"slope": -14.25, "p": 0.0001, "r_squared": 0.38},
                interpretation="Reaction time decreases by 14.25ms per hour of sleep.",
            ),
        ],
        literature_review=[
            LiteratureItem(
                title="Neurocognitive Dynamics of Sleep Deprivation",
                citation="Alvarez, C. et al. (2022). Neurocognitive Dynamics. J. Sleep Res.",
                review="Demonstrated that 24-hour sleep restriction severely degrades vigilance and working memory.",
            ),
            LiteratureItem(
                title="Circadian Rhythms and Cognitive Performance",
                citation="Belcher, K. (2023). Circadian Rhythms. Cognitive Science Rev.",
                review="Established baseline circadian variations in psychomotor processing speeds.",
            ),
        ],
        visualizations=[
            VisualizationSpec(
                title="Distribution of Reaction Latency",
                mark="Histogram",
                x_axis="reaction_time_ms",
            ),
            VisualizationSpec(
                title="Reaction Time vs Sleep Hours Regression",
                mark="ScatterPlot",
                x_axis="sleep_hours",
                y_axis="reaction_time_ms",
            ),
        ],
    )


def test_formatter_latex_escape():
    # Regular text with specials
    raw = "Cost is $50 & 50% discount for #1 user _test_ {foo}"
    # Without math delimiters, $ is escaped
    escaped = latex_escape(raw)
    assert r"\&" in escaped
    assert r"\%" in escaped
    assert r"\#" in escaped
    assert r"\_" in escaped

    # Math mode protection
    math_raw = "Significant difference found ($p < 0.05$ & $\\chi^2 = 12.4$)"
    math_escaped = latex_escape(math_raw)
    assert "$p < 0.05$" in math_escaped
    assert "$\\chi^2 = 12.4$" in math_escaped


def test_formatter_markdown_to_latex():
    md = """### Primary Hypotheses
This is **bold** and *italic* and `code`.
* Item one
* Item two
1. First
2. Second
"""
    tex = markdown_to_latex(md)
    assert r"\subsubsection{Primary Hypotheses}" in tex
    assert r"\textbf{bold}" in tex
    assert r"\textit{italic}" in tex
    assert r"\texttt{code}" in tex
    assert r"\begin{itemize}" in tex
    assert r"\item Item one" in tex
    assert r"\begin{enumerate}" in tex
    assert r"\item First" in tex


def test_formatter_generate_bibtex():
    items = [
        LiteratureItem(
            title="Deep Learning in Genomics",
            citation="Vaswani et al. (2021). Genomics Journal.",
            review="Transformer architectures in genomics.",
            doi="10.1234/test",metadata_verified=True,authors=[{"family":"Fixture","given":"A"}],year="2021",journal="Test Journal",
        )
    ]
    bib_text, cite_map = generate_bibtex(items)
    assert "@article{" in bib_text
    assert "Deep Learning in Genomics" in bib_text
    assert len(cite_map) >= 1


def test_formatter_tables(sample_research_context):
    # Statistical table
    test_dict = {
        "T-test": {"t": 3.14, "p": 0.002},
        "Regression": {"coef": 1.25, "p": 0.04},
    }
    table_tex = format_statistical_table(test_dict)
    assert r"\begin{table}" in table_tex
    assert r"\toprule" in table_tex
    assert "T-test" in table_tex
    assert "0.002" in table_tex

    # Variables table
    var_tex = format_variables_table(sample_research_context.model_dump())
    assert r"sleep\_hours" in var_tex
    assert r"reaction\_time\_ms" in var_tex


def test_template_manager_list():
    manager = TemplateManager()
    templates = manager.list_templates()
    template_ids = [t["id"] for t in templates]

    expected = ["acm", "elsevier", "generic_arxiv", "ieee", "nature", "plos"]
    for exp in expected:
        assert exp in template_ids, f"Expected template '{exp}' to be registered."


@pytest.mark.parametrize("template_name", ["ieee", "elsevier", "nature", "acm", "plos", "generic_arxiv"])
def test_generate_paper_all_templates(template_name, sample_research_context, sample_agent_output, tmp_path):
    tool = CreateLatexPaperTool()
    out_dir = tmp_path / f"test_{template_name}"

    res = tool.execute(
        template=template_name,
        context=sample_research_context,
        agent_output=sample_agent_output,
        output_dir=str(out_dir),
        compile_pdf=False,  # Skip compilation in CI/test environments without pdflatex
    )

    assert res.success is True
    data = res.data
    assert data["template"] == template_name
    assert os.path.exists(data["tex_file"])
    assert os.path.exists(data["bib_file"])

    with open(data["tex_file"], "r", encoding="utf-8") as f:
        tex_content = f.read()

    # Verify content fills
    assert "Sleep Deprivation" in tex_content
    assert "Introduction" in tex_content or "Introduction" in tex_content.lower()
    assert "Methodology" in tex_content or "Methods" in tex_content
    assert "Results" in tex_content
    assert "\\bibliography{references}" not in tex_content  # Unverified fixture sources must not become citations.
    assert data["publication_ready"] is False


def test_template_downloader_catalog():
    downloader = TemplateDownloader()
    catalog = downloader.get_catalog()
    assert "ieee" in catalog
    assert "elsevier" in catalog
    assert "nature" in catalog
    assert "acm" in catalog
    assert "plos" in catalog


def test_agent_integration(sample_research_context, sample_agent_output, tmp_path):
    agent = ResearchAgent()
    assert "create_latex_paper" in agent.tools

    # Test high-level create_paper method on agent
    paper_res = agent.create_paper(
        context=sample_research_context,
        agent_output=sample_agent_output,
        template="ieee",
        output_dir=str(tmp_path / "agent_paper"),
        compile_pdf=False,
    )

    assert isinstance(paper_res, dict)
    assert "tex_file" in paper_res
    assert Path(paper_res["tex_file"]).exists()
    assert Path(paper_res["bib_file"]).exists()
