"""Regression tests for input loss, statistical integrity, and unsafe tool boundaries."""

import io
import json
import tarfile
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

from research_agent import bridge
from research_agent.agent import ResearchAgent
from research_agent.config import AgentConfig
from research_agent.graph.nodes import ResearchGraphNodes
from research_agent.lats.cleaner_pool import CleanerPool
from research_agent.lats.evaluator import LATSEvaluator
from research_agent.lats.tree import Trajectory
from research_agent.latex_templates.downloader import TemplateDownloader
from research_agent.latex_templates.manager import TemplateManager
from research_agent.memory.store import MemoryStore
from research_agent.schemas import ResearchContext, StructuralMetadata
from research_agent.skills.abstract import AbstractSkill
from research_agent.skills.literature_review import LiteratureReviewSkill
from research_agent.study_designs.dispatcher import dispatch_tests
from research_agent.study_facts import build_study_facts
from research_agent.tools.analysis_tools import AutonomousAnalysisTool
from research_agent.tools.base import ToolResult
from research_agent.tools.data_tools import DatasetQueryTool, read_file_safely
from research_agent.tools.latex_paper_tool import CreateLatexPaperTool
from research_agent.tools.search_tools import AcademicLiteratureSearchTool


@pytest.fixture
def data():
    rng = np.random.default_rng(123)
    return pd.DataFrame({"group": ["a"] * 30 + ["b"] * 30, "score": rng.normal(size=60)})


@pytest.fixture
def context():
    return ResearchContext(independent_variable="group", dependent_variable="score", research_type="cross_sectional")


@pytest.mark.parametrize("function,method", [
    (bridge.predict_hypothesis, "generate_hypothesis"),
    (bridge.predict_abstract, "generate_abstract"),
    (bridge.predict_methodology, "generate_methodology"),
    (bridge.predict_introduction, "generate_introduction"),
])
def test_bridge_preserves_model_fields(function, method):
    agent = MagicMock()
    with patch.object(bridge, "get_agent", return_value=agent):
        function(independent_variable="drug", dependent_variable="score", sample_size=42, unknown="ignored")
    ctx = getattr(agent, method).call_args.args[0]
    assert (ctx.independent_variable, ctx.dependent_variable, ctx.sample_size) == ("drug", "score", 42)


def test_full_analysis_preserves_graph_results(context):
    agent = ResearchAgent.__new__(ResearchAgent)
    state = {"context": context, "test_results": {"test": {"p_value": .3}},
             "test_interpretation": "No evidence of an effect.", "test_status": "completed",
             "discussion": "Discussion", "conclusion": "Conclusion", "errors": ["notice"]}
    with patch.object(ResearchAgent, "run_graph", return_value=state):
        output = agent.run_full_analysis(context, tasks={"tests"})
    assert output.test_results[0].results["p_value"] == .3
    assert output.test_interpretation == state["test_interpretation"]
    assert output.discussion == "Discussion"
    assert output.errors == ["notice"]


def test_p_values_require_explicit_labels():
    values = LATSEvaluator.extract_p_values({
        "OLS Regression": {"coef": .001, "std_err": .002, "p_value": .8},
        "Model": {"p_values": {"Intercept": .00001, "treatment": .4}},
        "_study_facts": {"p_value": .000001},
    })
    assert [p for _, p in values] == [.8, .4]


def test_route_score_is_independent_of_p_value():
    a = LATSEvaluator.evaluate({"test": {"p": .0001}}, 100, 95)
    b = LATSEvaluator.evaluate({"test": {"p": .8}}, 100, 95)
    assert a.composite_score == b.composite_score


def test_multiplicity_counts_each_test_in_each_path():
    paths = [Trajectory(test_results={"a": {"p": .02}, "b": {"p": .7}}, min_p_value=.02) for _ in range(2)]
    LATSEvaluator.apply_multiplicity_correction(paths)
    assert paths[0].bonferroni_alpha == .05 / 4
    assert not paths[0].is_significant_after_bonferroni


def test_hc3_extraction_matches_statsmodels(data, context):
    facts = build_study_facts(data, context)
    expected = smf.ols("score ~ C(group)", data=data).fit().get_robustcov_results(cov_type="HC3")
    actual = list(facts.ols_robust_se.values())
    assert len(actual) == 2
    assert [v["std_err_hc3"] for v in actual] == pytest.approx(expected.bse, abs=.00005)


def test_unsupported_facts_do_not_invent_anova(data, context):
    context.independent_variable_type = "numeric"
    assert build_study_facts(data, context) is None
    context.independent_variable_type = "categorical"
    context.research_type = "pre_post"
    assert build_study_facts(data, context) is None


def test_categorical_covariate_does_not_crash(data, context):
    data["category"] = pd.Categorical(["young", "old"] * 30)
    context.covariates = ["category"]
    assert build_study_facts(data, context).regression_coefficients


def test_cleaning_preserves_rare_group_and_missingness_provenance(data, context):
    data["group"] = ["common"] * 58 + ["rare"] * 2
    data.loc[0, "score"] = np.nan
    cleaned = CleanerPool().apply_strategy(data, "no_outlier_removal", "score", ["group"])
    facts = build_study_facts(cleaned.cleaned_df, context, raw_df=data)
    assert set(cleaned.cleaned_df["group"]) == {"common", "rare"}
    assert facts.manifest.outliers_removed == 0
    assert facts.manifest.missing_rows_removed == 1
    assert facts.manifest.outlier_method == "None"


def test_actual_cleaning_strategy_is_recorded(data, context):
    data.loc[0, "score"] = 100
    cleaned = CleanerPool().apply_strategy(data, "iqr_conservative", "score", ["group"])
    facts = build_study_facts(cleaned.cleaned_df, context, raw_df=data)
    assert facts.manifest.outliers_removed == 1
    assert facts.manifest.outlier_method == "IQR 3.0x rule"
    assert facts.manifest.upper_fence < 100


def test_excel_header_is_supported(tmp_path, data):
    path = tmp_path / "data.xlsx"
    data.to_excel(path, index=False)
    loaded = read_file_safely(str(path))
    assert loaded is not None and loaded.shape == data.shape


def test_named_rct_groups_produce_finite_tests(data):
    results = dispatch_tests(data, "individual_rct", "binary", "numeric", "group", "score")
    for name in ["T-test (unadjusted)", "OLS (unadjusted)", "OLS (adjusted)"]:
        assert isinstance(results[name], dict)
        assert 0 <= results[name]["p-value"] <= 1
    assert set(data["group"]) == {"a", "b"}


def test_metadata_prediction_prepares_new_variables(data):
    nodes = ResearchGraphNodes()
    state = nodes.ingest_and_clean_state({"context": ResearchContext(hypothesis="H"), "df": data})
    metadata = StructuralMetadata(independent_variable="group", dependent_variable="score", research_type="cross_sectional")
    with patch.object(nodes.metadata_skill, "run", return_value=metadata):
        result = nodes.metadata_prediction_state(state)
    assert result["context"].independent_variable_type == "binary"
    assert result["context"].dependent_variable_type == "numeric"


def test_abstract_does_not_assert_unperformed_adjustment(data, context):
    facts = build_study_facts(data, context)
    with patch.object(AbstractSkill, "_invoke_with_fallback", side_effect=RuntimeError("offline")):
        text = AbstractSkill(AgentConfig()).run({**context.model_dump(), "study_facts": facts})
    assert "After accounting for cluster structure" not in text
    assert "mixed-effects" not in text


def test_tost_conclusion_matches_estimates(data):
    data["score"] *= .01
    result = AutonomousAnalysisTool().execute(df=data, iv="group", dv="score", operation="equivalence_testing")
    assert all(v["statistically_equivalent"] for v in result.data["results"].values())
    assert "1 of 1 evaluated contrasts meet" in result.data["conclusion"]


def test_literature_cannot_invent_or_replace_citations():
    record = {"title":"Real paper","citation":"Actual Author (2020)","doi":"10.1234/real", "work_type":"journal-article", "metadata_verified":True,
              "abstract":"The actual source reports group comparisons in caves."}
    response=json.dumps({"accepted":[{"doi":"10.1234/fake","review":"Fake","evidence_quotes":["fabricated"]},
        {"doi":"10.1234/real","review":"The study compares cave groups.","relevance_reason":"Directly relevant to cave groups", "evidence_quotes":[record["abstract"]]}],"next_query":""})
    with patch.object(AcademicLiteratureSearchTool,"execute",return_value=ToolResult.ok([record])), patch.object(LiteratureReviewSkill,"_invoke_with_fallback",return_value=response):
        items=LiteratureReviewSkill(AgentConfig()).run({"name":"A research study"})
    assert len(items)==1
    assert items[0].citation==record["citation"]
    assert items[0].review!=record["abstract"]
    assert items[0].evidence_quotes==[record["abstract"]]


def test_missing_authors_produce_draft_not_fictional_identities():
    payload = TemplateManager().prepare_payload(title="Study")
    assert payload["authors"] == []
    assert payload["publication_ready"] is False
    assert "DRAFT" in payload["title"]


def test_invalid_manuscript_cannot_be_marked_ready(data, context):
    facts = build_study_facts(data, context)
    payload = TemplateManager().prepare_payload(
        title="Study", authors=[{"name": "Supplied Author"}], context=context,
        study_facts=facts, abstract="The study included N = 9999 observations.",
    )
    assert payload["publication_ready"] is False
    assert any(i["rule"] == "sample_size_match" for i in payload["validation_report"]["issues"])


def test_paper_runs_get_distinct_directories():
    tool = CreateLatexPaperTool()
    first = tool.execute(title="First", compile_pdf=False)
    second = tool.execute(title="Second", compile_pdf=False)
    assert first.success and second.success
    assert first.data["output_dir"] != second.data["output_dir"]


@pytest.mark.parametrize("code", ["__import__('os').getcwd()", "df.to_csv('bad.csv')", "df.__class__", "df.eval('1+1')", "result = df; result.to_csv('bad.csv')"])
def test_dataset_query_rejects_executable_code(data, code):
    assert not DatasetQueryTool().execute(df=data, code=code).success


def test_dataset_query_supports_grouped_summaries(data):
    result = DatasetQueryTool().execute(df=data, code="result = df.groupby('group')['score'].mean()")
    assert result.success and "a" in result.data and "b" in result.data


def test_archive_traversal_is_rejected_before_extraction(tmp_path):
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w:gz") as tar:
        member = tarfile.TarInfo("../escaped.txt")
        member.size = 4
        tar.addfile(member, io.BytesIO(b"test"))
    downloader = TemplateDownloader(str(tmp_path / "templates"))
    with patch("urllib.request.urlopen", return_value=io.BytesIO(archive.getvalue())):
        result = downloader.download_template("https://example.org/template.tar.gz", custom_name="test")
    assert not result["success"]
    assert not (tmp_path / "templates" / "escaped.txt").exists()


def test_memory_updates_merge_across_instances(tmp_path):
    path = str(tmp_path / "memory")
    a, b = MemoryStore(path=path), MemoryStore(path=path)
    a.save_learning("custom", "a", 1)
    b.save_learning("custom", "b", 2)
    assert MemoryStore(path=path).get_all_learnings("custom") == {"a": 1, "b": 2}
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda i: (a if i % 2 else b).record_variable_type_pattern("age", "numeric"), range(10)))
    assert MemoryStore(path=path).get_learning("variable_types", "age")["type_counts"]["numeric"] == 10


def test_review_failure_marks_rendered_payload_as_draft(data, context):
    payload = TemplateManager().prepare_payload(
        title="Study", authors=[{"name": "Investigator"}], context=context,
        study_facts=build_study_facts(data, context), review_gate_passed=False,
    )
    assert payload["publication_ready"] is False
    assert "DRAFT" in payload["title"]
    assert any(issue["rule"] == "review_gate" for issue in payload["validation_report"]["issues"])


def test_pdf_compilation_resolves_bibliography(tmp_path):
    import subprocess
    tex = tmp_path / "paper.tex"
    tex.write_text("test")
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        tex.with_suffix(".aux").write_text(r"\bibdata{references}")
        tex.with_suffix(".pdf").write_bytes(b"pdf")
        return subprocess.CompletedProcess(command, 0, b"", b"")

    def which(name):
        return name if name in {"pdflatex", "bibtex"} else None

    with patch("research_agent.latex_templates.manager.shutil.which", side_effect=which), \
         patch("research_agent.latex_templates.manager.subprocess.run", side_effect=run), \
         patch("sys.executable", str(tmp_path / "python.exe"), create=True):
        assert TemplateManager.compile_to_pdf(tex) == tex.with_suffix(".pdf")
    assert [command[0] for command in commands] == ["pdflatex", "bibtex", "pdflatex", "pdflatex"]
    assert "-no-shell-escape" in commands[0]
