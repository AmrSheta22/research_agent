"""Contracts that failed in the reviewed manuscript, including complete-document coverage."""
import json
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
from research_agent.config import AgentConfig
from research_agent.evidence import canonical_record, clean_text, title_matches
from research_agent.review import complete_batches
from research_agent.schemas import ResearchContext
from research_agent.study_facts import build_study_facts
from research_agent.manuscript import coefficient_table, abstract_text
from research_agent.skills.literature_review import LiteratureReviewSkill
from research_agent.tools.search_tools import AcademicLiteratureSearchTool
from research_agent.tools.analysis_tools import AutonomousAnalysisTool
from research_agent.tools.base import ToolResult
from research_agent.latex_templates.formatter import generate_bibtex
from research_agent.graph.nodes import ResearchGraphNodes
ACADEMIC_EXECUTE = AcademicLiteratureSearchTool.execute


def test_review_batches_preserve_every_character_and_tail():
    sections={"background":"A"*50001+"TAIL_CRITICAL_ERROR", "tables":"TABLE_ERROR", "declarations":"DISCLOSURE_ERROR"}
    batches=list(complete_batches(sections,max_chars=10000))
    recovered="".join(text for batch in batches for text in batch.values())
    assert recovered=="".join(sections.values())
    assert "TAIL_CRITICAL_ERROR" in recovered
    assert all(sum(map(len,b.values()))<=10000 for b in batches)


def test_adaptive_splitting_preserves_unicode_and_all_text():
    from research_agent.review import split_batch
    text="δ18O \\beta [0.1234,0.5678] "*100
    parts=split_batch({"dense table":text})
    assert "".join(value for part in parts for value in part.values())==text


def test_logical_review_components_preserve_whole_abstract_and_table():
    from research_agent.review import document_components
    abstract=r"\begin{abstract}Complete abstract sentence.\end{abstract}"
    table=r"\begin{table}\caption{Estimates}\begin{tabular}{l}complete rows\end{tabular}\end{table}"
    source="preamble\n"+abstract+r"\section{Results}"+"Complete results.\n"+table+r"\section{Methods}"+"Complete methods."
    components=document_components(source)
    assert "".join(components.values())==source
    assert abstract in components.values()
    assert table in components.values()


def test_methods_audit_includes_actual_cave_exclusion_results():
    from research_agent.review import component_audit
    from unittest.mock import MagicMock
    facts=MagicMock()
    facts.model_dump.return_value={"sample_size_total":561,"additional_diagnostics":{"test":"computed"}}
    audit=component_audit(facts,"Methods",{"sensitivity_analysis":{"sensitivity_by_level":{"marble":{"clusters_tested":11}}}})
    assert audit["cave_exclusion_audit"]["sensitivity_by_level"]["marble"]["clusters_tested"]==11


def test_doi_identity_is_exact_not_a_prefix_match():
    from research_agent.evidence import identity_issues
    source={"doi":"10.1234/real","citation":"A (2020) https://doi.org/10.1234/really-different", "metadata_verified":True,"evidence_quotes":["Evidence"],"reviewed_source":"Evidence"}
    assert any("differs" in issue for issue in identity_issues([source]))


def test_identity_checker_resolves_doi_instead_of_trusting_a_flag():
    from research_agent.evidence import verify_registry_identity
    message={"title":["Actual registry title"],"author":[{"family":"Actual"}],"issued":{"date-parts":[[2020]]},"container-title":["Real journal"],"DOI":"10.1234/real"}
    forged={"doi":"10.1234/real","title":"Unrelated invented title","authors":[{"family":"Fake"}],"year":"2023","journal":"Invented journal","metadata_verified":True}
    with patch("habanero.Crossref") as registry:
        registry.return_value.works.return_value={"message":message}
        issues=verify_registry_identity(forged)
        registry.return_value.works.assert_called_once_with(ids="10.1234/real")
    assert "title, year, authors, venue" in issues[0]


def test_final_review_includes_new_diagnostics_in_audit():
    from research_agent.review import review_document
    from research_agent.skills.critic import FinalPeerReviewCritic
    from unittest.mock import MagicMock
    facts=MagicMock()
    facts.model_dump.return_value={"additional_diagnostics":{"cave_model_test":"computed evidence"},"cluster_robust_ols":{"term":{"std_err":.3}}}
    skill=FinalPeerReviewCritic(AgentConfig())
    with patch("research_agent.validation.paper_validator.PaperValidator") as validator,patch.object(skill,"_invoke_with_fallback",return_value='{"passed":true,"critical_issues":[],"target_sections_to_revise":[]}') as invoke:
        validator.return_value.validate_section.return_value=[]
        result=review_document(skill,{"assembled_manuscript":r"\section{Results} FULL DOCUMENT TAIL"},facts,"COMPLETE BIBLIOGRAPHY")
    prompt="\n".join(call.args[0][0].content for call in invoke.call_args_list)
    assert "computed evidence" in prompt and "cluster_robust_ols" in prompt
    assert "FULL DOCUMENT TAIL" in prompt and "COMPLETE BIBLIOGRAPHY" in prompt
    assert result.passed


def test_review_findings_require_actual_manuscript_evidence():
    from research_agent.review import grounded_findings
    findings=grounded_findings([{"section":"results","quote":"invented text","problem":"Missing Table 1"}],"Table 1 is present.")
    assert "Missing Table 1" not in findings[0]
    assert "ungrounded" in findings[0]
    valid=grounded_findings([{"section":"declarations","quote":"Authors not supplied","problem":"Missing author identities"}],"Authors not supplied")
    assert "Missing author identities" in valid[0]
    false_missing=grounded_findings([{"section":"discussion","quote":"See Table 1","problem":"Table 1 is not present in this batch","fix":"add table"}],"See Table 1",{"labels":["tab:estimates"]})
    assert false_missing==[]


def test_review_retries_temporary_quota_with_bounded_wait():
    from research_agent.review import invoke_review
    from unittest.mock import MagicMock
    skill=MagicMock()
    skill._invoke_with_fallback.side_effect=[RuntimeError("429"),"review response"]
    skill.config.get_fallback_chain.return_value=[object()]
    skill.config.cooldown_manager.get_cooldown_remaining.return_value=.2
    with patch("time.sleep") as sleep:
        assert invoke_review(skill,"prompt")=="review response"
    assert 0<sleep.call_args.args[0]<=60


def test_explicit_reasoning_budget_reaches_chat_provider():
    from research_agent.config import ModelConfig,LLMProvider
    model=ModelConfig(provider=LLMProvider.GROQ,model_name="openai/gpt-oss-20b",reasoning_effort="low")
    with patch("langchain_openai.ChatOpenAI") as chat:
        AgentConfig()._build_single_chat(model,0)
    assert chat.call_args.kwargs["reasoning_effort"]=="low"


def test_registry_metadata_requires_identity_and_keeps_structured_venue():
    item={"title":["Actual Title"],"author":[{"family":"Author","given":"A","authenticated-orcid":False,"affiliation":[],"role":[{"role":"author"}]}],"issued":{"date-parts":[[2022]]},"DOI":"10.1234/Actual", "container-title":["Actual Journal"],"type":"journal-article","abstract":"<p>Source &#8216;finding&#8217;.</p>"}
    record=canonical_record(item)
    from research_agent.schemas import LiteratureItem
    LiteratureItem(**{k:v for k,v in record.items() if k in LiteratureItem.model_fields},review="Reviewed summary")
    assert record["abstract"]=="Source ‘finding’."
    assert record["doi"]=="10.1234/actual"
    bib,_=generate_bibtex([record])
    assert "journal = {Actual Journal}" in bib
    assert "Scientific Reports" not in bib
    assert canonical_record({"title":["Missing authors"]}) is None
    assert not title_matches("Different scientific work",record["title"])


def test_unverified_metadata_never_gets_invented_bibtex():
    assert generate_bibtex([{"title":"No identity","citation":"bad","review":"text"}]) == ("",{})


def test_regression_p_value_map_is_never_a_dictionary_dump():
    from research_agent.latex_templates.formatter import format_statistical_table
    table=format_statistical_table({"OLS regression":{"p_values":{"const":1e-9,"group":.2}}})
    assert "p_values" not in table
    assert "'const'" not in table
    assert "0.2000" in table


def test_search_uses_title_relevance_instead_of_broad_registry_text():
    from unittest.mock import MagicMock
    response=MagicMock()
    response.json.return_value={"message":{"items":[]}}
    with patch("requests.get",return_value=response) as request:
        result=ACADEMIC_EXECUTE(AcademicLiteratureSearchTool(),query="core topic",limit=6)
    assert result.success
    assert request.call_args.kwargs["params"]["query.title"]=="core topic"
    assert request.call_args.kwargs["params"]["filter"]=="type:journal-article"


def test_search_explores_distinct_planned_topics():
    responses=[json.dumps({"queries":["physical controls","global evidence","source dataset"]})]+[json.dumps({"accepted":[],"next_query":"physical controls"})]*3
    with patch.object(AcademicLiteratureSearchTool,"execute",return_value=ToolResult.ok([])) as search, patch.object(LiteratureReviewSkill,"_invoke_with_fallback",side_effect=responses):
        assert LiteratureReviewSkill(AgentConfig()).run({"name":"Paper"})==[]
    assert [call.kwargs["query"] for call in search.call_args_list]==["physical controls","global evidence","source dataset"]


@pytest.mark.parametrize("quote",["Invented support for this claim.",""])
def test_source_review_rejects_unsubstantiated_claim(quote):
    record={"title":"Cave study","doi":"10.1234/cave","citation":"A (2020)","work_type":"journal-article","abstract":"These measurements describe cave isotope variability."}
    decision={"accepted":[{"doi":record["doi"],"review":"Invented scientific finding.","relevance_reason":"Relevant", "evidence_quotes":[quote]}],"next_query":""}
    with patch.object(AcademicLiteratureSearchTool,"execute",return_value=ToolResult.ok([record])),patch.object(LiteratureReviewSkill,"_invoke_with_fallback",return_value=json.dumps(decision)):
        assert LiteratureReviewSkill(AgentConfig()).run({"name":"Cave mean isotope study"})==[]


def test_site_name_precedes_entity_id_in_all_auxiliary_analyses():
    data=pd.DataFrame({"entity_id":range(12),"site_name":["a","b","c"]*4,"group":["x"]*6+["y"]*6,"score":np.arange(12.)})
    tool=AutonomousAnalysisTool()
    with patch.object(tool,"_dispatch_operation",return_value=ToolResult.ok({})) as dispatch:
        tool.execute(df=data,iv="group",dv="score",analyses=["cohort_and_covariates"])
    assert dispatch.call_args.args[4]=="site_name"


def test_descriptives_cannot_masquerade_as_mineralogy_age_controls():
    data=pd.DataFrame({"group":["a","a","b","b"],"score":[1.,2.,3.,4.]})
    result=AutonomousAnalysisTool().execute(df=data,iv="group",dv="score",operation="mineralogy_and_age_controls")
    assert not result.success
    assert "not implemented" in result.error


def test_failed_cave_exclusions_cannot_claim_stability():
    from unittest.mock import MagicMock
    data=pd.DataFrame({"group":["ref"]*15+["rare"]*5,"score":np.arange(20.),"site_name":np.repeat(["a","b","c","d"],5)})
    base=MagicMock(converged=True)
    base.fe_params={"C(group, Treatment(reference='ref'))[T.rare]":.5}
    with patch("research_agent.tools.analysis_tools.smf.mixedlm") as model:
        model.return_value.fit.side_effect=[base,ValueError("did not converge")]
        result=AutonomousAnalysisTool()._compute_leave_one_cluster_out(data,"group","score","site_name",[])
    level=result.data["sensitivity_by_level"]["rare"]
    assert level["successful_fits"]==0
    assert level["jackknife_coefficient_range"] is None
    assert level["stability"]=="No converged exclusion fits"


def test_conclusion_distinguishes_primary_model_from_significant_sensitivity():
    from types import SimpleNamespace
    from research_agent.manuscript import conclusion_text
    facts=SimpleNamespace(factor_name="group",lmm_executed=True,
        lmm_fixed_effects=[SimpleNamespace(parameter="group: mixed",p_value=.6)],
        regression_coefficients=[],cluster_robust_ols={"group: mixed":{"p_value":.04}})
    conclusion=conclusion_text(facts)
    assert "primary random-intercept mixed model" in conclusion
    assert "Cluster-robust OLS supported at least one contrast" in conclusion
    assert "inference depends on model specification" in conclusion
    assert len(conclusion.split())<=120


def test_dataset_access_uses_reviewed_version_not_another_release():
    from research_agent.provenance import availability_statements
    sources=[{"title":"Datasetv3","reviewed_source":"Datasetv3 is available at https://doi.org/10.1234/wrong"},
             {"title":"Datasetv2: documentation","reviewed_source":"Datasetv2 is available at https://doi.org/10.1234/right"}]
    data,code=availability_statements({"dataset_source_name":"Dataset v2","dataset_version":"v2 local extract"},sources)
    assert "10.1234/right" in data
    assert "10.1234/wrong" not in data
    assert "no public archive of the local extract" in data


def test_successful_cave_fits_do_not_claim_failures():
    from types import SimpleNamespace
    from research_agent.manuscript import discussion_text
    facts=SimpleNamespace(factor_name="group",lmm_executed=True,
        lmm_fixed_effects=[SimpleNamespace(parameter="group: mixed",p_value=.6)],
        regression_coefficients=[],cluster_robust_ols={},additional_diagnostics={})
    text=discussion_text(facts,{"sensitivity_analysis":{"sensitivity_by_level":{"mixed":{"failed_fits":[]}}}})
    assert "All attempted exclusion fits converged" in text
    assert "Reported convergence failures" not in text


def test_background_uses_only_vetted_doi_identifiers():
    from research_agent.schemas import LiteratureItem
    source=LiteratureItem(title="Actual study",citation="A (2020)",review="Supported finding",doi="10.1234/real")
    bad="Unsupported "*110+"[@doi:10.1234/fake]"
    with patch.object(LiteratureReviewSkill,"_invoke_with_fallback",return_value=bad):
        text=LiteratureReviewSkill(AgentConfig()).synthesize_background({"name":"Paper"},[source])
    assert "10.1234/fake" not in text
    assert "[@doi:10.1234/real]" in text


def test_variables_table_reports_units_levels_and_clusters():
    from research_agent.latex_templates.formatter import format_variables_table
    table=format_variables_table({"independent_variable":"group","dependent_variable":"score","covariates":["height"],"variable_units":{"height":"m","score":"units"},"num_clusters_total":12,"factor_levels":["a","b"],"reference_level":"a"})
    assert "height (m)" in table
    assert "12 unique clusters" in table
    assert "reference: a" in table


def test_provider_retry_after_is_respected():
    from research_agent.config import CooldownManager
    manager=CooldownManager()
    manager.record_failure("provider",RuntimeError("429 Rate limit. Please try again in 472.5ms."))
    assert 0<manager.get_cooldown_remaining("provider")<1


def test_truncated_model_output_is_not_accepted_as_complete():
    from research_agent.config import ModelConfig,LLMProvider
    from types import SimpleNamespace
    from unittest.mock import MagicMock
    config=AgentConfig()
    models=[ModelConfig(provider=LLMProvider.GROQ,model_name="truncated"),ModelConfig(provider=LLMProvider.GROQ,model_name="complete")]
    first=MagicMock()
    first.invoke.return_value=SimpleNamespace(content="Incomplete scientific sentence",response_metadata={"finish_reason":"length"})
    second=MagicMock()
    second.invoke.return_value=SimpleNamespace(content="Complete scientific sentence.",response_metadata={"finish_reason":"stop"})
    skill=LiteratureReviewSkill(config)
    with patch.object(config,"get_fallback_chain",return_value=models),patch.object(skill,"_get_llm",side_effect=[first,second]):
        assert skill._invoke_with_fallback([])=="Complete scientific sentence."


def test_clustered_estimates_and_diagnostics_feed_table():
    rng=np.random.default_rng(71)
    data=pd.DataFrame({"entity_id":range(120),"site_name":np.repeat(np.arange(30),4),"group":np.tile(["a","b"],60),"score":rng.normal(size=120),"latitude":rng.normal(size=120)})
    context=ResearchContext(research_type="cross_sectional",independent_variable="group",dependent_variable="score",covariates=["latitude"])
    facts=build_study_facts(data,context)
    assert facts.cluster_variable=="site_name"
    assert facts.num_clusters_total==30
    assert facts.cluster_robust_ols
    assert "entities_per_cluster" in facts.additional_diagnostics
    assert "extreme_value_sensitivity" in facts.additional_diagnostics
    assert set(facts.mdes_by_factor_level).issubset({"b (vs a)","b"})
    table=coefficient_table(facts)
    assert "Cluster robust" in table and "HC3" in table
    assert "p_values" not in table
    assert len(abstract_text(facts).split())<=220


def test_revision_updates_all_requested_sections():
    node=ResearchGraphNodes.__new__(ResearchGraphNodes)
    class Actor:
        def revise_section(self,**kwargs):
            return "fixed "+kwargs["section_name"]
    node.paper_actor_reviser=Actor()
    state={"peer_review_critique":{"issues":["issues"],"target_sections":["introduction","results","discussion"]},"study_facts":object()}
    result=node.actor_revision_state(state)
    assert result["introduction"]=="fixed introduction"
    assert result["test_interpretation"]=="fixed results"
    assert result["discussion"]=="fixed discussion"
    assert result["peer_review_retry_count"]==1


def test_revision_receives_only_its_section_findings():
    node=ResearchGraphNodes.__new__(ResearchGraphNodes)
    from unittest.mock import MagicMock
    node.paper_actor_reviser=MagicMock()
    state={"peer_review_critique":{"target_sections":["abstract","discussion"],"section_issues":{"abstract":["abstract issue"],"discussion":["discussion issue"]}},"study_facts":object()}
    node.actor_revision_state(state)
    calls=node.paper_actor_reviser.revise_section.call_args_list
    assert calls[0].kwargs["issues"]==["abstract issue"]
    assert calls[1].kwargs["issues"]==["discussion issue"]


def test_introduction_fallback_is_substantive_and_does_not_invent_a_gap():
    from research_agent.skills.introduction import IntroductionSkill
    text=IntroductionSkill._grounded_fallback({"independent_variable":"group","dependent_variable":"score","covariates":["age"]},[])
    assert len(text.split())>=150
    assert "first global" not in text.lower()
    assert "confidence intervals" in text
