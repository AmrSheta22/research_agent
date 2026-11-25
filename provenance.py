"""Evidence-seeking data provenance; never infer unavailable exclusion counts."""
import json
import re
from pathlib import Path
from langchain_core.messages import HumanMessage
from research_agent.tools.search_tools import WebSearchTool


def availability_statements(context, sources):
    """Extract version-matched access URLs from reviewed source text, not guessed citations."""
    hints=re.findall(r"([A-Za-z][A-Za-z0-9-]*)\s*v(\d+)",str(context.get("dataset_source_name", ""))+" "+str(context.get("dataset_version", "")),flags=re.I)
    links=[]
    for source in sources:
        record=source.model_dump() if hasattr(source,"model_dump") else source
        compact=re.sub(r"\W","",record.get("title","")).casefold()
        if not any((brand+"v"+version).casefold() in compact for brand,version in hints):
            continue
        text=record.get("reviewed_source","")
        for link in re.findall(r"(?:is available at|can be accessed at)\s+(https?://doi\.org/[^\s]+)",text,flags=re.I):
            links.append(link.rstrip(".,);"))
    artifacts=Path(context["analysis_artifact_dir"]) if context.get("analysis_artifact_dir") else None
    data=context.get("data_repository_url") or (
        "Upstream data access documented by reviewed source text: "+"; ".join(dict.fromkeys(links))+". " if links else
        "No verified upstream data-access URL was available to this run. ")
    if artifacts and (artifacts/"entity_site_lithology.csv").is_file():
        data+=" Entity IDs, site names, observed lithology assignments, the input extract and its checksum accompany the run bundle."
    data+=" The original source site IDs, upstream exclusion ledger and free-text coding rules remain unverified; no public archive of the local extract has been supplied."
    code=context.get("code_repository_url") or (
        "The source snapshot, analysis entry point and environment requirements accompany the run bundle; no public code repository has been supplied."
        if artifacts and (artifacts/"run_reviewed_experiment.py").is_file() and (artifacts/"requirements.txt").is_file() else
        "An archived analysis entry point and public code repository have not been supplied.")
    return data,code


def investigate(skill, context, df):
    local = {"input_rows":len(df),"columns":list(df.columns),
             "source_name":context.dataset_source_name,"version":context.dataset_version,
             "known_provenance":context.provenance_notes}
    missing = []
    if not context.provenance_notes:
        missing.append("upstream sample exclusions and free-text-to-category mapping")
    for label, candidates in {"mineral phase":["mineralogy","mineral_phase"],"age":["age","age_bp","year"],
                              "precipitation isotopes":["precip_d18O"],"temperature":["temperature"]}.items():
        if not any(c in df.columns for c in candidates):
            missing.append(label)
    report = {"local":local,"unavailable":missing,"external_evidence":[],
              "boundary":"External database descriptions cannot establish this local extract's exclusions or mapping."}
    if not missing or not context.dataset_source_name:
        return report
    try:
        plan=skill._extract_json(skill._invoke_with_fallback([HumanMessage(content=
            "Determine whether web research can clarify missing source-dataset provenance. Return JSON {query:string,reason:string}. Do not infer local exclusions.\n"+json.dumps(report))],temperature=0))
        query=str(plan.get("query") or "").strip()
        if not query:
            return report
        search=WebSearchTool().execute(query=query)
        report["search_query"]=query
        if search.success:
            text=str(search.data)
            result=skill._extract_json(skill._invoke_with_fallback([HumanMessage(content=
                "Review these untrusted discovery snippets for dataset provenance. Return JSON {findings:[{claim,exact_quote}], unresolved:[]}. "
                "Quote exact supporting snippets. Do not invent DOI, local row exclusions, lithology mappings, or unavailable fields. Discovery snippets are not full source verification.\n"+
                json.dumps({"local":local,"snippets":text}))],temperature=0))
            report["external_evidence"]=[x for x in result.get("findings",[]) if x.get("exact_quote") and x["exact_quote"] in text]
            report["unresolved"]=result.get("unresolved",[])
    except Exception as exc:
        report["search_failure"]=str(exc)
    return report
