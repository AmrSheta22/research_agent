"""Reproduce the reviewed workflow, evidence audit, and numerical analysis artifacts.

Run from the backend directory with python -m research_agent.run_reviewed_experiment.
Use --analysis-only to reproduce numerical facts without network calls.
"""
import argparse
import copy
import hashlib
import json
import logging
import shutil
import sys
import zipfile
import importlib.metadata
from datetime import datetime
from pathlib import Path

import pandas as pd
from research_agent.agent import ResearchAgent
from research_agent.config import AgentConfig
from research_agent.schemas import ResearchContext
from research_agent.study_facts import build_study_facts


def serialize(value):
    if hasattr(value,"model_dump"):
        return value.model_dump()
    if isinstance(value,pd.DataFrame):
        return {"rows":len(value),"columns":list(value.columns)}
    if hasattr(value,"item"):
        return value.item()
    return str(value)


def save(path, value):
    path.write_text(json.dumps(value,default=serialize,ensure_ascii=False,indent=2),encoding="utf-8")


def archive_sources(out):
    package=Path(__file__).resolve().parent
    with zipfile.ZipFile(out/"research_agent_source.zip","w",zipfile.ZIP_DEFLATED) as archive:
        for path in package.rglob("*"):
            if not path.is_file():
                continue
            relative=path.relative_to(package)
            if relative.parts[0] in {"data",".research_agent","__pycache__"}:
                continue
            if path.suffix==".py" or (relative.parts[0]=="latex_templates" and path.suffix==".tex") or path.name in {"requirements.txt","README.md","MANUSCRIPT_FIXES.md"}:
                archive.write(path,Path("research_agent")/relative)
        data=package/"data/noaa_sisal_speleothem_lithology.csv"
        archive.write(data,Path("research_agent/data")/data.name)
    versions={}
    for name in ("numpy","pandas","scipy","statsmodels","pydantic","langchain-core","langchain-openai","langgraph","requests","habanero","tiktoken","beautifulsoup4"):
        try:
            versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name]="not installed"
    save(out/"environment_versions.json",versions)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output-dir")
    parser.add_argument("--analysis-only",action="store_true")
    parser.add_argument("--model",default="openrouter-inkling")
    args=parser.parse_args()
    package=Path(__file__).resolve().parent
    out=Path(args.output_dir).resolve() if args.output_dir else package.parent/"workflow_runs"/datetime.now().strftime("reviewed_%Y%m%d_%H%M%S")
    out.mkdir(parents=True,exist_ok=False)
    logging.basicConfig(filename=out/"workflow.log",level=logging.INFO,encoding="utf-8",force=True)
    csv=package/"data/noaa_sisal_speleothem_lithology.csv"
    data=pd.read_csv(csv)
    context=ResearchContext(
        name="Mean speleothem oxygen-isotope offsets across host-rock lithologies",
        problem_statement="Estimate mean speleothem oxygen-isotope offsets across host-rock lithologies in the SISAL v2 extract, distinguishing mean offsets from within-cave variability and adjusting for elevation, latitude, and cave clustering.",
        hypothesis="Estimate lithology-associated mean offsets and confidence intervals conditional on elevation, latitude, and cave clustering.",
        research_type="cross_sectional",independent_variable="geology",dependent_variable="mean_d18O",
        independent_variable_type="categorical",dependent_variable_type="numeric",
        covariates=["elevation","latitude"],subject_group="site_name",file_path=str(csv),
        dataset_source_name="SISAL v2 speleothem isotope database",dataset_version="SISAL v2 local analytical extract",
        variable_units={"mean_d18O":"per mil VPDB","elevation":"m","latitude":"degrees"},
        provenance_notes=None,analysis_artifact_dir=str(out))
    save(out/"context.json",context)
    shutil.copyfile(csv,out/"analysis_input.csv")
    data[[c for c in ["entity_id","site_name","geology"] if c in data]].to_csv(out/"entity_site_lithology.csv",index=False)
    save(out/"run_manifest.json",{"input_sha256":hashlib.sha256(csv.read_bytes()).hexdigest(),"rows":len(data),"columns":list(data),
        "python":sys.version,"selection_boundary":"The supplied extract starts at 561 rows; upstream exclusion counts and original free-text lithology mapping are unavailable.",
        "coding":"Observed geology labels are preserved; entity_site_lithology.csv records existing assignments, not a reconstructed mapping from original free text.",
        "public_repository":None,"analysis_command":"python -m research_agent.run_reviewed_experiment --analysis-only"})
    shutil.copyfile(__file__,out/"run_reviewed_experiment.py")
    shutil.copyfile(package/"requirements.txt",out/"requirements.txt")
    archive_sources(out)
    print("Run directory:",out,flush=True)
    if args.analysis_only:
        save(out/"study_facts.json",build_study_facts(data,context,raw_df=data))
        print("Numerical analysis complete",flush=True)
        return
    config=AgentConfig(memory_path=str(out/"memory"))
    config.model_preference=[args.model]+[k for k in config.model_preference if k not in {args.model,"cohere-command-a-plus"}]
    config.custom_models={k:copy.deepcopy(v) for k,v in config.all_models.items()}
    for model in config.custom_models.values():
        model.timeout=90
        model.max_retries=0
        model.max_tokens=(900 if "qwen" in model.model_name else 1800) if model.provider.value == "groq" else 12000
        if model.provider.value=="groq":
            model.reasoning_effort="none" if "qwen" in model.model_name else "low"
    try:
        agent=ResearchAgent(config)
        state=agent.run_graph(context=context,df=data,paper_template="nature",output_dir=str(out/"paper"),compile_pdf=True)
        save(out/"final_state.json",state)
        save(out/"study_facts.json",state.get("study_facts"))
        save(out/"source_evidence.json",state.get("literature_review",[]))
        save(out/"data_provenance.json",state.get("autonomous_analysis",{}).get("data_provenance",{}))
        save(out/"summary.json",{k:v for k,v in state.items() if k.endswith("_status") or k in {"errors","publication_ready","validation_report","latex_paper_path","latex_pdf_path","review_coverage"}})
        print((out/"summary.json").read_text(encoding="utf-8"),flush=True)
    except Exception:
        import traceback
        (out/"failure.txt").write_text(traceback.format_exc(),encoding="utf-8")
        raise


if __name__=="__main__":
    main()
