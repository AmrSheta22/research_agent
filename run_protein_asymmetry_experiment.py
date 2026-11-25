"""Run the full ResearchAgent StateGraph pipeline on the Zenodo protein chain asymmetry dataset."""

import argparse
import copy
import hashlib
import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from research_agent.agent import ResearchAgent
from research_agent.config import AgentConfig
from research_agent.schemas import ResearchContext
from research_agent.study_facts import build_study_facts


def serialize(value):
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, pd.DataFrame):
        return {"rows": len(value), "columns": list(value.columns)}
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def save(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, default=serialize, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir")
    parser.add_argument("--analysis-only", action="store_true")
    parser.add_argument("--model", default="openrouter-inkling")
    args = parser.parse_args()

    package = Path(__file__).resolve().parent
    out = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else package.parent / "workflow_runs" / datetime.now().strftime("protein_asymmetry_%Y%m%d_%H%M%S")
    )
    out.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(filename=out / "workflow.log", level=logging.INFO, encoding="utf-8", force=True)

    csv = package / "data" / "protein_chain_asymmetry.csv"
    data = pd.read_csv(csv)

    context = ResearchContext(
        name="Local structural asymmetry across crystal symmetry groups in identical-sequence protein chains",
        problem_statement=(
            "Investigating whether local structural asymmetry (ASYM) between identical-sequence protein "
            "chain pairs differs across crystallographic symmetry groups (CristGenGroup), controlling "
            "for X-ray resolution (RESOLUTION), sequence length (SEQ_LENGTH), and protein-structure clustering (PROT)."
        ),
        hypothesis=(
            "Estimate mean local structural asymmetry offsets across crystallographic symmetry groups "
            "with confidence intervals, adjusting for X-ray resolution, sequence length, and PDB structure clustering."
        ),
        research_type="cross_sectional",
        independent_variable="CristGenGroup",
        dependent_variable="ASYM",
        independent_variable_type="categorical",
        dependent_variable_type="numeric",
        covariates=["RESOLUTION", "SEQ_LENGTH"],
        subject_group="PROT",
        file_path=str(csv),
        dataset_source_name="Zenodo Protein Chain Structural Asymmetry Dataset",
        dataset_version="10.5281/zenodo.24504161",
        data_repository_url="https://doi.org/10.5281/zenodo.24504161",
        variable_units={
            "ASYM": "fraction of asymmetric residues",
            "RESOLUTION": "Angstrom",
            "SEQ_LENGTH": "amino acid residues",
        },
        provenance_notes=(
            "Public Zenodo deposit (DOI: 10.5281/zenodo.24504161, asymetry_perProt.csv) containing "
            "1,558 identical-sequence chain-pair comparisons across 301 PDB structures."
        ),
        analysis_artifact_dir=str(out),
    )

    save(out / "context.json", context)
    shutil.copyfile(csv, out / "analysis_input.csv")
    save(
        out / "run_manifest.json",
        {
            "input_sha256": hashlib.sha256(csv.read_bytes()).hexdigest(),
            "rows": len(data),
            "columns": list(data),
            "python": sys.version,
            "source_doi": "10.5281/zenodo.24504161",
        },
    )

    print("Run directory:", out, flush=True)
    if args.analysis_only:
        save(out / "study_facts.json", build_study_facts(data, context, raw_df=data))
        print("Numerical analysis complete", flush=True)
        return

    config = AgentConfig(memory_path=str(out / "memory"))
    config.model_preference = [args.model] + [
        k for k in config.model_preference if k not in {args.model, "cohere-command-a-plus"}
    ]
    config.custom_models = {k: copy.deepcopy(v) for k, v in config.all_models.items()}
    for model in config.custom_models.values():
        model.timeout = 90
        model.max_retries = 0
        model.max_tokens = (
            (900 if "qwen" in model.model_name else 1800)
            if model.provider.value == "groq"
            else 12000
        )
        if model.provider.value == "groq":
            model.reasoning_effort = "none" if "qwen" in model.model_name else "low"

    try:
        agent = ResearchAgent(config)
        state = agent.run_graph(
            context=context,
            df=data,
            paper_template="nature",
            output_dir=str(out / "paper"),
            compile_pdf=True,
        )
        save(out / "final_state.json", state)
        save(out / "study_facts.json", state.get("study_facts"))
        save(out / "source_evidence.json", state.get("literature_review", []))
        save(out / "data_provenance.json", state.get("autonomous_analysis", {}).get("data_provenance", {}))
        save(
            out / "summary.json",
            {
                k: v
                for k, v in state.items()
                if k.endswith("_status")
                or k
                in {
                    "errors",
                    "publication_ready",
                    "validation_report",
                    "latex_paper_path",
                    "latex_pdf_path",
                    "review_coverage",
                }
            },
        )
        print((out / "summary.json").read_text(encoding="utf-8"), flush=True)
    except Exception:
        import traceback

        (out / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
