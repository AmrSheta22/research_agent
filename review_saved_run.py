"""Resume complete-document review/export without repeating analyses or source retrieval."""
import argparse
import copy
import json
import logging
from pathlib import Path
import pandas as pd
from research_agent.config import AgentConfig
from research_agent.graph.nodes import ResearchGraphNodes
from research_agent.schemas import ResearchContext, StudyFacts, LiteratureItem
from research_agent.run_reviewed_experiment import save,archive_sources


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("run_directory")
    parser.add_argument("--document-only",action="store_true")
    parser.add_argument("--max-revisions",type=int,default=2)
    parser.add_argument("--refresh-introduction",action="store_true")
    parser.add_argument("--refresh-background",action="store_true")
    args=parser.parse_args()
    out=Path(args.run_directory).resolve()
    state=json.loads((out/"final_state.json").read_text(encoding="utf-8"))
    save(out/"state_before_resumed_review.json",state)
    logging.basicConfig(filename=out/"resumed_review.log",level=logging.INFO,encoding="utf-8",force=True)
    state["context"]=ResearchContext(**state["context"])
    state["context"].analysis_artifact_dir=str(out)
    save(out/"context.json",state["context"])
    archive_sources(out)
    state["study_facts"]=StudyFacts(**state["study_facts"])
    state["literature_review"]=[LiteratureItem(**x) for x in state["literature_review"]]
    state["df"]=pd.read_csv(out/"analysis_input.csv")
    state["output_dir"]=str(out/"paper")
    state["peer_review_retry_count"]=0
    state["max_peer_review_retries"]=args.max_revisions
    config=AgentConfig(memory_path=str(out/"memory"))
    config.model_preference=["groq-qwen","groq-120b","groq-20b","openrouter-inkling"]
    config.custom_models={k:copy.deepcopy(v) for k,v in config.all_models.items()}
    for model in config.custom_models.values():
        model.timeout=90
        model.max_retries=0
        model.max_tokens=(900 if "qwen" in model.model_name else 1800) if model.provider.value=="groq" else 12000
        if model.provider.value=="groq":
            model.reasoning_effort="none" if "qwen" in model.model_name else "low"
    from research_agent.manuscript import abstract_text,methods_text,results_text,discussion_text,conclusion_text
    state["abstract"]=abstract_text(state["study_facts"])
    state["methodology"]=methods_text(state["study_facts"])
    state["test_interpretation"]=results_text(state["study_facts"],state.get("autonomous_analysis"))
    state["discussion"]=discussion_text(state["study_facts"],state.get("autonomous_analysis"))
    state["conclusion"]=conclusion_text(state["study_facts"])
    nodes=ResearchGraphNodes(config=config)
    if args.refresh_background:
        state["background"]=nodes.lit_review_skill.synthesize_background(state["context"].model_dump(),state["literature_review"],state.get("peer_review_critique",{}).get("section_issues",{}).get("background",[]))
    if args.refresh_introduction:
        state["introduction"]=nodes.introduction_skill.run({**state["context"].model_dump(),"literature_review":state["literature_review"],"study_facts":state["study_facts"]})
    print("Resuming review of:",out,flush=True)
    if not args.document_only:
        grounding=nodes.web_grounded_critic_state(state)
        state.update(grounding)
    for attempt in range(3):
        state.update(nodes.final_peer_review_state(state))
        save(out/"review_checkpoint.json",{k:v for k,v in state.items() if k not in {"df","raw_df"}})
        print("Review pass",attempt+1,state.get("peer_review_status"),"components",len(state.get("review_coverage",[])),flush=True)
        if nodes.route_peer_review(state)=="proceed":
            break
        state.update(nodes.actor_revision_state(state))
    state.update(nodes.create_latex_paper_state(state))
    save(out/"final_state.json",state)
    save(out/"summary.json",{k:v for k,v in state.items() if k.endswith("_status") or k in {"errors","publication_ready","validation_report","latex_paper_path","latex_pdf_path","review_coverage"}})
    print((out/"summary.json").read_text(encoding="utf-8"),flush=True)


if __name__=="__main__":
    main()
