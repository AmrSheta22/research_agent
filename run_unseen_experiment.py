"""
Execution script to test the complete Agentic StateGraph on unseen NOAA SISAL paleoclimate data.
Tests:
  - Data ingestion, type inference, and cleaning
  - LATS cyclic retry loop (evaluating empirical p-values & retention)
  - Statistical testing and interpretation
  - Literature review and contextual background synthesis
  - Academic methodology and abstract writing
  - Visualization planning
  - Quality Gate validation and refinement
  - Continuous learning memory storage
  - Final publication-ready LaTeX academic paper generation
"""

import os
import sys
import json
import pandas as pd

sys.path.insert(0, os.path.abspath("."))

from research_agent.agent import ResearchAgent
from research_agent.schemas import LiteratureItem, ResearchContext

def main():
    agent = ResearchAgent()

    csv_path = os.path.abspath("research_agent/data/noaa_sisal_speleothem_lithology.csv")
    df = pd.read_csv(csv_path)

    context = ResearchContext(
        name="Global Speleothem Calcite Isotope Analysis Across Karst Bedrock Lithologies",
        problem_statement="Investigating whether subterranean speleothem calcite stable oxygen isotope compositions (mean d18O) differ significantly across distinct karst host-rock lithologies (limestone, dolomite, marble, and mixed carbonate-siliciclastic), controlling for topographic elevation and geographic latitude.",
        research_type="cross_sectional",
        independent_variable="geology",
        dependent_variable="mean_d18O",
        independent_variable_type="categorical",
        dependent_variable_type="numeric",
        covariates=["elevation", "latitude"],
        hypothesis="Estimate mean speleothem oxygen-isotope offsets between lithologies with confidence intervals, adjusting for elevation, latitude, and cave clustering.",
        file_path=csv_path,
        dataset_source_name="NOAA NCEI Paleoclimatology / SISAL v2 Database",
        dataset_version="v2.0 (Comas-Bru et al., 2020)",
        variable_units={
            "mean_d18O": "‰ VPDB",
            "elevation": "m a.s.l.",
            "latitude": "° decimal",
        },
        factor_level_definitions={
            "limestone": "Calcitic limestone karst host bedrock (reference baseline)",
            "dolomite": "Dolomitic / Mg-rich carbonate host bedrock",
            "marble": "Metamorphic marble carbonate host bedrock",
            "mixed": "Mixed carbonate-siliciclastic host rock lithology",
        },
        unmeasured_confounders_note="bulk speleothem records not screened for aragonite versus calcite polymorphism",
        sampling_boundaries_note="composite mean records spanning multi-millennial temporal windows without Holocene-only filtering",

    )

    paper_dir = os.path.abspath("research_agent/data/speleothem_paper_output")
    os.makedirs(paper_dir, exist_ok=True)

    print("=" * 70)
    print("RUNNING AGENTIC STATEGRAPH ON UNSEEN NOAA SISAL DATASET")
    print("=" * 70)
    print(f"Dataset path: {csv_path}")
    print(f"Input records: {len(df)} speleothems across global caves")
    print(f"Features: {df.columns.tolist()}")
    print(f"Hypothesis: {context.hypothesis}")
    print("=" * 70)

    final_state = agent.run_graph(
        context=context,
        df=df,
        paper_template="nature",
        output_dir=paper_dir,
        compile_pdf=True,
    )

    print("\n" + "=" * 70)
    print("1. EMPIRICAL CORE & LATS CYCLIC RETRY RESULTS")
    print("=" * 70)
    print(f"LATS Route Decision:    {final_state.get('lats_route_decision')}")
    print(f"LATS Retries Executed: {final_state.get('lats_retry_count', 0)}")
    print(f"Cleaning Strategy Used: {final_state.get('cleaning_strategy')}")
    retention = final_state.get('retention_ratio', 1.0)
    print(f"Sample Retention Ratio: {retention:.1%}")
    print(f"Min p-value Evaluated:  {final_state.get('min_p_value')}")
    print(f"LATS Reflection:        {final_state.get('lats_reflection')}")

    print("\n" + "=" * 70)
    print("2. STATISTICAL TEST OUTCOMES")
    print("=" * 70)
    test_results = final_state.get("test_results", {})
    for t_name, t_data in test_results.items():
        print(f"\n[Test: {t_name}]")
        if isinstance(t_data, dict):
            for k, v in t_data.items():
                print(f"  - {k}: {v}")
        else:
            print(f"  {t_data}")

    print("\n" + "=" * 70)
    print("3. TEST INTERPRETATION")
    print("=" * 70)
    print(final_state.get("test_interpretation"))

    print("\n" + "=" * 70)
    print("4. PUBLICATION QUALITY GATE EVALUATION")
    print("=" * 70)
    print(f"Quality Status:         {final_state.get('quality_status')}")
    print(f"Quality Retries:        {final_state.get('quality_retry_count', 0)}")
    print(f"Methodology Passed:     {final_state.get('methodology_quality_passed')}")
    print(f"Abstract Passed:        {final_state.get('abstract_quality_passed')}")
    print(f"Critique / Feedback:    {final_state.get('quality_critique')}")

    print("\n" + "=" * 70)
    print("5. VISUALIZATIONS PLANNED")
    print("=" * 70)
    vis_specs = final_state.get("visualizations", [])
    for idx, v in enumerate(vis_specs, 1):
        if hasattr(v, "model_dump"):
            vd = v.model_dump()
        elif isinstance(v, dict):
            vd = v
        else:
            vd = {"title": str(v)}
        print(f" - Visualization {idx}: {vd.get('title')} [{vd.get('mark')}] (X: {vd.get('x_axis')}, Y: {vd.get('y_axis')})")

    print("\n" + "=" * 70)
    print("6. GENERATED ABSTRACT")
    print("=" * 70)
    print(final_state.get("abstract"))

    print("\n" + "=" * 70)
    print("7. GENERATED METHODOLOGY")
    print("=" * 70)
    print(final_state.get("methodology"))

    print("\n" + "=" * 70)
    print("8. GENERATED DISCUSSION & CONCLUSION")
    print("=" * 70)
    print("--- DISCUSSION ---")
    print(final_state.get("discussion"))
    print("\n--- CONCLUSION ---")
    print(final_state.get("conclusion"))

    print("\n" + "=" * 70)
    print("9. PAIRED REFLEXION & CROSS-SECTION CRITIQUE SUITE")
    print("=" * 70)
    print("Methods <-> Results Critique:", json.dumps(final_state.get("methods_results_critique"), indent=2))
    print("Results <-> Discussion Critique:", json.dumps(final_state.get("results_discussion_critique"), indent=2))
    print("Intro <-> Conclusion Critique:", json.dumps(final_state.get("intro_conclusion_critique"), indent=2))
    print("Global Paper Critique:", json.dumps(final_state.get("global_paper_critique"), indent=2))
    print("Reflexion History:", json.dumps(final_state.get("reflexion_history"), indent=2))

    print("\n" + "=" * 70)
    print("10. FINAL LATEX PAPER PACKAGE")
    print("=" * 70)
    print(f"Paper Generation Status: {final_state.get('latex_paper_status')}")
    print(f"TeX File Path:           {final_state.get('latex_paper_path')}")
    print(f"PDF File Path:           {final_state.get('latex_pdf_path')}")

    tex_path = final_state.get("latex_paper_path")
    if tex_path and os.path.exists(tex_path):
        bib_path = os.path.join(os.path.dirname(tex_path), "references.bib")
        tex_size = os.path.getsize(tex_path)
        bib_size = os.path.getsize(bib_path) if os.path.exists(bib_path) else 0
        print(f"main.tex size:           {tex_size} bytes")
        print(f"references.bib size:     {bib_size} bytes")

        with open(tex_path, "r", encoding="utf-8") as f:
            tex_content = f.read()

        print("\n--- LaTeX Source Preview (first 800 chars) ---")
        print(tex_content[:800])
        print("...\n--- End of Preview ---")

    print("\n" + "=" * 70)
    print("EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    main()
