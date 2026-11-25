# Manuscript integrity changes

1. Source discovery now uses up to three agent-directed search/review rounds. Candidate abstracts are read in full. Accepted summaries require exact source quotations and an explicit relevance explanation. Search snippets are discovery context, not paper evidence. Missing abstracts or rejected evidence do not become Background placeholders.
2. Registration is separated from relevance and scientific quality. Source screening explicitly rejects unrelated provider material and unsupported work. Abstract review is labelled as such; the workflow does not claim full-text verification.
3. Built-in domain and methodological reference lists were removed. Structured Crossref author/title/year/venue/DOI fields generate BibTeX without guessed authors, journals, or years. DOI identity is deduplicated.
4. Citation checking compares claims against structured identities and reviewed evidence. Introduction writing distinguishes the present estimand from prior work and cannot use the earlier invented global-research-gap fallback.
5. Final review receives the actual assembled LaTeX source, including Background, tables, declarations and front matter, plus the entire bibliography. Every character is covered.
6. Oversized reviews use labelled continuation batches and an additional cross-component findings review. All requested editable sections can be revised in one bounded revision pass; blockers requiring investigator input are retained.
7. Coefficient tables use the structured facts directly and include OLS, HC3, cave-cluster-robust OLS and LMM estimates with intervals. The unsupported predictor-share table was removed.
8. Primary and auxiliary analyses share the cave identifier. Entity IDs are not inferred as clusters. Cave-out fits report successful and failed fits. Clustered OLS, LMM residual/random-effect diagnostics and a prespecified 1.5-IQR sensitivity fit were added.
9. A provenance agent can request web research for unavailable source-dataset information. Exact snippet quotations are retained, with an explicit distinction between external database descriptions and unverified local exclusions. The runner exports entity/site/lithology assignments, input checksum, raw input snapshot and run manifest.
10. Compact deterministic Abstract and Methods use the facts contract. Results reference the estimate table, narrative sections have word budgets, covariate MDES rows are excluded, small p-values use thresholds, and the AI disclosure no longer claims investigator verification. Missing authors or repository provenance remain explicit draft blockers.

## Reproduction

From the backend directory:

```powershell
.\venv\Scripts\python.exe -X utf8 -B -m research_agent.run_reviewed_experiment
.\venv\Scripts\python.exe -X utf8 -B -m research_agent.run_reviewed_experiment --analysis-only
```

Each run saves its context, numerical facts, source evidence, source-search trace (in run memory), complete reviewer coverage, data provenance findings, TeX/BibTeX/PDF when compilation succeeds, and a summary. The public repository URL and author identities must come from the investigators. The source snapshot contains entity IDs and site names; original SISAL site IDs and free-text coding provenance are not reconstructed.

## Verification

113 offline tests passed after these changes, including tests for rejected evidence, exact metadata, complete review coverage, cave/individual ID distinction, structured tables, uncertainty estimates, and multi-section revision. Live run outcomes are recorded in the run-specific summary; successful execution is not scientific approval.

## Live-run corrections

The live experiment exposed and led to additional fixes for Crossref title ranking, nested author metadata, missing registry abstracts, provider token limits, and reviewer claims about sections outside a batch. Source access can use complete registry, OpenAlex or publisher abstracts, with the access scope recorded. Review requests are token-counted and split adaptively without losing text; findings need exact manuscript quotations and receive the document inventory and complete bibliography. Plot plans remain workflow artifacts rather than fabricated manuscript figures.

The saved analysis can be re-reviewed without repeating statistical fits or source retrieval:

```powershell
.\venv\Scripts\python.exe -X utf8 -B -m research_agent.review_saved_run workflow_runs/REPLACE_WITH_RUN --document-only
```

Reasoning budgets are configurable per provider. Groq model settings follow its [documented reasoning controls](https://console.groq.com/docs/reasoning). Quota waits are bounded; failed or incomplete reviews cannot approve a manuscript.

## Final experiment outcome

The numerical analysis was rerun against the same 561-record input with the latest code. It uses 245 site-name clusters, four evidence-backed sources, structured coefficient tables, cave-level sensitivity checks, and version-matched upstream data access. The final revised manuscript was compiled after deterministic full-document consistency checks. Prior live model reviews and their limitations are preserved separately; no new LLM approval is claimed by the deterministic export. The result remains a draft requiring independent review, author identities/declarations, public archive links, and verification of upstream selection/coding provenance. Source access was abstract-level, explicitly recorded in the evidence audit.

Default standalone paper creation cannot assume that the review gate passed. The supported graph supplies that decision explicitly.

Final artifact checks: 113 offline tests passed; numerical rerun matched the prior primary estimates within 1e-12; 561 records, 245 clusters, four registry-verified sources; PDF compiled successfully. See artifact_checks.json and review_report.json in the run bundle.
