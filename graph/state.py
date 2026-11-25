"""
State definition for the Research Agent StateGraph.

Mirrors the workflow states of the original platform:
  • test_status
  • literature_review_status
  • abstract_status
  • introduction_status
  • methodology_status
  • visualization_status
  • playground_status
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, Optional, Set
from typing_extensions import TypedDict
import pandas as pd

from research_agent.schemas import (
    LiteratureItem,
    ResearchContext,
    StudyFacts,
    VisualizationSpec,
)


class ResearchGraphState(TypedDict, total=False):
    """The central state flowing through all nodes in the StateGraph."""

    # Core Research Definition & Inputs
    context: ResearchContext

    # Task Toggles (matching the original API triggers)
    do_tests: bool
    do_literature_review: bool
    do_abstract: bool
    do_introduction: bool
    do_methodology: bool
    do_visualization: bool
    do_playground: bool

    # Workflow States (matching original status fields)
    test_status: str               # not_started | processing | completed | failed
    literature_review_status: str  # not_started | processing | completed | failed
    abstract_status: str           # not_started | processing | completed | failed
    introduction_status: str       # not_started | processing | completed | failed
    methodology_status: str        # not_started | processing | completed | failed
    discussion_status: str
    conclusion_status: str
    web_grounded_critic_status: str
    visualization_status: str      # not_started | processing | completed | failed
    playground_status: str         # not_started | started | completed | failed
    latex_paper_status: str        # not_started | processing | completed | failed | skipped

    # Paper Generation Options
    paper_template: Optional[str]  # generic_arxiv | ieee | elsevier | nature | acm | plos
    output_dir: Optional[str]
    compile_pdf: Optional[bool]
    authors: Optional[List[Dict[str, str]]]
    keywords: Optional[List[str]]
    publication_ready: bool
    validation_report: Optional[Dict[str, Any]]

    # Working Data & In-Memory Artefacts
    df: Optional[pd.DataFrame]
    dataset_preview: Optional[str]
    sample_size: Optional[int]
    study_facts: Optional[StudyFacts]
    autonomous_analysis: Optional[Dict[str, Any]]

    # Generated Section Outputs
    test_results: Dict[str, Any]
    test_names: List[str]
    test_interpretation: str
    literature_review: List[LiteratureItem]
    background: Optional[str]
    introduction: str
    methodology: str
    abstract: str
    discussion: Optional[str]
    conclusion: Optional[str]
    visualizations: List[VisualizationSpec]
    confounding_variables: List[str]

    # Paired Reflexion & Cross-Section Critic State
    methods_results_critique: Optional[Dict[str, Any]]
    results_discussion_critique: Optional[Dict[str, Any]]
    intro_conclusion_critique: Optional[Dict[str, Any]]
    web_grounded_critique: Optional[Dict[str, Any]]
    global_paper_critique: Optional[Dict[str, Any]]
    reflexion_history: Annotated[List[Dict[str, Any]], operator.add]
    critic_retry_count: int
    max_critic_retries: int

    # Generated LaTeX Paper Artefacts
    latex_paper_bundle: Optional[Dict[str, Any]]
    latex_paper_path: Optional[str]
    latex_pdf_path: Optional[str]
    latex_source: Optional[str]

    # LATS Cyclic Retry Loop State
    raw_df: Optional[pd.DataFrame]
    cleaning_strategy: Optional[str]
    cleaning_history: Annotated[List[str], operator.add]
    hypothesis_history: Annotated[List[str], operator.add]
    lats_retry_count: int
    max_lats_retries: int
    min_p_value: Optional[float]
    retention_ratio: Optional[float]
    lats_route_decision: Optional[str]   # significant | retry_cleaning | retry_hypothesis | retries_exhausted
    lats_reflection: Optional[str]

    # Quality Gate Loop State
    quality_retry_count: int
    max_quality_retries: int
    abstract_quality_passed: bool
    methodology_quality_passed: bool
    quality_status: str                  # not_started | passed | needs_refinement | exhausted
    quality_critique: Optional[str]
    quality_target: Optional[str]        # methodology | abstract

    # Final Peer Review & Actor Revision Loop State
    peer_review_critique: Optional[Dict[str, Any]]
    peer_review_retry_count: int
    max_peer_review_retries: int
    peer_review_status: str              # not_started | passed | needs_revision | exhausted
    review_coverage: List[str]
    peer_review_target: Optional[str]

    # Execution Trace & Errors
    tool_history: Annotated[List[Dict[str, Any]], operator.add]
    errors: Annotated[List[str], operator.add]
