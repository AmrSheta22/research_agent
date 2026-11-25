"""
StateGraph compilation for the Research Agent.

Defines the graph topology, state transitions, tool execution bindings,
and task execution flow adhering to the original system's state machine.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from langgraph.graph import END, START, StateGraph

from research_agent.config import AgentConfig
from research_agent.graph.nodes import ResearchGraphNodes
from research_agent.graph.state import ResearchGraphState
from research_agent.memory.store import MemoryStore

logger = logging.getLogger(__name__)


def build_research_graph(
    config: Optional[AgentConfig] = None,
    memory: Optional[MemoryStore] = None,
    tools: Optional[dict[str, Any]] = None,
    skills: Optional[dict[str, Any]] = None,
) -> StateGraph:
    """
    Constructs and compiles the Research Agent StateGraph with active agentic loops:
      1. LATS Cyclic Retry Loop (Empirical Core):
         - Tests p-values (p < alpha) and outlier retention (>= 75%).
         - If non-significant or excessive data dropped, routes back to ingest_and_clean
           with an alternative cleaning strategy or metadata_prediction.
      2. Autonomous Analysis State:
         - Computes power analysis, MDES at 80% power, Nakagawa variance partitioning,
           and equivalence testing.
      3. Synthesis Order (Architectural Enhancement D):
         - Methodology -> Discussion -> Conclusion -> Introduction -> Abstract.
         - Introduction and Abstract are written AFTER empirical findings and conclusions
           are settled, preventing results leakage, phantom citations, and contradicted hypotheses.
      4. Quality Gate Loop:
         - Validates methodology and abstract completeness, length, and error avoidance.
      5. Full Critique Suite & Grounding:
         - Intro-Conclusion Critic -> Web-Grounded Critic -> Global Paper Critic.
      6. Paper Generation:
         - Playground records continuous memory, then create_latex_paper creates
           the final publication-ready LaTeX paper bundle.
    """
    nodes = ResearchGraphNodes(config=config, memory=memory, tools=tools, skills=skills)
    builder = StateGraph(ResearchGraphState)

    # 1. Register State Nodes
    builder.add_node("ingest_and_clean", nodes.ingest_and_clean_state)
    builder.add_node("metadata_prediction", nodes.metadata_prediction_state)
    builder.add_node("conduct_tests", nodes.conduct_tests_state)
    builder.add_node("handle_lats_retry", nodes.handle_lats_retry_state)
    builder.add_node("autonomous_analysis", nodes.autonomous_analysis_state)
    builder.add_node("literature_review", nodes.literature_review_state)
    builder.add_node("methodology", nodes.methodology_state)
    builder.add_node("methods_results_critic", nodes.methods_results_critic_state)
    builder.add_node("discussion", nodes.discussion_state)
    builder.add_node("results_discussion_critic", nodes.results_discussion_critic_state)
    builder.add_node("conclusion", nodes.conclusion_state)
    builder.add_node("introduction", nodes.introduction_state)
    builder.add_node("abstract", nodes.abstract_state)
    builder.add_node("visualization", nodes.visualization_state)
    builder.add_node("quality_gate", nodes.quality_gate_state)
    builder.add_node("prepare_quality_refinement", nodes.prepare_quality_refinement_state)
    builder.add_node("intro_conclusion_critic", nodes.intro_conclusion_critic_state)
    builder.add_node("web_grounded_critic", nodes.web_grounded_critic_state)
    builder.add_node("global_paper_critic", nodes.global_paper_critic_state)
    builder.add_node("playground", nodes.playground_state)
    builder.add_node("final_peer_review", nodes.final_peer_review_state)
    builder.add_node("actor_revision", nodes.actor_revision_state)
    builder.add_node("create_latex_paper", nodes.create_latex_paper_state)

    # 2. Empirical Core with LATS Cyclic Loop
    builder.add_edge(START, "ingest_and_clean")
    builder.add_edge("ingest_and_clean", "metadata_prediction")
    builder.add_edge("metadata_prediction", "conduct_tests")

    # LATS conditional routing from tests
    builder.add_conditional_edges(
        "conduct_tests",
        nodes.route_after_tests,
        {
            "retry_cleaning": "handle_lats_retry",
            "retry_hypothesis": "handle_lats_retry",
            "proceed": "autonomous_analysis",
        },
    )
    builder.add_conditional_edges(
        "handle_lats_retry",
        nodes.route_after_lats_retry,
        {
            "ingest_and_clean": "ingest_and_clean",
            "metadata_prediction": "metadata_prediction",
        },
    )

    # 3. Downstream Research Synthesis: Empirical & Literature Core First
    builder.add_edge("autonomous_analysis", "literature_review")
    builder.add_edge("literature_review", "methodology")
    builder.add_edge("methodology", "methods_results_critic")
    builder.add_edge("methods_results_critic", "discussion")
    builder.add_edge("discussion", "results_discussion_critic")
    builder.add_edge("results_discussion_critic", "conclusion")

    # 4. Framing Sections executed AFTER Results, Discussion & Conclusion
    builder.add_edge("conclusion", "introduction")
    builder.add_edge("introduction", "abstract")
    builder.add_edge("abstract", "visualization")
    builder.add_edge("visualization", "quality_gate")

    # 5. Publication Quality Gate Loop
    builder.add_conditional_edges(
        "quality_gate",
        nodes.route_quality_gate,
        {
            "passed": "intro_conclusion_critic",
            "prepare_quality_refinement": "prepare_quality_refinement",
        },
    )
    builder.add_conditional_edges(
        "prepare_quality_refinement",
        nodes.route_quality_target,
        {
            "methodology": "methodology",
            "abstract": "abstract",
        },
    )

    # 6. Full Critique Suite & Grounding
    builder.add_edge("intro_conclusion_critic", "web_grounded_critic")
    builder.add_edge("web_grounded_critic", "global_paper_critic")
    builder.add_edge("global_paper_critic", "playground")

    # 7. Final peer review & actor revision loop before paper compilation
    builder.add_edge("playground", "final_peer_review")
    builder.add_conditional_edges(
        "final_peer_review",
        nodes.route_peer_review,
        {
            "proceed": "create_latex_paper",
            "revise": "actor_revision",
        },
    )
    builder.add_edge("actor_revision", "final_peer_review")
    builder.add_edge("create_latex_paper", END)

    return builder.compile()
