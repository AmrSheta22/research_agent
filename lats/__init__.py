"""
LATS (Language Agent Tree Search) subsystem for the Research Agent.

Enables the agent to search through multiple data cleaning options and
hypothesis formulations, execute statistical tests to gather empirical
p-values, evaluate trajectories, and take the optimal route.
"""

from research_agent.lats.tree import SearchTree, Trajectory, LATSNode, DecisionLevel
from research_agent.lats.cleaner_pool import CleanerPool, CleaningStrategy, CleaningResult
from research_agent.lats.evaluator import LATSEvaluator, EvaluationScore
from research_agent.lats.hypothesis_generator import HypothesisGenerator, HypothesisCandidate
from research_agent.lats.mcts import LATSSearchEngine
from research_agent.lats.agent import LATSAgent, LATSResult

__all__ = [
    "LATSAgent",
    "LATSResult",
    "LATSSearchEngine",
    "SearchTree",
    "Trajectory",
    "LATSNode",
    "DecisionLevel",
    "CleanerPool",
    "CleaningStrategy",
    "CleaningResult",
    "LATSEvaluator",
    "EvaluationScore",
    "HypothesisGenerator",
    "HypothesisCandidate",
]
