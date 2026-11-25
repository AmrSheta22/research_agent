"""
Tree and Node representations for Language Agent Tree Search (LATS).

Represents decision states, actions, value evaluations, and trajectories
for searching across data cleaning options and hypotheses.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import pandas as pd


class DecisionLevel(str, Enum):
    ROOT = "root"
    CLEANING = "cleaning"
    HYPOTHESIS = "hypothesis"
    EVALUATION = "evaluation"


@dataclass
class Trajectory:
    """A complete path through the decision tree."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    cleaning_strategy: str = ""
    cleaning_params: Dict[str, Any] = field(default_factory=dict)
    initial_sample_size: int = 0
    retained_sample_size: int = 0
    retention_ratio: float = 1.0

    hypothesis: str = ""
    hypothesis_angle: str = ""

    test_results: Dict[str, Any] = field(default_factory=dict)
    test_names: List[str] = field(default_factory=list)
    min_p_value: float = 1.0
    significant_tests: List[str] = field(default_factory=list)

    score: float = 0.0
    reflection: str = ""
    cleaned_df: Optional[pd.DataFrame] = None

    # Multiplicity / Garden-of-forking-paths control
    k_paths_explored: int = 1
    bonferroni_alpha: float = 0.05
    is_significant_after_bonferroni: bool = False
    is_significant_after_fdr: bool = False
    fdr_adjusted_min_p: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trajectory_id": self.id,
            "cleaning_strategy": self.cleaning_strategy,
            "retained_sample_size": self.retained_sample_size,
            "retention_ratio": round(self.retention_ratio, 3),
            "hypothesis": self.hypothesis,
            "hypothesis_angle": self.hypothesis_angle,
            "min_p_value": self.min_p_value,
            "significant_tests": self.significant_tests,
            "score": round(self.score, 4),
            "reflection": self.reflection,
            "k_paths_explored": self.k_paths_explored,
            "bonferroni_alpha": round(self.bonferroni_alpha, 6),
            "is_significant_after_bonferroni": self.is_significant_after_bonferroni,
            "is_significant_after_fdr": self.is_significant_after_fdr,
            "fdr_adjusted_min_p": round(self.fdr_adjusted_min_p, 6),
        }


class LATSNode:
    """A node in the Monte Carlo / Language Agent Tree Search."""

    def __init__(
        self,
        level: DecisionLevel,
        name: str,
        parent: Optional[LATSNode] = None,
        action: Any = None,
        state_data: Optional[Dict[str, Any]] = None,
    ):
        self.id: str = str(uuid.uuid4())[:8]
        self.level: DecisionLevel = level
        self.name: str = name
        self.parent: Optional[LATSNode] = parent
        self.children: List[LATSNode] = []
        self.action: Any = action
        self.state_data: Dict[str, Any] = state_data or {}

        # MCTS Statistics
        self.visits: int = 0
        self.value_sum: float = 0.0
        self.best_value: float = 0.0

        # Node specific evaluation
        self.reward: float = 0.0
        self.reflection: str = ""
        self.is_terminal: bool = False

    @property
    def average_value(self) -> float:
        return (self.value_sum / self.visits) if self.visits > 0 else 0.0

    def ucb_score(self, c_param: float = 1.414) -> float:
        """Upper Confidence Bound for Trees (UCT)."""
        if self.visits == 0:
            return float("inf")
        if not self.parent or self.parent.visits == 0:
            return self.average_value

        exploitation = self.average_value
        exploration = c_param * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration

    def add_child(self, child: LATSNode) -> LATSNode:
        child.parent = self
        self.children.append(child)
        return child

    def backpropagate(self, reward: float):
        """Update node statistics up to the root."""
        self.visits += 1
        self.value_sum += reward
        self.best_value = max(self.best_value, reward)
        if self.parent:
            self.parent.backpropagate(reward)

    def is_fully_expanded(self, expected_actions_count: int) -> bool:
        return len(self.children) >= expected_actions_count

    def best_child(self, c_param: float = 1.414) -> Optional[LATSNode]:
        if not self.children:
            return None
        return max(self.children, key=lambda c: c.ucb_score(c_param))


class SearchTree:
    """Manages the full tree search and trajectory ranking."""

    def __init__(self, root_name: str = "ResearchProblem"):
        self.root: LATSNode = LATSNode(level=DecisionLevel.ROOT, name=root_name)
        self.trajectories: List[Trajectory] = []

    def register_trajectory(self, traj: Trajectory):
        self.trajectories.append(traj)

    def get_best_trajectory(self) -> Optional[Trajectory]:
        if not self.trajectories:
            return None
        # Rank by score (which prioritizes statistical validity, significance, and sample power)
        return max(self.trajectories, key=lambda t: t.score)

    def get_ranked_trajectories(self) -> List[Trajectory]:
        return sorted(self.trajectories, key=lambda t: t.score, reverse=True)
