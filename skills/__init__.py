"""Skills — higher-order workflows that compose tools via LLM calls."""

from research_agent.skills.base import BaseSkill
from research_agent.skills.hypothesis import HypothesisSkill
from research_agent.skills.abstract import AbstractSkill
from research_agent.skills.methodology import MethodologySkill
from research_agent.skills.introduction import IntroductionSkill
from research_agent.skills.literature_review import LiteratureReviewSkill
from research_agent.skills.metadata_prediction import MetadataPredictionSkill
from research_agent.skills.test_interpretation import TestInterpretationSkill
from research_agent.skills.confounding_variables import ConfoundingVariablesSkill
from research_agent.skills.discussion import DiscussionSkill
from research_agent.skills.conclusion import ConclusionSkill
from research_agent.skills.critic import (
    CriticEvaluation,
    MethodsResultsCritic,
    ResultsDiscussionCritic,
    IntroConclusionCritic,
    GlobalPaperCritic,
    CrossSectionConsistencyCritic,
    CitationGroundingCritic,
    FinalPeerReviewCritic,
)
from research_agent.skills.actor import PaperActorReviser

__all__ = [
    "BaseSkill",
    "HypothesisSkill",
    "AbstractSkill",
    "MethodologySkill",
    "IntroductionSkill",
    "LiteratureReviewSkill",
    "MetadataPredictionSkill",
    "TestInterpretationSkill",
    "ConfoundingVariablesSkill",
    "DiscussionSkill",
    "ConclusionSkill",
    "CriticEvaluation",
    "MethodsResultsCritic",
    "ResultsDiscussionCritic",
    "IntroConclusionCritic",
    "GlobalPaperCritic",
    "CrossSectionConsistencyCritic",
    "CitationGroundingCritic",
    "FinalPeerReviewCritic",
    "PaperActorReviser",
]
