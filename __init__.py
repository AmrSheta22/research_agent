"""
research_agent - A modular, provider-agnostic research assistant agent.

Extracts the AI/LLM layer from the research-helper backend into a
standalone, framework-aligned package built around:
  • Tools   – self-describing, composable capabilities
  • Skills  – higher-order workflows that orchestrate tools
  • Memory  – session + persistent learning across research instances
"""

from research_agent.agent import ResearchAgent
from research_agent.config import AgentConfig

__all__ = ["ResearchAgent", "AgentConfig"]
__version__ = "0.1.0"
