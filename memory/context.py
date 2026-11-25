"""
Session context for tracking state within a single agent execution.

Provides working memory, tool execution history, and skill output caching
for the current research instance.
"""

from __future__ import annotations

import time
from typing import Any, Optional

from research_agent.schemas import ResearchContext
from research_agent.tools.base import ToolResult


class SessionContext:
    """Working memory for a single research analysis session."""

    def __init__(self, research_context: Optional[ResearchContext] = None):
        self.research_context: ResearchContext = research_context or ResearchContext()
        self.tool_history: list[dict[str, Any]] = []
        self.skill_outputs: dict[str, Any] = {}
        self.messages: list[dict[str, str]] = []
        self.iteration: int = 0
        self.created_at: float = time.time()
        self.learnings_applied: list[str] = []

    def record_tool_call(self, tool_name: str, args: dict[str, Any], result: ToolResult):
        """Append a tool call event to the session history."""
        self.tool_history.append({
            "tool": tool_name,
            "args": {k: str(v)[:100] for k, v in args.items() if k != "df"},
            "success": result.success,
            "error": result.error,
            "timestamp": time.time(),
        })

    def set_skill_output(self, skill_name: str, output: Any):
        """Cache the output of a skill."""
        self.skill_outputs[skill_name] = output

    def get_skill_output(self, skill_name: str) -> Optional[Any]:
        """Retrieve cached output from a previous skill in this session."""
        return self.skill_outputs.get(skill_name)

    def add_message(self, role: str, content: str):
        """Add a conversation message."""
        self.messages.append({"role": role, "content": content})

    def update_context(self, **kwargs):
        """Dynamically update research context fields."""
        for key, value in kwargs.items():
            if hasattr(self.research_context, key) and value is not None:
                setattr(self.research_context, key, value)

    def to_summary(self) -> str:
        """Produce a compact text summary of the current session state for LLM injection."""
        rc = self.research_context
        lines = [
            f"Study Design: {rc.research_type or 'Unknown'}",
            f"IV: {rc.independent_variable} ({rc.independent_variable_type or 'unknown'})",
            f"DV: {rc.dependent_variable} ({rc.dependent_variable_type or 'unknown'})",
            f"Hypothesis: {rc.hypothesis or 'Not set'}",
        ]
        if rc.covariates:
            lines.append(f"Covariates: {', '.join(rc.covariates)}")
        if self.skill_outputs:
            lines.append(f"Completed Skills: {', '.join(self.skill_outputs.keys())}")
        return "\n".join(lines)
