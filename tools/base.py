"""
Base class for all agent tools.

Every tool is self-describing so the LLM can discover and invoke it.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional, Type

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ToolResult(BaseModel):
    """Standard wrapper for tool outputs."""

    success: bool = True
    data: Any = None
    error: Optional[str] = None

    @classmethod
    def ok(cls, data: Any) -> ToolResult:
        return cls(success=True, data=data)

    @classmethod
    def fail(cls, error: str) -> ToolResult:
        return cls(success=False, error=error)


class BaseTool(ABC):
    """
    Abstract base for every agent tool.

    Subclasses must define:
      • name          – unique identifier (snake_case)
      • description   – natural-language explanation for the LLM
      • input_schema  – Pydantic model describing expected input
      • execute()     – the actual logic
    """

    name: str = ""
    description: str = ""
    input_schema: Optional[Type[BaseModel]] = None

    # ------------------------------------------------------------------
    # Introspection helpers (used by agent for tool-calling schemas)
    # ------------------------------------------------------------------
    def to_schema(self) -> dict:
        """Return an OpenAI / Gemini compatible function-calling schema."""
        schema: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
        }
        if self.input_schema:
            schema["parameters"] = self.input_schema.model_json_schema()
        return schema

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """Run the tool and return a ToolResult."""
        ...

    def __call__(self, **kwargs) -> ToolResult:
        """Convenience: tool instances are callable."""
        try:
            return self.execute(**kwargs)
        except Exception as exc:
            logger.exception("Tool %s failed", self.name)
            return ToolResult.fail(str(exc))

    def __repr__(self) -> str:
        return f"<Tool {self.name}>"
