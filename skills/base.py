"""
Base class for all agent skills.

A Skill is a higher-order capability that:
  1. Selects and configures tools
  2. Constructs prompts from templates + context
  3. Calls an LLM (with tool-calling or structured output)
  4. Post-processes the result
  5. Optionally loops for refinement
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Any, Optional

from json_repair import repair_json

from research_agent.config import AgentConfig, ModelConfig

logger = logging.getLogger(__name__)


class BaseSkill(ABC):
    """
    Abstract base for every agent skill.

    Subclasses must define:
      • name          – unique identifier
      • description   – what this skill does
      • run()         – execute the skill given a research context
    """

    name: str = ""
    description: str = ""

    def __init__(self, config: AgentConfig):
        self.config = config

    # ------------------------------------------------------------------
    # LLM helpers (shared by all skills)
    # ------------------------------------------------------------------
    def _get_llm(self, model_config: Optional[ModelConfig] = None, temperature: Optional[float] = None):
        """
        Instantiate a LangChain chat model from a ModelConfig or primary model via AgentConfig.
        """
        return self.config.get_chat_model(model_config=model_config, temperature=temperature)

    def _invoke_with_fallback(self, messages, temperature: Optional[float] = None) -> str:
        """
        Call the LLM with automatic fallback across the configured chain.
        Respects cooldown windows on 429, 402, and 5xx errors and triggers failover.
        Returns the raw content string.
        """
        chain = self.config.get_fallback_chain()
        last_error: Optional[Exception] = None

        for model_cfg in chain:
            model_key = self.config.get_model_key(model_cfg)
            try:
                logger.info("Invoking LLM model %s via %s...", model_cfg.model_name, model_cfg.provider)
                llm = self._get_llm(model_cfg, temperature=temperature)
                response = llm.invoke(messages)
                metadata=getattr(response,"response_metadata",{}) or {}
                if metadata.get("finish_reason") in {"length","max_tokens","max_output_tokens"}:
                    raise RuntimeError("Model output was truncated by its token limit; refusing partial text")
                content = response.content if hasattr(response, "content") else str(response)
                if isinstance(content, list):
                    content = "".join([b.get("text", "") if isinstance(b, dict) else str(b) for b in content])
                content_str = str(content).strip() if content else ""
                if not content_str and hasattr(response, "additional_kwargs"):
                    reasoning = response.additional_kwargs.get("reasoning") or response.additional_kwargs.get("reasoning_content")
                    if reasoning:
                        content_str = str(reasoning).strip()
                if content_str:
                    logger.info("Model %s successfully returned %d chars.", model_cfg.model_name, len(content_str))
                    return content_str
                logger.warning("Model %s (%s) returned empty content.", model_cfg.model_name, model_key)
            except Exception as exc:
                self.config.cooldown_manager.record_failure(model_cfg.model_name, exc, model_cfg)
                self.config.cooldown_manager.record_failure(model_key, exc, model_cfg)
                logger.warning(
                    "Model %s (%s) failed: %s. Placed in cooldown; failing over to next model.",
                    model_cfg.model_name, model_key, exc,
                )
                last_error = exc

        raise RuntimeError(
            f"All models in fallback chain failed. Last error: {last_error}"
        )

    # ------------------------------------------------------------------
    # Output parsing helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_json(text: str) -> Any:
        """
        Robustly extract JSON from LLM output that may contain markdown
        fences, preamble text, etc.
        """
        # Strip markdown fences
        md_match = re.search(r"```(?:json)?(.*?)```", text, re.DOTALL)
        if md_match:
            text = md_match.group(1).strip()

        # Try array
        arr_start = text.find("[")
        arr_end = text.rfind("]")
        obj_start = text.find("{")
        obj_end = text.rfind("}")

        if arr_start != -1 and arr_end != -1 and arr_end >= arr_start:
            if obj_start == -1 or arr_start < obj_start:
                text = text[arr_start : arr_end + 1]
        elif obj_start != -1 and obj_end != -1 and obj_end >= obj_start:
            text = text[obj_start : obj_end + 1]

        text = repair_json(text)
        return json.loads(text)

    @staticmethod
    def _format_json_as_prose(json_data) -> str:
        """Flatten a JSON dict/list into a single prose string separated by paragraph breaks."""
        if isinstance(json_data, str):
            return json_data
        if isinstance(json_data, list):
            return "\n\n".join(str(item) for item in json_data if item)
        if isinstance(json_data, dict):
            parts = []
            for key, value in json_data.items():
                if not value:
                    continue
                parts.append(str(value).strip())
            return "\n\n".join(parts)
        return str(json_data)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------
    @abstractmethod
    def run(self, context: dict[str, Any], memory: Optional[Any] = None) -> Any:
        """
        Execute the skill.

        Args:
            context: A dict with keys relevant to this skill
                     (typically derived from ResearchContext).
            memory:  Optional reference to the agent's memory store
                     so the skill can read/write learned patterns.

        Returns:
            Skill-specific result (string, dict, list, etc.)
        """
        ...

    def __repr__(self) -> str:
        return f"<Skill {self.name}>"
