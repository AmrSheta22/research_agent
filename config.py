"""
Provider-agnostic configuration for the Research Agent.

Supports Cohere, OpenRouter, Gonka Broker, SiliconFlow, Mistral, Google Gemini,
OpenAI, and Groq with automatic fallback chains, cooldown routing, and rate-limit recovery.
No API keys are hardcoded — they are resolved from env vars at runtime.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Provider enum
# ---------------------------------------------------------------------------
class LLMProvider(str, Enum):
    MISTRAL = "mistral"
    GEMINI = "gemini"
    OPENAI = "openai"
    GROQ = "groq"
    COHERE = "cohere"
    OPENROUTER = "openrouter"
    GONKA = "gonka"
    SILICONFLOW = "siliconflow"


# ---------------------------------------------------------------------------
# Cooldown Manager for rate-limits, quota exhaustion, and server errors
# ---------------------------------------------------------------------------
class CooldownManager:
    """
    Thread-safe manager for tracking transient and quota failure cooldowns
    across LLM providers and models.
    """

    def __init__(self):
        self._cooldowns: dict[str, float] = {}  # model key / identifier -> expiry timestamp
        self._lock = threading.Lock()

    def is_in_cooldown(self, key: str) -> bool:
        """Check whether a model identifier is currently in cooldown."""
        with self._lock:
            expiry = self._cooldowns.get(key)
            if expiry is None:
                return False
            if time.time() >= expiry:
                del self._cooldowns[key]
                return False
            return True

    def get_cooldown_remaining(self, key: str) -> float:
        """Return remaining seconds in cooldown for a model, or 0.0 if active."""
        with self._lock:
            expiry = self._cooldowns.get(key)
            if expiry is None:
                return 0.0
            rem = expiry - time.time()
            if rem <= 0:
                del self._cooldowns[key]
                return 0.0
            return rem

    def trigger_cooldown(self, key: str, duration_seconds: float, reason: str = "") -> None:
        """Place a model in cooldown for a specified duration in seconds."""
        with self._lock:
            expiry = time.time() + duration_seconds
            self._cooldowns[key] = expiry
            logger.warning(
                "Placed model '%s' in cooldown for %.1fs (reason: %s).",
                key, duration_seconds, reason or "unspecified",
            )

    def record_failure(self, key: str, exc: Exception, cfg: Optional[ModelConfig] = None) -> None:
        """
        Inspect error status codes (429, 402, 5xx) or error messages and trigger
        the appropriate cooldown duration.
        """
        err_str = str(exc)
        status_code = getattr(exc, "status_code", None)
        if status_code is None and hasattr(exc, "response"):
            status_code = getattr(exc.response, "status_code", None)

        rate_limit_secs = cfg.cooldown_seconds_rate_limit if cfg else 60.0
        quota_secs = cfg.cooldown_seconds_quota if cfg else 3600.0
        server_err_secs = cfg.cooldown_seconds_server_error if cfg else 30.0

        if status_code == 429 or "429" in err_str or "rate limit" in err_str.lower() or "too many requests" in err_str.lower():
            match=re.search(r"try again in ([0-9.]+)(ms|s)\b",err_str,re.I)
            if match:
                rate_limit_secs=max(.1,float(match.group(1))/(1000 if match.group(2).lower()=="ms" else 1))
            self.trigger_cooldown(key, rate_limit_secs, reason="HTTP 429 / Rate Limit")
        elif status_code == 402 or "402" in err_str or "quota" in err_str.lower() or "payment required" in err_str.lower() or "credits" in err_str.lower():
            self.trigger_cooldown(key, quota_secs, reason="HTTP 402 / Quota Exhaustion")
        elif (status_code and 500 <= status_code < 600) or any(code in err_str for code in ["500", "502", "503", "504"]):
            self.trigger_cooldown(key, server_err_secs, reason=f"HTTP {status_code or '5xx'} Upstream Error")
        elif "timed out" in err_str.lower() or "timeout" in err_str.lower():
            self.trigger_cooldown(key, 120.0, reason="Request Timed Out")
        else:
            self.trigger_cooldown(key, 60.0, reason=f"General Error: {exc}")

    def reset(self, key: Optional[str] = None) -> None:
        """Clear cooldown for a specific key, or reset all cooldowns if key is None."""
        with self._lock:
            if key:
                self._cooldowns.pop(key, None)
            else:
                self._cooldowns.clear()

    def get_status(self) -> dict[str, float]:
        """Return a mapping of active model keys to their remaining cooldown seconds."""
        with self._lock:
            now = time.time()
            return {
                k: round(exp - now, 1)
                for k, exp in self._cooldowns.items()
                if exp > now
            }


_GLOBAL_COOLDOWN_MANAGER = CooldownManager()


def get_default_cooldown_manager() -> CooldownManager:
    """Return the global CooldownManager singleton."""
    return _GLOBAL_COOLDOWN_MANAGER


# ---------------------------------------------------------------------------
# Per-model descriptor
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    provider: LLMProvider
    model_name: str
    temperature: float = 0.0
    api_key_env: str = ""          # env-var name, resolved lazily
    base_url: Optional[str] = None # custom API endpoint (e.g. for Groq / self-hosted)
    reasoning_effort: Optional[str] = None
    max_retries: int = 1
    timeout: float = 60.0
    default_headers: Optional[dict[str, str]] = None
    max_tokens: Optional[int] = None
    cooldown_seconds_rate_limit: float = 60.0    # 429 burst / rate limit cooldown
    cooldown_seconds_quota: float = 3600.0       # 402 quota exhaustion cooldown
    cooldown_seconds_server_error: float = 30.0  # 5xx server error cooldown

    @property
    def api_key(self) -> Optional[str]:
        if not self.api_key_env:
            return None
        val = os.getenv(self.api_key_env)
        if not val and self.api_key_env == "GONKA_API_KEY":
            val = os.getenv("GONKA_BROKER_API_KEY")
        return val


# ---------------------------------------------------------------------------
# Built-in model presets
# ---------------------------------------------------------------------------
PRESET_MODELS: dict[str, ModelConfig] = {
    # --- Primary Rank 1: Cohere Command A+ ---
    "cohere-command-a-plus": ModelConfig(
        provider=LLMProvider.COHERE,
        model_name="command-a-plus-05-2026",
        api_key_env="COHERE_API_KEY",
        base_url="https://api.cohere.com/compatibility/v1",
        timeout=120.0,
        cooldown_seconds_rate_limit=60.0,
        cooldown_seconds_quota=3600.0,
        cooldown_seconds_server_error=30.0,
    ),
    # --- Free Overflow Rank 2: OpenRouter Inkling ---
    "openrouter-inkling": ModelConfig(
        provider=LLMProvider.OPENROUTER,
        model_name="thinkingmachines/inkling-2-small:free",
        api_key_env="OPENROUTER_API_KEY",
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "https://cursor.com",
            "X-Title": "Cursor",
            "User-Agent": "Cursor/0.45.0",
        },
        timeout=60.0,
        cooldown_seconds_rate_limit=60.0,
        cooldown_seconds_quota=3600.0,
        cooldown_seconds_server_error=30.0,
    ),
    # --- Survival Tier Rank 3: Gonka Broker GLM-5.3-Flash ---
    "gonka-glm": ModelConfig(
        provider=LLMProvider.GONKA,
        model_name="zai-org/GLM-5.3-Flash",
        api_key_env="GONKA_API_KEY",
        base_url="https://proxy.gonkabroker.com/v1",
        timeout=60.0,
        cooldown_seconds_rate_limit=60.0,
        cooldown_seconds_quota=3600.0,
        cooldown_seconds_server_error=30.0,
    ),
    # --- Backstop Rank 4: SiliconFlow DeepSeek-V3 ---
    "siliconflow-deepseek": ModelConfig(
        provider=LLMProvider.SILICONFLOW,
        model_name="deepseek-ai/DeepSeek-V3",
        api_key_env="SILICONFLOW_API_KEY",
        base_url="https://api.siliconflow.com/v1",
        timeout=60.0,
        cooldown_seconds_rate_limit=60.0,
        cooldown_seconds_quota=3600.0,
        cooldown_seconds_server_error=30.0,
    ),
    # --- Other Provider Presets ---
    "groq-20b": ModelConfig(
        provider=LLMProvider.GROQ,
        model_name="openai/gpt-oss-20b",
        api_key_env="GROQ_API_KEY",
        base_url="https://api.groq.com/openai/v1",
    ),
    "groq-120b": ModelConfig(
        provider=LLMProvider.GROQ,
        model_name="openai/gpt-oss-120b",
        api_key_env="GROQ_API_KEY",
        base_url="https://api.groq.com/openai/v1",
    ),
    "groq-qwen": ModelConfig(
        provider=LLMProvider.GROQ,
        model_name="qwen/qwen3.8-27b",
        api_key_env="GROQ_API_KEY",
        base_url="https://api.groq.com/openai/v1",
    ),
    "mistral-7b": ModelConfig(
        provider=LLMProvider.MISTRAL,
        model_name="open-mistral-7b",
        api_key_env="MISTRAL_API_KEY",
    ),
    "mistral-small": ModelConfig(
        provider=LLMProvider.MISTRAL,
        model_name="mistral-small-latest",
        api_key_env="MISTRAL_API_KEY",
    ),
    "mistral-large": ModelConfig(
        provider=LLMProvider.MISTRAL,
        model_name="mistral-large-latest",
        api_key_env="MISTRAL_API_KEY",
    ),
    "gemini-flash": ModelConfig(
        provider=LLMProvider.GEMINI,
        model_name="gemini-2.5-flash",
        api_key_env="GEMINI_API_KEY",
    ),
    "openai-gpt4o": ModelConfig(
        provider=LLMProvider.OPENAI,
        model_name="gpt-4o",
        api_key_env="OPENAI_API_KEY",
    ),
}

DEFAULT_MODEL_PREFERENCE: list[str] = [
    "cohere-command-a-plus",   # Primary Rank 1
    "openrouter-inkling",      # Free Overflow Rank 2
    "gonka-glm",               # Survival Tier Rank 3
    "siliconflow-deepseek",    # Backstop Rank 4
    "groq-120b",
    "groq-qwen",
    "mistral-7b",
    "gemini-flash",
    "groq-20b",
]


# ---------------------------------------------------------------------------
# Agent-level config
# ---------------------------------------------------------------------------
@dataclass
class AgentConfig:
    """Top-level configuration for a ResearchAgent instance."""

    # Ordered preference — first available and active is used, rest are fallbacks
    model_preference: list[str] = field(
        default_factory=lambda: list(DEFAULT_MODEL_PREFERENCE)
    )

    # Override / extend the built-in presets
    custom_models: dict[str, ModelConfig] = field(default_factory=dict)

    # Cooldown manager
    cooldown_manager: CooldownManager = field(default_factory=get_default_cooldown_manager)

    # Memory
    memory_backend: str = "json"        # "json" | "sqlite" | "redis"
    memory_path: str = ".research_agent/memory"

    # Misc
    default_significance_level: float = 0.05
    max_tool_retries: int = 2
    verbose: bool = False

    # ----- helpers -----

    @property
    def all_models(self) -> dict[str, ModelConfig]:
        return {**PRESET_MODELS, **self.custom_models}

    def get_model_key(self, cfg: ModelConfig) -> str:
        """Return preset or custom key associated with a ModelConfig."""
        for k, v in self.all_models.items():
            if v == cfg or (v.model_name == cfg.model_name and v.provider == cfg.provider):
                return k
        return cfg.model_name

    def get_fallback_chain(self, include_cooling: bool = False) -> list[ModelConfig]:
        """
        Return all configured models with valid API keys in preference order.
        If include_cooling is False, models currently under cooldown are excluded
        unless all models are under cooldown.
        """
        all_available = [
            self.all_models[n]
            for n in self.model_preference
            if n in self.all_models and self.all_models[n].api_key
        ]
        if include_cooling:
            return all_available

        active = [
            m for m in all_available
            if not self.cooldown_manager.is_in_cooldown(m.model_name)
            and not self.cooldown_manager.is_in_cooldown(self.get_model_key(m))
        ]
        return active

    def get_primary_model(self) -> ModelConfig:
        """Return the first model whose API key is available and not in cooldown (if possible)."""
        chain = self.get_fallback_chain(include_cooling=False)
        if chain:
            return chain[0]
        raise RuntimeError(
            "No active LLM provider (missing credentials or all providers in cooldown). Configure: "
            + ", ".join(m.api_key_env for m in self.all_models.values())
        )

    def _build_single_chat(self, cfg: ModelConfig, temp: float):
        if cfg.provider in (
            LLMProvider.GROQ,
            LLMProvider.OPENAI,
            LLMProvider.COHERE,
            LLMProvider.OPENROUTER,
            LLMProvider.GONKA,
            LLMProvider.SILICONFLOW,
        ):
            from langchain_openai import ChatOpenAI
            kwargs: dict[str, Any] = {
                "model": cfg.model_name,
                "temperature": temp,
                "api_key": cfg.api_key,
                "max_retries": cfg.max_retries,
                "timeout": cfg.timeout or 60.0,
            }
            if cfg.base_url:
                kwargs["base_url"] = cfg.base_url
            if cfg.default_headers:
                kwargs["default_headers"] = cfg.default_headers
            if cfg.max_tokens:
                kwargs["max_tokens"] = cfg.max_tokens
            if cfg.reasoning_effort is not None:
                kwargs["reasoning_effort"] = cfg.reasoning_effort
            return ChatOpenAI(**kwargs)
        elif cfg.provider == LLMProvider.MISTRAL:
            from langchain_mistralai import ChatMistralAI
            return ChatMistralAI(
                model=cfg.model_name,
                temperature=temp,
                api_key=cfg.api_key,
                max_retries=cfg.max_retries,
                timeout=cfg.timeout or 30.0,
            )
        elif cfg.provider == LLMProvider.GEMINI:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=cfg.model_name,
                temperature=temp,
                api_key=cfg.api_key,
                timeout=cfg.timeout or 30.0,
            )
        raise ValueError(f"Unsupported provider: {cfg.provider}")

    def get_chat_model(
        self,
        model_config: Optional[ModelConfig] = None,
        temperature: Optional[float] = None,
    ):
        """Instantiate a LangChain chat model from ModelConfig or the fallback chain."""
        if model_config is not None:
            temp = temperature if temperature is not None else model_config.temperature
            return self._build_single_chat(model_config, temp)

        chain = self.get_fallback_chain(include_cooling=False)
        if not chain:
            primary_cfg = self.get_primary_model()
            temp = temperature if temperature is not None else primary_cfg.temperature
            return self._build_single_chat(primary_cfg, temp)

        primary_cfg = chain[0]
        temp = temperature if temperature is not None else primary_cfg.temperature
        primary_chat = self._build_single_chat(primary_cfg, temp)

        fallbacks = []
        for f_cfg in chain[1:]:
            try:
                f_temp = temperature if temperature is not None else f_cfg.temperature
                fallbacks.append(self._build_single_chat(f_cfg, f_temp))
            except Exception:
                pass

        if fallbacks:
            return primary_chat.with_fallbacks(fallbacks)
        return primary_chat
