"""
Tests for Cohere, OpenRouter, Gonka Broker, and SiliconFlow model configuration,
cooldown routing, and fallback chains.
"""

from unittest.mock import MagicMock, patch
import pytest

from research_agent.config import (
    AgentConfig,
    CooldownManager,
    LLMProvider,
    ModelConfig,
    PRESET_MODELS,
    DEFAULT_MODEL_PREFERENCE,
)
from research_agent.skills.base import BaseSkill


class DummySkill(BaseSkill):
    name = "dummy"
    description = "dummy skill for testing failover"

    def run(self, *args, **kwargs):
        return self._invoke_with_fallback([MagicMock()])


def test_new_providers_and_presets_exist():
    # Verify Enum values
    assert LLMProvider.COHERE == "cohere"
    assert LLMProvider.OPENROUTER == "openrouter"
    assert LLMProvider.GONKA == "gonka"
    assert LLMProvider.SILICONFLOW == "siliconflow"

    # Verify presets
    assert "cohere-command-a-plus" in PRESET_MODELS
    assert PRESET_MODELS["cohere-command-a-plus"].model_name == "command-a-plus-05-2026"
    assert PRESET_MODELS["cohere-command-a-plus"].provider == LLMProvider.COHERE
    assert PRESET_MODELS["cohere-command-a-plus"].base_url == "https://api.cohere.com/compatibility/v1"

    assert "openrouter-inkling" in PRESET_MODELS
    assert PRESET_MODELS["openrouter-inkling"].model_name == "thinkingmachines/inkling-2-small:free"
    assert PRESET_MODELS["openrouter-inkling"].provider == LLMProvider.OPENROUTER
    assert PRESET_MODELS["openrouter-inkling"].base_url == "https://openrouter.ai/api/v1"
    assert PRESET_MODELS["openrouter-inkling"].default_headers["HTTP-Referer"] == "https://cursor.com"

    assert "gonka-glm" in PRESET_MODELS
    assert PRESET_MODELS["gonka-glm"].model_name == "zai-org/GLM-5.3-Flash"
    assert PRESET_MODELS["gonka-glm"].provider == LLMProvider.GONKA
    assert PRESET_MODELS["gonka-glm"].base_url == "https://proxy.gonkabroker.com/v1"

    assert "siliconflow-deepseek" in PRESET_MODELS
    assert PRESET_MODELS["siliconflow-deepseek"].model_name == "deepseek-ai/DeepSeek-V3"
    assert PRESET_MODELS["siliconflow-deepseek"].provider == LLMProvider.SILICONFLOW
    assert PRESET_MODELS["siliconflow-deepseek"].base_url == "https://api.siliconflow.com/v1"


def test_default_preference_hierarchy():
    assert DEFAULT_MODEL_PREFERENCE[0] == "cohere-command-a-plus"
    assert DEFAULT_MODEL_PREFERENCE[1] == "openrouter-inkling"
    assert DEFAULT_MODEL_PREFERENCE[2] == "gonka-glm"
    assert DEFAULT_MODEL_PREFERENCE[3] == "siliconflow-deepseek"

    config = AgentConfig()
    assert config.model_preference[0] == "cohere-command-a-plus"
    assert config.model_preference[1] == "openrouter-inkling"
    assert config.model_preference[2] == "gonka-glm"
    assert config.model_preference[3] == "siliconflow-deepseek"


def test_cooldown_manager_status_and_triggers():
    cm = CooldownManager()
    assert not cm.is_in_cooldown("cohere-command-a-plus")

    # Trigger rate limit (429)
    exc_429 = Exception("HTTP 429 Too Many Requests: Rate limit exceeded")
    cm.record_failure("cohere-command-a-plus", exc_429, PRESET_MODELS["cohere-command-a-plus"])
    assert cm.is_in_cooldown("cohere-command-a-plus")
    assert cm.get_cooldown_remaining("cohere-command-a-plus") > 0

    # Reset
    cm.reset("cohere-command-a-plus")
    assert not cm.is_in_cooldown("cohere-command-a-plus")

    # Trigger quota exhaustion (402)
    exc_402 = Exception("HTTP 402 Payment Required: Quota exhausted")
    cm.record_failure("openrouter-inkling", exc_402, PRESET_MODELS["openrouter-inkling"])
    assert cm.is_in_cooldown("openrouter-inkling")
    assert cm.get_cooldown_remaining("openrouter-inkling") > 3000

    # Trigger 5xx upstream
    exc_503 = Exception("503 Service Unavailable")
    cm.record_failure("gonka-glm", exc_503, PRESET_MODELS["gonka-glm"])
    assert cm.is_in_cooldown("gonka-glm")


def test_fallback_chain_skips_cooling_models():
    cm = CooldownManager()
    config = AgentConfig(
        model_preference=["cohere-command-a-plus", "openrouter-inkling", "gonka-glm"],
        cooldown_manager=cm,
    )

    with patch.dict("os.environ", {
        "COHERE_API_KEY": "test_cohere",
        "OPENROUTER_API_KEY": "test_openrouter",
        "GONKA_API_KEY": "test_gonka",
    }):
        # Initial chain has all 3
        chain = config.get_fallback_chain()
        assert len(chain) == 3
        assert chain[0].model_name == "command-a-plus-05-2026"

        # Cohere fails with 429 -> cooldown
        cm.trigger_cooldown("command-a-plus-05-2026", 60.0)
        chain = config.get_fallback_chain()
        assert len(chain) == 2
        assert chain[0].model_name == "thinkingmachines/inkling-2-small:free"

        # OpenRouter fails with 402 -> cooldown
        cm.trigger_cooldown("thinkingmachines/inkling-2-small:free", 3600.0)
        chain = config.get_fallback_chain()
        assert len(chain) == 1
        assert chain[0].model_name == "zai-org/GLM-5.3-Flash"

        # All cooling models must remain unavailable to avoid immediate repeat requests
        cm.trigger_cooldown("zai-org/GLM-5.3-Flash", 60.0)
        chain = config.get_fallback_chain()
        assert chain == []


def test_skill_invoke_with_failover():
    cm = CooldownManager()
    config = AgentConfig(
        model_preference=["cohere-command-a-plus", "openrouter-inkling", "gonka-glm"],
        cooldown_manager=cm,
    )

    skill = DummySkill(config)

    with patch.dict("os.environ", {
        "COHERE_API_KEY": "test_cohere",
        "OPENROUTER_API_KEY": "test_openrouter",
        "GONKA_API_KEY": "test_gonka",
    }):
        # Mock LLM invocations:
        # Cohere throws 429
        # OpenRouter succeeds with content
        def mock_build_single_chat(cfg, temp):
            mock_llm = MagicMock()
            if cfg.provider == LLMProvider.COHERE:
                mock_llm.invoke.side_effect = Exception("429 RateLimitError")
            elif cfg.provider == LLMProvider.OPENROUTER:
                mock_res = MagicMock()
                mock_res.content = "OpenRouter success response"
                mock_llm.invoke.return_value = mock_res
            return mock_llm

        with patch.object(config, "_build_single_chat", side_effect=mock_build_single_chat):
            result = skill.run()
            assert result == "OpenRouter success response"

            # Verify Cohere is now in cooldown
            assert cm.is_in_cooldown("command-a-plus-05-2026")
            assert not cm.is_in_cooldown("thinkingmachines/inkling-2-small:free")
