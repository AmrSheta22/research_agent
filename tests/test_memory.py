"""Unit tests for research_agent memory subsystem."""

import os
import shutil
import tempfile
import pytest

from research_agent.memory.context import SessionContext
from research_agent.memory.store import MemoryStore
from research_agent.schemas import ResearchContext
from research_agent.tools.base import ToolResult


@pytest.fixture
def temp_memory_dir():
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_session_context():
    rc = ResearchContext(
        independent_variable="drug",
        dependent_variable="recovery",
        research_type="cross_sectional",
    )
    session = SessionContext(research_context=rc)

    # Tool call recording
    session.record_tool_call("test_tool", {"arg1": "val1"}, ToolResult.ok("data"))
    assert len(session.tool_history) == 1
    assert session.tool_history[0]["success"] is True

    # Skill output caching
    session.set_skill_output("hypothesis", "Drug increases recovery rate.")
    assert session.get_skill_output("hypothesis") == "Drug increases recovery rate."

    # Summary generation
    summary = session.to_summary()
    assert "cross_sectional" in summary
    assert "drug" in summary


def test_memory_store_continuous_learning(temp_memory_dir):
    store = MemoryStore(backend="json", path=temp_memory_dir)

    # 1. Variable type learning
    store.record_variable_type_pattern("patient_age", "numeric", was_correct=True)
    learned = store.get_learned_variable_type("patient_age")
    assert learned == "numeric"

    # 2. Test outcome recording
    store.record_test_outcome(
        design="individual_rct",
        iv_type="binary",
        dv_type="cont",
        test_used="two_sample_t_test",
        significant=True,
    )
    history = store.get_learning("test_outcomes", "individual_rct::binary::cont")
    assert len(history) == 1
    assert history[0]["significant"] is True

    # 3. User corrections
    store.record_user_correction("research_type", "cross_sectional", "individual_rct")
    enhancements = store.get_context_enhancements("study")
    assert any("individual_rct" in e for e in enhancements)

    # 4. Persistence reload test
    store2 = MemoryStore(backend="json", path=temp_memory_dir)
    assert store2.get_learned_variable_type("patient_age") == "numeric"
