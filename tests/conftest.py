"""Keep unit tests offline and isolate all generated artifacts from the checkout."""

from pathlib import Path
import shutil

import pytest


@pytest.fixture(autouse=True)
def isolated_agent_environment(monkeypatch, tmp_path):
    from research_agent.config import PRESET_MODELS
    from research_agent.tools.base import ToolResult
    from research_agent.tools.search_tools import AcademicLiteratureSearchTool, CitationLookupTool, WebSearchTool

    for model in PRESET_MODELS.values():
        if model.api_key_env:
            monkeypatch.delenv(model.api_key_env, raising=False)
    monkeypatch.delenv("GONKA_BROKER_API_KEY", raising=False)
    for tool in (AcademicLiteratureSearchTool, CitationLookupTool, WebSearchTool):
        monkeypatch.setattr(tool, "execute", lambda self, **kwargs: ToolResult.fail("External searches are disabled in unit tests"))
    data_dir = tmp_path / "research_agent" / "data"
    data_dir.mkdir(parents=True)
    source = Path(__file__).resolve().parents[1] / "data" / "noaa_sisal_speleothem_lithology.csv"
    shutil.copyfile(source, data_dir / source.name)
    monkeypatch.chdir(tmp_path)
