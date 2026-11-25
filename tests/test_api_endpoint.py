import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from uuid import uuid4
import pandas as pd

from main import app
from auth import get_current_user_id
from research_agent.lats.tree import Trajectory
from research_agent.lats.agent import LATSResult
from research_agent.schemas import ResearchContext


@pytest.fixture
def client():
    # Override auth dependency
    app.dependency_overrides[get_current_user_id] = lambda: "test-user-id"
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_lats_optimize_endpoint(client, tmp_path):
    # Create a small dummy CSV file
    dummy_csv = tmp_path / "test_data.csv"
    dummy_csv.write_text("age,treatment,outcome\n25,1,10\n30,1,12\n35,0,8\n40,0,7\n")

    # Mock trajectory & result
    mock_traj = Trajectory(
        id="traj_1",
        cleaning_strategy="iqr_standard",
        hypothesis_angle="directional",
        hypothesis="Treatment significantly increases outcome",
        initial_sample_size=4,
        retained_sample_size=4,
        retention_ratio=1.0,
        min_p_value=0.012,
        significant_tests=["Independent Two-Sample t-test"],
        score=0.92,
        reflection="Robust directional association.",
    )

    mock_result = MagicMock(spec=LATSResult)
    mock_result.best_trajectory = mock_traj
    mock_result.all_trajectories = [mock_traj]
    mock_result.audit_report = "# LATS Audit Report\nBest: iqr_standard"
    mock_result.optimized_context = ResearchContext(
        file_path=str(dummy_csv),
        hypothesis=mock_traj.hypothesis,
        sample_size=4,
    )

    with patch("routers.analysis.run_lats_optimization", return_value=mock_result), \
         patch("routers.analysis.decode_file_token", return_value=str(dummy_csv)):

        payload = {
            "file_token": "valid_token",
            "research_type": "cross_sectional",
            "independent_variable": "treatment",
            "dependent_variable": "outcome",
            "independent_variable_type": "binary",
            "dependent_variable_type": "continuous",
        }

        response = client.post("/api/lats-optimize", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["best_route"]["cleaning_strategy"] == "iqr_standard"
        assert data["best_route"]["min_p_value"] == 0.012
        assert data["optimized_hypothesis"] == "Treatment significantly increases outcome"
        assert len(data["all_trajectories"]) == 1
        assert "Audit Report" in data["audit_report"]
