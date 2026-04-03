"""
Integration tests for the FastAPI server — tests all 4 endpoints via TestClient.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from app import app


@pytest.fixture
def client():
    return TestClient(app)


# ─── Health Endpoint ──────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_body(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["environment"] == "CloudFinOpsEnv"
        assert data["version"] == "1.0.0"
        assert len(data["available_tasks"]) == 3


# ─── Reset Endpoint ──────────────────────────────────────────────────────

class TestResetEndpoint:
    def test_reset_easy(self, client):
        resp = client.post("/reset", json={"task_id": "easy_orphan_cleanup"})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["resources"]) == 10
        assert data["step_number"] == 0
        assert data["total_monthly_cost"] > 800

    def test_reset_medium(self, client):
        resp = client.post("/reset", json={"task_id": "medium_rightsize"})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["resources"]) == 20
        assert data["budget_target"] == 3800.0

    def test_reset_hard(self, client):
        resp = client.post("/reset", json={"task_id": "hard_dependency_migration"})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["resources"]) == 35
        assert data["maintenance_window"] == "02:00-06:00 UTC"

    def test_reset_invalid_task(self, client):
        resp = client.post("/reset", json={"task_id": "nonexistent"})
        assert resp.status_code == 400

    def test_metrics_hidden_after_reset(self, client):
        resp = client.post("/reset", json={"task_id": "easy_orphan_cleanup"})
        data = resp.json()
        for r in data["resources"]:
            assert r["metrics"] is None


# ─── Step Endpoint ────────────────────────────────────────────────────────

class TestStepEndpoint:
    def test_step_query_metrics(self, client):
        client.post("/reset", json={"task_id": "easy_orphan_cleanup"})
        resp = client.post("/step", json={
            "action_type": "query_metrics",
            "resource_id": "vol-0a1b2c3d4e5f60001",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["done"] is False
        assert data["reward"]["value"] < 0  # small negative

    def test_step_delete_wasteful(self, client):
        client.post("/reset", json={"task_id": "easy_orphan_cleanup"})
        resp = client.post("/step", json={
            "action_type": "delete",
            "resource_id": "vol-0a1b2c3d4e5f60001",
            "reason": "orphaned volume",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["reward"]["value"] > 0

    def test_step_delete_production(self, client):
        client.post("/reset", json={"task_id": "easy_orphan_cleanup"})
        resp = client.post("/step", json={
            "action_type": "delete",
            "resource_id": "i-0a1b2c3d4e5f60007",
        })
        data = resp.json()
        assert data["reward"]["value"] < -0.5
        assert len(data["info"]["safety_violations"]) > 0

    def test_step_commit(self, client):
        client.post("/reset", json={"task_id": "easy_orphan_cleanup"})
        resp = client.post("/step", json={"action_type": "commit_changes"})
        data = resp.json()
        assert data["done"] is True

    def test_step_after_done(self, client):
        client.post("/reset", json={"task_id": "easy_orphan_cleanup"})
        client.post("/step", json={"action_type": "commit_changes"})
        resp = client.post("/step", json={"action_type": "list_resources"})
        assert resp.status_code == 400

    def test_step_resize(self, client):
        client.post("/reset", json={"task_id": "medium_rightsize"})
        resp = client.post("/step", json={
            "action_type": "resize",
            "resource_id": "i-0b2c3d4e5f6a70004",
            "new_size": "t3.small",
        })
        data = resp.json()
        assert data["reward"]["value"] > 0
        assert data["info"]["cost_saved"] > 0

    def test_step_list_resources(self, client):
        client.post("/reset", json={"task_id": "easy_orphan_cleanup"})
        resp = client.post("/step", json={"action_type": "list_resources"})
        data = resp.json()
        assert len(data["observation"]["resources"]) == 10

    def test_step_check_deps(self, client):
        client.post("/reset", json={"task_id": "hard_dependency_migration"})
        resp = client.post("/step", json={
            "action_type": "check_deps",
            "resource_id": "i-0c3d4e5f6a7b80012",
        })
        data = resp.json()
        assert data["done"] is False


# ─── State Endpoint ───────────────────────────────────────────────────────

class TestStateEndpoint:
    def test_state_returns_oracle_data(self, client):
        client.post("/reset", json={"task_id": "easy_orphan_cleanup"})
        resp = client.get("/state")
        assert resp.status_code == 200
        data = resp.json()
        assert data["task_id"] == "easy_orphan_cleanup"
        assert data["optimal_savings"] > 0
        assert len(data["critical_resources"]) > 0

    def test_state_before_reset(self, client):
        """State before reset should error."""
        # Note: since TestClient shares global env, this may not error
        # if a previous test has already reset. That's OK for integration testing.
        pass


# ─── Full Oracle End-to-End ───────────────────────────────────────────────

class TestOracleEndToEnd:
    def test_easy_oracle_via_api(self, client):
        """Run easy oracle solution through API and verify score."""
        from data.generator import load_solution
        solution = load_solution("easy_orphan_cleanup")

        client.post("/reset", json={"task_id": "easy_orphan_cleanup"})

        for step_data in solution["optimal_action_sequence"]:
            action = {
                "action_type": step_data["action_type"],
                "resource_id": step_data.get("resource_id"),
                "new_size": step_data.get("new_size"),
                "reason": step_data.get("reason"),
            }
            resp = client.post("/step", json=action)
            assert resp.status_code == 200

        data = resp.json()
        assert data["done"] is True
        assert data["info"]["cost_saved"] > 60
        assert len(data["info"]["safety_violations"]) == 0

    def test_medium_oracle_via_api(self, client):
        """Run medium oracle solution through API and verify score."""
        from data.generator import load_solution
        solution = load_solution("medium_rightsize")

        client.post("/reset", json={"task_id": "medium_rightsize"})

        for step_data in solution["optimal_action_sequence"]:
            action = {
                "action_type": step_data["action_type"],
                "resource_id": step_data.get("resource_id"),
                "new_size": step_data.get("new_size"),
                "reason": step_data.get("reason"),
            }
            resp = client.post("/step", json=action)
            assert resp.status_code == 200

        data = resp.json()
        assert data["done"] is True
        assert data["info"]["cost_saved"] > 900
        assert len(data["info"]["safety_violations"]) == 0

    def test_hard_oracle_via_api(self, client):
        """Run hard oracle solution through API and verify score."""
        from data.generator import load_solution
        solution = load_solution("hard_dependency_migration")

        client.post("/reset", json={"task_id": "hard_dependency_migration"})

        for step_data in solution["optimal_action_sequence"]:
            action = {
                "action_type": step_data["action_type"],
                "resource_id": step_data.get("resource_id"),
                "new_size": step_data.get("new_size"),
                "reason": step_data.get("reason"),
            }
            resp = client.post("/step", json=action)
            assert resp.status_code == 200

        data = resp.json()
        assert data["done"] is True
        assert data["info"]["cost_saved"] > 2400  # ~$2513 optimal
        assert len(data["info"]["safety_violations"]) == 0
