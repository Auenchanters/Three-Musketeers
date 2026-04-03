"""
Tests for engine/environment.py — Core environment logic.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import Action, ActionType
from engine.environment import CloudFinOpsEnvironment


from collections import namedtuple

# Dummy reward for backwards compatibility in tests
DummyReward = namedtuple("DummyReward", ["value", "message", "is_valid"])

class _TestEnvWrapper:
    """Wraps the fully compliant OpenEnv to behave like our old local tuple version strictly for testing."""
    def __init__(self):
        self._env = CloudFinOpsEnvironment()
        
    def __getattr__(self, name):
        return getattr(self._env, name)
        
    def step(self, action):
        obs = self._env.step(action)
        # Re-pack the Observation properties back into the 4-part tuple the tests expect
        reward_obj = DummyReward(obs.reward, obs.message, True)
        return obs, reward_obj, obs.done, getattr(obs, "metadata", {})

@pytest.fixture
def env():
    return _TestEnvWrapper()

@pytest.fixture
def easy_env(env):
    env.reset(task_id="easy_orphan_cleanup")
    return env


@pytest.fixture
def medium_env(env):
    env.reset(task_id="medium_rightsize")
    return env


@pytest.fixture
def hard_env(env):
    env.reset(task_id="hard_dependency_migration")
    return env


# ─── Reset Tests ──────────────────────────────────────────────────────────

class TestReset:
    def test_reset_returns_observation(self, env):
        obs = env.reset(task_id="easy_orphan_cleanup")
        assert obs is not None
        assert obs.step_number == 0
        assert len(obs.resources) == 10

    def test_reset_medium(self, env):
        obs = env.reset(task_id="medium_rightsize")
        assert len(obs.resources) == 20
        assert obs.budget_target == 3800.0

    def test_reset_hard(self, env):
        obs = env.reset(task_id="hard_dependency_migration")
        assert len(obs.resources) == 35
        assert obs.maintenance_window == "02:00-06:00 UTC"

    def test_metrics_hidden_after_reset(self, easy_env):
        obs = easy_env.reset(task_id="easy_orphan_cleanup")
        for r in obs.resources:
            assert r.metrics is None, f"Metrics should be hidden for {r.resource_id}"

    def test_reset_invalid_task(self, env):
        with pytest.raises(ValueError):
            env.reset(task_id="nonexistent")

    @pytest.mark.parametrize("task_id", ["easy_orphan_cleanup", "medium_rightsize", "hard_dependency_migration"])
    def test_reset_all_tasks(self, env, task_id):
        obs = env.reset(task_id=task_id)
        assert obs.task_description != ""
        assert obs.total_monthly_cost > 0


# ─── Query Metrics Tests ─────────────────────────────────────────────────

class TestQueryMetrics:
    def test_query_reveals_metrics(self, easy_env):
        action = Action(action_type=ActionType.QUERY_METRICS, resource_id="vol-0a1b2c3d4e5f60001")
        obs, reward, done, info = easy_env.step(action)
        # Find the resource in observation
        vol = next(r for r in obs.resources if r.resource_id == "vol-0a1b2c3d4e5f60001")
        assert vol.metrics is not None
        assert vol.metrics.cpu_avg_7d == 0.0

    def test_query_metrics_reward(self, easy_env):
        action = Action(action_type=ActionType.QUERY_METRICS, resource_id="vol-0a1b2c3d4e5f60001")
        obs, reward, done, info = easy_env.step(action)
        assert reward.value < 0  # small negative cost
        assert done is False

    def test_query_invalid_resource(self, easy_env):
        action = Action(action_type=ActionType.QUERY_METRICS, resource_id="nonexistent")
        obs, reward, done, info = easy_env.step(action)
        assert "not found" in reward.message.lower()


# ─── Delete Tests ─────────────────────────────────────────────────────────

class TestDelete:
    def test_delete_orphaned_resource(self, easy_env):
        """Deleting an orphaned/wasteful resource should give positive reward."""
        action = Action(action_type=ActionType.DELETE, resource_id="vol-0a1b2c3d4e5f60001")
        obs, reward, done, info = easy_env.step(action)
        assert reward.value > 0  # positive (savings - step_cost)
        assert info["cost_saved"] > 0
        # Resource should be removed from observation
        ids = [r.resource_id for r in obs.resources]
        assert "vol-0a1b2c3d4e5f60001" not in ids

    def test_delete_production_resource_penalty(self, easy_env):
        """Deleting a production resource should give catastrophic penalty."""
        action = Action(action_type=ActionType.DELETE, resource_id="i-0a1b2c3d4e5f60007")
        obs, reward, done, info = easy_env.step(action)
        assert reward.value < -0.5
        assert len(info["safety_violations"]) > 0
        assert "production" in reward.message.lower() or "critical" in reward.message.lower()

    def test_delete_already_deleted(self, easy_env):
        """Deleting the same resource twice should fail gracefully."""
        action = Action(action_type=ActionType.DELETE, resource_id="vol-0a1b2c3d4e5f60001")
        easy_env.step(action)
        obs, reward, done, info = easy_env.step(action)
        assert "already" in reward.message.lower() or "deleted" in reward.message.lower()


# ─── Resize Tests ─────────────────────────────────────────────────────────

class TestResize:
    def test_resize_over_provisioned(self, medium_env):
        """Resizing an over-provisioned resource should give positive reward."""
        action = Action(
            action_type=ActionType.RESIZE,
            resource_id="i-0b2c3d4e5f6a70004",
            new_size="t3.small",
        )
        obs, reward, done, info = medium_env.step(action)
        assert reward.value > 0
        assert info["cost_saved"] > 0

    def test_resize_invalid_target(self, medium_env):
        """Resizing to an invalid target should fail."""
        action = Action(
            action_type=ActionType.RESIZE,
            resource_id="i-0b2c3d4e5f6a70004",
            new_size="p4d.24xlarge",  # not a valid resize target
        )
        obs, reward, done, info = medium_env.step(action)
        assert "invalid" in reward.message.lower() or "valid" in reward.message.lower()

    def test_resize_no_size_specified(self, medium_env):
        """Resize without new_size should fail gracefully."""
        action = Action(action_type=ActionType.RESIZE, resource_id="i-0b2c3d4e5f6a70004")
        obs, reward, done, info = medium_env.step(action)
        assert "no new_size" in reward.message.lower() or "not specified" in reward.message.lower()


# ─── Stop Tests ───────────────────────────────────────────────────────────

class TestStop:
    def test_stop_instance(self, easy_env):
        """Stopping a non-critical instance should work."""
        # The stopped dev instance
        action = Action(action_type=ActionType.STOP, resource_id="i-0a1b2c3d4e5f60006")
        obs, reward, done, info = easy_env.step(action)
        # Already stopped — should report that
        assert "already stopped" in reward.message.lower()

    def test_stop_production_penalty(self, easy_env):
        """Stopping a production instance should give penalty."""
        action = Action(action_type=ActionType.STOP, resource_id="i-0a1b2c3d4e5f60007")
        obs, reward, done, info = easy_env.step(action)
        assert reward.value < 0
        assert len(info["safety_violations"]) > 0


# ─── Commit Tests ─────────────────────────────────────────────────────────

class TestCommit:
    def test_commit_ends_episode(self, easy_env):
        action = Action(action_type=ActionType.COMMIT_CHANGES)
        obs, reward, done, info = easy_env.step(action)
        assert done is True

    def test_commit_with_savings(self, easy_env):
        """Delete some resources, then commit and check score."""
        # Delete a few wasteful resources
        easy_env.step(Action(action_type=ActionType.DELETE, resource_id="vol-0a1b2c3d4e5f60001"))
        easy_env.step(Action(action_type=ActionType.DELETE, resource_id="vol-0a1b2c3d4e5f60002"))
        # Now commit
        obs, reward, done, info = easy_env.step(Action(action_type=ActionType.COMMIT_CHANGES))
        assert done is True
        assert info["cost_saved"] > 0

    def test_step_after_done_raises(self, easy_env):
        easy_env.step(Action(action_type=ActionType.COMMIT_CHANGES))
        with pytest.raises(RuntimeError, match="done"):
            easy_env.step(Action(action_type=ActionType.COMMIT_CHANGES))


# ─── Check Dependencies Tests ────────────────────────────────────────────

class TestCheckDeps:
    def test_check_deps_returns_info(self, hard_env):
        action = Action(action_type=ActionType.CHECK_DEPS, resource_id="i-0c3d4e5f6a7b80012")
        obs, reward, done, info = hard_env.step(action)
        assert "kafka" in obs.message.lower() or "cluster" in obs.message.lower() or "depend" in obs.message.lower()

    def test_check_deps_cost(self, hard_env):
        action = Action(action_type=ActionType.CHECK_DEPS, resource_id="i-0c3d4e5f6a7b80012")
        obs, reward, done, info = hard_env.step(action)
        assert reward.value < 0  # small cost


# ─── List Resources Tests ────────────────────────────────────────────────

class TestListResources:
    def test_list_resources(self, easy_env):
        action = Action(action_type=ActionType.LIST_RESOURCES)
        obs, reward, done, info = easy_env.step(action)
        assert len(obs.resources) == 10
        assert reward.value < 0  # small cost


# ─── State Tests ──────────────────────────────────────────────────────────

class TestState:
    def test_state_returns_full_info(self, easy_env):
        state = easy_env.state()
        assert state.task_id == "easy_orphan_cleanup"
        assert state.optimal_savings > 0
        assert len(state.critical_resources) > 0
        assert state.done is False

    def test_state_not_initialized_raises(self):
        env = CloudFinOpsEnvironment()
        with pytest.raises(RuntimeError, match="not initialized"):
            env.state()


# ─── Oracle Solution Tests ────────────────────────────────────────────────

class TestOracleSolution:
    def test_easy_oracle_solution(self, env):
        """Running the oracle solution should achieve near-perfect score."""
        from data.generator import load_solution
        solution = load_solution("easy_orphan_cleanup")
        env.reset(task_id="easy_orphan_cleanup")
        
        for step_data in solution["optimal_action_sequence"]:
            action = Action(
                action_type=step_data["action_type"],
                resource_id=step_data.get("resource_id"),
                new_size=step_data.get("new_size"),
                reason=step_data.get("reason"),
            )
            obs, reward, done, info = env.step(action)
        
        assert done is True
        assert info["cost_saved"] > 60  # Should save ~$63/mo
        assert len(info["safety_violations"]) == 0
        
        score = env.get_final_score()
        assert score > 0.8, f"Expected score > 0.8, got {score}"

    def test_medium_oracle_solution(self, env):
        """Running the medium oracle solution should achieve good score."""
        from data.generator import load_solution
        solution = load_solution("medium_rightsize")
        
        env.reset(task_id="medium_rightsize")
        
        for step_data in solution["optimal_action_sequence"]:
            action = Action(
                action_type=step_data["action_type"],
                resource_id=step_data.get("resource_id"),
                new_size=step_data.get("new_size"),
                reason=step_data.get("reason"),
            )
            obs, reward, done, info = env.step(action)
        
        assert done is True
        assert info["cost_saved"] > 900  # Should save ~$1037/mo
        assert len(info["safety_violations"]) == 0

    def test_hard_oracle_solution(self, env):
        """Running the hard oracle solution should achieve good score with no violations."""
        from data.generator import load_solution
        solution = load_solution("hard_dependency_migration")
        
        env.reset(task_id="hard_dependency_migration")
        
        for step_data in solution["optimal_action_sequence"]:
            action = Action(
                action_type=step_data["action_type"],
                resource_id=step_data.get("resource_id"),
                new_size=step_data.get("new_size"),
                reason=step_data.get("reason"),
            )
            obs, reward, done, info = env.step(action)
        
        assert done is True
        assert info["cost_saved"] > 2400  # ~$2513 optimal
        assert len(info["safety_violations"]) == 0, f"Oracle should have 0 violations, got: {info['safety_violations']}"
        
        score = env.get_final_score()
        assert score > 0.8, f"Expected score > 0.8, got {score}"


# ─── Maintenance Eligible Tests ───────────────────────────────────────────

class TestMaintenanceEligible:
    def test_resize_critical_with_maintenance_eligible(self, hard_env):
        """Resizing a critical resource WITH maintenance_eligible should NOT penalize."""
        action = Action(
            action_type=ActionType.RESIZE,
            resource_id="rds-0c3d4e5f6a7b80015",  # critical + maintenance_eligible
            new_size="db.r5.large",
        )
        obs, reward, done, info = hard_env.step(action)
        assert reward.value > 0  # should succeed
        assert len(info["safety_violations"]) == 0
