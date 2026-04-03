"""
Tests for data/generator.py — Verify all scenarios, solutions, and pricing load correctly.
"""

import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.generator import (
    load_scenario, load_solution, load_pricing,
    get_available_tasks, get_optimal_savings, get_valid_resize_targets,
)
from models import Resource, ResourceType, ResourceStatus, UsageMetrics


# ─── Scenario Loading ─────────────────────────────────────────────────────

class TestScenarioLoading:
    def test_available_tasks(self):
        tasks = get_available_tasks()
        assert len(tasks) == 3
        assert "easy_orphan_cleanup" in tasks
        assert "medium_rightsize" in tasks
        assert "hard_dependency_migration" in tasks

    @pytest.mark.parametrize("task_id", get_available_tasks())
    def test_load_scenario_returns_dict(self, task_id):
        scenario = load_scenario(task_id)
        assert isinstance(scenario, dict)
        assert scenario["task_id"] == task_id

    @pytest.mark.parametrize("task_id", get_available_tasks())
    def test_scenario_has_required_keys(self, task_id):
        scenario = load_scenario(task_id)
        required = ["task_id", "task_difficulty", "task_description", "max_steps",
                     "resources", "critical_resources", "dependency_graph",
                     "wasteful_resources", "_cost_analysis"]
        for key in required:
            assert key in scenario, f"Missing key: {key}"

    def test_easy_has_10_resources(self):
        scenario = load_scenario("easy_orphan_cleanup")
        assert len(scenario["resources"]) == 10

    def test_medium_has_20_resources(self):
        scenario = load_scenario("medium_rightsize")
        assert len(scenario["resources"]) == 20

    def test_hard_has_35_resources(self):
        scenario = load_scenario("hard_dependency_migration")
        assert len(scenario["resources"]) == 35

    def test_invalid_task_raises(self):
        with pytest.raises(ValueError, match="Unknown task_id"):
            load_scenario("nonexistent_task")

    @pytest.mark.parametrize("task_id", get_available_tasks())
    def test_critical_resources_exist_in_resources(self, task_id):
        scenario = load_scenario(task_id)
        resource_ids = {r["resource_id"] for r in scenario["resources"]}
        for crit_id in scenario["critical_resources"]:
            assert crit_id in resource_ids, f"Critical resource {crit_id} not in resources"

    @pytest.mark.parametrize("task_id", get_available_tasks())
    def test_wasteful_resources_exist_in_resources(self, task_id):
        scenario = load_scenario(task_id)
        resource_ids = {r["resource_id"] for r in scenario["resources"]}
        for waste_id in scenario["wasteful_resources"]:
            assert waste_id in resource_ids, f"Wasteful resource {waste_id} not in resources"

    @pytest.mark.parametrize("task_id", get_available_tasks())
    def test_no_overlap_critical_wasteful(self, task_id):
        """Critical and wasteful sets must not overlap."""
        scenario = load_scenario(task_id)
        crit = set(scenario["critical_resources"])
        waste = set(scenario["wasteful_resources"])
        overlap = crit & waste
        assert len(overlap) == 0, f"Overlap: {overlap}"


# ─── Solution Loading ─────────────────────────────────────────────────────

class TestSolutionLoading:
    @pytest.mark.parametrize("task_id", get_available_tasks())
    def test_load_solution_returns_dict(self, task_id):
        solution = load_solution(task_id)
        assert isinstance(solution, dict)
        assert solution["task_id"] == task_id

    @pytest.mark.parametrize("task_id", get_available_tasks())
    def test_solution_has_optimal_savings(self, task_id):
        solution = load_solution(task_id)
        assert "optimal_savings_monthly" in solution
        assert solution["optimal_savings_monthly"] > 0

    @pytest.mark.parametrize("task_id", get_available_tasks())
    def test_solution_has_action_sequence(self, task_id):
        solution = load_solution(task_id)
        assert "optimal_action_sequence" in solution
        assert len(solution["optimal_action_sequence"]) > 0

    @pytest.mark.parametrize("task_id", get_available_tasks())
    def test_solution_ends_with_commit(self, task_id):
        solution = load_solution(task_id)
        last_action = solution["optimal_action_sequence"][-1]
        assert last_action["action_type"] == "commit_changes"

    @pytest.mark.parametrize("task_id", get_available_tasks())
    def test_optimal_savings_matches_cost_analysis(self, task_id):
        scenario = load_scenario(task_id)
        solution = load_solution(task_id)
        assert abs(
            solution["optimal_savings_monthly"]
            - scenario["_cost_analysis"]["optimal_savings_monthly"]
        ) < 0.01

    def test_get_optimal_savings_helper(self):
        savings = get_optimal_savings("easy_orphan_cleanup")
        assert savings == 63.22


# ─── Pricing Loading ──────────────────────────────────────────────────────

class TestPricingLoading:
    def test_load_pricing_returns_dict(self):
        pricing = load_pricing()
        assert isinstance(pricing, dict)

    def test_pricing_has_ec2_instances(self):
        pricing = load_pricing()
        ec2 = pricing["ec2_instances"]
        assert "t3.micro" in ec2
        assert "m5.xlarge" in ec2

    def test_pricing_has_rds_instances(self):
        pricing = load_pricing()
        rds = pricing["rds_instances"]
        assert "db.t3.medium" in rds
        assert "db.r5.2xlarge" in rds

    def test_pricing_has_resize_paths(self):
        pricing = load_pricing()
        assert "valid_resize_paths" in pricing
        assert "m5.xlarge" in pricing["valid_resize_paths"]

    def test_get_valid_resize_targets(self):
        targets = get_valid_resize_targets("m5.xlarge")
        assert len(targets) > 0
        assert "t3.small" in targets  # cross-family resize

    def test_ec2_pricing_values(self):
        pricing = load_pricing()
        t3_micro = pricing["ec2_instances"]["t3.micro"]
        assert t3_micro["cost_per_hour"] == 0.0104
        assert t3_micro["vcpu"] == 2


# ─── Resource Model Parsing ───────────────────────────────────────────────

class TestResourceParsing:
    @pytest.mark.parametrize("task_id", get_available_tasks())
    def test_resources_parse_to_pydantic(self, task_id):
        """All scenario resources should parse into valid Resource models."""
        scenario = load_scenario(task_id)
        for rdata in scenario["resources"]:
            # Remove internal field
            clean = {k: v for k, v in rdata.items() if not k.startswith("_")}
            if clean.get("metrics"):
                clean["metrics"] = UsageMetrics(**clean["metrics"])
            resource = Resource(**clean)
            assert resource.resource_id is not None
            assert resource.cost_per_hour >= 0
