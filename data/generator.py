"""
CloudFinOpsEnv — Data Loader

Loads curated scenario JSON files and returns parsed resources + oracle data.
No randomness — all data is hand-crafted JSON fixtures with real AWS pricing.
"""

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, Any, List, Optional

# Base paths
DATA_DIR = Path(__file__).parent
SCENARIOS_DIR = DATA_DIR / "scenarios"
SOLUTIONS_DIR = DATA_DIR / "solutions"
PRICING_DIR = DATA_DIR / "pricing"

# Task ID → filename mapping
SCENARIO_FILES = {
    "easy_orphan_cleanup": "easy_orphan_cleanup.json",
    "medium_rightsize": "medium_rightsize.json",
    "hard_dependency_migration": "hard_dependency_migration.json",
}

SOLUTION_FILES = {
    "easy_orphan_cleanup": "easy_solution.json",
    "medium_rightsize": "medium_solution.json",
    "hard_dependency_migration": "hard_solution.json",
}


def load_scenario(task_id: str) -> Dict[str, Any]:
    """
    Load a scenario by task_id.

    Returns a dict with keys:
        - task_id: str
        - task_difficulty: str
        - task_description: str
        - max_steps: int
        - budget_target: Optional[float]
        - maintenance_window: Optional[str]
        - resources: List[dict]
        - critical_resources: List[str]
        - dependency_graph: Dict[str, List[str]]
        - wasteful_resources: List[str]
        - rightsize_targets: Dict (medium/hard only)
        - _cost_analysis: Dict
    """
    if task_id not in SCENARIO_FILES:
        available = ", ".join(SCENARIO_FILES.keys())
        raise ValueError(f"Unknown task_id: '{task_id}'. Available: {available}")

    scenario_path = SCENARIOS_DIR / SCENARIO_FILES[task_id]
    with open(scenario_path, "r", encoding="utf-8") as f:
        scenario = json.load(f)

    return scenario


def load_solution(task_id: str) -> Dict[str, Any]:
    """
    Load the oracle solution for a task.

    Returns a dict with keys:
        - task_id: str
        - optimal_savings_monthly: float
        - optimal_action_sequence: List[dict]
    """
    if task_id not in SOLUTION_FILES:
        available = ", ".join(SOLUTION_FILES.keys())
        raise ValueError(f"Unknown task_id: '{task_id}'. Available: {available}")

    solution_path = SOLUTIONS_DIR / SOLUTION_FILES[task_id]
    with open(solution_path, "r", encoding="utf-8") as f:
        solution = json.load(f)

    return solution


@lru_cache(maxsize=1)
def load_pricing() -> Dict[str, Any]:
    """
    Load the AWS pricing reference data (cached after first call).

    Returns a dict with keys:
        - ec2_instances: Dict[instance_type → {vcpu, memory_gb, cost_per_hour}]
        - rds_instances: Dict[instance_type → {vcpu, memory_gb, cost_per_hour}]
        - ebs_volumes: Dict[volume_type → {cost_per_gb_month, cost_per_gb_hour}]
        - other_services: Dict[service → {cost_per_hour, note}]
        - valid_resize_paths: Dict[current_type → List[valid_target_types]]
    """
    pricing_path = PRICING_DIR / "aws_instance_pricing.json"
    with open(pricing_path, "r", encoding="utf-8") as f:
        pricing = json.load(f)

    return pricing


def get_available_tasks() -> List[str]:
    """Return list of available task IDs."""
    return list(SCENARIO_FILES.keys())


def get_optimal_savings(task_id: str) -> float:
    """Get the oracle-computed optimal savings for a task."""
    solution = load_solution(task_id)
    return solution["optimal_savings_monthly"]


def get_valid_resize_targets(current_type: str) -> List[str]:
    """Get valid resize targets for a given instance type."""
    pricing = load_pricing()
    paths = pricing.get("valid_resize_paths", {})
    return paths.get(current_type, [])
