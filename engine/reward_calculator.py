"""
CloudFinOpsEnv — Reward Calculator

Implements the per-step reward shaping strategy from the blueprint.
Pure functions — no side effects, fully deterministic, fully testable.
"""

from models.reward import Reward
from typing import Dict, Optional


# ─── Reward Constants (from Blueprint Section 6) ─────────────────────────
REWARD_CORRECT_DELETE = 0.10        # correct delete of orphaned resource
REWARD_CORRECT_RESIZE = 0.08       # correct resize (saves money)
REWARD_CORRECT_STOP = 0.05         # correct stop of idle resource
REWARD_COMMIT_EXCELLENT = 0.30     # commit with savings > 80% optimal
REWARD_COMMIT_GOOD = 0.15          # commit with savings > 50% optimal

PENALTY_DELETE_PRODUCTION = -1.00  # delete production resource
PENALTY_DELETE_WITH_DEPS = -0.50   # delete resource with active dependencies
PENALTY_BAD_RESIZE = -0.20         # resize that would cause performance issues
PENALTY_STEP = -0.01               # per-step cost

COST_QUERY_METRICS = -0.005        # investigation action
COST_CHECK_DEPS = -0.003           # dependency check
COST_LIST_RESOURCES = -0.002       # refresh resource list

# Savings → reward scaling: +reward per $10/month saved
SAVINGS_PER_UNIT = 10.0


class RewardCalculator:
    """Computes per-step rewards for the CloudFinOpsEnv."""

    @staticmethod
    def query_metrics_reward() -> Reward:
        """Reward for querying metrics (small investigation cost)."""
        return Reward(
            value=COST_QUERY_METRICS + PENALTY_STEP,
            breakdown={
                "investigation_cost": COST_QUERY_METRICS,
                "step_cost": PENALTY_STEP,
            },
            message="Queried resource metrics.",
        )

    @staticmethod
    def check_deps_reward() -> Reward:
        """Reward for checking dependencies (very small cost)."""
        return Reward(
            value=COST_CHECK_DEPS + PENALTY_STEP,
            breakdown={
                "investigation_cost": COST_CHECK_DEPS,
                "step_cost": PENALTY_STEP,
            },
            message="Checked resource dependencies.",
        )

    @staticmethod
    def list_resources_reward() -> Reward:
        """Reward for refreshing the resource list."""
        return Reward(
            value=COST_LIST_RESOURCES + PENALTY_STEP,
            breakdown={
                "investigation_cost": COST_LIST_RESOURCES,
                "step_cost": PENALTY_STEP,
            },
            message="Refreshed resource list.",
        )

    @staticmethod
    def delete_production_reward(resource_id: str) -> Reward:
        """Catastrophic penalty for deleting a production resource."""
        return Reward(
            value=PENALTY_DELETE_PRODUCTION + PENALTY_STEP,
            breakdown={
                "safety_penalty": PENALTY_DELETE_PRODUCTION,
                "step_cost": PENALTY_STEP,
            },
            message=f"CRITICAL: You deleted production resource {resource_id}! Safety multiplier set to 0.0.",
        )

    @staticmethod
    def delete_with_deps_reward(resource_id: str, dep_ids: list) -> Reward:
        """Penalty for deleting a resource with active dependencies."""
        deps_str = ", ".join(dep_ids)
        return Reward(
            value=PENALTY_DELETE_WITH_DEPS + PENALTY_STEP,
            breakdown={
                "safety_penalty": PENALTY_DELETE_WITH_DEPS,
                "step_cost": PENALTY_STEP,
            },
            message=f"WARNING: Resource {resource_id} has active dependencies: [{deps_str}]",
        )

    @staticmethod
    def successful_delete_reward(resource_id: str, monthly_savings: float) -> Reward:
        """Positive reward for correctly deleting a wasteful resource."""
        savings_reward = (monthly_savings / SAVINGS_PER_UNIT) * REWARD_CORRECT_DELETE
        return Reward(
            value=savings_reward + PENALTY_STEP,
            breakdown={
                "savings": round(savings_reward, 4),
                "step_cost": PENALTY_STEP,
            },
            message=f"Deleted {resource_id}. Saving ${monthly_savings:.2f}/month.",
        )

    @staticmethod
    def successful_stop_reward(resource_id: str, monthly_savings: float) -> Reward:
        """Positive reward for correctly stopping an idle resource."""
        savings_reward = (monthly_savings / SAVINGS_PER_UNIT) * REWARD_CORRECT_STOP
        return Reward(
            value=savings_reward + PENALTY_STEP,
            breakdown={
                "savings": round(savings_reward, 4),
                "step_cost": PENALTY_STEP,
            },
            message=f"Stopped {resource_id}. Saving ${monthly_savings:.2f}/month.",
        )

    @staticmethod
    def successful_resize_reward(resource_id: str, monthly_savings: float) -> Reward:
        """Positive reward for correctly resizing an over-provisioned resource."""
        savings_reward = (monthly_savings / SAVINGS_PER_UNIT) * REWARD_CORRECT_RESIZE
        return Reward(
            value=savings_reward + PENALTY_STEP,
            breakdown={
                "savings": round(savings_reward, 4),
                "step_cost": PENALTY_STEP,
            },
            message=f"Resized {resource_id}. Saving ${monthly_savings:.2f}/month.",
        )

    @staticmethod
    def bad_resize_reward(resource_id: str) -> Reward:
        """Penalty for a resize that would cause performance issues."""
        return Reward(
            value=PENALTY_BAD_RESIZE + PENALTY_STEP,
            breakdown={
                "safety_penalty": PENALTY_BAD_RESIZE,
                "step_cost": PENALTY_STEP,
            },
            message=f"WARNING: Resizing {resource_id} would cause performance issues (CPU avg > 80% after downsize).",
        )

    @staticmethod
    def invalid_action_reward(message: str) -> Reward:
        """Penalty for an invalid action (e.g., resource not found)."""
        return Reward(
            value=PENALTY_STEP,
            breakdown={"step_cost": PENALTY_STEP},
            message=message,
        )

    @staticmethod
    def commit_reward(
        actual_savings: float,
        optimal_savings: float,
    ) -> Reward:
        """
        Reward for committing changes (episode end).
        
        Includes bonus for excellent or good performance.
        """
        ratio = actual_savings / optimal_savings if optimal_savings > 0 else 0.0
        bonus = 0.0
        msg = ""

        if ratio >= 0.8:
            bonus = REWARD_COMMIT_EXCELLENT
            msg = f"Excellent! Saved ${actual_savings:.2f} ({ratio:.0%} of optimal ${optimal_savings:.2f})."
        elif ratio >= 0.5:
            bonus = REWARD_COMMIT_GOOD
            msg = f"Good. Saved ${actual_savings:.2f} ({ratio:.0%} of optimal ${optimal_savings:.2f})."
        else:
            msg = f"Committed. Saved ${actual_savings:.2f} ({ratio:.0%} of optimal ${optimal_savings:.2f})."

        return Reward(
            value=bonus,
            breakdown={
                "commit_bonus": bonus,
                "savings_ratio": round(ratio, 4),
            },
            message=msg,
        )

    @staticmethod
    def detach_reward(resource_id: str) -> Reward:
        """Reward for detaching a volume (neutral action)."""
        return Reward(
            value=PENALTY_STEP,
            breakdown={"step_cost": PENALTY_STEP},
            message=f"Detached {resource_id} from its parent.",
        )
