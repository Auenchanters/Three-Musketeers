"""Core environment engine. Implements reset(), step(), and state()."""

from typing import Dict, List, Optional, Any, Set

from models import (
    UsageMetrics,
    Resource, Observation, Action, ActionType, Reward, EnvironmentState,
)
from data.generator import load_scenario, load_solution, load_pricing, get_valid_resize_targets
from engine.dependency_graph import DependencyGraph
from engine.reward_calculator import RewardCalculator
from engine.grader import Grader

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

class CloudFinOpsEnvironment(Environment[Action, Observation, EnvironmentState]):
    """
    The CloudFinOpsEnv environment.
    
    Lifecycle:
        1. reset(task_id) → Observation  (start a new episode)
        2. step(action) → (Observation, Reward, done, info)  (repeat)
        3. state() → EnvironmentState  (inspect god-mode state)
    """

    def __init__(self):
        self._task_id: str = ""
        self._difficulty: str = ""
        self._task_description: str = ""
        self._max_steps: int = 15
        self._budget_target: Optional[float] = None
        self._maintenance_window: Optional[str] = None

        self._resources: Dict[str, dict] = {}
        self._hidden_metrics: Dict[str, dict] = {}
        self._removed_resources: Set[str] = set()

        self._optimal_savings: float = 0.0
        self._critical_resources: List[str] = []
        self._wasteful_resources: List[str] = []
        self._rightsize_targets: Dict[str, dict] = {}
        self._dep_graph: Optional[DependencyGraph] = None

        self._cost_saved: float = 0.0
        self._cascade_penalty: float = 0.0
        self._penalties_incurred: float = 0.0
        self._steps_taken: int = 0
        self._done: bool = False
        self._safety_violations: List[str] = []
        self._actions_taken: List[str] = []
        self._message: str = ""

        self._pricing: Dict[str, Any] = {}
        self._initialized: bool = False

    # --- metadata ---

    def get_metadata(self) -> EnvironmentMetadata:
        """Return rich metadata for the /metadata endpoint."""
        return EnvironmentMetadata(
            name="CloudFinOpsEnv",
            description=(
                "An environment where LLM agents optimize cloud infrastructure costs "
                "by identifying orphaned resources, right-sizing over-provisioned instances, "
                "and safely pruning waste — without breaking production systems."
            ),
            version="1.0.0",
            author="Three Musketeers (Utkarsh, Mohit, Tanush)",
        )

    # --- reset ---

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, task_id: str = "easy_orphan_cleanup", **kwargs) -> Observation:
        """
        Start a new episode for the given task.
        
        Loads the scenario, initializes state, hides metrics,
        and returns the initial observation.
        """
        scenario = load_scenario(task_id)
        solution = load_solution(task_id)
        if not self._pricing:
            self._pricing = load_pricing()

        self._task_id = task_id
        self._difficulty = scenario["task_difficulty"]
        self._task_description = scenario["task_description"]
        self._max_steps = scenario["max_steps"]
        self._budget_target = scenario.get("budget_target")
        self._maintenance_window = scenario.get("maintenance_window")

        # Load resources, hiding metrics initially
        self._resources = {}
        self._hidden_metrics = {}
        for r in scenario["resources"]:
            rid = r["resource_id"]
            # Store metrics separately (hidden until queried)
            if "metrics" in r and r["metrics"] is not None:
                self._hidden_metrics[rid] = r["metrics"]
            # Store resource WITHOUT metrics
            resource_copy = {k: v for k, v in r.items() if k not in ("metrics", "_role")}
            resource_copy["metrics"] = None
            self._resources[rid] = resource_copy

        # Oracle data
        self._optimal_savings = solution["optimal_savings_monthly"]
        self._critical_resources = list(scenario.get("critical_resources", []))
        self._wasteful_resources = list(scenario.get("wasteful_resources", []))
        self._rightsize_targets = dict(scenario.get("rightsize_targets", {}))
        self._dep_graph = DependencyGraph(scenario.get("dependency_graph", {}))

        # Reset episode tracking
        self._removed_resources = set()
        self._cost_saved = 0.0
        self._cascade_penalty = 0.0
        self._penalties_incurred = 0.0
        self._steps_taken = 0
        self._done = False
        self._safety_violations = []
        self._actions_taken = []
        self._message = f"Episode started. Task: {self._task_description}"

        self._initialized = True
        return self._build_observation()

    # --- step ---

    def step(self, action: Action, timeout_s: Optional[float] = None, **kwargs) -> Observation:
        """
        Process an agent action and return (observation, reward, done, info).
        """
        if not self._initialized:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        self._steps_taken += 1
        action_str = f"{action.action_type.value}"
        if action.resource_id:
            action_str += f"({action.resource_id})"
        if action.new_size:
            action_str += f" → {action.new_size}"
        if action.reason:
            action_str += f" [{action.reason}]"
        self._actions_taken.append(action_str)

        # Dispatch to handler
        handler = {
            ActionType.QUERY_METRICS: self._handle_query_metrics,
            ActionType.DELETE: self._handle_delete,
            ActionType.STOP: self._handle_stop,
            ActionType.RESIZE: self._handle_resize,
            ActionType.DETACH: self._handle_detach,
            ActionType.COMMIT_CHANGES: self._handle_commit,
            ActionType.LIST_RESOURCES: self._handle_list_resources,
            ActionType.CHECK_DEPS: self._handle_check_deps,
        }.get(action.action_type)

        if handler is None:
            reward = RewardCalculator.invalid_action_reward(
                f"Unknown action type: {action.action_type}"
            )
        else:
            reward = handler(action)

        # Always update message from reward so obs.message reflects this action
        self._message = reward.message

        # Check if max steps reached
        if self._steps_taken >= self._max_steps and not self._done:
            self._done = True
            self._message += " | Max steps reached. Episode ended."

        info = {
            "steps_taken": self._steps_taken,
            "cost_saved": self._cost_saved,
            "safety_violations": list(self._safety_violations),
        }

        obs = self._build_observation()
        # The OpenEnv validator requires obs.reward to be strictly inside
        # (0, 1) on EVERY step response — not just the final one.
        # Use the Grader's clamped final score when done, otherwise clamp
        # the per-step reward (which can be negative or > 1).
        if self._done:
            raw_reward = self.get_final_score()
        else:
            raw_reward = reward.value
        obs.reward = round(min(max(float(raw_reward), 0.01), 0.99), 4)
        obs.done = self._done

        # Add metadata internally or append the info object
        obs.metadata = info

        return obs

    # --- state ---

    @property
    def state(self) -> EnvironmentState:
        """Return full internal state (god-mode, for grading/debugging)."""
        if not self._initialized:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        all_resources = []
        for rid, rdata in self._resources.items():
            if rid not in self._removed_resources:
                r_copy = dict(rdata)
                # Include hidden metrics in state view
                if r_copy.get("metrics") is None and rid in self._hidden_metrics:
                    r_copy["metrics"] = self._hidden_metrics[rid]
                all_resources.append(Resource(**r_copy))

        return EnvironmentState(
            task_id=self._task_id,
            task_difficulty=self._difficulty,
            resources=all_resources,
            optimal_savings=self._optimal_savings,
            critical_resources=self._critical_resources,
            dependency_graph={
                k: v for k, v in (
                    self._dep_graph._adjacency if self._dep_graph else {}
                ).items()
            },
            cost_saved=self._cost_saved,
            penalties_incurred=self._penalties_incurred,
            steps_taken=self._steps_taken,
            done=self._done,
            safety_violations=list(self._safety_violations),
        )

    # --- action handlers ---

    def _handle_query_metrics(self, action: Action) -> Reward:
        """Reveal hidden metrics for a resource."""
        rid = action.resource_id
        if not rid or rid not in self._resources:
            return self._invalid_resource(rid)
        if rid in self._removed_resources:
            return RewardCalculator.invalid_action_reward(
                f"Resource {rid} has been deleted."
            )

        # Reveal metrics
        if rid in self._hidden_metrics:
            self._resources[rid]["metrics"] = self._hidden_metrics[rid]
            metrics = self._hidden_metrics[rid]
            self._message = (
                f"Metrics for {rid}: "
                f"CPU avg={metrics['cpu_avg_7d']}%, peak={metrics['cpu_peak_7d']}%, "
                f"Mem avg={metrics['memory_avg_7d']}%, "
                f"Net in={metrics['network_in_gb_7d']}GB, out={metrics['network_out_gb_7d']}GB, "
                f"Last access: {metrics['last_accessed_days_ago']} days ago."
            )
        else:
            self._message = f"No metrics available for {rid}."

        reward = RewardCalculator.query_metrics_reward()
        self._message = reward.message + " " + self._message
        return reward

    def _handle_delete(self, action: Action) -> Reward:
        """Delete a resource. Check safety first."""
        rid = action.resource_id
        if not rid or rid not in self._resources:
            return self._invalid_resource(rid)
        if rid in self._removed_resources:
            return RewardCalculator.invalid_action_reward(
                f"Resource {rid} has already been deleted."
            )

        resource = self._resources[rid]

        # CHECK 1: Is it a critical/production resource?
        if rid in self._critical_resources:
            violation = f"Deleted production/critical resource {rid} ({resource.get('name', '')})"
            self._safety_violations.append(violation)
            self._penalties_incurred += 1.0
            self._removed_resources.add(rid)
            reward = RewardCalculator.delete_production_reward(rid)
            self._message = reward.message
            return reward

        # CHECK 2: Does it have active dependents?
        if self._dep_graph:
            active_deps = [
                d for d in self._dep_graph.get_dependents(rid)
                if d not in self._removed_resources
            ]
            if active_deps:
                self._penalties_incurred += 0.5
                self._removed_resources.add(rid)
                # Also track cascade penalties for hard grading
                resource_cost = resource.get("cost_per_hour", 0) * 730
                self._cascade_penalty += resource_cost * 0.1  # partial cascade
                reward = RewardCalculator.delete_with_deps_reward(rid, active_deps)
                self._message = reward.message
                return reward

        # CHECK 3: Would it break cluster quorum?
        if self._dep_graph:
            all_resource_dicts = [
                r for r in self._resources.values()
                if r["resource_id"] not in self._removed_resources
            ]
            if self._dep_graph.would_break_cluster_quorum(rid, all_resource_dicts, self._removed_resources):
                violation = f"Deleted {rid} which would break cluster quorum"
                self._safety_violations.append(violation)
                self._penalties_incurred += 0.5
                self._removed_resources.add(rid)
                reward = RewardCalculator.delete_with_deps_reward(rid, ["cluster-quorum"])
                self._message = reward.message
                return reward

        # SUCCESS: Safe to delete
        monthly_cost = round(resource.get("cost_per_hour", 0) * 730, 2)
        self._cost_saved += monthly_cost
        self._removed_resources.add(rid)
        reward = RewardCalculator.successful_delete_reward(rid, monthly_cost)
        self._message = reward.message
        return reward

    def _handle_stop(self, action: Action) -> Reward:
        """Stop a running instance."""
        rid = action.resource_id
        if not rid or rid not in self._resources:
            return self._invalid_resource(rid)
        if rid in self._removed_resources:
            return RewardCalculator.invalid_action_reward(f"Resource {rid} has been deleted.")

        resource = self._resources[rid]

        # Check if it's a critical resource
        if rid in self._critical_resources:
            violation = f"Stopped production/critical resource {rid} ({resource.get('name', '')})"
            self._safety_violations.append(violation)
            self._penalties_incurred += 1.0
            resource["status"] = "stopped"
            reward = RewardCalculator.delete_production_reward(rid)
            self._message = reward.message
            return reward

        # Check if already stopped
        if resource.get("status") == "stopped":
            return RewardCalculator.invalid_action_reward(f"Resource {rid} is already stopped.")

        # Stop it (saves money but less than delete since reversible)
        monthly_cost = round(resource.get("cost_per_hour", 0) * 730, 2)
        # Stopped instances still incur some cost (EBS, EIPs), so savings ≈ 70%
        effective_savings = round(monthly_cost * 0.7, 2)
        self._cost_saved += effective_savings
        resource["status"] = "stopped"

        reward = RewardCalculator.successful_stop_reward(rid, effective_savings)
        self._message = reward.message
        return reward

    def _handle_resize(self, action: Action) -> Reward:
        """Resize a resource to a different tier."""
        rid = action.resource_id
        new_size = action.new_size
        if not rid or rid not in self._resources:
            return self._invalid_resource(rid)
        if rid in self._removed_resources:
            return RewardCalculator.invalid_action_reward(f"Resource {rid} has been deleted.")
        if not new_size:
            return RewardCalculator.invalid_action_reward("No new_size specified for resize action.")

        resource = self._resources[rid]
        current_type = resource.get("tags", {}).get("instance_type", "")

        # For S3 buckets, handle storage class changes
        if resource.get("resource_type") == "s3_bucket":
            return self._handle_s3_resize(rid, resource, new_size)

        # Validate resize path
        valid_targets = get_valid_resize_targets(current_type)
        if new_size not in valid_targets:
            return RewardCalculator.invalid_action_reward(
                f"Invalid resize: {current_type} → {new_size}. Valid targets: {valid_targets}"
            )

        # Check if critical resource
        # Critical resources tagged 'maintenance_eligible' CAN be resized safely
        if rid in self._critical_resources:
            is_maintenance_eligible = resource.get("tags", {}).get("maintenance_eligible") == "true"
            if not is_maintenance_eligible:
                violation = f"Resized production/critical resource {rid} ({resource.get('name', '')})"
                self._safety_violations.append(violation)
                self._penalties_incurred += 0.2

        # Check if resize would cause performance issues
        metrics = resource.get("metrics") or self._hidden_metrics.get(rid, {})
        cpu_avg = metrics.get("cpu_avg_7d", 0)
        if cpu_avg > 80:
            reward = RewardCalculator.bad_resize_reward(rid)
            self._message = reward.message
            return reward

        # Calculate savings
        old_cost_hr = resource.get("cost_per_hour", 0)
        new_cost_hr = self._get_instance_cost(new_size, resource.get("resource_type", ""))
        if new_cost_hr is None:
            return RewardCalculator.invalid_action_reward(
                f"Unknown instance type: {new_size}"
            )

        monthly_savings = round((old_cost_hr - new_cost_hr) * 730, 2)
        if monthly_savings < 0:
            return RewardCalculator.invalid_action_reward(
                f"Resize {current_type} → {new_size} would increase costs by ${abs(monthly_savings):.2f}/month."
            )

        # Apply resize
        self._cost_saved += monthly_savings
        resource["cost_per_hour"] = new_cost_hr
        resource["tags"]["instance_type"] = new_size

        reward = RewardCalculator.successful_resize_reward(rid, monthly_savings)
        self._message = reward.message
        return reward

    def _handle_s3_resize(self, rid: str, resource: dict, new_class: str) -> Reward:
        """Handle S3 storage class changes (treated as resize)."""
        valid_classes = ["STANDARD", "STANDARD_IA", "GLACIER", "DEEP_ARCHIVE"]
        if new_class not in valid_classes:
            return RewardCalculator.invalid_action_reward(
                f"Invalid S3 storage class: {new_class}. Valid: {valid_classes}"
            )

        # Check if critical resource without maintenance eligibility
        if rid in self._critical_resources:
            is_maintenance_eligible = resource.get("tags", {}).get("maintenance_eligible") == "true"
            if not is_maintenance_eligible:
                violation = f"Modified production/critical S3 bucket {rid} ({resource.get('name', '')})"
                self._safety_violations.append(violation)
                self._penalties_incurred += 0.2

        # Check rightsize targets to get optimal pricing
        if rid in self._rightsize_targets:
            target = self._rightsize_targets[rid]
            old_cost_hr = target["current_cost_hr"]
            new_cost_hr = target["optimal_cost_hr"] if new_class == target["optimal"] else old_cost_hr * 0.5
        else:
            old_cost_hr = resource.get("cost_per_hour", 0)
            # Rough cost ratios for S3 classes
            ratios = {"STANDARD": 1.0, "STANDARD_IA": 0.5, "GLACIER": 0.22, "DEEP_ARCHIVE": 0.05}
            new_cost_hr = old_cost_hr * ratios.get(new_class, 0.5)

        monthly_savings = round((old_cost_hr - new_cost_hr) * 730, 2)
        self._cost_saved += monthly_savings
        resource["cost_per_hour"] = new_cost_hr
        resource["tags"]["storage_class"] = new_class

        reward = RewardCalculator.successful_resize_reward(rid, monthly_savings)
        self._message = reward.message
        return reward

    def _handle_detach(self, action: Action) -> Reward:
        """Detach a volume from its parent instance."""
        rid = action.resource_id
        if not rid or rid not in self._resources:
            return self._invalid_resource(rid)
        if rid in self._removed_resources:
            return RewardCalculator.invalid_action_reward(f"Resource {rid} has been deleted.")

        resource = self._resources[rid]
        if not resource.get("attached_to"):
            return RewardCalculator.invalid_action_reward(
                f"Resource {rid} is not attached to anything."
            )

        resource["attached_to"] = None
        resource["status"] = "detached"
        reward = RewardCalculator.detach_reward(rid)
        self._message = reward.message
        return reward

    def _handle_commit(self, action: Action) -> Reward:
        """Commit changes and end the episode."""
        self._done = True

        # Compute final score
        final_score = Grader.compute_final_score(
            actual_savings=self._cost_saved,
            optimal_savings=self._optimal_savings,
            steps_taken=self._steps_taken,
            safety_violations=self._safety_violations,
            difficulty=self._difficulty,
            cascade_penalty=self._cascade_penalty,
        )

        reward = RewardCalculator.commit_reward(self._cost_saved, self._optimal_savings)
        self._message = (
            f"{reward.message} | Final score: {final_score:.3f} | "
            f"Steps: {self._steps_taken} | Violations: {len(self._safety_violations)}"
        )
        return reward

    def _handle_list_resources(self, action: Action) -> Reward:
        """Refresh the resource list (no-op, agent can see resources in observation)."""
        active_count = sum(
            1 for rid in self._resources if rid not in self._removed_resources
        )
        reward = RewardCalculator.list_resources_reward()
        self._message = f"Resources refreshed. {active_count} active resources."
        return reward

    def _handle_check_deps(self, action: Action) -> Reward:
        """Check what depends on a resource."""
        rid = action.resource_id
        if not rid or rid not in self._resources:
            return self._invalid_resource(rid)
        if rid in self._removed_resources:
            return RewardCalculator.invalid_action_reward(f"Resource {rid} has been deleted.")

        resource = self._resources[rid]
        deps = self._dep_graph.get_dependencies(rid) if self._dep_graph else []
        dependents = self._dep_graph.get_dependents(rid) if self._dep_graph else []
        resource_deps = resource.get("dependencies", [])

        # Check for replication pairs
        rep_partner = self._dep_graph.is_replication_pair(resource) if self._dep_graph else None
        # Check for circular dependencies
        peer = self._dep_graph.get_circular_peer(resource) if self._dep_graph else None

        parts = [f"Dependencies for {rid} ({resource.get('name', '')})"]
        if resource_deps:
            parts.append(f"  Depends on: {resource_deps}")
        if deps:
            parts.append(f"  Graph dependencies: {deps}")
        if dependents:
            parts.append(f"  Depended on by: {dependents}")
        if rep_partner:
            parts.append(f"  Replication partner: {rep_partner}")
        if peer:
            parts.append(f"  Circular dependency peer: {peer}")
        if resource.get("tags", {}).get("cluster"):
            cluster = resource["tags"]["cluster"]
            quorum = resource["tags"].get("quorum_min", "?")
            parts.append(f"  Cluster: {cluster} (quorum min: {quorum})")
        if not deps and not dependents and not resource_deps and not rep_partner and not peer:
            parts.append("  No dependencies found.")

        reward = RewardCalculator.check_deps_reward()
        self._message = " | ".join(parts)
        return reward

    # --- helpers ---

    def _build_observation(self) -> Observation:
        """Build the agent-visible observation."""
        active_resources = []
        total_cost_hourly = 0.0

        for rid, rdata in self._resources.items():
            if rid in self._removed_resources:
                continue
            # Build Resource model (with or without metrics depending on visibility)
            r = Resource(
                resource_id=rdata["resource_id"],
                resource_type=rdata["resource_type"],
                name=rdata["name"],
                status=rdata["status"],
                cost_per_hour=rdata["cost_per_hour"],
                tags={k: v for k, v in rdata.get("tags", {}).items() if not k.startswith("_")},
                created_days_ago=rdata["created_days_ago"],
                attached_to=rdata.get("attached_to"),
                dependencies=rdata.get("dependencies", []),
                metrics=UsageMetrics(**rdata["metrics"]) if rdata.get("metrics") else None,
            )
            active_resources.append(r)
            total_cost_hourly += rdata["cost_per_hour"]

        total_monthly = round(total_cost_hourly * 730, 2)

        # Compute cost heatmap by resource type
        cost_by_type: Dict[str, float] = {}
        for r in active_resources:
            rtype = r.resource_type.value if hasattr(r.resource_type, "value") else str(r.resource_type)
            cost_by_type[rtype] = round(cost_by_type.get(rtype, 0.0) + r.cost_per_hour * 730, 2)

        return Observation(
            task_description=self._task_description,
            resources=active_resources,
            total_monthly_cost=total_monthly,
            budget_target=self._budget_target,
            maintenance_window=self._maintenance_window,
            step_number=self._steps_taken,
            max_steps=self._max_steps,
            message=self._message,
            cost_saved_so_far=round(self._cost_saved, 2),
            actions_taken=list(self._actions_taken),
            cost_breakdown=cost_by_type,
        )

    def _invalid_resource(self, rid: Optional[str]) -> Reward:
        """Return invalid action reward for missing/invalid resource."""
        msg = f"Resource '{rid}' not found." if rid else "No resource_id specified."
        reward = RewardCalculator.invalid_action_reward(msg)
        self._message = msg
        return reward

    def _get_instance_cost(self, instance_type: str, resource_type: str) -> Optional[float]:
        """Look up cost per hour for an instance type from pricing data."""
        if instance_type.startswith("db."):
            instances = self._pricing.get("rds_instances", {})
        else:
            instances = self._pricing.get("ec2_instances", {})

        if instance_type in instances:
            return instances[instance_type].get("cost_per_hour")
        return None

    def get_final_score(self) -> float:
        """Get the final graded score for the episode."""
        return Grader.compute_final_score(
            actual_savings=self._cost_saved,
            optimal_savings=self._optimal_savings,
            steps_taken=self._steps_taken,
            safety_violations=self._safety_violations,
            difficulty=self._difficulty,
            cascade_penalty=self._cascade_penalty,
        )
