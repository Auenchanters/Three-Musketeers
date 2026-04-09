"""Deep check: test EVERY possible reward path and edge case for (0, 1) compliance."""
import os, sys, json
os.environ.setdefault("HF_TOKEN", "dummy")

from engine.environment import CloudFinOpsEnvironment
from models.action import Action, ActionType
from openenv.core.env_server.serialization import serialize_observation

FAIL = False

def check(label, obs, context=""):
    global FAIL
    serialized = serialize_observation(obs)
    r = serialized["reward"]
    d = serialized["done"]
    status = "OK"
    if r is None:
        status = "*** NONE ***"
        FAIL = True
    elif not (0 < r < 1):
        status = f"*** OUT OF RANGE: {r} ***"
        FAIL = True
    print(f"  {label}: reward={r!r} done={d} {status} {context}")

def run_full_episode(task_id):
    """Run through a realistic episode hitting many code paths."""
    print(f"\n{'='*60}")
    print(f"TASK: {task_id}")
    print(f"{'='*60}")
    
    env = CloudFinOpsEnvironment()
    
    # RESET
    obs = env.reset(task_id=task_id)
    check("reset", obs)
    
    # LIST_RESOURCES
    obs = env.step(Action(action_type=ActionType.LIST_RESOURCES))
    check("list_resources", obs)
    
    # Get all resource IDs
    resources = obs.resources
    print(f"  -> {len(resources)} resources found")
    
    # QUERY_METRICS on every resource
    for r in resources[:5]:  # limit to 5
        obs = env.step(Action(action_type=ActionType.QUERY_METRICS, resource_id=r.resource_id))
        check(f"query_metrics({r.resource_id})", obs)
        if obs.done:
            break
    
    if not obs.done:
        # CHECK_DEPS on first resource
        if resources:
            obs = env.step(Action(action_type=ActionType.CHECK_DEPS, resource_id=resources[0].resource_id))
            check(f"check_deps({resources[0].resource_id})", obs)
    
    if not obs.done:
        # Try DELETE on a non-critical resource (find one)
        for r in resources:
            tags = r.tags or {}
            if tags.get("env") not in ("production", "critical"):
                obs = env.step(Action(action_type=ActionType.DELETE, resource_id=r.resource_id, reason="test"))
                check(f"delete({r.resource_id})", obs, f"tags={tags}")
                break
    
    if not obs.done:
        # Try invalid resource
        obs = env.step(Action(action_type=ActionType.DELETE, resource_id="nonexistent-123", reason="test"))
        check("delete(invalid)", obs)
    
    if not obs.done:
        # Try STOP on a resource
        for r in resources:
            tags = r.tags or {}
            if tags.get("env") not in ("production", "critical") and r.status.value == "running":
                obs = env.step(Action(action_type=ActionType.STOP, resource_id=r.resource_id, reason="test"))
                check(f"stop({r.resource_id})", obs)
                break
    
    if not obs.done:
        # Try DETACH on a resource with attached_to
        for r in resources:
            if r.attached_to:
                obs = env.step(Action(action_type=ActionType.DETACH, resource_id=r.resource_id))
                check(f"detach({r.resource_id})", obs)
                break
    
    if not obs.done:
        # COMMIT
        obs = env.step(Action(action_type=ActionType.COMMIT_CHANGES, reason="done"))
        check("commit", obs)
    
    # Check final state
    state = env.state
    state_dict = state.model_dump()
    print(f"\n  Final state:")
    print(f"    cost_saved={state_dict['cost_saved']}")
    print(f"    optimal_savings={state_dict['optimal_savings']}")
    print(f"    steps_taken={state_dict['steps_taken']}")
    print(f"    safety_violations={state_dict['safety_violations']}")
    if state_dict['optimal_savings'] > 0:
        ratio = state_dict['cost_saved'] / state_dict['optimal_savings']
        print(f"    raw_ratio={ratio}")
        if ratio == 0.0 or ratio == 1.0:
            print(f"    *** WARNING: raw ratio is {ratio} — if validator reads this, it FAILS ***")

def test_production_delete(task_id):
    """Test deleting a critical resource — catastrophic penalty."""
    print(f"\n--- Testing production delete on {task_id} ---")
    env = CloudFinOpsEnvironment()
    obs = env.reset(task_id=task_id)
    
    # Find a critical resource
    state = env.state
    critical = state.critical_resources
    if not critical:
        print("  No critical resources, skipping")
        return
    
    rid = critical[0]
    print(f"  Deleting critical resource: {rid}")
    obs = env.step(Action(action_type=ActionType.DELETE, resource_id=rid, reason="test"))
    check("delete_production", obs)
    
    if not obs.done:
        obs = env.step(Action(action_type=ActionType.COMMIT_CHANGES, reason="done"))
        check("commit_after_violation", obs)
    
    state = env.state
    sd = state.model_dump()
    print(f"  violations={sd['safety_violations']}")
    from engine.grader import Grader
    score = Grader.compute_final_score(
        actual_savings=sd['cost_saved'],
        optimal_savings=sd['optimal_savings'],
        steps_taken=sd['steps_taken'],
        safety_violations=sd['safety_violations'],
        difficulty=sd['task_difficulty'],
    )
    print(f"  grader score={score}")
    if score <= 0 or score >= 1:
        print(f"  *** GRADER SCORE OUT OF RANGE: {score} ***")
        global FAIL
        FAIL = True

def test_max_steps_exhaustion(task_id):
    """Exhaust max steps by doing list_resources repeatedly."""
    print(f"\n--- Testing max steps exhaustion on {task_id} ---")
    env = CloudFinOpsEnvironment()
    obs = env.reset(task_id=task_id)
    
    max_steps = obs.max_steps
    print(f"  max_steps={max_steps}")
    
    for i in range(max_steps + 2):  # try going over
        if obs.done:
            print(f"  Done at step {i}")
            break
        obs = env.step(Action(action_type=ActionType.LIST_RESOURCES))
        check(f"step_{i+1}", obs)
    
    state = env.state
    sd = state.model_dump()
    from engine.grader import Grader
    score = Grader.compute_final_score(
        actual_savings=sd['cost_saved'],
        optimal_savings=sd['optimal_savings'],
        steps_taken=sd['steps_taken'],
        safety_violations=sd['safety_violations'],
        difficulty=sd['task_difficulty'],
    )
    print(f"  grader score after exhaustion={score}")
    if score <= 0 or score >= 1:
        print(f"  *** GRADER SCORE OUT OF RANGE: {score} ***")
        global FAIL
        FAIL = True

# Run everything
for task in ["easy_orphan_cleanup", "medium_rightsize", "hard_dependency_migration"]:
    run_full_episode(task)
    test_production_delete(task)
    test_max_steps_exhaustion(task)

print(f"\n{'='*60}")
if FAIL:
    print("*** SOME CHECKS FAILED ***")
    sys.exit(1)
else:
    print("ALL DEEP CHECKS PASSED")
