"""Quick sanity check: verify ALL possible obs.reward values are in (0, 1)."""
import os, sys
os.environ.setdefault("HF_TOKEN", "dummy")

from engine.environment import CloudFinOpsEnvironment
from models.action import Action, ActionType
from openenv.core.env_server.serialization import serialize_observation

def check(label, obs):
    """Check obs.reward and what serialize_observation produces."""
    serialized = serialize_observation(obs)
    r = serialized["reward"]
    print(f"  {label}: obs.reward={obs.reward!r}  serialized.reward={r!r}  done={obs.done}", end="")
    if r is None:
        print("  *** NONE - could be problem! ***")
        return False
    elif not (0 < r < 1):
        print(f"  *** OUT OF RANGE! ***")
        return False
    else:
        print("  OK")
        return True

all_ok = True

for task_id in ["easy_orphan_cleanup", "medium_rightsize", "hard_dependency_migration"]:
    print(f"\n=== {task_id} ===")
    env = CloudFinOpsEnvironment()
    
    # Check RESET observation
    obs = env.reset(task_id=task_id)
    ok = check("reset", obs)
    if not ok:
        all_ok = False
    
    # Check a few step observations
    obs = env.step(Action(action_type=ActionType.LIST_RESOURCES))
    ok = check("list_resources", obs)
    if not ok:
        all_ok = False
    
    # Query metrics on first resource
    res_id = obs.resources[0].resource_id if obs.resources else None
    if res_id:
        obs = env.step(Action(action_type=ActionType.QUERY_METRICS, resource_id=res_id))
        ok = check("query_metrics", obs)
        if not ok:
            all_ok = False
    
    # Commit
    obs = env.step(Action(action_type=ActionType.COMMIT_CHANGES, reason="test"))
    ok = check("commit (done)", obs)
    if not ok:
        all_ok = False

print()
if all_ok:
    print("ALL CHECKS PASSED")
else:
    print("*** SOME CHECKS FAILED ***")
    sys.exit(1)
