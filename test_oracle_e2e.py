"""
CloudFinOpsEnv — Oracle End-to-End Validation

Runs the pre-computed optimal solutions against the environment to prove
that the full pipeline works correctly: reset -> step -> grading -> scoring.

NO LLM or API credits required. This uses deterministic oracle solutions.

Usage:
    python test_oracle_e2e.py                     # Test against local server
    python test_oracle_e2e.py --url https://...   # Test against deployed HF Space
"""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from client import CloudFinOpsClient
from models.action import Action, ActionType
from data.generator import load_solution, get_available_tasks


def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step, action, reward, done, error=None):
    print(
        f"[STEP] step={step} action={json.dumps(action)} reward={reward} done={done} error={error}",
        flush=True,
    )


def log_end(success, steps, score, rewards):
    print(
        f"[END] success={success} steps={steps} score={score} rewards={json.dumps(rewards)}",
        flush=True,
    )


def run_oracle_task(env_url, task_id):
    """Run the oracle solution for a single task and return results."""
    solution = load_solution(task_id)
    optimal_savings = solution["optimal_savings_monthly"]
    actions = solution["optimal_action_sequence"]

    log_start(task=task_id, env="CloudFinOpsEnv", model="oracle-solution")

    sync_client = CloudFinOpsClient(base_url=env_url).sync()
    rewards = []
    steps_taken = 0

    with sync_client:
        result = sync_client.reset(task_id=task_id)
        obs = result.observation
        print(f"  Resources: {len(obs.resources)} | Monthly cost: ${obs.total_monthly_cost:.2f}", flush=True)

        for i, step_data in enumerate(actions, 1):
            action = Action(
                action_type=step_data["action_type"],
                resource_id=step_data.get("resource_id"),
                new_size=step_data.get("new_size"),
                reason=step_data.get("reason"),
            )

            result = sync_client.step(action)
            obs = result.observation
            reward = result.reward if isinstance(result.reward, (int, float)) else 0.0
            done = result.done
            rewards.append(reward)
            steps_taken = i

            action_dict = {
                "action_type": step_data["action_type"],
                "resource_id": step_data.get("resource_id"),
            }
            if step_data.get("new_size"):
                action_dict["new_size"] = step_data["new_size"]

            log_step(step=i, action=action_dict, reward=reward, done=done)

            if done:
                break

    # Compute score from observation data (cost_saved_so_far from last obs).
    # Hackathon validator requires scores strictly inside (0, 1), so clamp
    # to [0.01, 0.99] regardless of source value.
    actual_savings = obs.cost_saved_so_far if hasattr(obs, "cost_saved_so_far") else 0
    # No safety violations if all rewards were non-catastrophic (no -1.0 penalties)
    has_violations = any(r <= -0.9 for r in rewards)

    if optimal_savings > 0 and not has_violations:
        score = actual_savings / optimal_savings
    else:
        score = 0.01
    score = max(0.01, min(0.99, float(score)))

    success = score >= 0.5
    log_end(success=success, steps=steps_taken, score=round(score, 4), rewards=rewards)

    return {
        "task_id": task_id,
        "score": round(score, 3),
        "success": success,
        "steps": steps_taken,
        "cost_saved": round(actual_savings, 2),
        "optimal_savings": optimal_savings,
        "safety_violations": [] if not has_violations else ["catastrophic penalty detected"],
        "savings_ratio": round(actual_savings / optimal_savings, 3) if optimal_savings > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Oracle E2E Validation for CloudFinOpsEnv")
    parser.add_argument("--url", default="http://localhost:7860", help="Environment server URL")
    args = parser.parse_args()

    env_url = os.environ.get("ENV_URL", args.url)

    print("=" * 60)
    print("CloudFinOpsEnv - Oracle End-to-End Validation")
    print(f"Environment: {env_url}")
    print("Model: oracle-solution (deterministic, no LLM needed)")
    print("=" * 60)
    print()

    # Health check
    try:
        import httpx
        health = httpx.get(f"{env_url}/health", timeout=10.0)
        print(f"Health check: {health.json()}")
    except Exception as e:
        print(f"ERROR: Cannot connect to {env_url}: {e}")
        print("Start the server first: uvicorn app:app --host 0.0.0.0 --port 7860")
        print("Or run via Docker:      docker run -p 7860:7860 cloudfinopsenv")
        sys.exit(1)

    print()
    results = {}
    tasks = get_available_tasks()

    for task_id in tasks:
        print(f"\n{'='*60}")
        try:
            result = run_oracle_task(env_url, task_id)
            results[task_id] = result
        except Exception as e:
            print(f"ERROR running {task_id}: {e}")
            import traceback
            traceback.print_exc()
            results[task_id] = {"score": 0.0, "success": False, "error": str(e)}

    # Final report
    print(f"\n\n{'='*60}")
    print("ORACLE VALIDATION RESULTS")
    print("=" * 60)

    all_passed = True
    for task_id, result in results.items():
        score = result.get("score", 0)
        status = "PASS" if result.get("success") else "FAIL"
        violations = len(result.get("safety_violations", []))
        savings = result.get("cost_saved", 0)
        optimal = result.get("optimal_savings", 0)
        ratio = result.get("savings_ratio", 0)

        print(f"\n  {task_id}:")
        print(f"    Score:      {score:.3f} ({status})")
        print(f"    Savings:    ${savings:.2f} / ${optimal:.2f} ({ratio:.0%} of optimal)")
        print(f"    Steps:      {result.get('steps', 0)}")
        print(f"    Violations: {violations}")

        if not result.get("success"):
            all_passed = False

    avg = sum(r.get("score", 0) for r in results.values()) / len(results) if results else 0
    print(f"\n  Average Score: {avg:.3f}")
    print("=" * 60)

    if all_passed:
        print("\n  ALL TASKS PASSED - Environment is working correctly!")
        print("  Ready for competition submission.")
    else:
        print("\n  Some tasks did not reach 0.5 threshold.")
        print("  Check safety violations and savings above.")

    print("=" * 60)


if __name__ == "__main__":
    main()
