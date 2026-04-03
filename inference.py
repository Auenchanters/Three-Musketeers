"""
CloudFinOpsEnv — Baseline Inference Script

Uses OpenAI-compatible client per hackathon requirements.
Emits [START], [STEP], [END] structured logs.
Calls the environment server via HTTP.

Environment Variables:
    API_BASE_URL  — Base URL for the LLM API
    HF_TOKEN / OPENAI_API_KEY  — API key
    MODEL_NAME    — Model to use (default: meta-llama/Meta-Llama-3-8B-Instruct)
    ENV_URL       — CloudFinOpsEnv server URL (default: http://localhost:7860)
"""

import os
import sys
import json
import httpx
from openai import OpenAI

# ─── Environment Variables ────────────────────────────────────────────────

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY", "")
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860")

# ─── Constants ────────────────────────────────────────────────────────────

MAX_STEPS = 30
TASKS = ["easy_orphan_cleanup", "medium_rightsize", "hard_dependency_migration"]
TEMPERATURE = 0.2
MAX_TOKENS = 500

SYSTEM_PROMPT = """You are a FinOps agent optimizing cloud infrastructure costs.
You have access to these actions:
- query_metrics(resource_id): Get 7-day CPU/memory/network usage for a resource
- check_deps(resource_id): Check what other resources depend on this one
- delete(resource_id, reason): Permanently remove a resource
- stop(resource_id, reason): Stop a running instance
- resize(resource_id, new_size, reason): Change the size/tier of a resource
- list_resources(): Refresh the resource list
- commit_changes(): Finalize your changes and end the episode

RULES:
- NEVER delete resources tagged "production" or "critical"
- ALWAYS query_metrics and check_deps before deleting or resizing
- Optimize for maximum cost savings with zero production impact
- Provide a reason for every action

Respond with a single JSON action, e.g.:
{"action_type": "query_metrics", "resource_id": "vol-abc123"}
{"action_type": "delete", "resource_id": "vol-abc123", "reason": "Detached volume, unused for 200 days"}
{"action_type": "resize", "resource_id": "i-abc123", "new_size": "t3.small", "reason": "CPU avg 2%"}
{"action_type": "commit_changes", "reason": "All optimizations complete"}
"""


# ─── Structured Logging ──────────────────────────────────────────────────

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


# ─── Observation Formatting ──────────────────────────────────────────────

def format_observation(obs: dict) -> str:
    """Convert observation dict to LLM-friendly text."""
    lines = [
        f"=== TASK: {obs['task_description']} ===",
        f"Step: {obs['step_number']}/{obs['max_steps']}",
        f"Total monthly cost: ${obs['total_monthly_cost']:.2f}",
        f"Cost saved so far: ${obs['cost_saved_so_far']:.2f}",
    ]
    if obs.get("budget_target"):
        lines.append(f"Budget target: ${obs['budget_target']:.2f}/month")
    if obs.get("maintenance_window"):
        lines.append(f"Maintenance window: {obs['maintenance_window']}")
    if obs.get("message"):
        lines.append(f"Last message: {obs['message']}")

    lines.append(f"\n--- RESOURCES ({len(obs['resources'])} active) ---")
    for r in obs["resources"]:
        monthly = r["cost_per_hour"] * 730
        line = f"  [{r['resource_type']}] {r['resource_id']} | {r['name']} | status={r['status']} | ${monthly:.2f}/mo"
        tags = r.get("tags", {})
        tag_str = ", ".join(f"{k}={v}" for k, v in tags.items() if not k.startswith("_"))
        if tag_str:
            line += f" | tags: {tag_str}"
        if r.get("attached_to"):
            line += f" | attached_to: {r['attached_to']}"
        if r.get("dependencies"):
            line += f" | deps: {r['dependencies']}"
        if r.get("metrics"):
            m = r["metrics"]
            line += f" | CPU avg={m['cpu_avg_7d']}% peak={m['cpu_peak_7d']}% | Mem avg={m['memory_avg_7d']}%"
        lines.append(line)

    return "\n".join(lines)


# ─── Action Parsing ──────────────────────────────────────────────────────

def parse_action(text: str) -> dict:
    """Parse LLM response text into action dict."""
    text = text.strip()

    # Try to find JSON in the response
    # Handle markdown code blocks
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    # Try to find JSON object
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        json_str = text[start:end]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    # Fallback: commit if we can't parse
    return {"action_type": "commit_changes", "reason": "Could not parse action, committing."}


# ─── Agent Logic ─────────────────────────────────────────────────────────

def get_agent_action(client: OpenAI, observation_text: str, history: list) -> str:
    """Ask the LLM to decide the next action."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": observation_text},
    ]
    # Include recent history for context
    for h in history[-5:]:
        messages.append({"role": "assistant", "content": h["action"]})
        messages.append({"role": "user", "content": h["result"]})

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return text
    except Exception as e:
        print(f"[ERROR] LLM call failed: {e}", flush=True)
        return '{"action_type": "commit_changes", "reason": "LLM error, committing."}'


# ─── Environment API Client ──────────────────────────────────────────────

class EnvClient:
    """HTTP client for the CloudFinOpsEnv server."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=30.0)

    def reset(self, task_id: str) -> dict:
        """POST /reset"""
        resp = self.client.post(f"{self.base_url}/reset", json={"task_id": task_id})
        resp.raise_for_status()
        return resp.json()

    def step(self, action: dict) -> dict:
        """POST /step"""
        resp = self.client.post(f"{self.base_url}/step", json=action)
        resp.raise_for_status()
        return resp.json()

    def state(self) -> dict:
        """GET /state"""
        resp = self.client.get(f"{self.base_url}/state")
        resp.raise_for_status()
        return resp.json()

    def health(self) -> dict:
        """GET /health"""
        resp = self.client.get(f"{self.base_url}/health")
        resp.raise_for_status()
        return resp.json()

    def close(self):
        self.client.close()


# ─── Main Loop ────────────────────────────────────────────────────────────

def run_task(llm_client: OpenAI, env_client: EnvClient, task_name: str) -> float:
    """Run a single task and return the score."""
    history = []
    rewards = []

    log_start(task=task_name, env="CloudFinOpsEnv", model=MODEL_NAME)

    # Reset environment
    obs = env_client.reset(task_name)

    for step_num in range(1, MAX_STEPS + 1):
        # Format observation for LLM
        obs_text = format_observation(obs)

        # Get LLM action
        action_text = get_agent_action(llm_client, obs_text, history)
        action = parse_action(action_text)

        # Execute action
        result = env_client.step(action)
        obs = result["observation"]
        reward = result["reward"]["value"] if isinstance(result["reward"], dict) else result["reward"]
        done = result["done"]
        rewards.append(reward)

        log_step(step=step_num, action=action, reward=reward, done=done)

        history.append({
            "action": action_text,
            "result": obs.get("message", ""),
        })

        if done:
            break

    # Get final state for scoring
    final_state = env_client.state()
    score = min(max(sum(rewards), 0.0), 1.0)

    # Use actual graded score from state if available
    actual_savings = final_state.get("cost_saved", 0)
    optimal_savings = final_state.get("optimal_savings", 1)
    if optimal_savings > 0:
        ratio = actual_savings / optimal_savings
        has_violations = len(final_state.get("safety_violations", [])) > 0
        if has_violations:
            score = 0.0
        else:
            score = min(max(ratio - (step_num * 0.005), 0.0), 1.0)

    success = score >= 0.5
    log_end(success=success, steps=step_num, score=round(score, 3), rewards=rewards)

    return score


def main():
    """Run all tasks and report scores."""
    if not API_KEY:
        print("[WARNING] No API key found. Set HF_TOKEN or OPENAI_API_KEY.", flush=True)
        print("[INFO] Running in dry-run mode — will test env connectivity only.", flush=True)

    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "dummy")
    env_client = EnvClient(ENV_URL)

    # Health check
    try:
        health = env_client.health()
        print(f"[INFO] Environment healthy: {health}", flush=True)
    except Exception as e:
        print(f"[ERROR] Cannot connect to environment at {ENV_URL}: {e}", flush=True)
        sys.exit(1)

    scores = {}
    for task in TASKS:
        try:
            score = run_task(llm_client, env_client, task)
            scores[task] = score
            print(f"\n{'='*50}")
            print(f"Task {task}: {score:.3f}")
            print(f"{'='*50}\n")
        except Exception as e:
            print(f"[ERROR] Task {task} failed: {e}", flush=True)
            scores[task] = 0.0

    env_client.close()

    print("\n" + "=" * 60)
    print("FINAL SCORES")
    print("=" * 60)
    for task, score in scores.items():
        status = "✓ PASS" if score >= 0.5 else "✗ FAIL"
        print(f"  {task}: {score:.3f} {status}")
    avg = sum(scores.values()) / len(scores) if scores else 0
    print(f"\n  Average: {avg:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
