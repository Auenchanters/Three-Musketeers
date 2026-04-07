"""
Baseline inference script for CloudFinOpsEnv.

Uses an OpenAI-compatible client. Emits [START], [STEP], [END] structured
logs to stdout. All other output goes to stderr.

Env vars: API_BASE_URL, HF_TOKEN / OPENAI_API_KEY, MODEL_NAME, ENV_URL
"""

import os
import sys
import re
import json
import time
import traceback
import httpx
from openai import OpenAI
from client import CloudFinOpsClient
from models.action import Action, ActionType

INFERENCE_DEBUG = os.environ.get("INFERENCE_DEBUG", "0") == "1"

def _debug(msg: str):
    """Print debug info to stderr (never pollutes structured stdout)."""
    if INFERENCE_DEBUG:
        print(f"[DEBUG] {msg}", file=sys.stderr, flush=True)

def _info(msg: str):
    print(f"[INFO] {msg}", file=sys.stderr, flush=True)

def _warn(msg: str):
    print(f"[WARN] {msg}", file=sys.stderr, flush=True)

def _error(msg: str):
    print(f"[ERROR] {msg}", file=sys.stderr, flush=True)


API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY", "")
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860")


MAX_STEPS = 30
TASKS = ["easy_orphan_cleanup", "medium_rightsize", "hard_dependency_migration"]
TEMPERATURE = 0.2
MAX_TOKENS = 256
RETRY_DELAYS = [3, 6, 12]  # seconds between retries (21s worst case)

SYSTEM_PROMPT = """You are a FinOps agent optimizing cloud infrastructure costs.

IMPORTANT: You MUST respond with ONLY a single JSON object. No explanation, no reasoning, no text before or after. Just the JSON.

Available actions:
- query_metrics(resource_id): Get 7-day CPU/memory/network usage
- check_deps(resource_id): Check dependencies
- delete(resource_id, reason): Permanently remove a resource
- stop(resource_id, reason): Stop a running instance
- resize(resource_id, new_size, reason): Change size/tier
- list_resources(): Refresh resource list
- detach(resource_id): Detach a volume
- commit_changes(): Finalize changes and end episode

RULES:
- NEVER delete or stop resources tagged "production" or "critical"
- ALWAYS query_metrics and check_deps before deleting or resizing
- Optimize for maximum cost savings with zero production impact
- Attack the highest-cost resources first! Priority is to maximize savings.
- Do NOT re-check or re-resize resources you already optimized.
- Do NOT repeatedly query_metrics on the same resource once its metrics are available in the observation.
- Valid EC2 resize path example: m5.xlarge -> t3.small, m5.large -> t3.medium. Valid RDS Path: db.r5.xlarge -> db.t3.medium. S3 paths: STANDARD -> STANDARD_IA -> GLACIER.
- When you have optimized all wasteful resources, IMMEDIATELY use commit_changes
- If running low on steps, commit_changes to save your progress

WORKFLOW:
1. Identify high-cost non-production resources.
2. If metrics/deps are unknown, use query_metrics/check_deps.
3. If waste, use delete() or stop(). If over-provisioned (low CPU), use resize().
4. Repeat. Once all waste is handled, commit_changes().

RESPOND WITH ONLY JSON. Examples:
{"action_type": "query_metrics", "resource_id": "vol-abc123"}
{"action_type": "delete", "resource_id": "vol-abc123", "reason": "Detached volume, unused"}
{"action_type": "resize", "resource_id": "i-abc123", "new_size": "t3.small", "reason": "CPU avg 2%"}
{"action_type": "commit_changes", "reason": "All optimizations complete"}
"""



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



def format_observation(obs: dict) -> str:
    """Convert observation dict to LLM-friendly text."""
    steps_remaining = obs['max_steps'] - obs['step_number']
    lines = [
        f"=== TASK: {obs['task_description']} ===",
        f"Step: {obs['step_number']}/{obs['max_steps']} ({steps_remaining} steps remaining)",
        f"Total monthly cost: ${obs['total_monthly_cost']:.2f}",
        f"Cost saved so far: ${obs['cost_saved_so_far']:.2f}",
    ]
    # Urgency warning when running low on steps
    if steps_remaining <= 2:
        lines.append("*** URGENT: You must commit_changes NOW or lose all progress! ***")
    elif steps_remaining <= 4:
        lines.append("*** WARNING: Running low on steps. Consider using commit_changes soon. ***")
    if obs.get("budget_target"):
        lines.append(f"Budget target: ${obs['budget_target']:.2f}/month")
    if obs.get("maintenance_window"):
        lines.append(f"Maintenance window: {obs['maintenance_window']}")
    if obs.get("message"):
        lines.append(f"Last message: {obs['message']}")

    # Cost heatmap by resource type (helps agent prioritize)
    if obs.get("cost_breakdown"):
        lines.append("\n--- COST BREAKDOWN BY TYPE ---")
        for rtype, cost in sorted(obs["cost_breakdown"].items(), key=lambda x: -x[1]):
            lines.append(f"  {rtype}: ${cost:.2f}/mo")

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

    # Fallback: list_resources so we don't accidentally end the episode
    _warn(f"Could not parse LLM response: {text[:200]}")
    return {"action_type": "list_resources", "reason": "Could not parse action, listing resources."}



def get_agent_action(client: OpenAI, observation_text: str, history: list, parse_failures: int = 0) -> str:
    """Ask the LLM to decide the next action, with retry on failure."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": observation_text},
    ]
    # Include recent history for context
    # Filter out duplicate historical actions to avoid context flooding
    deduped_history = []
    seen_actions = set()
    for h in reversed(history):
        if h["action"] not in seen_actions:
            seen_actions.add(h["action"])
            deduped_history.insert(0, h)
        if len(deduped_history) >= 5:
            break

    for h in deduped_history:
        messages.append({"role": "assistant", "content": h["action"]})
        messages.append({"role": "user", "content": h["result"]})

    # If we've had parse failures, add a strong reminder
    if parse_failures > 0:
        messages.append({"role": "user", "content": (
            "REMINDER: Respond with ONLY a JSON object, no other text. "
            "Example: {\"action_type\": \"query_metrics\", \"resource_id\": \"vol-abc123\"}"
        )})

    last_err = None
    for attempt, delay in enumerate([0] + RETRY_DELAYS):
        if delay > 0:
            _warn(f"Retry {attempt}/{len(RETRY_DELAYS)} after {delay}s...")
            time.sleep(delay)
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False,
            )
            text = (completion.choices[0].message.content or "").strip()
            _debug(f"Raw LLM response: {text[:500]}")
            # Strip <think>...</think> tags from thinking models
            if "<think>" in text:
                text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
                _debug(f"After think-strip: {text[:300]}")
            return text
        except Exception as e:
            last_err = e
            _error(f"LLM call failed (attempt {attempt+1}): {e}")

    _error(f"All retries exhausted. Last error: {last_err}")
    return '{"action_type": "list_resources", "reason": "LLM error, listing resources."}'



def obs_to_dict(obs) -> dict:
    """Convert an Observation object (or dict) to a plain dict for format_observation."""
    if isinstance(obs, dict):
        return obs
    # Pydantic model → dict
    d = obs.model_dump() if hasattr(obs, "model_dump") else obs.__dict__
    # Flatten resources that are Pydantic models
    if "resources" in d:
        d["resources"] = [
            r.model_dump() if hasattr(r, "model_dump") else r
            for r in d["resources"]
        ]
    return d



def run_task(llm_client: OpenAI, env_url: str, task_name: str) -> float:
    """Run a single task using the OpenEnv WebSocket client and return the score."""
    history = []
    rewards = []
    parse_failures = 0

    log_start(task=task_name, env="CloudFinOpsEnv", model=MODEL_NAME)

    # Use the sync wrapper of the OpenEnv WebSocket client
    sync_client = CloudFinOpsClient(base_url=env_url).sync()

    with sync_client:
        # Reset environment
        result = sync_client.reset(task_id=task_name)
        obs = obs_to_dict(result.observation)

        for step_num in range(1, MAX_STEPS + 1):
            # Format observation for LLM
            obs_text = format_observation(obs)

            # Safety net: force commit on last step to preserve savings
            task_max_steps = obs.get("max_steps", MAX_STEPS)
            if obs.get("step_number", 0) >= task_max_steps - 1:
                _info(f"Auto-committing on step {step_num} (last step safety net)")
                action_text = '{"action_type": "commit_changes", "reason": "Final step — committing to preserve savings."}'
                action_dict = parse_action(action_text)
            else:
                # Get LLM action
                action_text = get_agent_action(llm_client, obs_text, history, parse_failures)
                action_dict = parse_action(action_text)

            # Track parse failures
            if action_dict.get("reason", "").startswith("Could not parse"):
                parse_failures += 1
            else:
                parse_failures = 0

            # If stuck in parse failure loop (3+ in a row), commit to save progress
            if parse_failures >= 3:
                _warn(f"3 consecutive parse failures — committing to save progress.")
                action_dict = {"action_type": "commit_changes", "reason": "Committing after repeated parse failures."}
                parse_failures = 0

            # Build typed Action object for the OpenEnv client
            action = Action(
                action_type=action_dict.get("action_type", "commit_changes"),
                resource_id=action_dict.get("resource_id"),
                new_size=action_dict.get("new_size"),
                reason=action_dict.get("reason"),
            )

            # Execute action
            result = sync_client.step(action)
            obs = obs_to_dict(result.observation)
            reward = result.reward if isinstance(result.reward, (int, float)) else 0.0
            done = result.done
            rewards.append(reward)

            log_step(step=step_num, action=action_dict, reward=reward, done=done)

            history.append({
                "action": action_text,
                "result": obs.get("message", ""),
            })

            if done:
                break

        # Get final state for scoring
        try:
            final_state = sync_client.state()
            state_dict = final_state.model_dump() if hasattr(final_state, "model_dump") else vars(final_state)
            _debug(f"Final state keys: {list(state_dict.keys())}")
            _debug(f"cost_saved={state_dict.get('cost_saved')}, optimal={state_dict.get('optimal_savings')}, violations={state_dict.get('safety_violations')}")
        except Exception as e:
            _error(f"Failed to get final state: {e}")
            state_dict = {}

    # Score calculation: use grader state if available, otherwise sum rewards
    score = min(max(sum(rewards), 0.0), 1.0)

    actual_savings = state_dict.get("cost_saved", 0)
    optimal_savings = state_dict.get("optimal_savings", 0)
    if optimal_savings > 0:
        ratio = actual_savings / optimal_savings
        has_violations = len(state_dict.get("safety_violations", [])) > 0
        _info(f"Savings: ${actual_savings:.2f} / ${optimal_savings:.2f} ({ratio:.0%}), violations={has_violations}")
        if has_violations:
            score = 0.0
        else:
            score = min(max(ratio - (step_num * 0.005), 0.0), 1.0)
    else:
        _warn(f"No optimal_savings in state — using reward-sum score: {score:.3f}")

    success = score >= 0.5
    log_end(success=success, steps=step_num, score=round(score, 3), rewards=rewards)

    return score


def main():
    """Run all tasks and report scores."""
    if not API_KEY:
        _warn("No API key found. Set HF_TOKEN or OPENAI_API_KEY.")
        _info("Running in dry-run mode -- will test env connectivity only.")

    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "dummy")

    # Health check
    try:
        health = httpx.get(f"{ENV_URL}/health", timeout=10.0)
        _info(f"Environment healthy: {health.json()}")
    except Exception as e:
        _error(f"Cannot connect to environment at {ENV_URL}: {e}")
        sys.exit(1)

    scores = {}
    for task in TASKS:
        try:
            score = run_task(llm_client, ENV_URL, task)
            scores[task] = score
            _info(f"Task {task}: {score:.3f}")
        except Exception as e:
            _error(f"Task {task} failed: {e}")
            traceback.print_exc(file=sys.stderr)
            scores[task] = 0.0

    # Print summary to stderr (stdout is reserved for structured logs)
    print("\n" + "=" * 60, file=sys.stderr)
    print("FINAL SCORES", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    for task, score in scores.items():
        status = "PASS" if score >= 0.5 else "FAIL"
        print(f"  {task}: {score:.3f} {status}", file=sys.stderr)
    avg = sum(scores.values()) / len(scores) if scores else 0
    print(f"\n  Average: {avg:.3f}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)


if __name__ == "__main__":
    main()
