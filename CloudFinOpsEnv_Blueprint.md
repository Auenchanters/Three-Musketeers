# CloudFinOpsEnv — Complete Architecture Blueprint
## Meta PyTorch OpenEnv Hackathon | Team: Three Musketeers

---

## 1. ENVIRONMENT IDENTITY

**Name:** `CloudFinOpsEnv`  
**Tagline:** "An OpenEnv environment where LLM agents learn to optimize cloud infrastructure costs by identifying waste, right-sizing resources, and safely pruning orphaned assets — without breaking production."

**Why this wins (mapped to rubric):**
- **Real-world utility (30%):** Cloud waste is a $100B+ problem. Every DevOps/FinOps team does this manually.
- **Task/grader quality (25%):** Deterministic oracle grader with mathematical formula. No ambiguity.
- **Environment design (20%):** Rich multi-step episodes with query→analyze→act→verify loop.
- **Code quality (15%):** FastAPI backend, typed Pydantic models, clean OpenEnv spec compliance.
- **Creativity (10%):** Not in any example domain list. Fresh, relatable, immediately impressive.

---

## 2. PYDANTIC MODELS (OpenEnv Typed Models)

### Observation Model
```python
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from enum import Enum

class ResourceType(str, Enum):
    EC2_INSTANCE = "ec2_instance"
    EBS_VOLUME = "ebs_volume"
    RDS_INSTANCE = "rds_instance"
    S3_BUCKET = "s3_bucket"
    ELASTIC_IP = "elastic_ip"
    NAT_GATEWAY = "nat_gateway"
    LOAD_BALANCER = "load_balancer"

class ResourceStatus(str, Enum):
    RUNNING = "running"
    STOPPED = "stopped"
    DETACHED = "detached"
    IDLE = "idle"

class Resource(BaseModel):
    resource_id: str                          # e.g., "i-0abc123", "vol-xyz789"
    resource_type: ResourceType
    name: str                                 # human-readable name tag
    status: ResourceStatus
    cost_per_hour: float                      # USD
    tags: Dict[str, str]                      # e.g., {"env": "production", "team": "backend"}
    created_days_ago: int
    attached_to: Optional[str] = None         # parent resource_id (e.g., volume → instance)
    dependencies: List[str] = []              # resource_ids that depend on this
    metrics: Optional[Dict[str, float]] = None  # only populated after QueryMetrics

class UsageMetrics(BaseModel):
    cpu_avg_7d: float          # 0.0 - 100.0
    cpu_peak_7d: float
    memory_avg_7d: float       # 0.0 - 100.0
    memory_peak_7d: float
    network_in_gb_7d: float
    network_out_gb_7d: float
    disk_iops_avg_7d: float
    last_accessed_days_ago: int

class Observation(BaseModel):
    """What the agent sees at each step."""
    task_description: str                     # natural language task brief
    resources: List[Resource]                 # current cloud state
    total_monthly_cost: float                 # current total $/month
    budget_target: Optional[float] = None     # desired $/month (for medium/hard)
    maintenance_window: Optional[str] = None  # time window for changes (hard task)
    step_number: int
    max_steps: int
    message: str                              # environment feedback after last action
    cost_saved_so_far: float
    actions_taken: List[str]                  # history of agent's actions this episode
```

### Action Model
```python
class ActionType(str, Enum):
    QUERY_METRICS = "query_metrics"       # inspect a resource's usage
    DELETE = "delete"                      # permanently remove a resource
    STOP = "stop"                         # stop a running instance
    RESIZE = "resize"                     # change instance/db tier
    DETACH = "detach"                     # detach a volume from instance
    COMMIT_CHANGES = "commit_changes"     # end episode, finalize savings
    LIST_RESOURCES = "list_resources"     # re-list all resources (refresh view)
    CHECK_DEPENDENCIES = "check_deps"    # check what depends on a resource

class Action(BaseModel):
    """What the agent does at each step."""
    action_type: ActionType
    resource_id: Optional[str] = None     # target resource
    new_size: Optional[str] = None        # for resize: "t3.small" → "t3.micro"
    reason: Optional[str] = None          # agent's justification (for logging)
```

### Reward Model
```python
class Reward(BaseModel):
    """Feedback after each step."""
    value: float                          # the numeric reward
    breakdown: Dict[str, float]           # {"savings": 0.1, "safety_penalty": 0.0, "step_cost": -0.01}
    message: str                          # human-readable explanation
```

---

## 3. ENVIRONMENT STATE MACHINE

### State Model
```python
class EnvironmentState(BaseModel):
    """Internal state returned by state()."""
    task_id: str
    task_difficulty: str                  # "easy", "medium", "hard"
    resources: List[Resource]             # full resource list with hidden metadata
    optimal_savings: float                # god-mode: max possible savings
    critical_resources: List[str]         # god-mode: resource_ids that must NOT be deleted
    dependency_graph: Dict[str, List[str]]  # resource_id → [dependent_ids]
    cost_saved: float
    penalties_incurred: float
    steps_taken: int
    done: bool
    safety_violations: List[str]          # log of critical mistakes
```

---

## 4. CORE API IMPLEMENTATION

### reset() → Observation
```
1. Select task (easy/medium/hard) based on task_id parameter
2. Generate fresh cloud environment state from task template
3. Compute optimal_savings (oracle pre-calculates best possible outcome)
4. Initialize step counter, cost tracker, violation log
5. Return initial Observation with task description + resource list
```

### step(action) → (Observation, Reward, Done, Info)
```
For each action type:

QUERY_METRICS:
  - Populate the target resource's metrics field with UsageMetrics
  - Reward: -0.005 (small step cost, encourages efficiency)
  - This is the "investigation" action — agent should do this before acting

DELETE:
  - If resource is in critical_resources → SAFETY VIOLATION
    - Reward: -1.0, safety_multiplier drops to 0.0
    - Message: "CRITICAL: You deleted production resource {id}!"
  - If resource has active dependencies → PARTIAL PENALTY  
    - Reward: -0.5
    - Message: "WARNING: Resource {id} has active dependencies: {deps}"
  - If resource is genuinely orphaned/wasteful → SUCCESS
    - cost_saved += resource.cost_per_hour * 730 (monthly)
    - Reward: +0.1 per $10/month saved
    - Message: "Deleted {id}. Saving ${amount}/month."

STOP:
  - Similar logic to delete but less risky (can be restarted)
  - Reward: +0.05 per $10/month saved (less than delete since reversible)

RESIZE:
  - Check if new_size is valid for the resource type
  - Compute cost difference
  - Reward: +0.08 per $10/month saved
  - If resize breaks performance (cpu_avg > 80% and downsizing) → penalty

CHECK_DEPENDENCIES:
  - Returns dependency info in observation message
  - Reward: -0.003 (very small cost, encourages due diligence)

COMMIT_CHANGES:
  - Episode ends (done = True)
  - Final score computed by oracle grader
  - Reward: final_score value

LIST_RESOURCES:
  - Refreshes resource list in observation
  - Reward: -0.002
```

### state() → EnvironmentState
```
Returns full internal state including oracle data (for debugging/grading)
```

---

## 5. THREE TASKS (Easy → Medium → Hard)

### Task 1: EASY — "Orphan Cleanup"
**Objective:** Delete orphaned/detached resources that cost money but serve no purpose.

**Cloud state (10 resources):**
- 3 detached EBS volumes (obvious waste — no attached instance)
- 2 unused Elastic IPs (not attached to any instance)
- 1 stopped EC2 instance (stopped 90 days ago, no recent access)
- 4 production resources (must NOT touch)

**What makes it easy:**
- Orphaned resources have `status: "detached"` or `attached_to: null`
- No dependencies to worry about
- Tags clearly separate production from waste

**Grader formula:**
```
optimal_savings = sum(cost of 6 wasteful resources) * 730  # monthly
score = (actual_savings / optimal_savings) * safety_multiplier
safety_multiplier = 0.0 if any production resource deleted, else 1.0
```

**Expected baseline score:** 0.7–0.9 (LLM should handle this well)

---

### Task 2: MEDIUM — "Right-Size & Prune"
**Objective:** Reduce costs by right-sizing oversized instances AND cleaning up waste, while respecting environment tags (don't touch production).

**Cloud state (20 resources):**
- 3 orphaned volumes (same as easy)
- 2 EC2 instances with avg CPU < 5% (massively over-provisioned, can downsize)
- 1 RDS instance on db.r5.2xlarge but only using 10% capacity
- 2 idle NAT gateways (no traffic in 30 days)
- 1 load balancer with zero targets
- 11 production resources (must NOT touch, some with tricky tags)

**What makes it medium:**
- Agent must QUERY metrics before deciding (can't tell from status alone)
- Resize decisions require understanding instance tiers (t3.large → t3.small)
- Some resources look idle but are tagged "staging" (safe to touch) vs "production" (not safe)
- Budget target provided — agent must figure out which combination of actions meets it

**Grader formula:**
```
optimal_savings = pre-computed best resize + delete combination
score = (actual_savings / optimal_savings) * safety_multiplier - (steps * 0.005)
safety_multiplier = 0.0 if production deleted, 0.7 if staging wrongly deleted
```

**Expected baseline score:** 0.4–0.6

---

### Task 3: HARD — "Dependency-Aware Migration"
**Objective:** Optimize costs in a complex environment with inter-resource dependencies, a maintenance window constraint, and cascading effects.

**Cloud state (35 resources):**
- Everything from medium PLUS:
- 3 EC2 instances forming a cluster (delete one → others lose quorum)
- 1 RDS primary with a read replica (must migrate both or neither)
- 2 resources with circular dependency tags (A depends on B, B depends on A — agent must figure out it's safe to resize both)
- A "maintenance window" constraint: some changes can only be made to resources tagged "maintenance_eligible"
- Hidden waste: an S3 bucket with lifecycle policy that should be changed (requires multi-step: check → analyze → resize)
- Decoy resources: look expensive but are critical (high cost + "production" tag + active dependencies)

**What makes it hard for frontier LLMs:**
- **Long-horizon planning:** Agent must check dependencies BEFORE acting, build a mental model of the graph
- **Constraint satisfaction:** Maintenance window limits which resources can be touched
- **Cascading effects:** Deleting resource A changes the state of resources B and C
- **Ambiguity:** Some resources appear wasteful but are critical (requires reading tags carefully + checking deps)
- **Optimal path is non-obvious:** The highest-savings action might trigger a cascade that costs more than it saves

**Grader formula:**
```
optimal_savings = pre-computed (accounts for dependencies and cascades)
cascade_penalty = sum(cost of unintended side-effects)
score = ((actual_savings - cascade_penalty) / optimal_savings) * safety_mult - (steps * 0.003)
safety_multiplier = 0.0 if any critical resource impacted
score = clamp(score, 0.0, 1.0)
```

**Expected baseline score:** 0.1–0.3 (frontier models will struggle with dependency reasoning)

---

## 6. REWARD SHAPING STRATEGY

### Per-Step Rewards (continuous signal throughout trajectory)
| Event                                 | Reward   | Rationale                                    |
|---------------------------------------|----------|----------------------------------------------|
| Correct delete of orphaned resource   | +0.10    | Direct progress toward goal                  |
| Correct resize (saves money)          | +0.08    | Good action, slightly less impactful         |
| Correct stop of idle resource         | +0.05    | Conservative but valid                       |
| Query metrics (investigation)         | -0.005   | Small cost to discourage spam querying       |
| Check dependencies                    | -0.003   | Even cheaper, encourage due diligence        |
| List resources                        | -0.002   | Nearly free refresh                          |
| Delete production resource            | -1.00    | Catastrophic — instant heavy penalty         |
| Delete resource with dependencies     | -0.50    | Broke something but not production           |
| Bad resize (undersized for workload)  | -0.20    | Would cause performance issues               |
| Each step taken                       | -0.01    | Time pressure, encourages efficiency         |
| Commit with savings > 80% optimal     | +0.30    | Bonus for excellent performance              |
| Commit with savings > 50% optimal     | +0.15    | Bonus for good performance                   |

### Why this works:
- **Not binary:** Agent gets feedback at EVERY step, not just at the end
- **Partial credit:** Saving 60% of optimal waste still gets a decent score
- **Penalizes degenerate behavior:** Querying everything costs steps, deleting blindly costs heavily
- **Rewards investigation:** The small query cost means smart agents query first, act second

---

## 7. SYNTHETIC DATA GENERATION

### Resource Templates (Python dict factories)
```python
# Example: generate an orphaned EBS volume
def make_orphaned_volume():
    return Resource(
        resource_id=f"vol-{uuid4().hex[:8]}",
        resource_type=ResourceType.EBS_VOLUME,
        name=f"old-backup-{random.choice(['dev', 'test', 'staging'])}-{randint(1,99)}",
        status=ResourceStatus.DETACHED,
        cost_per_hour=round(random.uniform(0.01, 0.15), 3),
        tags={"env": random.choice(["dev", "test"]), "team": random.choice(["backend", "data", "ml"])},
        created_days_ago=random.randint(30, 365),
        attached_to=None,       # KEY SIGNAL: not attached to anything
        dependencies=[],
    )

# Example: generate a production EC2 instance (must not touch)
def make_production_instance():
    return Resource(
        resource_id=f"i-{uuid4().hex[:8]}",
        resource_type=ResourceType.EC2_INSTANCE,
        name=f"prod-api-{random.choice(['primary', 'secondary', 'worker'])}-{randint(1,10)}",
        status=ResourceStatus.RUNNING,
        cost_per_hour=round(random.uniform(0.20, 1.50), 3),
        tags={"env": "production", "team": "platform", "critical": "true"},
        created_days_ago=random.randint(1, 180),
        attached_to=None,
        dependencies=[...],     # other resources depend on this
        metrics=None,           # hidden until queried
    )
```

### Data Generation Rules:
- Each task has a TEMPLATE that defines the mix of resources
- Templates are seeded (deterministic) so scores are reproducible
- Metrics are pre-generated and only revealed when agent calls QUERY_METRICS
- Dependencies are defined in a graph (adjacency list) stored in state
- Oracle pre-computes optimal_savings by finding the best safe action sequence

---

## 8. INFERENCE.PY STRUCTURE

```python
"""
CloudFinOpsEnv Baseline Inference Script
Uses OpenAI client per hackathon requirements.
Emits [START], [STEP], [END] structured logs.
"""
import asyncio, os, json
from openai import OpenAI

# --- ENV VARS (hackathon requirement) ---
API_BASE_URL = os.environ.get("API_BASE_URL")
API_KEY = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY")
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")

# --- CONSTANTS ---
MAX_STEPS = 30          # per episode
TASKS = ["easy_orphan_cleanup", "medium_rightsize", "hard_dependency_migration"]
TEMPERATURE = 0.2       # low temp for consistent baseline
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
"""

def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error=None):
    print(f"[STEP] step={step} action={json.dumps(action)} reward={reward} done={done} error={error}", flush=True)

def log_end(success, steps, score, rewards):
    print(f"[END] success={success} steps={steps} score={score} rewards={json.dumps(rewards)}", flush=True)

def get_agent_action(client, observation_text, history):
    """Ask the LLM to decide the next action."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": observation_text},
    ]
    # Include recent history for context
    for h in history[-5:]:
        messages.append({"role": "assistant", "content": h["action"]})
        messages.append({"role": "user", "content": h["result"]})

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        stream=False,
    )
    text = (completion.choices[0].message.content or "").strip()
    return text

async def run_task(client, env, task_name):
    """Run a single task and return the score."""
    history = []
    rewards = []
    
    log_start(task=task_name, env="CloudFinOpsEnv", model=MODEL_NAME)
    
    result = await env.reset(task=task_name)
    
    for step in range(1, MAX_STEPS + 1):
        if result.done:
            break
        
        # Format observation as text for the LLM
        obs_text = format_observation(result.observation)
        
        # Get LLM's action
        action_text = get_agent_action(client, obs_text, history)
        action = parse_action(action_text)  # JSON → Action pydantic model
        
        # Execute action
        result = await env.step(action)
        reward = result.reward or 0.0
        rewards.append(reward)
        
        log_step(step=step, action=action_text, reward=reward, done=result.done, error=None)
        
        history.append({"action": action_text, "result": result.observation.message})
        
        if result.done:
            break
    
    score = compute_final_score(rewards, result)
    score = min(max(score, 0.0), 1.0)
    success = score >= 0.5
    
    log_end(success=success, steps=step, score=score, rewards=rewards)
    return score

async def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = await CloudFinOpsEnv.from_docker_image(IMAGE_NAME)
    
    for task in TASKS:
        score = await run_task(client, env, task)
        print(f"Task {task}: {score:.3f}")
    
    await env.close()

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 9. DOCKERFILE

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Copy requirements first (Docker cache optimization)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port for HF Spaces
EXPOSE 7860

# Health check for automated validator
HEALTHCHECK --interval=30s --timeout=10s \
    CMD curl -f http://localhost:7860/health || exit 1

# Run the FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
```

### requirements.txt
```
fastapi>=0.104.0
uvicorn>=0.24.0
pydantic>=2.5.0
openai>=1.3.0
numpy>=1.24.0
```

---

## 10. OPENENV.YAML

```yaml
name: CloudFinOpsEnv
version: "1.0.0"
description: >
  An environment where LLM agents optimize cloud infrastructure costs
  by identifying orphaned resources, right-sizing over-provisioned instances,
  and safely pruning waste — without breaking production systems.
author: Three Musketeers
tags:
  - openenv
  - finops
  - cloud-optimization
  - cost-management

observation_model: app.models.Observation
action_model: app.models.Action
reward_model: app.models.Reward

tasks:
  - id: easy_orphan_cleanup
    name: "Orphan Cleanup"
    difficulty: easy
    description: "Delete orphaned and detached resources costing money with no purpose."
    max_steps: 15

  - id: medium_rightsize
    name: "Right-Size & Prune"
    difficulty: medium
    description: "Right-size over-provisioned instances and clean up waste while respecting environment tags."
    max_steps: 25

  - id: hard_dependency_migration
    name: "Dependency-Aware Migration"
    difficulty: hard
    description: "Optimize costs in a complex environment with dependencies, maintenance windows, and cascading effects."
    max_steps: 40

endpoints:
  reset: /reset
  step: /step
  state: /state
  health: /health
```

---

## 11. PROJECT STRUCTURE

```
CloudFinOpsEnv/
├── app.py                    # FastAPI server (reset/step/state endpoints)
├── inference.py              # Baseline inference script (root, per rules)
├── openenv.yaml              # Environment metadata
├── Dockerfile
├── requirements.txt
├── README.md
├── models/
│   ├── __init__.py
│   ├── observation.py        # Observation, Resource, UsageMetrics
│   ├── action.py             # Action, ActionType
│   ├── reward.py             # Reward
│   └── state.py              # EnvironmentState
├── engine/
│   ├── __init__.py
│   ├── environment.py        # Core env logic (reset, step, state)
│   ├── grader.py             # Oracle grader (deterministic scoring)
│   ├── reward_calculator.py  # Per-step reward computation
│   └── dependency_graph.py   # Resource dependency management
├── data/
│   ├── __init__.py
│   ├── generator.py          # Synthetic cloud state generator
│   ├── templates/
│   │   ├── easy.py           # Task 1 resource templates
│   │   ├── medium.py         # Task 2 resource templates
│   │   └── hard.py           # Task 3 resource templates
│   └── instance_types.json   # Valid resize options (t3.micro → t3.small etc.)
└── tests/
    ├── test_environment.py
    ├── test_grader.py
    └── test_generator.py
```

---

## 12. README OUTLINE (per hackathon requirements)

### Sections Required:
1. **Motivation** — Cloud waste is $100B+ problem, every DevOps team faces it daily
2. **Environment Description** — Mock cloud infra, agent optimizes costs via text commands
3. **Action Space** — 8 actions: query_metrics, delete, stop, resize, detach, check_deps, list_resources, commit_changes
4. **Observation Space** — JSON resource list with types, costs, tags, metrics, dependencies
5. **Tasks** — Easy (orphan cleanup), Medium (right-sizing), Hard (dependency-aware migration)
6. **Reward Function** — Continuous per-step rewards with safety penalties
7. **Grading** — Oracle formula: (savings/optimal) × safety_multiplier - step_penalty
8. **Setup Instructions** — docker build, docker run, env vars
9. **Baseline Scores** — Easy: ~0.8, Medium: ~0.5, Hard: ~0.2
10. **Team** — Three Musketeers (Utkarsh, Mohit, Tanush)

---

## 13. IMPLEMENTATION PRIORITY (48-hour sprint plan)

### Day 1 (Hours 1-8): Foundation
- [ ] Set up repo + Dockerfile + requirements.txt
- [ ] Define all Pydantic models (observation, action, reward, state)
- [ ] Build data/generator.py for Easy task (10 resources)
- [ ] Implement reset() and step() for basic actions (delete, query)
- [ ] Test locally: reset → query → delete → commit flow works

### Day 2 (Hours 9-16): Core Logic
- [ ] Implement grader.py (oracle scoring formula)
- [ ] Implement reward_calculator.py (per-step rewards)
- [ ] Build Easy + Medium task templates
- [ ] Add resize and stop actions
- [ ] Build dependency_graph.py for Hard task

### Day 3 (Hours 17-24): Hard Task + Inference
- [ ] Build Hard task template (35 resources + dependencies + maintenance window)
- [ ] Write inference.py with [START]/[STEP]/[END] logging
- [ ] Test all 3 tasks end-to-end
- [ ] Deploy to Hugging Face Spaces

### Day 4 (Hours 25-32): Polish + Validate
- [ ] Write README
- [ ] Write openenv.yaml
- [ ] Run openenv validate
- [ ] Run pre-submission validation script
- [ ] Fix any issues
- [ ] Final HF Spaces deploy + test

### Buffer (Hours 33-40): Emergency fixes
- [ ] Address any validator failures
- [ ] Tune reward values based on baseline runs
- [ ] Polish README with actual baseline scores

---

## 14. RISK MITIGATION

| Risk | Mitigation |
|------|------------|
| Oracle grader miscalculates optimal | Write unit tests for grader with known inputs/outputs |
| Hard task too easy for LLMs | Add more dependency complexity + decoy resources |
| Hard task too hard (baseline scores 0.0) | Ensure at least SOME orphaned resources exist that any agent can find |
| HF Spaces deploy fails | Test Dockerfile locally FIRST, deploy by Day 3 |
| openenv validate fails | Read OpenEnv source code for exact expectations, test early |
| Inference exceeds 20min | Cap MAX_STEPS at 40, use temperature=0.2 for fast responses |
| Memory exceeds 8GB | All data is Python dicts in memory, nowhere near 8GB |

---

## 15. WHAT MAKES THIS WIN

1. **Judges feel the pain:** Every Meta engineer has seen a $10K/month orphaned RDS instance
2. **Grader is bulletproof:** Mathematical formula, no LLM-as-judge, fully deterministic
3. **Rich agent interaction:** Query → investigate → analyze → act → verify → commit
4. **Hard task is genuinely hard:** Dependency graphs + maintenance windows + cascading effects
5. **Clean code:** FastAPI backend (your team's strength), typed Pydantic models, clear project structure
6. **Runs offline:** Zero external APIs, all synthetic data, fits in Docker with 8GB
7. **Reproducible:** Seeded random generation, deterministic grading
