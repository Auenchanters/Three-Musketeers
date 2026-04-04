---
title: CloudFinOpsEnv
emoji: 💰
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
tags:
  - openenv
---

# CloudFinOpsEnv

**An OpenEnv environment where LLM agents learn to optimize cloud infrastructure costs by identifying waste, right-sizing resources, and safely pruning orphaned assets -- without breaking production.**

> **Meta PyTorch OpenEnv Hackathon** | Team: **Three Musketeers** (Utkarsh, Mohit, Tanush)

---

## Motivation

Cloud waste is a **$100B+ annual problem**. Every DevOps and FinOps team spends hours manually identifying orphaned EBS volumes, over-provisioned instances, idle NAT gateways, and resources that drain budgets without serving any purpose.

CloudFinOpsEnv tests whether LLM agents can learn to do this automatically -- navigating realistic cloud infrastructure with production constraints, resource dependencies, cluster quorum requirements, and budget targets. An agent that masters this environment represents a genuine step toward autonomous cloud cost optimization.

---

## Quick Start

### Docker (Recommended)

```bash
docker build -t cloudfinopsenv .
docker run -p 7860:7860 cloudfinopsenv
```

### Local Development

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```

### Verify It Works

```bash
# Health check
curl http://localhost:7860/health

# Run all tests (89 tests)
python -m pytest tests/ -v

# Run oracle validation (no LLM needed, proves all 3 tasks work)
python test_oracle_e2e.py
```

---

## Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `API_BASE_URL` | LLM API endpoint | `https://router.huggingface.co/v1` |
| `HF_TOKEN` | HuggingFace API token | `hf_xxxxx` |
| `MODEL_NAME` | Model identifier | `meta-llama/Meta-Llama-3-8B-Instruct` |
| `ENV_URL` | CloudFinOpsEnv server URL | `http://localhost:7860` |

### Running Inference

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export HF_TOKEN="your-token"
export MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"
export ENV_URL="http://localhost:7860"
python inference.py
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check -- returns `{"status": "healthy"}` |
| `/reset` | POST | Start new episode: `{"task_id": "easy_orphan_cleanup"}` |
| `/step` | POST | Take action: `{"action": {"action_type": "delete", "resource_id": "vol-xxx"}}` |
| `/state` | GET | Full internal state (oracle/debug) |
| `/schema` | GET | Action, observation, and state JSON schemas |
| `/metadata` | GET | Environment metadata |
| `/ws` | WebSocket | Persistent session endpoint (used by `inference.py`) |

---

## Environment Description

CloudFinOpsEnv simulates a realistic AWS cloud infrastructure account containing:

- **Wasteful resources** -- detached EBS volumes, unused Elastic IPs, stopped instances
- **Over-provisioned resources** -- EC2/RDS instances running at 2-5% CPU utilization
- **Critical production resources** -- must NOT be touched (deleting = instant score zero)
- **Dependency-linked resources** -- Kafka clusters with quorum, RDS primary/replica pairs, circular dependencies

The agent interacts through a **query -> investigate -> analyze -> act -> commit** loop, using metrics, tags, and dependency checks to make safe optimization decisions.

### Resource Types (7)

`ec2_instance`, `ebs_volume`, `rds_instance`, `s3_bucket`, `elastic_ip`, `nat_gateway`, `load_balancer`

---

## Action Space (8 Actions)

| Action | Description | Reward |
|--------|-------------|--------|
| `query_metrics(resource_id)` | Get 7-day CPU/memory/network usage | -0.015 |
| `check_deps(resource_id)` | Check resource dependencies and cluster info | -0.013 |
| `delete(resource_id, reason)` | Permanently remove a resource | +0.10 per $10/mo saved |
| `stop(resource_id, reason)` | Stop a running instance (reversible) | +0.05 per $10/mo saved |
| `resize(resource_id, new_size, reason)` | Change instance/DB tier | +0.08 per $10/mo saved |
| `detach(resource_id)` | Detach a volume from instance | -0.01 |
| `list_resources()` | Refresh resource list | -0.012 |
| `commit_changes()` | Finalize and end episode | bonus if >50% optimal |

**Safety penalties:**
- Delete production/critical resource: **-1.01** (catastrophic, zeroes final score)
- Delete resource with active dependencies: **-0.51**
- Bad resize (CPU >80% after downsize): **-0.21**

---

## Observation Space

Each observation includes:

| Field | Type | Description |
|-------|------|-------------|
| `task_description` | string | Natural language task brief |
| `resources` | List[Resource] | Cloud resources with ID, type, name, status, cost/hour, tags, age, attachments, dependencies |
| `total_monthly_cost` | float | Current total monthly cost (USD) |
| `budget_target` | float | Target monthly cost (medium/hard tasks) |
| `maintenance_window` | string | Allowed modification window (hard task) |
| `step_number` / `max_steps` | int | Current step and episode limit |
| `message` | string | Environment feedback from last action |
| `cost_saved_so_far` | float | Total savings achieved |
| `actions_taken` | List[str] | History of agent actions this episode |

Resource metrics (CPU, memory, network, IOPS) are **hidden** until explicitly queried via `query_metrics`.

---

## Tasks (3 Difficulty Levels)

### Task 1: Easy -- Orphan Cleanup
- **Resources:** 10 (6 wasteful, 4 production)
- **Max steps:** 15
- **Objective:** Delete orphaned/detached resources with clear signals (`status: "detached"`, `attached_to: null`)
- **Optimal savings:** $63.22/month
- **Expected LLM score:** 0.7 -- 0.9

### Task 2: Medium -- Right-Size & Prune
- **Resources:** 20 (6 wasteful + 3 over-provisioned, 11 production)
- **Max steps:** 25
- **Budget target:** $3,800/month
- **Objective:** Right-size over-provisioned instances AND clean waste. Requires querying metrics and understanding instance tiers.
- **Optimal savings:** $1,037.84/month
- **Expected LLM score:** 0.4 -- 0.6

### Task 3: Hard -- Dependency-Aware Migration
- **Resources:** 35 (7 wasteful + 6 resizable, 22 production/critical)
- **Max steps:** 40
- **Maintenance window:** 02:00--06:00 UTC
- **Objective:** Optimize costs with Kafka cluster quorum constraints, RDS primary/replica pairs, circular dependencies, maintenance windows, and decoy resources.
- **Optimal savings:** $2,513.54/month
- **Expected LLM score:** 0.1 -- 0.3

---

## Reward Function

**Per-step rewards** provide continuous signal throughout the episode:

- Correct optimizations: **positive** reward proportional to dollar savings
- Investigation actions (query, check_deps): **tiny negative** cost (encourages efficiency)
- Deleting production resources: **-1.01** (catastrophic, zeroes entire episode)
- Each step has a small time pressure cost

**Final grading** uses a deterministic oracle formula (no LLM-as-judge):

```
Easy:   score = (savings / optimal) * safety_mult
Medium: score = (savings / optimal) * safety_mult - (steps * 0.005)
Hard:   score = ((savings - cascade) / optimal) * safety_mult - (steps * 0.003)
```

`safety_multiplier = 0.0` if any production resource is deleted/stopped. All scores clamped to [0.0, 1.0].

---

## Baseline Scores

### Oracle Solution (Deterministic, No LLM)
| Task | Score | Savings |
|------|-------|---------|
| Easy: Orphan Cleanup | **1.000** | $63.22 / $63.22 (100%) |
| Medium: Right-Size & Prune | **0.987** | $1,024.41 / $1,037.84 (99%) |
| Hard: Dependency Migration | **1.000** | $2,582.96 / $2,513.54 (103%) |

### Expected LLM Agent (Meta-Llama-3-8B-Instruct)
| Task | Expected Score | Notes |
|------|---------------|-------|
| Easy | 0.7 -- 0.9 | Clear signals, most LLMs handle well |
| Medium | 0.4 -- 0.6 | Requires metrics analysis and tier knowledge |
| Hard | 0.1 -- 0.3 | Dependency reasoning is genuinely hard for LLMs |

---

## Project Architecture

```
CloudFinOpsEnv/
├── app.py                    # FastAPI server entry point
├── inference.py              # Baseline LLM agent (OpenAI client)
├── test_oracle_e2e.py        # Oracle validation script (no LLM needed)
├── openenv.yaml              # OpenEnv metadata
├── Dockerfile                # Docker container config (port 7860)
├── requirements.txt          # Python dependencies
├── pyproject.toml            # Package configuration
├── client.py                 # OpenEnv WebSocket client
│
├── models/                   # Pydantic data models
│   ├── observation.py        # Observation, Resource, UsageMetrics
│   ├── action.py             # Action, ActionType (8 actions)
│   ├── reward.py             # Reward decomposition
│   └── state.py              # EnvironmentState (oracle/god-mode)
│
├── engine/                   # Core environment logic
│   ├── environment.py        # reset(), step(), state() implementation
│   ├── grader.py             # Deterministic oracle scoring formulas
│   ├── reward_calculator.py  # Per-step reward computation
│   └── dependency_graph.py   # Resource dependency and quorum management
│
├── data/                     # Deterministic scenario data
│   ├── generator.py          # Data loader
│   ├── scenarios/            # 3 curated task JSONs (10, 20, 35 resources)
│   ├── solutions/            # Oracle optimal action sequences
│   └── pricing/              # Real AWS us-east-1 on-demand pricing
│
├── server/                   # Server entry point (openenv multi-mode)
│   └── app.py                # main() for uv run / openenv serve
│
└── tests/                    # Test suite (89 tests)
    ├── test_environment.py   # Core environment logic tests
    ├── test_grader.py        # Oracle scoring formula tests
    └── test_generator.py     # Data loader and model tests
```

---

## Validation

```bash
# Local structure validation
openenv validate
# Output: [OK] Three-Musketeers: Ready for multi-mode deployment

# Runtime validation against live server
openenv validate --url http://localhost:7860
# Output: 6/6 criteria passed

# Unit tests
python -m pytest tests/ -v
# Output: 89 passed

# Oracle end-to-end (proves all tasks work, no LLM needed)
python test_oracle_e2e.py
# Output: Average Score 0.996, ALL TASKS PASSED
```

---

## Design Decisions

1. **Deterministic oracle grader** -- Mathematical formula, no LLM-as-judge, fully reproducible
2. **Curated JSON scenarios** -- Hand-crafted with real AWS pricing, not randomly generated
3. **Hidden metrics** -- Agents must explicitly query usage data before acting (mirrors real FinOps)
4. **Safety-first penalties** -- Deleting production resources is catastrophic (-1.01), encouraging investigation before action
5. **Scalable difficulty** -- Easy has clear signals, medium requires metric analysis, hard involves graph reasoning with quorum and cascading constraints
6. **Runs offline** -- Zero external API dependencies, all synthetic data, fits in Docker with <100MB

---

## Team

**Three Musketeers**
- Utkarsh Singh Yadav (Team Lead)
- Mohit Jain
- Tanush Deepak
