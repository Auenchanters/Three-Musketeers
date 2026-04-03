# CloudFinOpsEnv 🌩️💰

**An OpenEnv environment where LLM agents learn to optimize cloud infrastructure costs by identifying waste, right-sizing resources, and safely pruning orphaned assets — without breaking production.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Meta PyTorch OpenEnv Hackathon** | Team: **Three Musketeers** (Utkarsh, Mohit, Tanush)

---

## 🎯 Motivation

Cloud waste is a **$100B+ annual problem**. Every DevOps and FinOps team spends countless hours manually identifying orphaned EBS volumes, over-provisioned instances, idle NAT gateways, and other resources that drain budgets without serving any purpose.

CloudFinOpsEnv tests whether LLM agents can **learn to do this automatically** — navigating complex cloud environments with production constraints, resource dependencies, and budget targets. An agent that masters this environment would represent a genuine step toward autonomous cloud cost optimization.

---

## 🏗️ Environment Description

CloudFinOpsEnv simulates a realistic cloud infrastructure account with a mix of:
- **Wasteful resources** — detached volumes, unused Elastic IPs, stopped instances
- **Over-provisioned resources** — instances running at 2-5% CPU utilization
- **Critical production resources** — must NOT be touched
- **Dependency-linked resources** — clusters, replication pairs, service meshes

The agent interacts through a **query → investigate → analyze → act → verify → commit** loop, using information from metrics, tags, and dependency checks to make safe optimization decisions.

---

## 🎮 Action Space

| Action | Description | Reward |
|--------|-------------|--------|
| `query_metrics(resource_id)` | Get 7-day CPU/memory/network usage | -0.005 |
| `check_deps(resource_id)` | Check resource dependencies | -0.003 |
| `delete(resource_id, reason)` | Permanently remove a resource | +0.10 per $10/mo saved |
| `stop(resource_id, reason)` | Stop a running instance | +0.05 per $10/mo saved |
| `resize(resource_id, new_size, reason)` | Change instance/db tier | +0.08 per $10/mo saved |
| `detach(resource_id)` | Detach a volume from instance | -0.01 |
| `list_resources()` | Refresh resource list | -0.002 |
| `commit_changes()` | Finalize and end episode | bonus if >50% optimal |

---

## 👁️ Observation Space

Each observation includes:
- **Task description** — Natural language brief
- **Resources** — List of cloud resources with: ID, type, name, status, cost/hour, tags, creation age, attachments, dependencies
- **Metrics** — CPU, memory, network, IOPS (revealed after `query_metrics`)
- **Cost tracking** — Total monthly cost, savings so far, budget target
- **Episode info** — Step number, max steps, action history, feedback message

---

## 📋 Tasks

### Task 1: Easy — "Orphan Cleanup" (10 resources, 15 steps)
Delete orphaned/detached resources. Clear signals: `status: "detached"`, `attached_to: null`, non-production tags.

### Task 2: Medium — "Right-Size & Prune" (20 resources, 25 steps)
Right-size over-provisioned instances AND clean waste. Requires querying metrics, understanding instance tiers, and respecting environment tags.

### Task 3: Hard — "Dependency-Aware Migration" (35 resources, 40 steps)
Complex environment with Kafka cluster quorum, RDS primary/replica pairs, circular dependencies, maintenance windows, decoy resources, and cascading effects.

---

## 📊 Reward Function

**Per-step rewards** provide continuous signal:
- Correct optimizations give **positive** reward proportional to savings
- Deleting production resources gives **-1.0** (catastrophic)
- Investigation actions have **tiny negative** cost (encourages efficiency)
- Each step costs **-0.01** (time pressure)

**Final grading** uses a deterministic oracle formula (no LLM-as-judge):

```
Easy:   score = (savings / optimal) × safety_mult
Medium: score = (savings / optimal) × safety_mult − (steps × 0.005)
Hard:   score = ((savings − cascade) / optimal) × safety_mult − (steps × 0.003)
```

`safety_multiplier = 0.0` if any production resource is deleted.

---

## 🚀 Setup Instructions

### Docker (Recommended)

```bash
# Build
docker build -t cloudfinopsenv .

# Run
docker run -p 7860:7860 cloudfinopsenv
```

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn app:app --host 0.0.0.0 --port 7860

# Run tests
python -m pytest tests/ -v

# Run data verification
python test_data_verify.py
```

### Environment Variables (for inference)

```bash
export API_BASE_URL="https://api-inference.huggingface.co/v1"
export HF_TOKEN="your-token"
export MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"
export ENV_URL="http://localhost:7860"
```

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Start new episode: `{"task_id": "easy_orphan_cleanup"}` |
| `/step` | POST | Take action: `{"action_type": "delete", "resource_id": "vol-xxx"}` |
| `/state` | GET | Full internal state (oracle/debug) |
| `/health` | GET | Health check |

---

## 📈 Baseline Scores

| Task | Expected LLM Score | Notes |
|------|-------------------|-------|
| Easy: Orphan Cleanup | 0.7–0.9 | Clear signals, most LLMs handle well |
| Medium: Right-Size & Prune | 0.4–0.6 | Requires metrics analysis |
| Hard: Dependency Migration | 0.1–0.3 | Dependency reasoning is genuinely hard |

---

## 🏛️ Architecture

```
CloudFinOpsEnv/
├── app.py                    # FastAPI server (4 endpoints)
├── inference.py              # Baseline LLM agent
├── openenv.yaml              # OpenEnv metadata
├── Dockerfile                # Container config
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── models/                   # Pydantic data models
│   ├── observation.py        # Observation, Resource, UsageMetrics
│   ├── action.py             # Action, ActionType
│   ├── reward.py             # Reward
│   └── state.py              # EnvironmentState
├── engine/                   # Core game logic
│   ├── environment.py        # reset(), step(), state()
│   ├── grader.py             # Oracle scoring formulas
│   ├── reward_calculator.py  # Per-step reward computation
│   └── dependency_graph.py   # Resource dependency management
├── data/                     # Deterministic scenario data
│   ├── generator.py          # Data loader
│   ├── scenarios/            # 3 task JSONs (curated, not random)
│   ├── solutions/            # Oracle solutions
│   └── pricing/              # Real AWS pricing reference
└── tests/                    # Comprehensive test suite
    ├── test_environment.py
    ├── test_grader.py
    └── test_generator.py
```

---

## 🏆 Why This Wins

1. **Judges feel the pain** — Every engineer has seen a $10K/month orphaned RDS instance
2. **Grader is bulletproof** — Mathematical formula, no LLM-as-judge, fully deterministic
3. **Rich agent interaction** — Query → investigate → analyze → act → verify → commit
4. **Hard task is genuinely hard** — Dependency graphs + maintenance windows + cascading effects
5. **Clean code** — FastAPI + typed Pydantic models + comprehensive tests
6. **Runs offline** — Zero external APIs, all synthetic data, fits in Docker with <100MB
7. **Reproducible** — Curated JSON scenarios, deterministic grading

---

## 👥 Team

**Three Musketeers**
- Utkarsh
- Mohit
- Tanush
