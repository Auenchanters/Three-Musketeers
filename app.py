"""
CloudFinOpsEnv — FastAPI Server

The OpenEnv-compliant API server exposing 4 endpoints:
    POST /reset     — Start a new episode
    POST /step      — Take an action
    GET  /state     — Inspect internal state (god-mode)
    GET  /health    — Health check

Runs on port 7860 for HuggingFace Spaces compatibility.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

from models import Observation, Action, Reward, EnvironmentState
from engine import CloudFinOpsEnvironment
from data.generator import get_available_tasks


# ─── FastAPI App ──────────────────────────────────────────────────────────

app = FastAPI(
    title="CloudFinOpsEnv",
    description="An OpenEnv environment where LLM agents optimize cloud infrastructure costs.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global environment instance (single-session, per hackathon requirements)
env = CloudFinOpsEnvironment()


# ─── Request/Response Models ─────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = Field(
        description="ID of the task to run. Options: easy_orphan_cleanup, medium_rightsize, hard_dependency_migration"
    )


class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any] = {}


class HealthResponse(BaseModel):
    status: str
    environment: str
    version: str
    available_tasks: List[str]


# ─── Endpoints ────────────────────────────────────────────────────────────

@app.post("/reset", response_model=Observation)
async def reset(request: ResetRequest):
    """
    Start a new episode for the specified task.
    
    Returns the initial Observation with task description and resource list.
    """
    try:
        observation = env.reset(request.task_id)
        return observation
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.post("/step", response_model=StepResponse)
async def step(action: Action):
    """
    Take an action in the current episode.
    
    Returns observation, reward, done flag, and info dict.
    """
    try:
        observation, reward, done, info = env.step(action)
        return StepResponse(
            observation=observation,
            reward=reward,
            done=done,
            info=info,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.get("/state", response_model=EnvironmentState)
async def state():
    """
    Return the full internal environment state (god-mode).
    
    Includes oracle data for grading and debugging.
    """
    try:
        return env.state()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint for automated validators."""
    return HealthResponse(
        status="healthy",
        environment="CloudFinOpsEnv",
        version="1.0.0",
        available_tasks=get_available_tasks(),
    )
