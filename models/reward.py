"""
CloudFinOpsEnv — Reward Model

Defines the feedback structure returned after each agent action.
Includes a numeric value, a breakdown of components, and a human-readable message.
"""

from pydantic import BaseModel, Field
from typing import Dict


class Reward(BaseModel):
    """
    Feedback after each step.

    The reward is decomposed into components so agents (and humans)
    can understand what drove the score:
    - savings: positive reward for cost reductions
    - safety_penalty: negative reward for breaking production
    - step_cost: small negative per step to encourage efficiency
    - investigation_cost: small negative for query/check actions
    """
    value: float = Field(description="The numeric reward value")
    breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Component breakdown, e.g. {'savings': 0.1, 'safety_penalty': 0.0, 'step_cost': -0.01}"
    )
    message: str = Field(default="", description="Human-readable explanation of the reward")
