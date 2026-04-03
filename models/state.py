"""
CloudFinOpsEnv — Environment State Model

Internal state returned by state(). Contains oracle/god-mode data
that the agent does NOT see — used for grading and debugging.
"""

from pydantic import BaseModel, Field
from typing import List, Dict
from .observation import Resource


class EnvironmentState(BaseModel):
    """
    Full internal environment state (god-mode view).

    This is returned by the state() endpoint for debugging and grading.
    It includes data the agent never sees, such as:
    - optimal_savings: the best possible savings (oracle-computed)
    - critical_resources: resource IDs that must NOT be deleted
    - dependency_graph: full resource dependency adjacency list
    - safety_violations: log of critical mistakes the agent made
    """
    task_id: str = Field(description="ID of the current task")
    task_difficulty: str = Field(description="'easy', 'medium', or 'hard'")
    resources: List[Resource] = Field(description="Full resource list with all metadata")
    optimal_savings: float = Field(description="Maximum possible monthly savings (oracle-computed)")
    critical_resources: List[str] = Field(
        default_factory=list,
        description="Resource IDs that must NOT be deleted (production/critical)"
    )
    dependency_graph: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Adjacency list: resource_id → [dependent_resource_ids]"
    )
    cost_saved: float = Field(default=0.0, description="Total cost saved so far")
    penalties_incurred: float = Field(default=0.0, description="Total penalties from mistakes")
    steps_taken: int = Field(default=0, description="Number of steps taken")
    done: bool = Field(default=False, description="Whether the episode is complete")
    safety_violations: List[str] = Field(
        default_factory=list,
        description="Log of critical mistakes (e.g., 'Deleted production resource i-abc123')"
    )
