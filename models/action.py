"""Action models for the CloudFinOpsEnv agent."""

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum
from openenv.core.env_server.types import Action as BaseAction


class ActionType(str, Enum):
    """The 8 actions available to the agent."""
    QUERY_METRICS = "query_metrics"
    DELETE = "delete"
    STOP = "stop"
    RESIZE = "resize"
    DETACH = "detach"
    COMMIT_CHANGES = "commit_changes"
    LIST_RESOURCES = "list_resources"
    CHECK_DEPS = "check_deps"

class Action(BaseAction):
    """
    What the agent does at each step.

    Every action requires an action_type. Some actions also need:
    - resource_id: which resource to act on
    - new_size: what to resize to (only for RESIZE)
    - reason: agent's justification (for logging/debugging)
    """
    action_type: ActionType = Field(description="The type of action to perform")
    resource_id: Optional[str] = Field(default=None, description="Target resource ID")
    new_size: Optional[str] = Field(default=None, description="New size/tier for RESIZE action (e.g., 't3.micro')")
    reason: Optional[str] = Field(default=None, description="Agent's justification for the action")
