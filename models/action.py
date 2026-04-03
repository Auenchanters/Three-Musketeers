"""
CloudFinOpsEnv — Action Models

Defines the 8 actions an agent can take:
- QUERY_METRICS: inspect resource usage (investigation)
- DELETE: permanently remove a resource
- STOP: stop a running instance (reversible)
- RESIZE: change instance/db tier
- DETACH: detach a volume from instance
- COMMIT_CHANGES: end episode and finalize savings
- LIST_RESOURCES: refresh the resource list
- CHECK_DEPS: check what depends on a resource
"""

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class ActionType(str, Enum):
    """The 8 actions available to the agent."""
    QUERY_METRICS = "query_metrics"       # Inspect a resource's 7-day usage
    DELETE = "delete"                      # Permanently remove a resource
    STOP = "stop"                          # Stop a running instance (can be restarted)
    RESIZE = "resize"                      # Change instance/db tier (e.g., t3.large → t3.small)
    DETACH = "detach"                      # Detach a volume from an instance
    COMMIT_CHANGES = "commit_changes"      # End episode, finalize all savings
    LIST_RESOURCES = "list_resources"      # Re-list all resources (refresh view)
    CHECK_DEPS = "check_deps"             # Check what other resources depend on this one


from openenv.core.env_server.types import Action as BaseAction

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
