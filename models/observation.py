"""Observation models: what the agent sees at each step."""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from enum import Enum
from openenv.core.env_server.types import Observation as BaseObservation


class ResourceType(str, Enum):
    """Types of cloud resources in the environment."""
    EC2_INSTANCE = "ec2_instance"
    EBS_VOLUME = "ebs_volume"
    RDS_INSTANCE = "rds_instance"
    S3_BUCKET = "s3_bucket"
    ELASTIC_IP = "elastic_ip"
    NAT_GATEWAY = "nat_gateway"
    LOAD_BALANCER = "load_balancer"


class ResourceStatus(str, Enum):
    """Current operational status of a resource."""
    RUNNING = "running"
    STOPPED = "stopped"
    DETACHED = "detached"
    IDLE = "idle"


class UsageMetrics(BaseModel):
    """
    7-day usage metrics for a resource.
    Only populated after the agent calls QUERY_METRICS on the resource.
    """
    cpu_avg_7d: float = Field(ge=0.0, le=100.0, description="Average CPU utilization over 7 days (%)")
    cpu_peak_7d: float = Field(ge=0.0, le=100.0, description="Peak CPU utilization over 7 days (%)")
    memory_avg_7d: float = Field(ge=0.0, le=100.0, description="Average memory utilization over 7 days (%)")
    memory_peak_7d: float = Field(ge=0.0, le=100.0, description="Peak memory utilization over 7 days (%)")
    network_in_gb_7d: float = Field(ge=0.0, description="Inbound network traffic over 7 days (GB)")
    network_out_gb_7d: float = Field(ge=0.0, description="Outbound network traffic over 7 days (GB)")
    disk_iops_avg_7d: float = Field(ge=0.0, description="Average disk IOPS over 7 days")
    last_accessed_days_ago: int = Field(ge=0, description="Days since last meaningful access")


class Resource(BaseModel):
    """
    A single cloud resource in the environment.

    The agent sees these in the observation. Some fields (like metrics)
    are hidden until explicitly queried via QUERY_METRICS.
    """
    resource_id: str = Field(description="Unique ID, e.g. 'i-0abc123', 'vol-xyz789'")
    resource_type: ResourceType
    name: str = Field(description="Human-readable name tag")
    status: ResourceStatus
    cost_per_hour: float = Field(ge=0.0, description="Cost in USD per hour")
    tags: Dict[str, str] = Field(default_factory=dict, description="Key-value tags, e.g. {'env': 'production'}")
    created_days_ago: int = Field(ge=0, description="Days since resource was created")
    attached_to: Optional[str] = Field(default=None, description="Parent resource_id (e.g., volume → instance)")
    dependencies: List[str] = Field(default_factory=list, description="Resource IDs that depend on this resource")
    metrics: Optional[UsageMetrics] = Field(default=None, description="Usage metrics (populated after QUERY_METRICS)")

    @property
    def monthly_cost(self) -> float:
        """Calculate monthly cost (730 hours/month)."""
        return round(self.cost_per_hour * 730, 2)


class Observation(BaseObservation):
    """
    What the agent sees at each step.

    Contains the task description, current cloud state, cost tracking,
    and feedback from the last action taken.
    """
    task_description: str = Field(description="Natural language task brief")
    resources: List[Resource] = Field(description="Current cloud resource state")
    total_monthly_cost: float = Field(description="Current total monthly cost in USD")
    budget_target: Optional[float] = Field(default=None, description="Desired monthly cost target (medium/hard)")
    maintenance_window: Optional[str] = Field(default=None, description="Time window for changes (hard task)")
    step_number: int = Field(ge=0, description="Current step in the episode")
    max_steps: int = Field(gt=0, description="Maximum steps allowed")
    message: str = Field(default="", description="Environment feedback after last action")
    cost_saved_so_far: float = Field(default=0.0, description="Total savings achieved so far")
    actions_taken: List[str] = Field(default_factory=list, description="History of agent's actions this episode")
    cost_breakdown: Optional[Dict[str, float]] = Field(default=None, description="Monthly cost by resource type (e.g. {'ec2_instance': 1200.50, 'ebs_volume': 45.00})")
