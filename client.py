from openenv.core.env_client import EnvClient
from models.action import Action
from models.observation import Observation
from models.state import EnvironmentState

class CloudFinOpsClient(EnvClient[Action, Observation, EnvironmentState]):
    """
    Client for interacting with the CloudFinOpsEnv environment.
    Use this class to communicate with the deployed HuggingFace Space.
    """
    
    def __init__(self, base_url: str = "http://localhost:7860"):
        super().__init__(base_url, Action, Observation)
