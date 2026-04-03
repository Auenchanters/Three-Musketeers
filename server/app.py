"""
CloudFinOpsEnv — Server Entry Point

Exposes the CloudFinOpsEnvironment over HTTP and WebSocket endpoints,
compatible with EnvClient and openenv validate.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - GET /health: Health check
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Via uv run:
    uv run --project . server

    # Via uvicorn:
    uvicorn server.app:app --host 0.0.0.0 --port 7860

    # Direct execution:
    python -m server.app
"""

try:
    from openenv.core.env_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required. Install with: pip install openenv-core"
    ) from e

try:
    from models.action import Action
    from models.observation import Observation
    from engine.environment import CloudFinOpsEnvironment
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from models.action import Action
    from models.observation import Observation
    from engine.environment import CloudFinOpsEnvironment


app = create_app(
    env=CloudFinOpsEnvironment,
    action_cls=Action,
    observation_cls=Observation,
)


def main(host: str = "0.0.0.0", port: int = 7860):
    """
    Entry point for direct execution via uv run or python -m.

    Enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8000
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
