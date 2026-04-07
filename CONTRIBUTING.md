# Contributing to CloudFinOpsEnv

Thank you for your interest in contributing to CloudFinOpsEnv! This environment simulates cloud infrastructure optimization scenarios.

## Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Auenchanters/Three-Musketeers.git
   cd Three-Musketeers
   ```

2. **Create a virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Running Tests

We use `pytest` for testing. You can run the entire suite with:
```bash
python3 -m pytest tests/ -v
```

To run end-to-end oracle validation (which tests the core logic without needing an LLM):
```bash
python3 test_oracle_e2e.py
```

## Running the Server Locally

FastAPI powers the OpenEnv environment. Start it using:
```bash
uvicorn app:app --host 0.0.0.0 --port 7860
```
Then navigate to `http://localhost:7860/` or `http://localhost:7860/docs` to see the API.

## Project Structure
- `app.py`: FastAPI server entry point.
- `engine/`: Handles the core simulation of cloud infrastructure and dependency tracking.
- `models/`: Pydantic models for resources, states, and actions.
- `data/`: Curated AWS scenarios and oracle solutions.
- `tests/`: Pytest suite covering all environment rules.

If you encounter issues or have suggestions, please open an issue or a pull request!
