"""Point the agent and the judge at an endpoint (env vars read by src.agent.config and src.evaluation.judges)."""
import os


def use_vllm(agent_port: int = 8000, judge_port: int = 8001) -> None:
    """Self-hosted vLLM servers on the GPU node (see launch.py)."""
    os.environ.update(AGENT_BASE_URL=f"http://localhost:{agent_port}/v1", AGENT_API_KEY="EMPTY",
                      JUDGE_BASE_URL=f"http://localhost:{judge_port}/v1", JUDGE_API_KEY="EMPTY",
                      JUDGE_MODEL="judge-llama-3.3-70b")


def use_foundation_models(host: str, token: str, judge_model: str = "databricks-meta-llama-3-3-70b-instruct") -> None:
    """Databricks Foundation Model API for both agent and judge: no GPUs needed (development path)."""
    url = host.rstrip("/") + "/serving-endpoints"
    os.environ.update(AGENT_BASE_URL=url, AGENT_API_KEY=token, JUDGE_BASE_URL=url, JUDGE_API_KEY=token, JUDGE_MODEL=judge_model)
