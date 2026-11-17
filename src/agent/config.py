"""Tunable knobs. A PR that changes this file or prompts.py triggers the CI eval gate."""
import os
from dataclasses import dataclass

RETRIEVAL_K = 3  # runbook chunks returned per search
MAX_STEPS = 8  # ReAct tool-loop budget


@dataclass(frozen=True)
class AgentConfig:
    name: str
    model: str  # served-model-name on the OpenAI-compatible endpoint
    mode: str = "react"  # "react" | "plan_execute"
    retry: bool = False  # retry middleware for transient tool errors


# The four architectures benchmarked in the sweep.
CONFIGS = {
    "A": AgentConfig("A_8b_react", "llama-3.1-8b", "react", False),
    "B": AgentConfig("B_32b_react", "qwen2.5-32b", "react", False),
    "C": AgentConfig("C_32b_plan_retry", "qwen2.5-32b", "plan_execute", True),
    "D": AgentConfig("D_70b_plan_retry", "llama-3.3-70b", "plan_execute", True),
}


def resolve(config: "str | AgentConfig") -> AgentConfig:
    return CONFIGS[config] if isinstance(config, str) else config


def endpoint() -> tuple[str, str]:
    """(base_url, api_key) of the agent model: vLLM, Mosaic AI Model Serving, or Foundation Model API."""
    return os.getenv("AGENT_BASE_URL", "http://localhost:8000/v1"), os.getenv("AGENT_API_KEY", "EMPTY")
