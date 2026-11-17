import json
import os

os.environ["AEGIS_BACKEND"] = "sqlite"
os.environ["MLFLOW_ENABLE_ASYNC_TRACE_LOGGING"] = "false"  # tests read traces back immediately

import mlflow
import pytest

from src.agent.llm import LLM
from src.data_gen.kb import build_kb, demo_incidents

USAGE = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}


def tc(name: str, **args) -> dict:
    return {"id": f"call_{name}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def calls(*tool_calls) -> dict:
    return {"role": "assistant", "content": "", "tool_calls": list(tool_calls)}


def say(text: str) -> dict:
    return {"role": "assistant", "content": text}


def scripted(*replies) -> LLM:
    """An LLM that returns the given assistant messages in order (no network)."""
    it = iter(replies)
    return LLM("scripted", complete=lambda messages, tools, choice: (next(it), USAGE))


@pytest.fixture(scope="session")
def incidents():
    inc = demo_incidents()
    build_kb(inc)
    return inc


@pytest.fixture(autouse=True)
def tracking(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.set_experiment(experiment_id=mlflow.create_experiment("tests", artifact_location=(tmp_path / "artifacts").as_uri()))


def last_trace():
    return mlflow.get_trace(mlflow.get_last_active_trace_id())
