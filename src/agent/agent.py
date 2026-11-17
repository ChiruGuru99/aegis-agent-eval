"""Incident-triage agent: ReAct or Plan-and-Execute, optional retry middleware. Fully traced."""
import json
import re

import mlflow
from mlflow.entities import SpanType

from . import prompts as P
from .config import MAX_STEPS, RETRIEVAL_K, AgentConfig, resolve
from .llm import LLM
from .tools import SPEC_BY_NAME, TOOL_NAMES, TOOL_SPECS, call_tool

DEFAULT_PLAN = ["query_service_logs", "inspect_recent_deployments", "check_upstream_health", "search_runbooks"]


def _transient(out: str) -> bool:
    return '"error"' in out and any(s in out for s in ("503", "timed out", "Timeout"))


def with_retry(call, attempts: int = 3):
    """Middleware: re-run a tool call on transient errors. Each attempt is its own TOOL span."""

    @mlflow.trace(name="retry_middleware", span_type=SpanType.CHAIN)
    def wrapped(name, arguments):
        out = call(name, arguments)
        for _ in range(attempts - 1):
            if not _transient(out):
                break
            out = call(name, arguments)
        return out

    return wrapped


def _tool_msg(tc: dict, call) -> dict:
    out = call(tc["function"]["name"], tc["function"]["arguments"])
    return {"role": "tool", "tool_call_id": tc["id"], "content": f"<tool_output>{out}</tool_output>"}


def _react(messages, llm, call) -> str:
    for _ in range(MAX_STEPS):
        msg = llm(messages, TOOL_SPECS)
        messages.append(msg)
        if not msg.get("tool_calls"):
            return msg["content"]
        messages += [_tool_msg(tc, call) for tc in msg["tool_calls"]]
    messages.append({"role": "user", "content": "Step budget exhausted. " + P.SYNTH_PROMPT})
    return llm(messages)["content"]


@mlflow.trace(name="plan", span_type=SpanType.CHAIN)
def make_plan(messages, llm) -> list[str]:
    raw = llm(messages + [{"role": "user", "content": P.PLANNER_PROMPT}])["content"]
    try:
        steps = json.loads(re.search(r"\{.*\}", raw, re.S).group())["steps"]
        steps = [s for s in dict.fromkeys(steps) if s in TOOL_NAMES]
    except (AttributeError, KeyError, TypeError, json.JSONDecodeError):
        steps = []
    return steps or DEFAULT_PLAN


def _plan_execute(messages, llm, call) -> str:
    for tool in make_plan(messages, llm):
        step = {"role": "user", "content": P.STEP_PROMPT.format(tool=tool)}
        msg = llm(messages + [step], [SPEC_BY_NAME[tool]], {"type": "function", "function": {"name": tool}})
        if msg.get("tool_calls"):
            messages.append(msg)
            messages += [_tool_msg(tc, call) for tc in msg["tool_calls"]]
    messages.append({"role": "user", "content": P.SYNTH_PROMPT})
    return llm(messages)["content"]


def log_config(cfg: AgentConfig) -> None:
    """Log the agent configuration (and prompts) on the active MLflow run."""
    mlflow.log_params({"config": cfg.name, "model": cfg.model, "mode": cfg.mode, "retry": cfg.retry,
                       "retrieval_k": RETRIEVAL_K, "max_steps": MAX_STEPS})
    mlflow.log_text(P.SYSTEM_PROMPT, "prompts/system.txt")
    mlflow.log_text(P.PLANNER_PROMPT, "prompts/planner.txt")


@mlflow.trace(name="incident_triage_agent", span_type=SpanType.AGENT)
def run_agent(question: str, config: str | AgentConfig = "C", llm: LLM | None = None) -> str:
    cfg = resolve(config)
    llm = llm or LLM(cfg.model)
    call = with_retry(call_tool) if cfg.retry else call_tool
    if mlflow.get_current_active_span():
        mlflow.update_current_trace(tags={"config": cfg.name, "model": cfg.model})
    messages = [{"role": "system", "content": P.SYSTEM_PROMPT}, {"role": "user", "content": question}]
    return (_plan_execute if cfg.mode == "plan_execute" else _react)(messages, llm, call)
