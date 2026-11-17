"""The four agent tools. Every tool is an MLflow span; fault injection simulates flaky upstreams."""
import contextvars
import json
import os
import re
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timedelta
from functools import lru_cache

import mlflow
from mlflow.entities import Document, SpanType

from . import backend as B
from .config import RETRIEVAL_K

FMT = "%Y-%m-%dT%H:%M:%S"
_faults = contextvars.ContextVar("faults", default=None)


class UpstreamError(Exception):
    pass


@contextmanager
def inject_faults(fail_first: int = 0, partial: bool = False):
    """Make check_upstream_health fail its first `fail_first` calls, then optionally return partial data."""
    tok = _faults.set({"fail_first": int(fail_first), "partial": bool(partial), "calls": 0})
    try:
        yield
    finally:
        _faults.reset(tok)


def annotate(**attrs) -> None:
    """Attach attributes to the current span (no-op when tracing is disabled, e.g. evaluate's dry run)."""
    span = mlflow.get_current_active_span()
    if span:
        span.set_attributes(attrs)


def _window(end_time: str, delta: timedelta) -> tuple[str, str]:
    end = datetime.fromisoformat(end_time.rstrip("Z"))
    return (end - delta).strftime(FMT), end.strftime(FMT)


@mlflow.trace(name="query_service_logs", span_type=SpanType.TOOL)
def query_service_logs(service_name: str, end_time: str, time_window_mins: int = 30) -> str:
    start, end = _window(end_time, timedelta(minutes=min(int(time_window_mins), 240)))
    sql = (
        f"SELECT ts, level, exception, message FROM {B.table('service_logs')} "
        "WHERE service = :service AND ts >= :start AND ts <= :end AND level IN ('ERROR', 'WARN') "
        "ORDER BY ts DESC LIMIT 500"
    )
    params = {"service": service_name, "start": start, "end": end}
    rows = B.run_sql(sql, params)
    annotate(sql=sql, sql_params=params, rows_returned=len(rows))
    errors = [r for r in rows if r["level"] == "ERROR"]
    return json.dumps(
        {
            "service": service_name,
            "window": [start, end],
            "error_count": len(errors),
            "top_exceptions": [{"exception": e, "count": c} for e, c in Counter(r["exception"] for r in errors if r["exception"]).most_common(5)],
            "sample_messages": list(dict.fromkeys(r["message"] for r in rows))[:10],
        }
    )


@mlflow.trace(name="inspect_recent_deployments", span_type=SpanType.TOOL)
def inspect_recent_deployments(service_name: str, end_time: str, lookback_hours: int = 24) -> str:
    start, end = _window(end_time, timedelta(hours=min(int(lookback_hours), 72)))
    sql = (
        f"SELECT ts, commit_hash, change_type, author, message, diff FROM {B.table('deployments')} "
        "WHERE service = :service AND ts >= :start AND ts <= :end ORDER BY ts DESC LIMIT 10"
    )
    params = {"service": service_name, "start": start, "end": end}
    rows = B.run_sql(sql, params)
    annotate(sql=sql, sql_params=params, rows_returned=len(rows))
    return json.dumps({"service": service_name, "window": [start, end], "deployments": rows})


@mlflow.trace(name="check_upstream_health", span_type=SpanType.TOOL)
def check_upstream_health(dependency: str, at_time: str) -> str:
    f = _faults.get()
    if f:
        f["calls"] += 1
        if f["calls"] <= f["fail_first"]:
            raise UpstreamError("HTTP 503 Service Unavailable: telemetry request timed out after 5000ms")
    start, end = _window(at_time, timedelta(minutes=15))
    sql = (
        f"SELECT ts, status, p99_latency_ms, error_rate FROM {B.table('upstream_health')} "
        "WHERE dependency = :dep AND ts >= :start AND ts <= :end ORDER BY ts DESC LIMIT 1"
    )
    rows = B.run_sql(sql, {"dep": dependency, "start": start, "end": end})
    annotate(sql=sql, rows_returned=len(rows))
    if not rows:
        return json.dumps({"dependency": dependency, "status": "no_data"})
    out = {"dependency": dependency, **rows[0]}
    if f and f["partial"] and f["calls"] > f["fail_first"]:
        out.update(p99_latency_ms=None, error_rate=None, note="partial telemetry: latency and error rate unavailable")
    return json.dumps(out)


def _tokens(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", s.lower()))


@lru_cache(maxsize=1)
def _vs_index():
    from databricks.vector_search.client import VectorSearchClient

    return VectorSearchClient(disable_notice=True).get_index(
        endpoint_name=os.getenv("AEGIS_VS_ENDPOINT", "aegis_vs"),
        index_name=f"{B.CATALOG}.{B.SCHEMA}.runbooks_index",
    )


def _search(query: str, k: int) -> list[dict]:
    if os.getenv("AEGIS_RETRIEVER", "vector" if B.BACKEND == "spark" else "keyword") == "vector":
        cols = ["chunk_id", "title", "text"]
        res = _vs_index().similarity_search(query_text=query, columns=cols, num_results=k)
        return [dict(zip(cols, r[:3])) for r in res["result"]["data_array"]]
    # ponytail: keyword overlap ranking; vector search is the production path on Databricks
    q = _tokens(query)
    scored = [(len(q & _tokens(r["title"] + " " + r["text"])), r) for r in B.read_table("runbooks")]
    return [r for s, r in sorted(scored, key=lambda x: -x[0])[:k] if s > 0]


@mlflow.trace(name="runbook_retriever", span_type=SpanType.RETRIEVER)
def _retrieve(query: str, k: int) -> list[Document]:
    return [
        Document(page_content=r["text"], id=r["chunk_id"], metadata={"doc_uri": r["chunk_id"], "title": r["title"]})
        for r in _search(query, k)
    ]


@mlflow.trace(name="search_runbooks", span_type=SpanType.TOOL)
def search_runbooks(query: str) -> str:
    docs = _retrieve(query, RETRIEVAL_K)
    return json.dumps([{"chunk_id": d.id, "title": d.metadata["title"], "text": d.page_content} for d in docs])


def _spec(name, desc, props, required):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {"type": "object", "properties": props, "required": required, "additionalProperties": False},
        },
    }


_S, _I = {"type": "string"}, {"type": "integer"}
TOOL_SPECS = [
    _spec(
        "query_service_logs",
        "Error/warn log summary for a service in the window ending at end_time (ISO 8601, e.g. 2026-01-03T12:00:00).",
        {"service_name": _S, "end_time": _S, "time_window_mins": _I},
        ["service_name", "end_time"],
    ),
    _spec(
        "inspect_recent_deployments",
        "Recent commits, config diffs and feature-flag toggles for a service before end_time.",
        {"service_name": _S, "end_time": _S, "lookback_hours": _I},
        ["service_name", "end_time"],
    ),
    _spec(
        "check_upstream_health",
        "Telemetry (status, p99 latency, error rate) for an upstream dependency around at_time. May time out; retry.",
        {"dependency": _S, "at_time": _S},
        ["dependency", "at_time"],
    ),
    _spec(
        "search_runbooks",
        "Search incident runbooks, architecture docs and SLA/approval policies.",
        {"query": _S},
        ["query"],
    ),
]
TOOLS = {f.__name__: f for f in (query_service_logs, inspect_recent_deployments, check_upstream_health, search_runbooks)}
TOOL_NAMES = list(TOOLS)
SPEC_BY_NAME = {s["function"]["name"]: s for s in TOOL_SPECS}


def validate_call(name: str, arguments) -> bool:
    """True if the model emitted a known tool with JSON args that match the schema (used by schema_validity)."""
    spec = SPEC_BY_NAME.get(name)
    try:
        args = json.loads(arguments) if isinstance(arguments, str) else arguments
    except json.JSONDecodeError:
        return False
    if not spec or not isinstance(args, dict):
        return False
    p = spec["function"]["parameters"]
    types = {"string": str, "integer": int}
    return (
        all(k in args for k in p["required"])
        and all(k in p["properties"] for k in args)
        and all(isinstance(v, types[p["properties"][k]["type"]]) and not isinstance(v, bool) for k, v in args.items())
    )


def call_tool(name: str, arguments) -> str:
    """Execute a tool call from the model. Failures become JSON error strings the model can react to."""
    try:
        args = json.loads(arguments) if isinstance(arguments, str) else arguments
        return TOOLS[name](**args)
    except Exception as e:  # noqa: BLE001 - the model must see every failure mode
        return json.dumps({"error": f"{type(e).__name__}: {e}"})
