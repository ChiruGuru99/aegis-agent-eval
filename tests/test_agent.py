import json

from mlflow.entities import SpanType

from src.agent import tools
from src.agent.agent import run_agent
from src.agent.tools import inject_faults, validate_call
from tests.conftest import calls, last_trace, say, scripted, tc


def _inc(incidents, rca):
    return next(i for i in incidents if i["rca_type"] == rca)


def test_tools_return_grounded_data(incidents):
    inc = _inc(incidents, "pool_exhaustion")
    logs = json.loads(tools.query_service_logs(inc["service"], inc["alert_time"]))
    assert logs["top_exceptions"][0]["exception"] == inc["rca"]["exc"]
    deps = json.loads(tools.inspect_recent_deployments(inc["service"], inc["alert_time"]))["deployments"]
    assert inc["commit"] in {d["commit_hash"] for d in deps}
    assert json.loads(tools.check_upstream_health(inc["dep"], inc["alert_time"]))["status"] == "healthy"
    hits = json.loads(tools.search_runbooks("rollback procedure database connection pool exhaustion"))
    assert hits[0]["chunk_id"].startswith("RB-DB-01")


def test_fault_injection_then_partial(incidents):
    inc = _inc(incidents, "upstream_timeout")
    with inject_faults(fail_first=1, partial=True):
        assert "error" in json.loads(tools.call_tool("check_upstream_health", {"dependency": inc["dep"], "at_time": inc["alert_time"]}))
        out = json.loads(tools.check_upstream_health(inc["dep"], inc["alert_time"]))
        assert out["p99_latency_ms"] is None and out["status"] == "degraded"


def test_validate_call():
    assert validate_call("search_runbooks", '{"query": "x"}')
    assert not validate_call("search_runbooks", '{"q": "x"}')
    assert not validate_call("drop_table", "{}")
    assert not validate_call("query_service_logs", '{"service_name": "a", "end_time": "t", "time_window_mins": "30"}')


def test_react_trace_tree(incidents):
    inc = _inc(incidents, "pool_exhaustion")
    llm = scripted(
        calls(tc("query_service_logs", service_name=inc["service"], end_time=inc["alert_time"])),
        say("Root cause: connection pool exhausted."),
    )
    out = run_agent("alert", "A", llm)
    assert "pool" in out
    spans = last_trace().data.spans
    by_type = {t: [s.name for s in spans if s.span_type == t] for t in (SpanType.AGENT, SpanType.LLM, SpanType.TOOL)}
    assert by_type[SpanType.AGENT] == ["incident_triage_agent"]
    assert by_type[SpanType.TOOL] == ["query_service_logs"]
    assert len(by_type[SpanType.LLM]) == 2
    tool = next(s for s in spans if s.name == "query_service_logs")
    assert "service_logs" in tool.attributes["sql"]
    assert last_trace().info.token_usage["total_tokens"] == 30


def test_plan_execute_retry_middleware(incidents):
    inc = _inc(incidents, "upstream_timeout")
    llm = scripted(
        say('{"steps": ["check_upstream_health"]}'),
        calls(tc("check_upstream_health", dependency=inc["dep"], at_time=inc["alert_time"])),
        say("Root cause: upstream degraded."),
    )
    with inject_faults(fail_first=1):
        run_agent("alert", "C", llm)
    spans = [s for s in last_trace().data.spans if s.name == "check_upstream_health"]
    assert [s.status.status_code.name for s in sorted(spans, key=lambda s: s.start_time_ns)] == ["ERROR", "OK"]
