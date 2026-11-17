"""Write the knowledge-base tables (Delta on Databricks, sqlite locally) the agent's tools query."""
import random

from ..agent import backend as B
from .catalog import RCA_TYPES, SERVICES, incident_rows, runbook_chunks


def build_kb(incidents: list[dict]) -> dict[str, int]:
    """incidents: dicts returned by catalog.incident_rows. Overwrites all four KB tables."""
    tables = {
        "runbooks": runbook_chunks(),
        "service_logs": [r for i in incidents for r in i["logs"]],
        "deployments": [r for i in incidents for r in i["deployments"]],
        "upstream_health": [r for i in incidents for r in i["upstream"]],
    }
    for name, rows in tables.items():
        B.write_table(name, rows)
    return {n: len(r) for n, r in tables.items()}


def demo_incidents() -> list[dict]:
    """One causal incident per RCA type: enough to run the agent end to end."""
    return [incident_rows(i, t, random.Random(f"demo-{i}"), SERVICES[i % len(SERVICES)]) for i, t in enumerate(RCA_TYPES)]
