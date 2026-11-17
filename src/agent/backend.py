"""Data access. Unity Catalog (Spark) on Databricks, in-memory sqlite locally.

Same SQL runs on both, so tools are testable on a laptop and identical in the workspace.
Override with AEGIS_BACKEND=sqlite|spark, AEGIS_CATALOG, AEGIS_SCHEMA.
"""
import os
import sqlite3
import threading

BACKEND = os.getenv("AEGIS_BACKEND") or ("spark" if "DATABRICKS_RUNTIME_VERSION" in os.environ else "sqlite")
CATALOG = os.getenv("AEGIS_CATALOG", "main")
SCHEMA = os.getenv("AEGIS_SCHEMA", "aegis")  # knowledge base tables
EVAL_SCHEMA = os.getenv("AEGIS_EVAL_SCHEMA", "eval")  # benchmark tables

_lock = threading.Lock()
_db = sqlite3.connect(":memory:", check_same_thread=False)
_db.row_factory = sqlite3.Row


def table(name: str, schema: str | None = None) -> str:
    return f"{CATALOG}.{schema or SCHEMA}.{name}" if BACKEND == "spark" else name


def _spark():
    from pyspark.sql import SparkSession

    return SparkSession.builder.getOrCreate()


def write_table(name: str, rows: list[dict], schema: str | None = None) -> None:
    """Overwrite a table. rows must be non-empty, flat, and free of None."""
    if BACKEND == "sqlite":
        cols = list(rows[0])
        with _lock:
            _db.execute(f"DROP TABLE IF EXISTS {name}")
            _db.execute(f"CREATE TABLE {name} ({', '.join(cols)})")
            _db.executemany(f"INSERT INTO {name} VALUES ({', '.join(':' + c for c in cols)})", rows)
        return
    import pandas as pd

    spark = _spark()
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{schema or SCHEMA}")
    (
        spark.createDataFrame(pd.DataFrame(rows))
        .write.mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(table(name, schema))
    )


def run_sql(sql: str, params: dict | None = None) -> list[dict]:
    """Run a parameterised query (named :markers only) and return rows as dicts."""
    params = params or {}
    if BACKEND == "spark":
        return [r.asDict() for r in _spark().sql(sql, args=params).collect()]
    with _lock:
        return [dict(r) for r in _db.execute(sql, params)]


def read_table(name: str, schema: str | None = None) -> list[dict]:
    return run_sql(f"SELECT * FROM {table(name, schema)}")
