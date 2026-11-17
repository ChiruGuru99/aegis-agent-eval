# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Build the Unity Catalog knowledge base
# MAGIC Creates the Delta tables the agent's tools query (`service_logs`, `deployments`, `upstream_health`, `runbooks`) and a **Mosaic AI Vector Search** index over the runbooks.
# MAGIC Starts with one demo incident per root-cause type; notebook 02 overwrites the tables with the full 400-incident set.

# COMMAND ----------

# MAGIC %pip install -q -U "mlflow>=3.18" openai databricks-vectorsearch

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os, sys
sys.path.insert(0, os.path.abspath(".."))
dbutils.widgets.text("catalog", "main")
CATALOG = dbutils.widgets.get("catalog")
os.environ["AEGIS_CATALOG"] = CATALOG
SCHEMA = "aegis"

from src.data_gen.kb import build_kb, demo_incidents
counts = build_kb(demo_incidents())
print(counts)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Delta tables
# MAGIC Change Data Feed on `runbooks` is required for a Delta Sync vector index.

# COMMAND ----------

spark.sql(f"ALTER TABLE {CATALOG}.{SCHEMA}.runbooks SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
for t in ["service_logs", "deployments", "upstream_health", "runbooks"]:
    spark.sql(f"COMMENT ON TABLE {CATALOG}.{SCHEMA}.{t} IS 'Aegis incident-triage knowledge base (synthetic)'")
display(spark.sql(f"SELECT service, level, exception, message FROM {CATALOG}.{SCHEMA}.service_logs WHERE level = 'ERROR' LIMIT 5"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Vector Search index over runbooks
# MAGIC Managed embeddings (`databricks-gte-large-en`): Databricks embeds the `text` column for us.

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

ENDPOINT, INDEX = "aegis_vs", f"{CATALOG}.{SCHEMA}.runbooks_index"
vsc = VectorSearchClient()
if ENDPOINT not in [e["name"] for e in vsc.list_endpoints().get("endpoints", [])]:
    vsc.create_endpoint_and_wait(name=ENDPOINT, endpoint_type="STANDARD")

try:
    index = vsc.get_index(endpoint_name=ENDPOINT, index_name=INDEX)
    index.describe()
    index.sync()
except Exception:  # index does not exist yet
    index = vsc.create_delta_sync_index_and_wait(
        endpoint_name=ENDPOINT, index_name=INDEX, source_table_name=f"{CATALOG}.{SCHEMA}.runbooks",
        pipeline_type="TRIGGERED", primary_key="chunk_id", embedding_source_column="text",
        embedding_model_endpoint_name="databricks-gte-large-en")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Smoke test the retriever tool
# MAGIC `search_runbooks` uses the vector index on Databricks (`AEGIS_RETRIEVER=vector` is the default there). Set `AEGIS_RETRIEVER=keyword` to fall back to the keyword ranker.

# COMMAND ----------

from src.agent.tools import search_runbooks
print(search_runbooks("what is the rollback procedure for database connection pool exhaustion?"))
