"""
Shared helper for reading Cognee's graph, used by novelty.py, temporal.py,
chat.py, and app.py.

`load_graph()` reads via Cognee's own documented accessor -
`get_graph_engine().get_graph_data()` - rather than reading cognee_db's
SQLite `nodes`/`edges` tables directly. This project originally read
those SQLite tables directly (also avoiding SearchType.CYPHER, which was
found unreliable in this environment). That worked against the Cognee
version this was first built on, but as of Cognee 1.4.0 the graph itself
lives in a separate per-dataset graph-engine store (Kuzu / Neo4j /
NetworkX / Cognee's own "ladybug" engine, depending on config - check
`dataset_database.graph_database_provider` in cognee_db to see which one
a given install uses) - the SQLite `nodes`/`edges` tables are empty.
`get_graph_data()` is Cognee's own stable accessor for exactly this, used
internally by `cognee.visualize_graph()` itself, so it stays correct
across whichever backend is actually configured. It's still a pure local
read against the graph engine - no LLM/embedding calls, same zero-cost
property this module has always had.

`load_graph()` returns nodes/edges as flat dicts (not the raw
`(node_id, properties)` / `(source_id, target_id, label, properties)`
tuples `get_graph_data()` returns), so pick_col/node_property/
resolve_node_name below - and every caller in novelty.py, temporal.py,
chat.py, and app.py - keep working unchanged.

Run inspect_db.py first any time a real run's node/edge property names
don't match what's expected here - it prints live counts and sample
nodes/edges straight from the graph engine.
"""

import glob
import json
import os
import re
import sqlite3
import re as _re

from cognee.infrastructure.databases.graph import get_graph_engine

# Property-name candidates. get_graph_data()'s per-node/edge properties
# dict is expected to already use the first candidate in each list below
# (matching schema.py's DataPoint fields and Cognee's own edge property
# names) - the remaining candidates are a fallback for property-naming
# differences across Cognee versions/backends.
NODE_ID_CANDIDATES = ["id", "slug", "node_id", "uuid", "data_point_id"]
NODE_NAME_CANDIDATES = ["label", "name", "node_name", "text", "value"]
NODE_TYPE_CANDIDATES = ["type", "class_name", "label", "node_type", "category"]
NODE_PROPERTIES_BLOB_CANDIDATES = ["attributes", "properties", "metadata", "data", "payload", "extra"]

EDGE_SOURCE_CANDIDATES = ["source_node_id", "source_id", "from_id", "source"]
EDGE_TARGET_CANDIDATES = ["destination_node_id", "target_node_id", "target_id", "to_id", "target"]
EDGE_LABEL_CANDIDATES = ["relationship_name", "label", "relation", "type", "relationship_type"]
_CITATION_PATTERN = _re.compile(r'^[A-Z][a-zA-Z\-]+,\s+[A-Z]\.')


def looks_like_citation(text: str) -> bool:
    """Reference-list entries start with 'Lastname, F.:' author-initial
    style, unlike real paper titles. Filters phantom Paper/Method/Dataset
    nodes extracted from a chunk that landed in the References section."""
    if not text:
        return False
    return bool(_CITATION_PATTERN.match(str(text).strip()))


def _env_value(name):
    value = os.environ.get(name)
    if value:
        return value

    env_file = ".env"
    if not os.path.exists(env_file):
        return None

    with open(env_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, raw_value = line.split("=", 1)
            if key.strip() == name:
                return raw_value.strip().strip('"').strip("'")
    return None


def find_cognee_db():
    """Locates cognee_db, the SQLite METADATA store (data/datasets/
    pipeline_runs/permissions/etc.) - NOT where graph nodes/edges live as
    of Cognee 1.4.0 (see this module's docstring). Still useful for
    inspect_db.py's metadata dump and for confirming which graph-engine
    backend a given install is actually configured to use."""
    env_path = _env_value("COGNEE_SQLITE_PATH")
    if env_path:
        return os.path.expanduser(env_path)

    system_root = _env_value("SYSTEM_ROOT_DIRECTORY")
    if system_root:
        db_path = os.path.join(os.path.expanduser(system_root), "databases", "cognee_db")
        if os.path.exists(db_path):
            return db_path

    local_path = "./.cognee_system/databases/cognee_db"
    if os.path.exists(local_path):
        return local_path

    candidates = (
        glob.glob(os.path.expanduser("~/cognee-hypothesis-project/.cognee_system/databases/cognee_db"))
        + glob.glob(os.path.expanduser("~/.cognee_system/databases/cognee_db"))
    )
    return candidates[0] if candidates else None


def _load_table(conn, table):
    cur = conn.cursor()
    cur.execute(f"SELECT * FROM {table}")
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def pick_col(row, candidates, table_name, required=True):
    for c in candidates:
        if c in row:
            return c
    if required:
        raise RuntimeError(
            f"None of {candidates} found as columns in `{table_name}` "
            f"(actual columns: {list(row.keys())}). "
            f"Run inspect_db.py and update the *_CANDIDATES lists in graph_store.py."
        )
    return None


async def load_graph():
    """Returns (nodes, edges) as lists of flat dicts, read live from
    Cognee's graph engine via get_graph_data(). This is async because
    get_graph_engine()/get_graph_data() are - every caller already runs
    inside (or can be wrapped in) an async context; see app.py's
    run_async() for the Streamlit-side wrapper."""
    graph_engine = await get_graph_engine()
    raw_nodes, raw_edges = await graph_engine.get_graph_data()

    nodes = [
        {"id": node_id, **(properties or {})}
        for node_id, properties in raw_nodes
    ]
    edges = [
        {
            "source_node_id": source_id,
            "destination_node_id": target_id,
            "relationship_name": relationship_label,
            **(properties or {}),
        }
        for source_id, target_id, relationship_label, properties in raw_edges
    ]
    return nodes, edges


def node_property(node, key, blob_candidates=NODE_PROPERTIES_BLOB_CANDIDATES):
    """
    Reads a schema field (e.g. 'year') off a node dict. Tries a direct
    key first (e.g. node['year']), then falls back to parsing a JSON
    properties blob if the graph engine nests extra DataPoint fields
    that way instead of flattening them onto the node dict directly.
    """
    if key in node and node[key] is not None:
        return node[key]
    for blob_col in blob_candidates:
        raw = node.get(blob_col)
        if not raw:
            continue
        try:
            data = json.loads(raw) if isinstance(raw, str) else raw
            if isinstance(data, dict) and data.get(key) is not None:
                return data[key]
        except (json.JSONDecodeError, TypeError):
            continue
    return None


def resolve_node_name(node, name_col):
    """
    Prefers the exact 'name' field (matches schema.py's Method/
    DatasetNode/Task.name exactly) over the label/name column, which
    Cognee sometimes computes/concatenates from indexed fields rather
    than storing the raw field verbatim. Falls back to the label/name
    column if no 'name' property is found (e.g. for Paper nodes, which
    use 'title' instead of 'name').
    """
    exact_name = node_property(node, "name")
    if exact_name:
        return exact_name
    return node.get(name_col)


def infer_tasks_from_node(node):
    """
    Best-effort fallback for tiny/sparse graphs where Cognee extracted Paper,
    Method, and DatasetNode but did not materialize Task nodes/edges.

    This is deliberately conservative: it only recognizes explicit phrases like
    "for Task Y" or standalone "Task Y" in fields already stored on the node.
    """
    texts = [
        node_property(node, "title"),
        node_property(node, "abstract"),
        node_property(node, "description"),
        node_property(node, "text"),
        node.get("label"),
    ]

    tasks = set()
    for text in texts:
        if not text:
            continue
        for match in re.finditer(r"\bfor\s+(Task\s+[A-Za-z0-9_-]+)\b", str(text), re.IGNORECASE):
            tasks.add(" ".join(match.group(1).split()))
        for match in re.finditer(r"\b(Task\s+[A-Za-z0-9_-]+)\b", str(text), re.IGNORECASE):
            tasks.add(" ".join(match.group(1).split()))
    return tasks
