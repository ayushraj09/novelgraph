"""
Shared helper for reading Cognee's graph directly out of its SQLite
metadata store (cognee_db), used by both novelty.py and temporal.py.

This exists because SearchType.CYPHER and the Ladybug/Kuzu Cypher layer
were found to be unreliable in this environment - reading the `nodes` and
`edges` tables directly is a zero-cost (no LLM/embedding calls), reliable
alternative for structural graph queries.

Run inspect_db.py first any time these column-name guesses stop matching -
it dumps the actual schema so the *_CANDIDATES lists below can be corrected.
"""

import os
import glob
import json
import re
import sqlite3
import re as _re

# Cognee's `edges.source_node_id` / `destination_node_id` reference
# `nodes.slug`, not the relational row `nodes.id`.
NODE_ID_CANDIDATES = ["slug", "id", "node_id", "uuid", "data_point_id"]
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


def load_graph():
    """Returns (nodes, edges) as lists of dicts, straight from cognee_db."""
    db_path = find_cognee_db()
    if not db_path or not os.path.exists(db_path):
        raise RuntimeError(
            "Could not locate cognee_db. Run inspect_db.py to find it, then "
            "set COGNEE_SQLITE_PATH explicitly if needed."
        )
    conn = sqlite3.connect(db_path)
    nodes = _load_table(conn, "nodes")
    edges = _load_table(conn, "edges")
    conn.close()
    return nodes, edges


def node_property(node, key, blob_candidates=NODE_PROPERTIES_BLOB_CANDIDATES):
    """
    Reads a schema field (e.g. 'year') off a node dict. Tries a direct
    column first (e.g. node['year']), then falls back to parsing a JSON
    properties blob if the DB stores extra DataPoint fields that way.
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
    Prefers the exact 'name' field from the attributes/properties JSON blob
    (matches schema.py's Method/DatasetNode/Task.name exactly) over the
    `label` column, which Cognee sometimes computes/concatenates from
    metadata['index_fields'] rather than storing the raw field verbatim.
    Falls back to the label/name column if no 'name' property is found
    (e.g. for Paper nodes, which use 'title' instead of 'name').
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
