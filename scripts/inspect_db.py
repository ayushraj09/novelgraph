"""
Zero-cost inspection of the actual graph data plus Cognee's SQLite
metadata store.

As of Cognee 1.4.0, graph nodes/edges live in a separate per-dataset
graph-engine store (see `dataset_database.graph_database_provider` in the
metadata dump below - could be Kuzu, Neo4j, NetworkX, or Cognee's own
"ladybug" engine depending on config), NOT in cognee_db's `nodes`/`edges`
SQLite tables, which are typically empty. This script prints both:

1. Live node/edge data straight from the graph engine (via Cognee's own
   `get_graph_data()` accessor - the same one `graph_store.load_graph()`
   and `cognee.visualize_graph()` use) - this is what actually matters
   for confirming the graph looks right.
2. The SQLite metadata store's tables/columns/row counts, useful for
   confirming ingestion status, dataset IDs, and which graph-engine
   backend is configured.

Run this any time novelty.py/temporal.py return unexpectedly empty
results, to see whether the graph itself is empty/small, or whether
something downstream (property names, task clustering) is the problem.

Run with:
    uv run scripts/inspect_db.py
"""

import asyncio
import os
import sqlite3

from novelgraph.graph_store import find_cognee_db, load_graph


async def print_live_graph_data() -> None:
    print("=" * 60)
    print("LIVE GRAPH ENGINE DATA (via get_graph_data())")
    print("=" * 60)
    try:
        nodes, edges = await load_graph()
    except Exception as e:
        print(f"Could not read graph data: {e}")
        return

    print(f"Nodes: {len(nodes)} | Edges: {len(edges)}\n")

    if not nodes:
        print(
            "No nodes found. If graph_debug.html shows a populated graph "
            "but this is empty, check that SYSTEM_ROOT_DIRECTORY/dataset "
            "match what the pipeline actually ingested into (e.g. a stale "
            "or mismatched dataset name/path)."
        )
        return

    print("Sample nodes (first 5):")
    for node in nodes[:5]:
        print(" ", node)

    print("\nSample edges (first 5):")
    for edge in edges[:5]:
        print(" ", edge)

    node_types = {}
    for node in nodes:
        node_types[node.get("type")] = node_types.get(node.get("type"), 0) + 1
    print("\nNode counts by type:", node_types)

    edge_labels = {}
    for edge in edges:
        edge_labels[edge.get("relationship_name")] = edge_labels.get(edge.get("relationship_name"), 0) + 1
    print("Edge counts by relationship:", edge_labels)
    print()


def print_sqlite_metadata() -> None:
    print("=" * 60)
    print("SQLITE METADATA STORE (cognee_db)")
    print("=" * 60)
    db_path = find_cognee_db()
    print(f"Using: {db_path}\n")

    if not db_path or not os.path.exists(db_path):
        print(
            "Could not locate cognee_db automatically. Set COGNEE_SQLITE_PATH "
            "to the exact path and re-run, e.g.:\n"
            "  COGNEE_SQLITE_PATH=/absolute/path/to/.cognee_system/databases/cognee_db uv run scripts/inspect_db.py"
        )
        return

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # These tables carry the most useful signal for diagnosing a run;
    # the rest (users/roles/acls/etc.) are auth/multitenancy plumbing.
    tables_of_interest = ["datasets", "data", "dataset_data", "dataset_database", "pipeline_runs", "nodes", "edges"]

    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    all_tables = {row[0] for row in cur.fetchall()}

    for table in tables_of_interest:
        if table not in all_tables:
            continue
        print(f"--- {table} ---")
        cur.execute(f"PRAGMA table_info({table})")
        cols = [col[1] for col in cur.fetchall()]
        print(f"  columns: {cols}")

        try:
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            count = cur.fetchone()[0]
            print(f"  Row count: {count}")
            if count and table in ("datasets", "dataset_database", "pipeline_runs"):
                cur.execute(f"SELECT * FROM {table} LIMIT 3")
                for row in cur.fetchall():
                    print(" ", row)
        except sqlite3.OperationalError as e:
            print(f"  (could not read rows: {e})")
        print()

    conn.close()


async def main():
    await print_live_graph_data()
    print_sqlite_metadata()


if __name__ == "__main__":
    asyncio.run(main())
