"""
Zero-cost inspection of Cognee's SQLite metadata store.

Run this to confirm the exact column names in the `nodes` and `edges`
tables before trusting any SQL that reads them - no API calls, no
LLM/embedding cost, just local sqlite3 reads.

Run with:
    uv run scripts/inspect_db.py

If it can't find your cognee_db automatically, set the path explicitly:
    COGNEE_SQLITE_PATH=/absolute/path/to/cognee_db uv run scripts/inspect_db.py
"""

import os
import sqlite3

from novelgraph.graph_store import find_cognee_db


def main():
    db_path = find_cognee_db()
    print(f"Using: {db_path}\n")

    if not db_path or not os.path.exists(db_path):
        print(
            "Could not locate cognee_db automatically. Set COGNEE_SQLITE_PATH "
            "to the exact path and re-run, e.g.:\n"
            "  COGNEE_SQLITE_PATH=/Users/you/project/.cognee_system/databases/cognee_db uv run scripts/inspect_db.py"
        )
        return

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cur.fetchall()]
    print("Tables found:", tables, "\n")

    for table in tables:
        print(f"--- {table} ---")
        cur.execute(f"PRAGMA table_info({table})")
        for col in cur.fetchall():
            print(f"  {col[1]} ({col[2]})")

        try:
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            count = cur.fetchone()[0]
            print(f"  Row count: {count}")

            cur.execute(f"SELECT * FROM {table} LIMIT 3")
            rows = cur.fetchall()
            print("  Sample rows:")
            for row in rows:
                print(" ", row)
        except sqlite3.OperationalError as e:
            print(f"  (could not read rows: {e})")
        print()

    conn.close()


if __name__ == "__main__":
    main()
