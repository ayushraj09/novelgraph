"""
Standalone test for Stage 3 (novelty detection) against whatever graph
currently exists - does NOT ingest anything. Zero API cost (pure local
SQLite reads).

Run with:
    uv run scripts/debug_novelty.py
"""

import asyncio

from novelgraph.graph_store import load_graph
from novelgraph.novelty import find_novel_pairs


async def main():
    nodes, edges = load_graph()
    print(f"Loaded {len(nodes)} nodes and {len(edges)} edges.\n")

    print("--- Novel pairs ---")
    pairs = await find_novel_pairs()
    if not pairs:
        print(
            "(empty - if nodes/edges above are non-zero, run inspect_db.py "
            "to check column names, or check that graph_model=Paper "
            "actually produced Method/DatasetNode/Task labels as expected)"
        )
    for p in pairs:
        print(" ", p)


if __name__ == "__main__":
    asyncio.run(main())
