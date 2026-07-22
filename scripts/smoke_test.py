"""
STEP-BY-STEP COST-SAFETY PROTOCOL

Run this BEFORE ingesting real papers. It ingests only 2 tiny inline
sample strings (costs pennies - a couple of small LLM calls for entity
extraction, no embeddings-heavy corpus), then runs Stage 3 (novelty) and
Stage 5 (temporal) against that tiny graph and prints the results so you
can eyeball whether they make sense.

If something looks wrong (empty results, wrong column names, etc.), fix
it here - iterating on this tiny graph costs nothing further (novelty.py
and temporal.py are pure local SQLite reads, zero additional API cost
once the graph exists).

Only once this all looks correct should you run the real pipeline
(`scripts/main.py`) against your real `data/papers/` folder.

Run with:
    uv run scripts/smoke_test.py
"""

import asyncio

from dotenv import load_dotenv

from novelgraph.ingest import ingest_smoke_test
from novelgraph.novelty import find_novel_pairs
from novelgraph.pipeline import close_cognee_telemetry_session
from novelgraph.temporal import check_temporal_novelty

load_dotenv()


async def main():
    try:
        print("=" * 60)
        print("STEP 1: Ingesting 2 tiny sample documents (cheap, ~pennies)")
        print("=" * 60)
        await ingest_smoke_test(reset=True)
        print("Done.\n")

        print("=" * 60)
        print("STEP 2: Running novelty detection against the sample graph")
        print("=" * 60)
        print(
            "Expected (based on the 2 sample docs in ingest.py):\n"
            "  Sample doc 1: Method A x Dataset X x Task Y (already tried)\n"
            "  Sample doc 2: Method B x Dataset Z x Task Y (already tried)\n"
            "  -> Method A and Dataset Z BOTH involve Task Y but were never paired,\n"
            "     so ('Method A', 'Dataset Z') should show up as a novel pair here.\n"
        )
        pairs = await find_novel_pairs()
        if not pairs:
            print(
                "!! EMPTY RESULT. Run inspect_db.py now to check the actual "
                "nodes/edges table schema, and update graph_store.py's "
                "*_CANDIDATES lists if the column names don't match.\n"
            )
        for p in pairs:
            print(" ", p)
        print()

        print("=" * 60)
        print("STEP 3: Running temporal checks against the sample graph")
        print("=" * 60)
        print("Expected: Method A was used in 2022, so this should report a prior use before 2024.\n")
        result = await check_temporal_novelty("Method A", "Dataset Z", before_year=2024)
        print(" ", result)
        print()

        print("=" * 60)
        print("If STEP 2 and STEP 3 both look correct: you're clear to run")
        print("`uv run scripts/main.py` against your real data/papers/ folder.")
        print("If not: fix novelty.py / temporal.py / graph_store.py now -")
        print("re-running this script costs pennies, not your real corpus.")
        print("=" * 60)
    finally:
        await close_cognee_telemetry_session()


if __name__ == "__main__":
    asyncio.run(main())
