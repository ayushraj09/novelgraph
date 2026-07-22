"""
CLI entry point: runs the full pipeline (ingest -> novelty -> hypothesis
-> temporal -> agentic verification -> evidence -> eval) and prints a
report. All the actual logic lives in `novelgraph.pipeline`.

Run with:
    uv run scripts/main.py                  # incremental ingest (only new/changed papers) + full report
    SKIP_INGEST=true uv run scripts/main.py  # skip ingestion entirely, reuse existing graph
    RESET_GRAPH=true uv run scripts/main.py  # wipe and fully rebuild the graph from scratch
"""

import asyncio

from dotenv import load_dotenv

from novelgraph.pipeline import run_pipeline

load_dotenv()

if __name__ == "__main__":
    asyncio.run(run_pipeline())
