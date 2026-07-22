"""
End-to-end orchestration: runs every stage in order for each novel
(method, dataset) pair and assembles a final hypothesis report.

`scripts/main.py` is the thin CLI entry point around `run_pipeline()`;
the Streamlit app calls the same lower-level pieces (`find_novel_pairs`,
`run_refinement`, `get_evidence`, ...) directly for its own UI flow, so
this module is where the CLI's specific "process every pair, print a
report, write a graph visualization" behavior lives.

Incremental ingestion: `ingest()`/`cognify()` default to
`incremental_loading=True`, with two dedup layers - content-hash dedup in
`add()`, and pipeline-status dedup in `cognify()` - so re-running the
pipeline over a papers folder that is mostly already-ingested plus one
new paper only spends LLM/embedding calls on the new one, as long as
`reset=False`. Set `RESET_GRAPH=true` only when you deliberately want a
clean slate (e.g. you changed schema.py).

Run-history caching: pairs the Critic already approved in a past run are
reused directly (Stage 6/7 skipped entirely) instead of re-verified every
time; see `run_history.py`.
"""

import os
from pathlib import Path
from typing import List

import cognee

from .agents import run_refinement
from .eval import evaluate_run, format_eval_summary
from .evidence import get_evidence
from .ingest import DEFAULT_PAPERS_DIR, ingest_papers_folder
from .novelty import find_novel_pairs
from .run_history import find_prior_result, record_result
from .run_history import summary as history_summary
from .temporal import check_temporal_novelty

DATASET_NAME = "research"
GRAPH_VISUALIZATION_FILENAME = "graph_debug.html"


async def close_cognee_telemetry_session() -> None:
    """Best-effort cleanup for Cognee's fire-and-forget telemetry client."""
    try:
        from cognee.shared import utils as cognee_utils

        session = getattr(cognee_utils, "_telemetry_session", None)
        if session is not None and not session.closed:
            await session.close()
        cognee_utils._telemetry_session = None
        cognee_utils._telemetry_session_loop = None
    except Exception:
        pass


def print_report(report: List[dict]) -> None:
    for item in report:
        print("=" * 60)
        status = "APPROVED" if item["approved"] else "NOT APPROVED"
        if item.get("from_cache"):
            status += " [cached from prior run]"
        novelty_bits = []
        if item.get("shared_neighbors") is not None:
            novelty_bits.append(f"shared_neighbors={item['shared_neighbors']}")
        if item.get("similarity") is not None:
            novelty_bits.append(f"similarity={item['similarity']}")
        novelty_str = f" [{', '.join(novelty_bits)}]" if novelty_bits else ""

        print(f"{item['method']} x {item['dataset']} ({status}){novelty_str}")
        print()
        print(item["hypothesis"] or "(no hypothesis text returned)")
        print()
        print(f"Temporal note: {item['temporal_note']}")
        print()

        evidence = item.get("evidence") or []
        if evidence:
            print("Evidence:")
            for bullet in evidence:
                print(f"  - {bullet}")
        else:
            print("Evidence: (none retrieved)")
        print()


async def process_pair(pair: dict, index: int) -> dict:
    method_name = pair["method"]
    dataset_name = pair["dataset"]

    temporal_note = await check_temporal_novelty(method_name, dataset_name)

    prior = find_prior_result(method_name, dataset_name)
    if prior is not None and prior.get("approved"):
        print(
            f"  [cache] {method_name} x {dataset_name} was already approved "
            f"in a prior run - reusing it, skipping Stage 6/7 calls."
        )
        return {
            "method": method_name,
            "dataset": dataset_name,
            "hypothesis": prior["hypothesis"],
            "approved": True,
            "temporal_note": temporal_note,
            "evidence": prior.get("evidence", []),
            "shared_neighbors": pair.get("shared_neighbors"),
            "similarity": pair.get("similarity"),
            "from_cache": True,
        }

    refinement = await run_refinement(method_name, dataset_name, session_id=f"run-{index}")
    evidence = await get_evidence(method_name, dataset_name) if refinement["approved"] else []
    record_result(method_name, dataset_name, refinement["approved"], refinement["hypothesis"], evidence)

    return {
        "method": method_name,
        "dataset": dataset_name,
        "hypothesis": refinement["hypothesis"],
        "approved": refinement["approved"],
        "temporal_note": temporal_note,
        "evidence": evidence,
        "shared_neighbors": pair.get("shared_neighbors"),
        "similarity": pair.get("similarity"),
        "from_cache": False,
    }


async def run_pipeline(top_n: int = 3) -> None:
    """Runs Stages 1-8 end to end and prints the final report, mirroring
    what `scripts/main.py` exposes as a CLI. Also writes an interactive
    graph visualization (`graph_debug.html`) at the end of the run."""
    try:
        skip_ingest = os.environ.get("SKIP_INGEST", "false").lower() == "true"
        reset_graph = os.environ.get("RESET_GRAPH", "false").lower() == "true"

        if skip_ingest:
            print("SKIP_INGEST=true -> reusing existing graph, no add()/cognify() calls made.")
        else:
            if reset_graph:
                print(
                    "RESET_GRAPH=true -> wiping the graph and rebuilding from scratch "
                    "(full LLM extraction + embeddings for every paper)."
                )
            else:
                print(
                    "Ingesting data/papers/ (incremental: Cognee will skip any paper "
                    "whose content was already processed - only new/changed papers "
                    "cost LLM/embedding calls)."
                )
            await ingest_papers_folder(DEFAULT_PAPERS_DIR, dataset_name=DATASET_NAME, reset=reset_graph)

        pairs = await find_novel_pairs()
        top_pairs = pairs[:top_n]

        report = [await process_pair(pair, i) for i, pair in enumerate(top_pairs)]

        print_report(report)

        run_metrics = evaluate_run(report, novel_pairs=top_pairs)
        print(format_eval_summary(run_metrics))
        print()

        cumulative = history_summary()
        print(
            "Cumulative across all logged runs "
            f"(run_history.jsonl): {cumulative['total_runs_logged']} pairs verified, "
            f"{cumulative['unique_pairs_seen']} unique, "
            f"{cumulative['cumulative_approval_rate']:.0%} approval rate."
        )
        print()

        # Optional debug visualization of the full graph
        graph_debug_path = str(Path(GRAPH_VISUALIZATION_FILENAME).resolve())
        await cognee.visualize_graph(graph_debug_path, dataset=DATASET_NAME)
        print(f"Graph visualization written to: {graph_debug_path}")
    finally:
        await close_cognee_telemetry_session()
