"""
Lightweight persistent run history for the hypothesis-generation pipeline.

WHY THIS EXISTS: the knowledge graph itself already persists across runs
(Cognee writes SQLite/vector/graph stores to disk - that's what makes
SKIP_INGEST=true work). What was NOT persisted anywhere was the
hypothesis pipeline's own history: which (Method, Dataset) pairs Stage 6
already verified, and whether the Critic approved them. Without that,
re-running main.py re-spends Stage 6 (2 LLM agents, up to 2 rounds) and
Stage 7 (evidence retrieval) on pairs you've already resolved, every time.

This is a minimal append-only JSONL log - no new database, no new
service, consistent with the project's existing "zero-cost local reads"
philosophy (graph_store.py, novelty.py). main.py checks it before
spending Stage 6/7 calls on a pair, and appends to it after each run.

This deliberately does NOT try to be a general conversational memory
system (see README's "Persistent memory" section for why that scope was
kept out) - it only tracks this pipeline's own prior verdicts.
"""

import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

HISTORY_PATH = os.environ.get("RUN_HISTORY_PATH", "run_history.jsonl")


def load_history(history_path: str = None) -> List[dict]:
    path = history_path or HISTORY_PATH
    if not os.path.exists(path):
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def find_prior_result(method: str, dataset: str, history_path: str = None) -> Optional[dict]:
    """Returns the most recent recorded result for this (method, dataset)
    pair, or None if it's never been run before."""
    match = None
    for record in load_history(history_path):
        if record.get("method") == method and record.get("dataset") == dataset:
            match = record
    return match


def record_result(method: str, dataset: str, approved: bool, hypothesis: str,
                   evidence: Optional[List[str]] = None, history_path: str = None) -> None:
    path = history_path or HISTORY_PATH
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "method": method,
        "dataset": dataset,
        "approved": approved,
        "hypothesis": hypothesis,
        "evidence": evidence or [],
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def summary(history_path: str = None) -> Dict:
    """Cumulative stats across every run ever logged - not just the
    current process's report. Useful for the demo: 'across N runs over
    the whole project, X% of hypotheses were graph-verified.'"""
    history = load_history(history_path)
    total = len(history)
    approved = sum(1 for r in history if r.get("approved"))
    unique_pairs = {(r.get("method"), r.get("dataset")) for r in history}
    return {
        "total_runs_logged": total,
        "unique_pairs_seen": len(unique_pairs),
        "cumulative_approval_rate": round(approved / total, 3) if total else 0.0,
    }
