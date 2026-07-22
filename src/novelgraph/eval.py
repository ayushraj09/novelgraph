"""
Evaluation - lightweight, zero-cost metrics for the demo.

The original plan calls for novelty score, precision/recall against
held-out method-dataset pairs, human quality ratings, and graph-cohesion
metrics. Full versions of those need either labeled ground truth or human
raters, neither of which fits a hackathon timeline. This module computes
the subset that's mechanically derivable from a completed run's report
(no LLM/embedding calls, no ground truth needed), so there's a real answer
ready if judges ask "how do you know these hypotheses are good?"

Metrics:
- pairs_evaluated: how many novel (method, dataset) candidates were run
  through Stage 6/7.
- approval_rate: fraction the Critic approved (every cited claim checked
  out against the graph) - a cheap proxy for "hypothesis is grounded, not
  hallucinated".
- evidence_coverage: fraction of approved hypotheses that also got at
  least one Stage 7 evidence bullet - i.e. actually citable, not just
  approved in principle.
- avg_shared_neighbors / avg_similarity: from novelty.py's own scoring,
  reported here so the report and the eval summary agree on how "novel"
  the chosen pairs were.
"""

from typing import Dict, List


def evaluate_run(report: List[dict], novel_pairs: List[dict] = None) -> Dict:
    total = len(report)
    if total == 0:
        return {
            "pairs_evaluated": 0,
            "approval_rate": 0.0,
            "evidence_coverage": 0.0,
            "avg_shared_neighbors": 0.0,
            "avg_similarity": 0.0,
        }

    approved = sum(1 for r in report if r.get("approved"))
    with_evidence = sum(1 for r in report if r.get("approved") and r.get("evidence"))

    metrics = {
        "pairs_evaluated": total,
        "approval_rate": round(approved / total, 3),
        "evidence_coverage": round(with_evidence / approved, 3) if approved else 0.0,
    }

    if novel_pairs:
        used_pairs = novel_pairs[:total]
        if used_pairs:
            metrics["avg_shared_neighbors"] = round(
                sum(p.get("shared_neighbors", 0) for p in used_pairs) / len(used_pairs), 3
            )
            metrics["avg_similarity"] = round(
                sum(p.get("similarity", 0.0) for p in used_pairs) / len(used_pairs), 3
            )

    return metrics


def format_eval_summary(metrics: Dict) -> str:
    lines = ["=" * 60, "EVALUATION SUMMARY", "=" * 60]
    lines.append(f"Pairs evaluated:      {metrics.get('pairs_evaluated', 0)}")
    lines.append(f"Approval rate:        {metrics.get('approval_rate', 0.0):.0%}"
                 f"  (Critic verified every cited claim against the graph)")
    lines.append(f"Evidence coverage:    {metrics.get('evidence_coverage', 0.0):.0%}"
                 f"  (approved hypotheses with >=1 citable evidence bullet)")
    if "avg_shared_neighbors" in metrics:
        lines.append(f"Avg shared neighbors: {metrics['avg_shared_neighbors']}"
                     f"  (structural novelty signal, from novelty.py)")
    if "avg_similarity" in metrics:
        lines.append(f"Avg lexical similarity: {metrics['avg_similarity']}"
                     f"  (description-overlap proxy for embedding similarity)")
    return "\n".join(lines)
