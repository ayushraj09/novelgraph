"""
Stage 5: Temporal Framing - custom function version.

Answers: "has this method been applied to this dataset (or a task the
dataset shares) before a given year?" using Paper.year directly - NOT
Cognee's SearchType.TEMPORAL, which depends on temporal_cognify=True (now
disabled - see ingest.py for why).

This reads Paper nodes' `year` field via graph_store.py (direct SQLite
reads, zero cost - no LLM/embedding calls) and checks which Papers connect
to the given method and dataset before the cutoff year.
"""

from collections import defaultdict
from .graph_store import (
    load_graph, pick_col, node_property, resolve_node_name, infer_tasks_from_node,
    NODE_ID_CANDIDATES, NODE_NAME_CANDIDATES,
    EDGE_SOURCE_CANDIDATES, EDGE_TARGET_CANDIDATES, EDGE_LABEL_CANDIDATES,
)

METHOD_EDGE_LABEL = "method"      # Paper -> Method
DATASET_EDGE_LABEL = "dataset"    # Paper -> DatasetNode
USED_FOR_EDGE_LABEL = "used_for"  # Method -> Task
TASKS_EDGE_LABEL = "tasks"        # DatasetNode -> Task


async def check_temporal_novelty(method_name: str, dataset_name: str, before_year: int = 2024) -> str:
    nodes, edges = load_graph()
    if not nodes or not edges:
        return "No graph data available."

    id_col = pick_col(nodes[0], NODE_ID_CANDIDATES, "nodes")
    name_col = pick_col(nodes[0], NODE_NAME_CANDIDATES, "nodes")
    src_col = pick_col(edges[0], EDGE_SOURCE_CANDIDATES, "edges")
    tgt_col = pick_col(edges[0], EDGE_TARGET_CANDIDATES, "edges")
    label_col = pick_col(edges[0], EDGE_LABEL_CANDIDATES, "edges")

    id_to_name = {n[id_col]: resolve_node_name(n, name_col) for n in nodes}
    id_to_node = {n[id_col]: n for n in nodes}

    by_source = defaultdict(list)
    by_target = defaultdict(list)
    for e in edges:
        by_source[e[src_col]].append(e)
        by_target[e[tgt_col]].append(e)

    # Tasks associated with the target dataset (for the "or a shared task" check)
    dataset_id = next((nid for nid, name in id_to_name.items() if name == dataset_name), None)
    dataset_tasks = set()
    if dataset_id is not None:
        dataset_tasks = {
            id_to_name.get(e[tgt_col])
            for e in by_source.get(dataset_id, [])
            if e.get(label_col) == TASKS_EDGE_LABEL
        }
    dataset_tasks.discard(None)

    if not dataset_tasks:
        for paper_id, out_edges in by_source.items():
            paper_datasets = [
                id_to_name.get(e[tgt_col])
                for e in out_edges
                if e.get(label_col) == DATASET_EDGE_LABEL
            ]
            if dataset_name in paper_datasets:
                dataset_tasks.update(infer_tasks_from_node(id_to_node.get(paper_id, {})))

    # Find every Paper that links to this method, and check what it also links to
    prior_uses = []
    for paper_id, out_edges in by_source.items():
        method_names = [id_to_name.get(e[tgt_col]) for e in out_edges if e.get(label_col) == METHOD_EDGE_LABEL]
        if method_name not in method_names:
            continue

        paper_node = id_to_node.get(paper_id, {})
        year = node_property(paper_node, "year")
        if year is not None:
            try:
                year = int(year)
            except (TypeError, ValueError):
                year = None

        paper_datasets = [id_to_name.get(e[tgt_col]) for e in out_edges if e.get(label_col) == DATASET_EDGE_LABEL]

        # Direct match on the dataset itself, OR a task the dataset shares
        matched_directly = dataset_name in paper_datasets
        matched_via_task = False
        if not matched_directly and dataset_tasks:
            for pd in paper_datasets:
                pd_id = next((nid for nid, name in id_to_name.items() if name == pd), None)
                pd_tasks = set()
                if pd_id is not None:
                    pd_tasks = {
                        id_to_name.get(e[tgt_col])
                        for e in by_source.get(pd_id, [])
                        if e.get(label_col) == TASKS_EDGE_LABEL
                    }
                    pd_tasks.discard(None)
                if not pd_tasks:
                    pd_tasks = infer_tasks_from_node(paper_node)
                if pd_tasks & dataset_tasks:
                    matched_via_task = True
                    break

        if matched_directly or matched_via_task:
            if year is not None and year < before_year:
                prior_uses.append({
                    "paper_id": paper_id,
                    "dataset": paper_datasets,
                    "year": year,
                    "match": "direct" if matched_directly else "shared_task",
                })

    if not prior_uses:
        return (
            f"No prior use of '{method_name}' on '{dataset_name}' (or a related dataset "
            f"sharing its task) found with a recorded year before {before_year}."
        )

    lines = [
        f"'{method_name}' was applied to a related dataset in {u['year']} "
        f"({u['match']} match via {u['dataset']})."
        for u in sorted(prior_uses, key=lambda u: u["year"])
    ]
    return " ".join(lines)


if __name__ == "__main__":
    import asyncio
    print(asyncio.run(check_temporal_novelty("Method A", "Dataset Z")))
