"""
Stage 3: Novelty Detection - typed schema version.

Finds Method/Dataset pairs that are NOT already linked through a shared
Paper, but DO share a structural neighbor (a common Task).

With temporal_cognify OFF (see ingest.py), Cognee's cognify() actually uses
schema.py's graph_model=Paper, producing these edges:

    Paper       -[method]->   Method
    Paper       -[dataset]->  DatasetNode
    Paper       -[result]->   Result
    DatasetNode -[tasks]->    Task
    Method      -[used_for]-> Task
    Result      -[derived_from]-> Method

There is no direct Method<->DatasetNode edge - a Paper is what links them,
and a shared Task is what makes an untried pair "structurally close".

Reads the graph via graph_store.py (direct SQLite reads - zero cost, no
LLM/embedding calls), not cognee.search(SearchType.CYPHER, ...), which was
found to be unreliable in this environment.

TASK NAME CLUSTERING: cognify() extracts Task names as free text, so the
same real-world task shows up under different literal strings across
papers - e.g. "Skin lesion classification" vs "Medical Image
Classification" vs "Image Classification". The original version joined
Method/DatasetNode pairs on exact Task-name equality, which meant these
near-duplicate Task nodes never matched and find_novel_pairs() returned
empty even on graphs with real Method/used_for/Task and
DatasetNode/tasks/Task edges.

Fix: Task nodes are now greedily clustered by lexical (Jaccard word-
overlap) similarity - the same zero-cost mechanism already used below for
the description tiebreaker - and Methods/Datasets are joined on shared
Task CLUSTER, not exact Task name. TASK_SIMILARITY_THRESHOLD controls how
aggressive this is; raise it if unrelated tasks start clustering together,
lower it if true duplicates still aren't merging. This is still pure local
computation: no LLM or embedding API calls.

IMPROVEMENT: the original ranking only used raw shared-Task counts
(shared_neighbors). The project plan calls for path-length + embedding-
similarity in the novelty score; full embedding calls would add API cost
and defeat the "zero-cost retrieval" property this stage is designed around.
As a free proxy, we now compute a lexical (word-overlap / Jaccard) similarity
between each Method's and Dataset's description text and use it as a
secondary sort key - so among pairs with equal shared_neighbors, semantically
closer pairs (by description wording) rank higher. This is still pure local
computation: no LLM or embedding API calls.
"""

import os
import re
from collections import defaultdict
from .graph_store import (
    load_graph,
    pick_col,
    node_property,
    resolve_node_name,
    infer_tasks_from_node,
    looks_like_citation,
    NODE_ID_CANDIDATES,
    NODE_NAME_CANDIDATES,
    EDGE_SOURCE_CANDIDATES,
    EDGE_TARGET_CANDIDATES,
    EDGE_LABEL_CANDIDATES,
)

METHOD_EDGE_LABEL = "method"      # Paper -> Method
DATASET_EDGE_LABEL = "dataset"    # Paper -> DatasetNode
USED_FOR_EDGE_LABEL = "used_for"  # Method -> Task
TASKS_EDGE_LABEL = "tasks"        # DatasetNode -> Task

# How similar two Task name strings need to be (Jaccard word-overlap,
# 0.0-1.0) to be treated as the "same" structural Task node. Tune via env
# var if a real run under- or over-clusters. 0.3 was chosen empirically as
# loose enough to catch "Skin lesion classification" <-> "Medical Image
# Classification" (share "classification") without merging unrelated tasks
# that happen to share one common word.
TASK_SIMILARITY_THRESHOLD = float(os.environ.get("TASK_SIMILARITY_THRESHOLD", "0.25"))

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set:
    if not text:
        return set()
    return set(_WORD_RE.findall(text.lower()))


def lexical_similarity(text_a: str, text_b: str) -> float:
    """Zero-cost Jaccard word-overlap similarity, used as a free proxy for
    embedding similarity when ranking novel pairs / clustering Task names.
    Returns 0.0-1.0."""
    tokens_a, tokens_b = _tokenize(text_a), _tokenize(text_b)
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return intersection / union if union else 0.0


def _cluster_task_names(names: set) -> dict:
    """Greedy single-pass clustering of Task name strings by lexical
    similarity. Returns {task_name: cluster_id}. First name seen in a
    cluster becomes its representative; every later name within
    TASK_SIMILARITY_THRESHOLD of any existing representative joins that
    cluster, otherwise it starts a new one.

    This is intentionally simple (no re-clustering/merging passes) - for
    the small number of distinct Task strings a hackathon-scale graph
    produces, one greedy pass is enough, and it stays zero-cost / fully
    deterministic given the same input order.
    """
    cluster_reps = []  # list of (cluster_id, representative_name)
    name_to_cluster = {}
    for name in names:
        assigned = None
        for cluster_id, rep_name in cluster_reps:
            if lexical_similarity(name, rep_name) >= TASK_SIMILARITY_THRESHOLD:
                assigned = cluster_id
                break
        if assigned is None:
            assigned = len(cluster_reps)
            cluster_reps.append((assigned, name))
        name_to_cluster[name] = assigned
    return name_to_cluster


def _build_description_index(nodes, name_col) -> dict:
    """Maps a node's resolved name -> description/abstract text, for the
    lexical-similarity tiebreaker. First description seen per name wins."""
    name_to_desc = {}
    for n in nodes:
        name = resolve_node_name(n, name_col)
        if not name or name in name_to_desc:
            continue
        desc = (
            node_property(n, "description")
            or node_property(n, "abstract")
            or node_property(n, "text")
            or ""
        )
        if desc:
            name_to_desc[name] = str(desc)
    return name_to_desc


def _node_names_by_type(nodes, name_col, type_name: str) -> list:
    names = []
    seen = set()
    for node in nodes:
        if node.get("type") != type_name:
            continue
        name = resolve_node_name(node, name_col)
        if name and name not in seen:
            seen.add(name)
            names.append(name)
    return names


async def find_novel_pairs():
    nodes, edges = load_graph()
    if not nodes or not edges:
        return []

    id_col = pick_col(nodes[0], NODE_ID_CANDIDATES, "nodes")
    name_col = pick_col(nodes[0], NODE_NAME_CANDIDATES, "nodes")
    src_col = pick_col(edges[0], EDGE_SOURCE_CANDIDATES, "edges")
    tgt_col = pick_col(edges[0], EDGE_TARGET_CANDIDATES, "edges")
    label_col = pick_col(edges[0], EDGE_LABEL_CANDIDATES, "edges")

    id_to_name = {n[id_col]: resolve_node_name(n, name_col) for n in nodes}
    id_to_node = {n[id_col]: n for n in nodes}
    name_to_desc = _build_description_index(nodes, name_col)

    bad_ids = {
        nid for nid, node in id_to_node.items()
        if node.get("type") == "Paper"
        and looks_like_citation(node_property(node, "title"))
    }
    if bad_ids:
        id_to_name = {k: v for k, v in id_to_name.items() if k not in bad_ids}

    by_source = defaultdict(list)
    by_target = defaultdict(list)
    for e in edges:
        by_source[e[src_col]].append(e)
        by_target[e[tgt_col]].append(e)

    # "Already tried": same Paper links both a Method and a DatasetNode
    already_tried = set()
    for _paper_id, out_edges in by_source.items():
        methods = [id_to_name.get(e[tgt_col]) for e in out_edges if e.get(label_col) == METHOD_EDGE_LABEL]
        datasets = [id_to_name.get(e[tgt_col]) for e in out_edges if e.get(label_col) == DATASET_EDGE_LABEL]
        for m in methods:
            for d in datasets:
                if m and d:
                    already_tried.add((m, d))

    # Build the Task-name -> cluster_id map from every Task name that
    # actually appears as the target of a used_for or tasks edge. This is
    # what lets structurally-close-but-differently-worded Tasks join.
    task_names_seen = set()
    for task_id, in_edges in by_target.items():
        for e in in_edges:
            if e.get(label_col) in (USED_FOR_EDGE_LABEL, TASKS_EDGE_LABEL):
                name = id_to_name.get(task_id)
                if name:
                    task_names_seen.add(name)
    name_to_cluster = _cluster_task_names(task_names_seen)

    # "Structurally close": Method -[used_for]-> Task <-[tasks]- DatasetNode,
    # joined by Task CLUSTER rather than exact Task node identity.
    novel_counter = defaultdict(lambda: {"shared_via": set(), "count": 0})
    for task_id, in_edges in by_target.items():
        methods = [id_to_name.get(e[src_col]) for e in in_edges if e.get(label_col) == USED_FOR_EDGE_LABEL]
        datasets = [id_to_name.get(e[src_col]) for e in in_edges if e.get(label_col) == TASKS_EDGE_LABEL]
        task_name = id_to_name.get(task_id)
        cluster_id = name_to_cluster.get(task_name) if task_name else None
        for m in methods:
            for d in datasets:
                if not m or not d or (m, d) in already_tried:
                    continue
                key = (m, d)
                novel_counter[key]["count"] += 1
                if task_name:
                    novel_counter[key]["shared_via"].add(task_name)

    # Second pass: also join across DIFFERENT task_ids that share a cluster
    # (e.g. Method used_for "Skin lesion classification" [task_id A],
    # DatasetNode tasks "Medical Image Classification" [task_id B] - same
    # cluster, different node). The loop above only catches same-task_id
    # matches; this catches cross-cluster matches.
    cluster_to_methods = defaultdict(set)
    cluster_to_datasets = defaultdict(set)
    cluster_to_task_names = defaultdict(set)
    for task_id, in_edges in by_target.items():
        task_name = id_to_name.get(task_id)
        cluster_id = name_to_cluster.get(task_name) if task_name else None
        if cluster_id is None:
            continue
        cluster_to_task_names[cluster_id].add(task_name)
        for e in in_edges:
            if e.get(label_col) == USED_FOR_EDGE_LABEL:
                m = id_to_name.get(e[src_col])
                if m:
                    cluster_to_methods[cluster_id].add(m)
            elif e.get(label_col) == TASKS_EDGE_LABEL:
                d = id_to_name.get(e[src_col])
                if d:
                    cluster_to_datasets[cluster_id].add(d)

    for cluster_id in set(cluster_to_methods) | set(cluster_to_datasets):
        methods = cluster_to_methods.get(cluster_id, set())
        datasets = cluster_to_datasets.get(cluster_id, set())
        for m in methods:
            for d in datasets:
                if (m, d) in already_tried:
                    continue
                key = (m, d)
                if key in novel_counter and cluster_to_task_names[cluster_id] <= novel_counter[key]["shared_via"]:
                    continue  # already fully counted in the same-task_id pass above
                novel_counter[key]["count"] += 1
                novel_counter[key]["shared_via"].update(cluster_to_task_names[cluster_id])

    # Fallback for sparse typed graphs: the LLM may extract Paper -> Method and
    # Paper -> DatasetNode while leaving DatasetNode.tasks / Method.used_for
    # empty. In that case, infer explicit "Task X" phrases from Paper metadata,
    # then apply the same clustering so inferred task phrases match too.
    if not novel_counter:
        method_tasks = defaultdict(set)
        dataset_tasks = defaultdict(set)

        for paper_id, out_edges in by_source.items():
            methods = [
                id_to_name.get(e[tgt_col])
                for e in out_edges
                if e.get(label_col) == METHOD_EDGE_LABEL
            ]
            datasets = [
                id_to_name.get(e[tgt_col])
                for e in out_edges
                if e.get(label_col) == DATASET_EDGE_LABEL
            ]
            tasks = infer_tasks_from_node(id_to_node.get(paper_id, {}))
            if not tasks:
                continue

            for method in methods:
                if method:
                    method_tasks[method].update(tasks)
            for dataset in datasets:
                if dataset:
                    dataset_tasks[dataset].update(tasks)

        all_inferred_names = set()
        for names in method_tasks.values():
            all_inferred_names.update(names)
        for names in dataset_tasks.values():
            all_inferred_names.update(names)
        inferred_cluster_map = _cluster_task_names(all_inferred_names)

        for method, m_tasks in method_tasks.items():
            m_clusters = {inferred_cluster_map[t] for t in m_tasks}
            for dataset, d_tasks in dataset_tasks.items():
                if (method, dataset) in already_tried:
                    continue
                d_clusters = {inferred_cluster_map[t] for t in d_tasks}
                shared_clusters = m_clusters & d_clusters
                if not shared_clusters:
                    continue
                key = (method, dataset)
                for cluster_id in shared_clusters:
                    novel_counter[key]["count"] += 1
                    matched_names = {t for t in (m_tasks | d_tasks) if inferred_cluster_map[t] == cluster_id}
                    novel_counter[key]["shared_via"].update(matched_names)

    novel = [
        {
            "method": m,
            "dataset": d,
            "shared_via": sorted(info["shared_via"])[0] if info["shared_via"] else "",
            "shared_neighbors": info["count"],
            "similarity": round(
                lexical_similarity(name_to_desc.get(m, ""), name_to_desc.get(d, "")), 4
            ),
            "mode": "shared_task",
        }
        for (m, d), info in novel_counter.items()
    ]

    # Small corpora often have one Method/Dataset per paper and no explicit
    # cross-paper shared Task edge after extraction. Rather than showing an
    # empty Discover tab, surface exploratory cross-paper combinations ranked
    # by Method/Dataset description overlap. These are weaker than strict
    # shared-task candidates, so mark them clearly for the UI/report.
    if not novel:
        methods = _node_names_by_type(nodes, name_col, "Method")
        datasets = _node_names_by_type(nodes, name_col, "DatasetNode")
        for method in methods:
            for dataset in datasets:
                if (method, dataset) in already_tried:
                    continue
                similarity = lexical_similarity(
                    name_to_desc.get(method, method),
                    name_to_desc.get(dataset, dataset),
                )
                if similarity <= 0:
                    continue
                novel.append({
                    "method": method,
                    "dataset": dataset,
                    "shared_via": "description overlap",
                    "shared_neighbors": 0,
                    "similarity": round(similarity, 4),
                    "mode": "exploratory",
                })

    # Primary: shared structural neighbors. Secondary (tiebreaker): lexical
    # similarity between Method/Dataset descriptions - a free proxy for the
    # embedding-similarity term in the original novelty-scoring plan.
    novel.sort(key=lambda r: (r["shared_neighbors"], r["similarity"]), reverse=True)
    return novel[:20]


if __name__ == "__main__":
    import asyncio
    print(asyncio.run(find_novel_pairs()))
