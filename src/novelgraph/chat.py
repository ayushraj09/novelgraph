"""
Conversational Q&A over the ingested knowledge graph.

This is separate from the Stage 1-8 hypothesis pipeline (hypothesis.py,
evidence.py, etc. use specific search types for the hypothesis-generation
workflow). This module is for open-ended Q&A over whatever has already
been ingested via `ingest.py` / the pipeline entry point.

Requires that ingestion has already run at least once, so there is a
graph to query.

Two things this module fixes relative to a naive `cognee.search()` call:

1. `cognee.search()` returns one entry per dataset searched, shaped like
   `{'dataset_id': UUID(...), 'search_result': [...]}` - not a plain
   string. `search_utils.extract_search_text()` unpacks that correctly.
2. Lightweight in-session conversational memory: without it, "what about
   applying it to Task Z?" has no idea what "it" refers to, since nothing
   carries context between turns. The last few exchanges are prepended as
   context to each new query before it is sent to Cognee's
   `GRAPH_COMPLETION` search.

This is intentionally NOT persisted to disk across process restarts -
that is a different, heavier feature (a real session store) than this
exploratory REPL/UI helper needs. The knowledge graph itself is what
persists across restarts; this only fixes the "no memory even within one
sitting" gap. See the README's persistence section for the full
reasoning.
"""

import re
from collections import deque

import cognee
from cognee import SearchType

from .graph_store import (
    NODE_NAME_CANDIDATES,
    NODE_TYPE_CANDIDATES,
    load_graph,
    node_property,
    pick_col,
    resolve_node_name,
)
from .search_utils import extract_search_text

MAX_HISTORY_TURNS = 3
MAX_MEMORY_ANSWER_CHARS = 300

BRIEF_SYSTEM_PROMPT = (
    "Answer using the provided graph context. Default to brief answers: "
    "use at most 4 sentences or 3 bullets. Do not add introductions, paper "
    "inventories, or extra sections unless the user asks for them. If the "
    "user explicitly asks for detail, depth, a full plan, or step-by-step "
    "reasoning, then provide a longer answer."
)


def _wants_detailed_answer(query: str) -> bool:
    normalized = query.lower()
    return bool(
        re.search(
            r"\b(explain in detail|detail(?:ed)?|deep dive|in depth|full plan|"
            r"step[- ]?by[- ]?step|comprehensive|elaborate|long answer)\b",
            normalized,
        )
    )


def _compact_text(text: str, limit: int = MAX_MEMORY_ANSWER_CHARS) -> str:
    """Keep chat memory useful without resending whole generated answers."""
    one_line = " ".join(str(text).split())
    if len(one_line) <= limit:
        return one_line
    return one_line[: limit - 3].rstrip() + "..."


def _build_query_with_context(history: deque, query: str) -> str:
    detail_instruction = (
        "The user asked for detail, so a longer answer is allowed."
        if _wants_detailed_answer(query)
        else "The user did not ask for detail. Keep the answer brief."
    )

    if not history:
        return f"{detail_instruction}\n\nQuestion: {query}"

    context = "\n\n".join(
        f"Previous question: {q}\nPrevious answer summary: {a}" for q, a in history
    )
    return (
        f"Conversation so far:\n{context}\n\n"
        f"New question (use the conversation above only to resolve "
        f"pronouns/follow-ups. {detail_instruction}): {query}"
    )


async def _paper_inventory_answer(query: str) -> str | None:
    """Answer simple paper inventory questions locally, with zero LLM calls."""
    normalized = query.lower()
    asks_about_papers = re.search(r"\b(papers?|documents?|pdfs?)\b", normalized)
    asks_inventory = re.search(
        r"\b(how many|count|list|what papers|which papers|currently.*graph|access)\b",
        normalized,
    )
    if not (asks_about_papers and asks_inventory):
        return None

    try:
        nodes, _ = await load_graph()
    except Exception:
        return None

    if not nodes:
        return "I do not see any papers in the local graph yet."

    type_col = pick_col(nodes[0], NODE_TYPE_CANDIDATES, "nodes", required=False)
    name_col = pick_col(nodes[0], NODE_NAME_CANDIDATES, "nodes", required=False)
    if not type_col or not name_col:
        return None

    papers = []
    seen = set()
    for node in nodes:
        if node.get(type_col) != "Paper":
            continue
        title = node_property(node, "title") or resolve_node_name(node, name_col)
        if not title or title in seen:
            continue
        seen.add(title)
        papers.append(str(title))

    if not papers:
        return "I do not see any Paper nodes in the local graph yet."

    papers.sort()
    if re.search(r"\b(how many|count)\b", normalized) and not re.search(r"\b(list|what|which)\b", normalized):
        return f"There are {len(papers)} papers currently in the graph."

    paper_lines = "\n".join(f"{index}. {title}" for index, title in enumerate(papers, start=1))
    return f"There are {len(papers)} papers currently in the graph:\n{paper_lines}"


async def ask(query: str, history: deque | None = None) -> str:
    """Answer one question against the graph, optionally using prior-turn
    `history` (a deque of (question, compacted_answer) tuples) to resolve
    pronouns/follow-ups. Does not mutate `history` - callers append the
    returned answer themselves so both the CLI REPL and the Streamlit UI
    can manage their own session state."""
    local_answer = await _paper_inventory_answer(query)
    if local_answer is not None:
        return local_answer

    query_with_context = _build_query_with_context(history or deque(), query)

    # GRAPH_COMPLETION is Cognee's default conversational mode: it pulls
    # relevant graph context (entities + relationships) and asks an LLM
    # to answer grounded in that context.
    results = await cognee.search(
        query_type=SearchType.GRAPH_COMPLETION,
        query_text=query_with_context,
        system_prompt=BRIEF_SYSTEM_PROMPT,
    )
    return extract_search_text(results) or "(no answer found in the graph)"
