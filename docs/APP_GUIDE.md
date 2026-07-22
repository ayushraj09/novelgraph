# Research Novelty Explorer (Streamlit)

A local Streamlit front-end over the existing Cognee + LangGraph pipeline
(`novelgraph.ingest`, `novelgraph.novelty`, `novelgraph.temporal`, `novelgraph.agents`, `novelgraph.evidence`, `novelgraph.eval`, `novelgraph.run_history`, `novelgraph.chat`). It doesn't reimplement any of the
pipeline logic - it just wraps it with a UI.

## Run it

```bash
uv sync
cp .env.example .env   # fill in your OpenAI key, or paste it in the sidebar instead
uv run streamlit run app.py
```

## What's in each tab

- **Upload & Ingest**: drag in papers (PDF/DOCX/PPTX/TXT/MD/CSV). PDFs get
  the same pymupdf4llm pre-parsing as the CLI. Ingestion is incremental -
  re-uploading an already-ingested paper costs nothing further.
- **Chat**: same `GRAPH_COMPLETION` flow as `scripts/chat_cli.py`, including the paper
  inventory shortcut and short in-session memory, just rendered as a chat
  UI instead of a terminal loop. Use this to sanity-check ingestion and
  explore *before* running the expensive Novelty pipeline - it's a single
  LLM call per question, versus 6+ per pair in the pipeline below.
- **Novelty & Hypotheses**: runs Stages 3-8, in two modes:
  - *Auto-discover top pairs* - `novelty.py`'s shared-Task heuristic ranks
    candidate (Method, Dataset) pairs; on a small corpus this can surface
    just one pair (or none), so don't be surprised if the slider doesn't
    matter yet.
  - *Check a specific pair* - pick (or type) any Method/Dataset directly
    and run the same Stage 5-7 verification on it. This is the one worth
    using once you have your own paper in mind: type your method against
    an existing dataset node (or vice versa) and see whether the Critic
    can actually ground a hypothesis for it.

  Both modes generate a hypothesis, verify it through the Generator/Critic
  loop, pull evidence, and show a per-run + cumulative eval summary.
  Verified pairs are cached in `run_history.jsonl` so re-running doesn't
  re-spend LLM calls on pairs you've already resolved. Each result also
  gets an **automated citation check** (see below).
- **Graph Explorer**: lists every exact Method/DatasetNode/Task/Paper/Result
  name in the graph (zero-cost SQLite read), and can render the same
  interactive `cognee.visualize_graph()` output `scripts/main.py` writes to
  `graph_debug.html` at the end of its run - inline, instead of a file you
  have to go find and open separately.

## Why this is single-workspace, not multi-tenant

`graph_store.py` reads the `nodes`/`edges` tables directly out of Cognee's
SQLite store with no per-dataset filtering, and the pipeline's
`cognee.search()` calls don't scope by dataset either. So even though
`ingest()` accepts a `dataset_name`, the novelty/chat/evidence queries all
see everything ever ingested into the local Cognee system - there's no
code-level isolation between "workspaces" today. Rather than fake
multi-user isolation on top of that, this app is deliberately **one graph
at a time**: use the sidebar's "Reset graph" button before starting a new
set of papers if you want a clean slate. Making this properly multi-user
would mean either running separate Cognee system directories per user
(via `SYSTEM_ROOT_DIRECTORY`/`COGNEE_SQLITE_PATH`) or adding dataset-scoped
filtering into `graph_store.py` - worth doing if this grows past a
hackathon demo.

## A caveat about "APPROVED" hypotheses

A real logged run in `run_history.jsonl` (a `scripts/main.py` run over a small,
43-node graph) shows the failure mode precisely. Checking each citation
against the literal node names actually used in that run:

- `(Node: Class-conditioned diffusion model fine-tuning with semantic
  quality evaluation and filtering)` - **this one is fine.** It's
  character-for-character identical to the real Method node's name, which
  `novelty.py` resolved via `graph_store.py`'s exact-name lookup.
- `"HAM10000 and APTOS datasets"` - the real node is named `"HAM10000 and
  APTOS Medical Image Datasets"`. Close, but not exact - a paraphrase.
- `"prior preservation with EyePACS/APTOS"` - doesn't correspond to any
  node at all. It's the Critic/Generator describing a mechanism from a
  cited paper's Section 3.1 as if it were itself a citable node.

So the Critic isn't uniformly unreliable - it reproduced the one exact
string it was handed verbatim without issue. What it doesn't reliably
catch is an *invented* phrase that merely sounds plausible as a node,
which is exactly what `agents.py`'s prompt asks it to reject and exactly
what got through anyway. It checks this via its own LLM judgment against
a prompt instruction, not a code-level lookup.

The app now adds that code-level check on top: every result runs through
`check_citations()`, which pulls `(Node: ...)`/`(Nodes: ...)` spans out of
the generated text and compares each one against the graph's real entity
names from `graph_store.py` (same zero-cost SQLite read `novelty.py` and
`temporal.py` already rely on). It flags `"prior preservation with
EyePACS/APTOS"`-style phrases as **not an exact match** while correctly
passing citations like the Method name above. It's a heuristic regex over
varied LLM phrasing, not a formal parser, so treat a clean check as
reassuring and a flagged one as "verify manually" rather than either as
gospel - the Graph Explorer tab lists every real entity name for exactly
that manual spot-check.

## Known gotchas carried over from the CLI

- Needs `TRIPLET_EMBEDDING=true` (default in the sidebar) set before your
  *first* ingest for a graph, or Stage 7 evidence will come back empty.
- `CAPTION_FIGURES=true` costs one vision API call per embedded image on
  first ingest of a given PDF (cached by file hash after that).
- Async: every backend call goes through `asyncio.run()` per Streamlit
  interaction, since Streamlit's script-rerun model has no persistent
  event loop between reruns.
