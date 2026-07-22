# NovelGraph

> Turning a folder of research papers into a typed knowledge graph, then using structural graph analysis and an agentic verification loop to propose and ground novel research directions.

NovelGraph ingests research papers into a **typed entity graph** (Paper, Method, Dataset, Task, Result), detects **Method ↔ Dataset combinations that have never been tried together but are structurally close** (via shared-neighbor analysis and lexical task clustering), and then runs a **multi-hop reasoning + dual-agent verification loop** to turn each candidate pair into an evidence-backed hypothesis — with every citation checked against the literal graph, not just plausible-sounding.

I built this end-to-end as my own project during a hackathon organized by [Cognee](https://cognee.ai/), which provided the underlying graph-construction and retrieval engine; the schema design, novelty-detection heuristics, orchestration pipeline, agentic verification loop, and the Streamlit interface are my own work.

## Why this exists

Literature review doesn't scale past a few dozen papers. A researcher can hold maybe 10-15 papers' worth of Method/Dataset combinations in their head at once; a real corpus has hundreds. The interesting question isn't "what did paper X do" — it's "what combination of an existing method and an existing dataset has *nobody tried yet*, even though they're one hop apart in the literature." That's a graph-structure question, not a summarization question, which is why this is built around explicit typed nodes and edges rather than a flat vector index.

## How it works

```
Papers (PDF/DOCX/PPTX/TXT/MD/CSV)
        │
        ▼
┌────────────────────┐   1. Typed schema        Paper --method--> Method
│  Graph Construction│   2. Chunked extraction  Paper --dataset-> DatasetNode
│  (typed DataPoints)│      + incremental       DatasetNode --tasks--> Task
└────────────────────┘        ingestion          Method --used_for--> Task
        │                                       Result --derived_from--> Method
        ▼
┌────────────────────┐   3. Untested (Method, Dataset) pairs ranked by
│ Novelty Detection  │      shared structural neighbors (common Task),
│ (structural, local)│      with Jaccard-clustered task names and a
└────────────────────┘      lexical-similarity tiebreaker — zero LLM calls
        │
        ▼
┌────────────────────┐   4. Multi-hop chain-of-thought retrieval proposes
│ Hypothesis Seeding │      a first-draft hypothesis across the graph
└────────────────────┘
        │
        ▼
┌────────────────────┐   5/6. Temporal/prior-art check, then a Generator/
│ Agentic Refinement │       Critic loop (LangGraph) where the Critic
│  (Generator/Critic)│       rejects any citation that isn't an exact,
└────────────────────┘       literal node/edge identity in the graph
        │
        ▼
┌────────────────────┐   7/8. Provenance-grounded evidence bullets pulled
│ Evidence & Report  │       from triplet-level retrieval, plus run-level
└────────────────────┘       evaluation metrics (approval rate, evidence
                             coverage, novelty scores)
```

**Design choices worth calling out:**

- **No direct Method↔Dataset edge in the schema.** A Paper is what links a Method and a Dataset. This is deliberate: it's exactly what makes "these two are structurally close but never directly connected" a well-defined, computable condition rather than a fuzzy semantic judgment.
- **Task-name clustering, not exact-match.** The same real-world task shows up as different literal strings across papers ("skin lesion classification" vs. "medical image classification"). Greedy single-pass clustering by Jaccard word-overlap merges these before joining Method/Dataset pairs on a shared Task, without ever calling an embedding API for it.
- **Citation grounding is enforced, not assumed.** Both agent prompts define a valid citation as an *exact* node/edge identity as it appears in the graph — not a paraphrase of a node's description. A logged production run surfaced exactly the failure mode this guards against: an invented phrase that sounded plausible enough to pass a naive check. Read the full writeup in [`docs/APP_GUIDE.md`](docs/APP_GUIDE.md#a-caveat-about-approved-hypotheses).
- **Cost-aware by construction.** Content-hash deduplication at ingestion, pipeline-status dedup at graph-build time, and a persisted run-history log mean re-running the pipeline never re-spends LLM/embedding calls on papers or (Method, Dataset) pairs already processed.

### Graph visualization

Every pipeline run also writes an interactive HTML visualization of the constructed graph.

<!-- Replace this with a screenshot or embed of your generated graph_debug.html -->
![Knowledge graph visualization](docs/graph_screenshot.png)

## Project layout

```
.
├── src/novelgraph/          # Installable package — all core logic
│   ├── schema.py            # Typed DataPoint graph schema
│   ├── ingest.py            # PDF preprocessing + incremental ingestion
│   ├── pdf_preprocess.py    # Layout/table-aware PDF -> markdown, optional figure captioning
│   ├── graph_store.py       # Direct, zero-cost reads of the graph's SQLite store
│   ├── novelty.py           # Structural novelty detection + task clustering
│   ├── temporal.py          # Prior-art / "has this been tried before year X" checks
│   ├── hypothesis.py        # Multi-hop chain-of-thought hypothesis seeding
│   ├── agents.py            # LangGraph Generator/Critic verification loop
│   ├── evidence.py          # Triplet-level evidence/citation assembly
│   ├── eval.py              # Run-level evaluation metrics
│   ├── run_history.py       # Persistent verdict log (JSONL, append-only)
│   ├── search_utils.py      # Parses the graph engine's raw search response shape
│   ├── chat.py              # Conversational Q&A over the ingested graph
│   └── pipeline.py          # End-to-end orchestration used by scripts/main.py
├── scripts/                 # Thin CLI entry points around the package
│   ├── main.py               # Full pipeline: ingest -> novelty -> hypothesis -> verify -> report
│   ├── chat_cli.py           # Terminal REPL for Q&A over the graph
│   ├── smoke_test.py         # Cheap end-to-end sanity check (pennies, not your real corpus)
│   ├── debug_novelty.py      # Re-run novelty detection against the existing graph, zero cost
│   └── inspect_db.py         # Dump the graph engine's SQLite schema/rows
├── app.py                   # Streamlit UI (upload, chat, discover, graph explorer)
├── docs/APP_GUIDE.md         # Streamlit UI walkthrough + known caveats
├── data/papers/              # Drop research papers here
├── pyproject.toml
└── .env.example
```

## Setup

This project uses [`uv`](https://docs.astral.sh/uv/) for dependency management.

```bash
git clone https://github.com/ayushraj09/novelgraph.git
cd novelgraph

uv sync                 # creates .venv and installs everything from pyproject.toml
cp .env.example .env    # fill in your OpenAI key and storage paths
```

### Sanity check first (costs pennies)

```bash
uv run scripts/smoke_test.py
```

This ingests two tiny inline documents, then runs novelty detection and the temporal check against that toy graph so you can confirm the wiring works before spending real API cost on your own papers.

### Run the full pipeline

```bash
# Drop PDFs/DOCX/PPTX/TXT/MD/CSV into data/papers/, then:
uv run scripts/main.py

# Re-run without re-ingesting (reuse the existing graph):
SKIP_INGEST=true uv run scripts/main.py

# Force a full rebuild (e.g. after changing schema.py):
RESET_GRAPH=true uv run scripts/main.py
```

### Chat with your papers

```bash
uv run scripts/chat_cli.py
```

### Interactive UI

```bash
uv run streamlit run app.py
```

See [`docs/APP_GUIDE.md`](docs/APP_GUIDE.md) for a full tour of the UI's tabs, including the graph explorer and the automated citation checker.

## Configuration

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | Required. Used by the LangGraph agents. |
| `LLM_API_KEY` | — | Same key, separate variable required by the graph engine internally. |
| `TRIPLET_EMBEDDING` | `true` | Must be set before first ingestion, or evidence assembly returns empty. |
| `SYSTEM_ROOT_DIRECTORY` / `DATA_ROOT_DIRECTORY` | auto-resolved to `<project root>/.cognee_system` / `.cognee_data` | Absolute paths pinning graph storage to this project. Resolved automatically on a fresh clone by `novelgraph/config.py`; set explicitly only to override. |
| `SKIP_INGEST` | `false` | Skip ingestion entirely, reuse the existing graph. |
| `RESET_GRAPH` | `false` | Wipe and rebuild the graph from scratch. |
| `CAPTION_FIGURES` | `false` | Enable vision-model captioning of embedded PDF figures (costs one call per image, cached by file hash). |
| `TASK_SIMILARITY_THRESHOLD` | `0.25` | Jaccard similarity cutoff for clustering Task name strings. |
| `SKIP_COT_SEED` | `false` | Skip the multi-hop chain-of-thought seed and generate hypotheses from scratch instead. |
| `RESEARCH_CHUNK_SIZE` | `60000` | Chunk size used during graph construction. |

## Tech stack

- **Graph construction & retrieval:** [Cognee](https://cognee.ai/) as the underlying typed-graph engine and retrieval layer
- **Agentic reasoning:** LangGraph (Generator/Critic loop)
- **LLM / embeddings:** OpenAI (`gpt-4.1-mini`, `text-embedding-3-small`)
- **PDF parsing:** pymupdf / pymupdf4llm
- **Interface:** Streamlit
- **Persistence:** SQLite (graph engine's own store) + an append-only JSONL run-history log

## Acknowledgements

Built during a hackathon organized by Cognee, with [HardikShreays](https://github.com/HardikShreays) as a collaborator on the team.

## License

MIT
