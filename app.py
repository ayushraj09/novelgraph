"""Streamlit application for the research knowledge-graph pipeline.

Run with: streamlit run app.py
"""

from __future__ import annotations

import asyncio
import html
import os
import re
from pathlib import Path

# Import novelgraph (and therefore novelgraph.config) BEFORE cognee, so
# SYSTEM_ROOT_DIRECTORY/DATA_ROOT_DIRECTORY are resolved to absolute,
# project-anchored paths before cognee reads its own configuration.
import novelgraph  # noqa: F401
import cognee
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from novelgraph.chat import (
    BRIEF_SYSTEM_PROMPT,
    MAX_HISTORY_TURNS,
    _build_query_with_context,
    _compact_text,
    _paper_inventory_answer,
)
from novelgraph.graph_store import (
    EDGE_LABEL_CANDIDATES,
    EDGE_SOURCE_CANDIDATES,
    EDGE_TARGET_CANDIDATES,
    NODE_ID_CANDIDATES,
    NODE_NAME_CANDIDATES,
    NODE_TYPE_CANDIDATES,
    load_graph,
    node_property,
    pick_col,
    resolve_node_name,
)
from novelgraph.ingest import ingest
from novelgraph.novelty import find_novel_pairs
from novelgraph.search_utils import extract_search_text

load_dotenv()

APP_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = APP_DIR / "uploaded_papers"
DATASET_NAME = "research"
SUPPORTED_FILES = ["pdf", "docx", "pptx", "txt", "md", "csv"]
MAX_DIAGRAM_NODES = 18
MAX_DIAGRAM_EDGES = 28

st.set_page_config(
    page_title="Research Copilot",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {max-width: 1180px; padding-top: 3rem; padding-bottom: 4rem;}
      [data-testid="stMetric"] {background: rgba(120,120,120,.06); border: 1px solid rgba(120,120,120,.18); padding: 1rem; border-radius: .8rem;}
      .page-heading {padding-top:.35rem; margin-bottom:.35rem;}
      .eyebrow {display:block; font-size:.78rem; font-weight:700; line-height:1.45; letter-spacing:.12em; text-transform:uppercase; opacity:.62; margin:0 0 .35rem;}
      .hero {font-size:2.45rem; font-weight:750; line-height:1.1; margin:.25rem 0 .5rem;}
      .subtle {opacity:.7; font-size:1.02rem; margin-bottom:1.4rem;}
      .library-list {display:flex; flex-direction:column; gap:.7rem; margin-top:.7rem; max-width:880px;}
      .library-item {display:grid; grid-template-columns:1.45rem minmax(0,1fr); column-gap:.55rem; align-items:start; font-size:1.02rem; line-height:1.45;}
      .library-dot {font-size:1.1rem; line-height:1.45; opacity:.9;}
      .library-title {overflow-wrap:normal; word-break:normal; hyphens:none;}
    </style>
    """,
    unsafe_allow_html=True,
)


def run_async(coro):
    """Run one async backend operation from Streamlit's synchronous runner."""
    return asyncio.run(coro)


def safe_filename(name: str) -> str:
    """Keep uploads inside UPLOAD_DIR and make collisions predictable."""
    clean = Path(name).name
    return re.sub(r"[^A-Za-z0-9._ -]", "_", clean) or "upload"


def graph_snapshot() -> tuple[list, list]:
    try:
        return load_graph()
    except Exception:
        return [], []


def graph_entities() -> dict[str, list[str]]:
    nodes, _ = graph_snapshot()
    if not nodes:
        return {}
    type_col = pick_col(nodes[0], NODE_TYPE_CANDIDATES, "nodes", required=False)
    name_col = pick_col(nodes[0], NODE_NAME_CANDIDATES, "nodes", required=False)
    if not type_col or not name_col:
        return {}

    entities: dict[str, set[str]] = {}
    for node in nodes:
        kind = str(node.get(type_col) or "Unknown")
        name = node_property(node, "title") or resolve_node_name(node, name_col)
        if name:
            entities.setdefault(kind, set()).add(str(name))
    return {kind: sorted(names) for kind, names in sorted(entities.items())}


def paper_titles() -> list[str]:
    return graph_entities().get("Paper", [])


def clean_title(title: str) -> str:
    text = " ".join(str(title).replace("_", " ").split())
    text = re.sub(r"(?<=[a-z])(?=for\b)", " ", text)
    return text


def render_library_list(titles: list[str], limit: int = 8) -> None:
    rows = []
    for title in titles[:limit]:
        rows.append(
            '<div class="library-item">'
            '<span class="library-dot">•</span>'
            f'<span class="library-title">{html.escape(clean_title(title))}</span>'
            '</div>'
        )
    st.markdown(f'<div class="library-list">{"".join(rows)}</div>', unsafe_allow_html=True)
    if len(titles) > limit:
        st.caption(f"And {len(titles) - limit} more.")


def _node_display_name(node: dict, name_col: str) -> str:
    return str(node_property(node, "title") or resolve_node_name(node, name_col) or "")


def _mermaid_id(index: int) -> str:
    return f"N{index}"


def _mermaid_text(value: str, limit: int = 84) -> str:
    text = " ".join(str(value).split())
    if len(text) > limit:
        text = text[: limit - 3].rstrip() + "..."
    return text.replace('"', "'")


def _edge_label(value: str) -> str:
    label = re.sub(r"[^A-Za-z0-9_ /-]+", " ", str(value or "related_to"))
    return " ".join(label.split()) or "related_to"


def _strip_mermaid_fences(text: str) -> str:
    cleaned = str(text or "").strip()
    cleaned = re.sub(r"^```(?:mermaid)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _valid_mermaid(code: str) -> bool:
    return _mermaid_validation_error(code) is None


def _mermaid_validation_error(code: str) -> str | None:
    stripped = str(code or "").strip()
    if not stripped:
        return "Empty Mermaid code."
    if not re.match(r"^(flowchart|graph)\s+(TD|TB|LR|RL|BT)\b", stripped, re.IGNORECASE):
        return "The diagram must start with `flowchart TD`, `flowchart LR`, or another valid Mermaid direction."

    allowed_node_ref = re.compile(r"\b(?:N\d+|HYP)\b")
    unsafe_label = re.compile(r"\b(?:N\d+|HYP)\[[^\"]")
    for line_number, raw_line in enumerate(stripped.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("%%") or line.lower().startswith(("flowchart ", "graph ")):
            continue
        if unsafe_label.search(line):
            return (
                f"Line {line_number} uses an unquoted node label. "
                "Use N0[\"label\"] or HYP[\"label\"], never N0[label]."
            )
        if "`" in line:
            return f"Line {line_number} contains backticks, which often break Mermaid rendering."
        if "-->" in line:
            parts = line.split("-->", 1)
            if not allowed_node_ref.search(parts[0]) or not allowed_node_ref.search(parts[1]):
                return f"Line {line_number} has an edge whose source or target is not a known Mermaid node ID."
        elif not re.match(r"^(?:N\d+|HYP)\[\".+\"\]$", line):
            return (
                f"Line {line_number} is not a safe node declaration or edge. "
                "Use only quoted node declarations and simple `A -->|label| B` edges."
            )
    return None


def _chat_graph_context(question: str, answer: str) -> tuple[list[dict], list[dict]]:
    """Select real graph nodes/edges relevant to the latest chat turn."""
    nodes, edges = graph_snapshot()
    if not nodes or not edges:
        return [], []

    id_col = pick_col(nodes[0], NODE_ID_CANDIDATES, "nodes", required=False)
    name_col = pick_col(nodes[0], NODE_NAME_CANDIDATES, "nodes", required=False)
    type_col = pick_col(nodes[0], NODE_TYPE_CANDIDATES, "nodes", required=False)
    src_col = pick_col(edges[0], EDGE_SOURCE_CANDIDATES, "edges", required=False)
    tgt_col = pick_col(edges[0], EDGE_TARGET_CANDIDATES, "edges", required=False)
    label_col = pick_col(edges[0], EDGE_LABEL_CANDIDATES, "edges", required=False)
    if not all([id_col, name_col, src_col, tgt_col]):
        return [], []

    id_to_node = {node[id_col]: node for node in nodes}
    id_to_name = {node[id_col]: _node_display_name(node, name_col) for node in nodes}
    id_to_type = {
        node[id_col]: str(node.get(type_col) or "Entity")
        for node in nodes
    }
    context = f"{question}\n{answer}".lower()

    mentioned = []
    for node_id, name in id_to_name.items():
        if len(name) < 4:
            continue
        if name.lower() in context:
            mentioned.append(node_id)
    mentioned = sorted(set(mentioned), key=lambda node_id: len(id_to_name[node_id]), reverse=True)

    if not mentioned:
        return [], []

    selected = set(mentioned[:MAX_DIAGRAM_NODES])
    selected_edges = []

    # First include exact edges among mentioned nodes, then expand one hop
    # so the diagram can explain the path behind a chat answer.
    for edge in edges:
        source = edge.get(src_col)
        target = edge.get(tgt_col)
        if source in selected and target in selected:
            selected_edges.append(edge)

    for edge in edges:
        if len(selected) >= MAX_DIAGRAM_NODES:
            break
        source = edge.get(src_col)
        target = edge.get(tgt_col)
        if source in selected and target in id_to_node:
            selected.add(target)
            selected_edges.append(edge)
        elif target in selected and source in id_to_node:
            selected.add(source)
            selected_edges.append(edge)

    for edge in edges:
        source = edge.get(src_col)
        target = edge.get(tgt_col)
        if source in selected and target in selected and edge not in selected_edges:
            selected_edges.append(edge)

    ordered_nodes = sorted(
        selected,
        key=lambda node_id: (id_to_type.get(node_id, ""), id_to_name.get(node_id, "")),
    )
    node_context = [
        {
            "id": _mermaid_id(index),
            "graph_id": node_id,
            "type": id_to_type.get(node_id, "Entity"),
            "name": id_to_name.get(node_id, "Unnamed entity"),
        }
        for index, node_id in enumerate(ordered_nodes)
    ]
    graph_to_mermaid_id = {item["graph_id"]: item["id"] for item in node_context}

    edge_context = []
    seen_edges = set()
    for edge in selected_edges:
        source = edge.get(src_col)
        target = edge.get(tgt_col)
        if source not in graph_to_mermaid_id or target not in graph_to_mermaid_id:
            continue
        label = _edge_label(edge.get(label_col) if label_col else "")
        key = (source, target, label)
        if key in seen_edges:
            continue
        seen_edges.add(key)
        edge_context.append({
            "source": graph_to_mermaid_id[source],
            "target": graph_to_mermaid_id[target],
            "label": label,
        })
        if len(edge_context) >= MAX_DIAGRAM_EDGES:
            break

    return node_context, edge_context


def build_chat_flow_diagram(question: str, answer: str) -> str:
    """Create a deterministic Mermaid fallback from selected graph context."""
    node_context, edge_context = _chat_graph_context(question, answer)
    if not node_context:
        return ""

    lines = ["flowchart TD"]
    for node in node_context:
        kind = _mermaid_text(node["type"], 24)
        name = _mermaid_text(node["name"])
        lines.append(f'    {node["id"]}["{kind}: {name}"]')

    for edge in edge_context:
        lines.append(f'    {edge["source"]} -->|{edge["label"]}| {edge["target"]}')

    return "\n".join(lines)


def generate_llm_flow_diagram(question: str, answer: str, max_retry: int = 2) -> str:
    """Ask a small LLM call to turn selected graph context into neat Mermaid."""
    node_context, edge_context = _chat_graph_context(question, answer)
    if not node_context:
        return ""

    node_lines = "\n".join(
        f'- {node["id"]}: {node["type"]}: {node["name"]}' for node in node_context
    )
    edge_lines = "\n".join(
        f'- {edge["source"]} -> {edge["target"]}: {edge["label"]}' for edge in edge_context
    ) or "- No direct selected edges; use the nodes to show a conservative evidence map."

    prompt = f"""
Create a clean and user-friendly Mermaid flowchart for a research knowledge-graph explanation.

Use ONLY these node IDs and labels. Do not create new factual nodes.
You should omit less relevant nodes. Omit nodes with all numbers as they are not user-friendly.
You may add one final summary node named HYP["Hypothesis / takeaway"] only if it helps explain the chat answer.
Use short readable labels, but preserve the real method/dataset/paper/task names.
Prefer flowchart TD or LR. Return Mermaid code only, with no markdown fence.
Every node label must be quoted, like N0["DDR dataset"]. Never write N0[DDR dataset].
Use simple edges like N0 -->|method| N1. Avoid parentheses, backticks, markdown, HTML, or unsupported Mermaid syntax.

Question:
{question}

Answer:
{answer}

Allowed nodes:
{node_lines}

Allowed edges:
{edge_lines}
""".strip()

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
    if not api_key:
        return ""

    client = OpenAI(api_key=api_key)
    messages = [
        {
            "role": "system",
            "content": (
                "You write valid Mermaid flowcharts from provided graph context. "
                "Do not invent paper, method, dataset, task, or result nodes. "
                "Always quote node labels, for example N0[\"label\"]."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    attempts = max(1, max_retry + 1)
    last_code = ""
    last_error = ""
    for attempt in range(attempts):
        response = client.chat.completions.create(
            model=os.environ.get("DIAGRAM_MODEL", "gpt-4.1-mini"),
            temperature=0.1,
            max_tokens=900,
            messages=messages,
        )
        last_code = _strip_mermaid_fences(response.choices[0].message.content)
        last_error = _mermaid_validation_error(last_code) or ""
        if not last_error:
            return last_code
        if attempt < attempts - 1:
            messages.append({"role": "assistant", "content": last_code})
            messages.append({
                "role": "user",
                "content": (
                    "The Mermaid code above failed local validation before rendering:\n"
                    f"{last_error}\n\n"
                    "Return corrected Mermaid code only. Keep every node label quoted, "
                    "use only the allowed node IDs, and avoid advanced Mermaid syntax."
                ),
            })

    st.session_state.last_mermaid_error = last_error
    return ""


def render_mermaid(code: str) -> None:
    safe_code = html.escape(code)
    diagram_path = APP_DIR / "mermaid_diagram.html"
    diagram_path.write_text(
        f"""<!doctype html>
        <html>
        <head>
          <meta charset="utf-8" />
          <style>
            body {{
              margin: 0;
              background: transparent;
              color: inherit;
              font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            }}
            .mermaid-wrap {{
              width: 100%;
              min-height: 420px;
              overflow: auto;
              border: 1px solid rgba(120,120,120,.22);
              border-radius: 8px;
              padding: 18px;
              box-sizing: border-box;
              background: rgba(255,255,255,.02);
            }}
          </style>
        </head>
        <body>
        <div class="mermaid-wrap">
          <pre class="mermaid">{safe_code}</pre>
        </div>
        <script type="module">
          import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs";
          mermaid.initialize({{ startOnLoad: true, theme: "neutral", securityLevel: "loose" }});
        </script>
        </body>
        </html>
        """,
        encoding="utf-8",
    )
    st.iframe(diagram_path, height=520)


def initialise_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("last_candidates", [])
    st.session_state.setdefault("discovery_ran", False)
    st.session_state.setdefault("discovery_error", "")
    st.session_state.setdefault("last_chat_question", "")
    st.session_state.setdefault("last_chat_answer", "")
    st.session_state.setdefault("chat_mermaid", "")
    st.session_state.setdefault("last_mermaid_error", "")


async def reset_workspace() -> None:
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    (APP_DIR / "graph_debug.html").unlink(missing_ok=True)


async def refresh_graph_visualization() -> None:
    await cognee.visualize_graph(str(APP_DIR / "graph_debug.html"), dataset=DATASET_NAME)


def render_sidebar() -> str:
    with st.sidebar:
        st.title("Research Copilot")
        page = st.radio(
            "Navigation",
            ["Overview", "Add papers", "Chat", "Discover", "Graph"],
            label_visibility="collapsed",
        )

        st.divider()
        st.caption("OPENAI CONNECTION")
        api_key = st.text_input(
            "API key",
            type="password",
            placeholder="Uses .env when left blank",
            label_visibility="collapsed",
        )
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            os.environ["LLM_API_KEY"] = api_key
        configured = bool(os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY"))
        if configured:
            st.success("API key configured")
        else:
            st.warning("API key required")

        nodes, edges = graph_snapshot()
        st.divider()
        st.caption("WORKSPACE")
        left, right = st.columns(2)
        left.metric("Nodes", len(nodes))
        right.metric("Edges", len(edges))

        with st.expander("Advanced settings"):
            triplets = st.toggle("Evidence embeddings", value=True)
            figures = st.toggle("Caption PDF figures", value=False)
            os.environ["TRIPLET_EMBEDDING"] = str(triplets).lower()
            os.environ["CAPTION_FIGURES"] = str(figures).lower()
            st.caption("Figure captioning adds vision-model calls during ingestion.")

            if st.button("Reset workspace", use_container_width=True):
                try:
                    with st.spinner("Resetting…"):
                        run_async(reset_workspace())
                    st.session_state.messages = []
                    st.session_state.chat_history = []
                    st.session_state.last_candidates = []
                    st.session_state.discovery_ran = False
                    st.session_state.discovery_error = ""
                    st.session_state.last_chat_question = ""
                    st.session_state.last_chat_answer = ""
                    st.session_state.chat_mermaid = ""
                    st.session_state.last_mermaid_error = ""
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not reset: {exc}")
    return page


def page_header(label: str, title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="page-heading">
          <div class="eyebrow">{html.escape(label)}</div>
          <div class="hero">{html.escape(title)}</div>
          <div class="subtle">{html.escape(subtitle)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_overview() -> None:
    page_header("Knowledge workspace", "Turn papers into connected research insight", "Upload a corpus, ask grounded questions, and surface method–dataset combinations worth investigating.")
    nodes, edges = graph_snapshot()
    papers = paper_titles()
    c1, c2, c3 = st.columns(3)
    c1.metric("Papers", len(papers))
    c2.metric("Knowledge nodes", len(nodes))
    c3.metric("Relationships", len(edges))

    st.subheader("Get started")
    a, b, c = st.columns(3)
    with a:
        st.info("**1 · Add papers**\n\nUpload PDFs or documents and build the knowledge graph.")
    with b:
        st.info("**2 · Ask questions**\n\nExplore findings across papers with short conversational memory.")
    with c:
        st.info("**3 · Discover gaps**\n\nRank structurally novel method and dataset pairings.")

    if papers:
        st.subheader("Current library")
        render_library_list(papers)
    else:
        st.warning("Your workspace is empty. Open **Add papers** from the sidebar to begin.")


def render_upload() -> None:
    page_header("Ingestion", "Add research papers", "Files are added incrementally; unchanged content is not processed twice.")
    files = st.file_uploader(
        "Drop files here",
        type=SUPPORTED_FILES,
        accept_multiple_files=True,
        help="PDF, Word, PowerPoint, Markdown, text, and CSV are supported.",
    )
    if files:
        total_mb = sum(file.size for file in files) / 1_048_576
        st.caption(f"{len(files)} file(s) selected · {total_mb:.1f} MB")

    if st.button("Build knowledge graph", type="primary", disabled=not files):
        if not (os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")):
            st.error("Add an OpenAI API key in the sidebar or your .env file first.")
            return
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        paths = []
        for file in files:
            destination = UPLOAD_DIR / safe_filename(file.name)
            destination.write_bytes(file.getvalue())
            paths.append(str(destination))
        try:
            with st.status("Building your knowledge graph…", expanded=True) as status:
                st.write("Preparing documents")
                st.write("Extracting entities and relationships")
                run_async(ingest(paths, dataset_name=DATASET_NAME, reset=False))
                st.write("Refreshing graph visualization")
                run_async(refresh_graph_visualization())
                status.update(label="Knowledge graph ready", state="complete")
            st.success(f"Added {len(paths)} document(s).")
        except Exception as exc:
            st.error(f"Ingestion failed: {type(exc).__name__}: {exc}")

    titles = paper_titles()
    if titles:
        st.divider()
        st.subheader(f"Library · {len(titles)} papers")
        render_library_list(titles, limit=len(titles))


def answer_question(question: str) -> str:
    local = _paper_inventory_answer(question)
    if local is not None:
        return local
    context = st.session_state.chat_history[-MAX_HISTORY_TURNS:]
    query = _build_query_with_context(context, question)
    results = run_async(
        cognee.search(
            query_type=cognee.SearchType.GRAPH_COMPLETION,
            query_text=query,
            system_prompt=BRIEF_SYSTEM_PROMPT,
        )
    )
    return extract_search_text(results) or "I couldn't find a grounded answer in the graph."


def render_chat() -> None:
    page_header("Grounded Q&A", "Chat with your papers", "Answers use the local knowledge graph and the last few turns for follow-up context.")
    if not graph_snapshot()[0]:
        st.warning("Add and ingest at least one paper before starting a chat.")
        return

    if st.session_state.messages and st.button("Clear conversation"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.session_state.last_chat_question = ""
        st.session_state.last_chat_answer = ""
        st.session_state.chat_mermaid = ""
        st.session_state.last_mermaid_error = ""
        st.rerun()
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if question := st.chat_input("Ask about methods, results, datasets, or comparisons…"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching the graph…"):
                    answer = answer_question(question)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.session_state.chat_history.append((question, _compact_text(answer)))
                st.session_state.last_chat_question = question
                st.session_state.last_chat_answer = answer
                st.session_state.chat_mermaid = ""
                st.session_state.last_mermaid_error = ""
            except Exception as exc:
                st.error(f"Search failed: {type(exc).__name__}: {exc}")

    if st.session_state.last_chat_answer:
        st.divider()
        st.subheader("Flow diagram")
        st.caption("Generate a polished Mermaid view from exact graph nodes and relationships in the chat context.")
        if st.button("Generate flow diagram", type="secondary"):
            question = st.session_state.last_chat_question
            answer = st.session_state.last_chat_answer
            diagram = ""
            st.session_state.last_mermaid_error = ""
            if os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY"):
                try:
                    with st.spinner("Drafting a clean Mermaid diagram…"):
                        diagram = generate_llm_flow_diagram(question, answer, max_retry=2)
                except Exception as exc:
                    st.caption(f"LLM diagram polish failed, using local fallback: {type(exc).__name__}")
            if not diagram:
                diagram = build_chat_flow_diagram(question, answer)
                if diagram and st.session_state.last_mermaid_error:
                    st.caption(f"LLM Mermaid repair did not pass validation; using local fallback. Last issue: {st.session_state.last_mermaid_error}")
            if diagram:
                st.session_state.chat_mermaid = diagram
            else:
                st.session_state.chat_mermaid = ""
                st.warning("I could not find enough exact graph entities in the last answer to draw a grounded diagram.")
        if st.session_state.chat_mermaid:
            render_mermaid(st.session_state.chat_mermaid)
            with st.expander("Mermaid source"):
                st.code(st.session_state.chat_mermaid, language="mermaid")


def render_discover() -> None:
    page_header("Research discovery", "Find promising combinations", "Surface method-dataset pairings that may be worth testing next.")
    st.caption(
        "Best case: candidates share an extracted task but were not tried together. "
        "On smaller graphs, NovelGraph falls back to exploratory cross-paper matches "
        "ranked by method/dataset description overlap."
    )
    limit = st.slider("Candidates to show", 3, 20, 8)
    if st.button("Find research gaps", type="primary"):
        st.session_state.discovery_ran = True
        st.session_state.discovery_error = ""
        try:
            with st.spinner("Analysing graph structure…"):
                all_candidates = run_async(find_novel_pairs())
                st.session_state.last_candidates = all_candidates[:limit]
            if st.session_state.last_candidates:
                st.success(f"Found {len(st.session_state.last_candidates)} candidate(s).")
            else:
                st.warning("Discovery ran, but no candidate pairs were found in the current graph.")
        except Exception as exc:
            st.session_state.last_candidates = []
            st.session_state.discovery_error = f"{type(exc).__name__}: {exc}"
            st.error(f"Discovery failed: {st.session_state.discovery_error}")

    candidates = st.session_state.last_candidates
    if not candidates:
        if st.session_state.discovery_error:
            st.error(f"Last discovery error: {st.session_state.discovery_error}")
        elif st.session_state.discovery_ran:
            st.info(
                "Discovery completed, but no candidates were available. The graph likely "
                "does not yet contain enough Method and DatasetNode entities with usable "
                "descriptions; try resetting and rebuilding with the latest ingestion pipeline."
            )
        else:
            st.info("Click **Find research gaps** after ingesting papers to rank candidate method-dataset pairings.")
        return
    for rank, item in enumerate(candidates, 1):
        with st.container(border=True):
            st.markdown(f"**{rank}. {item['method']} × {item['dataset']}**")
            mode = item.get("mode", "shared_task")
            if mode == "exploratory":
                st.caption("Exploratory candidate: no shared Task edge found, ranked by description overlap.")
            else:
                st.caption(f"Shared via: {item.get('shared_via') or 'task connection'}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Shared tasks", item.get("shared_neighbors", 0))
            similarity = item.get("similarity")
            c2.metric("Lexical similarity", f"{similarity:.2f}" if isinstance(similarity, (int, float)) else "—")
            c3.metric("Mode", "Exploratory" if mode == "exploratory" else "Strict")


def render_graph() -> None:
    page_header("Knowledge graph", "Visualise and explore entities", "Browse extracted entities and view the interactive graph visualization.")

    # Render interactive graph visualization
    graph_html_path = APP_DIR / "graph_debug.html"
    if graph_html_path.exists():
        with st.expander("Interactive visualization", expanded=True):
            st.iframe(graph_html_path, height=600)
    elif graph_snapshot()[0]:
        st.info("Run the pipeline once with `python main.py` to generate the graph visualization.")

    st.divider()

    entities = graph_entities()
    if not entities:
        st.warning("No graph data yet. Add papers to build the knowledge graph.")
        return
    types = list(entities)
    selected = st.multiselect("Entity types", types, default=types[:3])
    query = st.text_input("Filter entities", placeholder="Search by name…").strip().lower()
    for kind in selected:
        names = [name for name in entities[kind] if not query or query in name.lower()]
        with st.expander(f"{kind} · {len(names)}", expanded=len(selected) == 1):
            if names:
                for name in names:
                    st.markdown(f"- {name}")
            else:
                st.caption("No matches")


def main() -> None:
    initialise_state()
    page = render_sidebar()
    {
        "Overview": render_overview,
        "Add papers": render_upload,
        "Chat": render_chat,
        "Discover": render_discover,
        "Graph": render_graph,
    }[page]()


if __name__ == "__main__":
    main()
