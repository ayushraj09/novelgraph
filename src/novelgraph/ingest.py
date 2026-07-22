"""
Stage 2: Ingestion

Loads data into Cognee and builds the knowledge graph.

IMPORTANT: temporal_cognify is now OFF (was True in an earlier version).
temporal_cognify=True silently DISABLES graph_model - Cognee's temporal
pipeline replaces schema-guided extraction entirely rather than layering on
top of it. With it off, graph_model=Paper actually takes effect and
produces the typed Paper/Method/DatasetNode/Task/Result graph novelty.py
and temporal.py now expect.

Stage 5's "has this been tried before year X" feature no longer uses
Cognee's built-in temporal search - it's a custom function in temporal.py
that reads the `year` field directly off Paper nodes (see that file's
docstring for why).

cognee.add() accepts raw text, a single file path, a directory path, an S3
path, or a list of any of these - it handles .pdf, .docx, .pptx, .txt, .md,
.csv and more natively, so real research papers can be pointed at directly
without any manual text extraction.

PDF HANDLING: PDFs are now pre-converted to rich markdown text (via
pdf_preprocess.pdf_to_rich_text) before being handed to cognee.add(),
instead of pointing add() at the raw .pdf path. Cognee's own PDF loaders
either do bare text extraction (PyPdfLoader, the default) or - only if
`unstructured[pdf]` happens to be installed - layout+table-aware
extraction (AdvancedPdfLoader), and neither extracts embedded
figures/diagrams/algorithm listings at all. Pre-converting sidesteps that
entirely: pymupdf4llm preserves headings/tables locally for free, and
figure captioning (optional, costs API calls, off by default - see
pdf_preprocess.py) fills in what would otherwise be silently dropped.
Non-PDF files (.docx, .txt, .md, .csv, etc.) are left untouched and go
through Cognee's native handling as before.

INCREMENTAL INGESTION: Cognee's add() and cognify() both default to
incremental_loading=True, with two dedup layers - content-hash dedup in
add(), and pipeline-status dedup in cognify() - so adding a folder that's
mostly already-ingested papers plus one new one only spends LLM/embedding
calls on the new one, automatically, as long as reset=False. See main.py's
RESET_GRAPH env var and its docstring for the full explanation - a
previous version of this project hardcoded reset=True, which defeated
this entirely.

Make sure TRIPLET_EMBEDDING=true is set in your .env BEFORE running this,
since Stage 7 (SearchType.TRIPLET_COMPLETION) needs triplet embeddings that
are only built during cognify().
"""

import asyncio
import os
import re
from pathlib import Path

import cognee
from cognee.tasks.ingestion.data_item import DataItem
from .schema import Paper
from .pdf_preprocess import pdf_to_rich_text

DEFAULT_PAPERS_DIR = "data/papers"
DEFAULT_RESEARCH_CHUNK_SIZE = int(os.environ.get("RESEARCH_CHUNK_SIZE", "60000"))

PAPER_EXTRACTION_PROMPT = """
Extract a research knowledge graph for the uploaded source document.

Critical rules for Paper nodes:
- Create exactly one Paper node for each uploaded source document.
- The Paper.title must be the uploaded source document title provided in
  SOURCE_DOCUMENT_TITLE. Do not create Paper nodes for cited work,
  bibliography entries, figure captions, related work, or examples.
- Use the source document's own abstract/year when available.
- Extract Method, DatasetNode, Task, and Result nodes only when they describe
  the source document's own contribution, experiments, datasets, or findings.
- Do not treat references, author lists, OCR artifacts, or generic figure
  descriptions as new papers.
""".strip()

REFERENCE_SECTION_RE = re.compile(
    r"(?im)^\s*(?:#*\s*)?(?:references|bibliography)\s*$"
)


def _source_title_from_path(path: str) -> str:
    """Human-readable source document title used to anchor one Paper node."""
    stem = Path(path).stem
    cleaned = re.sub(r"[_-]+", " ", stem)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or Path(path).name


def _strip_references(markdown: str) -> str:
    """Drop bibliography tail so cited papers do not become Paper nodes."""
    match = REFERENCE_SECTION_RE.search(markdown)
    if not match:
        return markdown
    return markdown[: match.start()].rstrip()


def _wrap_source_document(text: str, source_title: str, source_path: str) -> DataItem:
    """Give Cognee explicit source-document identity before chunk extraction."""
    body = _strip_references(text)
    wrapped = (
        f"SOURCE_DOCUMENT_TITLE: {source_title}\n"
        f"SOURCE_FILE_NAME: {Path(source_path).name}\n\n"
        "The text below belongs to this single uploaded source document. "
        "When extracting the Paper node, use SOURCE_DOCUMENT_TITLE as the "
        "Paper.title and do not create Paper nodes for cited or referenced work.\n\n"
        f"{body}"
    )
    return DataItem(
        data=wrapped,
        label=source_title,
        external_metadata={
            "source_title": source_title,
            "source_file_name": Path(source_path).name,
            "source_path": str(Path(source_path).resolve()),
        },
    )


def _resolve_item_for_add(item: str) -> list:
    """Expands a single ingest() input into one or more values ready for
    cognee.add(). Raw text (not an existing path) passes through
    unchanged. A directory is walked one level (Cognee's own add() would
    otherwise walk it internally, but we need to intercept PDFs
    specifically) and each entry is resolved recursively. A .pdf file is
    pre-converted to rich markdown via pdf_preprocess so tables/layout
    survive and figures can optionally be captioned; every other file
    type (.docx, .txt, .md, .csv, ...) is left as a path so Cognee's
    native loaders handle it, since those already work well as-is.
    """
    if os.path.isdir(item):
        resolved = []
        for path in sorted(Path(item).iterdir()):
            resolved.extend(_resolve_item_for_add(str(path)))
        return resolved

    if os.path.isfile(item) and item.lower().endswith(".pdf"):
        source_title = _source_title_from_path(item)
        return [_wrap_source_document(pdf_to_rich_text(item), source_title, item)]

    return [item]


async def ingest(data, dataset_name: str = "research", reset: bool = False):
    """
    data: a single string (raw text OR a file/directory path), or a list
    mixing raw text and file/directory paths. Examples:

        await ingest("data/papers")                        # whole folder
        await ingest(["data/papers/a.pdf", "data/papers/b.pdf"])
        await ingest(["Paper: ...", "data/papers/a.pdf"])   # mixed
    """
    if reset:
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)

    items = data if isinstance(data, list) else [data]
    resolved_items = []
    for item in items:
        resolved_items.extend(_resolve_item_for_add(item))

    for item in resolved_items:
        await cognee.add(item, dataset_name=dataset_name)

    await cognee.cognify(
        datasets=[dataset_name],
        graph_model=Paper,
        chunk_size=DEFAULT_RESEARCH_CHUNK_SIZE,
        custom_prompt=PAPER_EXTRACTION_PROMPT,
    )


async def ingest_papers_folder(papers_dir: str = DEFAULT_PAPERS_DIR, dataset_name: str = "research", reset: bool = False):
    """Convenience wrapper: point Cognee directly at a folder of real papers."""
    await ingest(papers_dir, dataset_name=dataset_name, reset=reset)


SAMPLE_DOCS = [
    "Paper: Applying Method A to Dataset X for Task Y in 2022, achieving 92% accuracy.",
    "Paper: Method B was evaluated on Dataset Z for Task Y in 2023.",
]


async def ingest_smoke_test(reset: bool = True):
    """
    Cheap end-to-end validation run using 2 tiny inline strings instead of
    real papers - costs pennies. Use this to verify schema.py, novelty.py,
    and temporal.py all work correctly BEFORE spending real API cost on
    ingesting data/papers/. See smoke_test.py for the full validation flow.
    """
    await ingest(SAMPLE_DOCS, dataset_name="research", reset=reset)


if __name__ == "__main__":
    asyncio.run(ingest_smoke_test(reset=True))
