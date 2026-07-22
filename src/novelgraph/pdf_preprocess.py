"""
Rich PDF preprocessing: text + layout + tables + (optional) figure captions.

WHY THIS EXISTS: Cognee's default PDF path uses PyPdfLoader - bare
pypdf-based text extraction - unless `unstructured[pdf]` is installed, in
which case it upgrades to AdvancedPdfLoader (layout + table structure
preserved, falls back to PyPdfLoader on error). Neither extracts embedded
images/figures/diagrams/algorithm listings at all - and research papers
routinely put exactly the content your schema cares about (benchmark
tables -> Result, architecture diagrams, pseudocode) inside those.

This module converts each PDF into a single markdown string up front -
text/headings/tables preserved via pymupdf4llm, with optional inline
figure captions from a vision-capable LLM - and that markdown is what
gets handed to cognee.add() as raw text, instead of pointing add() at the
PDF path. This means ingestion quality no longer depends on which loader
Cognee happens to pick internally, and works the same regardless of
whether `unstructured[pdf]` is installed.

COST CONTROL:
- Text/table extraction (pymupdf4llm) is free and local - always on.
- Figure captioning calls the OpenAI vision API once per embedded image
  above a small size threshold (skips tiny logos/bullets). It's OFF by
  default - set CAPTION_FIGURES=true to enable - in keeping with this
  project's existing cost-safety posture (see smoke_test.py).
- Both text extraction and captioning are cached to disk by file content
  hash (PDF_TEXT_CACHE_DIR, default ./.pdf_text_cache), so re-running
  ingestion on unchanged papers costs nothing further even before
  Cognee's own add()/cognify() dedup layers see the content - this is
  what makes it safe to call this on the whole data/papers/ folder every
  run without worrying about re-captioning images you've already paid for.

Requires: pip install pymupdf pymupdf4llm  (and `openai` if CAPTION_FIGURES=true)
Falls back to returning the original PDF path unchanged if pymupdf4llm
isn't installed, so this is a strict upgrade, never a hard dependency.
"""

import base64
import hashlib
import os

try:
    import pymupdf4llm
    import fitz  # PyMuPDF
    PDF_LIBS_AVAILABLE = True
except ImportError:
    PDF_LIBS_AVAILABLE = False

CACHE_DIR = os.environ.get("PDF_TEXT_CACHE_DIR", ".pdf_text_cache")

CAPTION_PROMPT = (
    "This image is a figure, diagram, chart, or algorithm/pseudocode "
    "listing extracted from a research paper. Describe it precisely and "
    "concisely for someone who cannot see it: what it shows, any labeled "
    "methods/datasets/metrics/numbers visible, and its likely role in the "
    "paper (architecture diagram, results chart, ablation table, "
    "algorithm listing, etc). 2-4 sentences, no preamble."
)

# Embedded images smaller than this (bytes) are treated as decorative
# (logos, bullet icons, letterhead) rather than real figures, and skipped
# to avoid wasting vision-API calls on them.
MIN_FIGURE_BYTES = 3000


def _file_hash(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def _cache_path(pdf_path: str, caption_figures: bool) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    suffix = "captioned" if caption_figures else "textonly"
    return os.path.join(CACHE_DIR, f"{_file_hash(pdf_path)}_{suffix}.md")


def _caption_image(image_bytes: bytes, client) -> str:
    """Caption a single embedded image via an OpenAI vision-capable model."""
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": CAPTION_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ],
        }],
        max_tokens=200,
    )
    return response.choices[0].message.content.strip()


def _extract_figure_captions(pdf_path: str) -> list:
    """Extracts every sufficiently-large embedded raster image from the
    PDF and captions it. Returns '[Figure N, page P]: <caption>' strings
    in document order. Requires OPENAI_API_KEY."""
    from openai import OpenAI
    client = OpenAI()

    captions = []
    doc = fitz.open(pdf_path)
    figure_num = 0
    try:
        for page_index in range(len(doc)):
            page = doc[page_index]
            for img in page.get_images(full=True):
                xref = img[0]
                try:
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    if len(image_bytes) < MIN_FIGURE_BYTES:
                        continue
                    figure_num += 1
                    caption = _caption_image(image_bytes, client)
                    captions.append(f"[Figure {figure_num}, page {page_index + 1}]: {caption}")
                except Exception as exc:
                    print(f"  [warn] could not caption an image on page {page_index + 1}: {exc}")
    finally:
        doc.close()
    return captions


def pdf_to_rich_text(pdf_path: str, caption_figures: bool = None) -> str:
    """Converts a PDF into a single markdown string: full text with
    headings/paragraphs/tables preserved (pymupdf4llm), plus optionally a
    'Figures' section with LLM-generated captions for embedded images.

    Cached to disk by file content hash - re-calling this on an unchanged
    PDF (even across separate script runs) costs nothing further.

    Falls back to returning pdf_path unchanged (so cognee.add() uses its
    own default loader) if pymupdf4llm isn't installed.
    """
    if not PDF_LIBS_AVAILABLE:
        print(f"  [warn] pymupdf4llm/pymupdf not installed - falling back to "
              f"Cognee's default PDF loader for {pdf_path}. "
              f"Run: pip install pymupdf pymupdf4llm")
        return pdf_path

    if caption_figures is None:
        caption_figures = os.environ.get("CAPTION_FIGURES", "false").lower() == "true"

    cache_path = _cache_path(pdf_path, caption_figures)
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return f.read()

    print(f"  Parsing {pdf_path} (text + tables via pymupdf4llm"
          f"{', captioning figures' if caption_figures else ''})...")

    markdown = pymupdf4llm.to_markdown(pdf_path)

    if caption_figures:
        try:
            captions = _extract_figure_captions(pdf_path)
            if captions:
                markdown += "\n\n## Figures\n\n" + "\n\n".join(captions)
        except Exception as exc:
            print(f"  [warn] figure captioning failed for {pdf_path}, "
                  f"continuing with text/tables only: {exc}")

    with open(cache_path, "w", encoding="utf-8") as f:
        f.write(markdown)

    return markdown
