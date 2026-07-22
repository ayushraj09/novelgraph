"""
Shared helper for parsing cognee.search() return values.

BUG THIS FIXES: cognee.search() does not return a list of plain strings.
It returns a list with one entry PER DATASET SEARCHED, shaped like:

    [
      {
        'dataset_id': UUID('...'),
        'dataset_name': 'research',
        'dataset_tenant_id': None,
        'search_result': ['<actual answer text>', ...],
      },
      ...
    ]

chat.py, hypothesis.py, and evidence.py previously did `results[0]` and
treated that as the answer string. In reality `results[0]` is the whole
dict above, so the raw dict (UUIDs and all) got printed / stored / passed
downstream instead of the answer text - visible directly in chat.py's
REPL output ("Cognee: {'dataset_id': UUID(...), ...}").

These two helpers are now the single place that unpacks this shape, so any
future SearchType or Cognee version change only needs to be fixed here.
"""

from typing import List


def extract_search_results(results) -> List[str]:
    """Flatten cognee.search() output into a flat list of result strings.

    Handles:
    - The normal shape: list[{'search_result': [str, ...]}]
    - A 'search_result' that's a single string instead of a list
    - Legacy/unexpected shapes (plain list of strings) as a fallback
    """
    if not results:
        return []

    texts: List[str] = []
    for item in results:
        if isinstance(item, dict) and "search_result" in item:
            sr = item["search_result"]
            if isinstance(sr, list):
                texts.extend(str(x).strip() for x in sr if str(x).strip())
            elif sr:
                texts.append(str(sr).strip())
        elif item:
            # Fallback for shapes without a 'search_result' key
            texts.append(str(item).strip())
    return texts


def extract_search_text(results) -> str:
    """Flatten cognee.search() output into a single joined string.

    Use this where you want one blob of text (chat.py, hypothesis.py).
    Use extract_search_results() where you want a list of separate
    bullet-able items (evidence.py).
    """
    return "\n\n".join(extract_search_results(results))
