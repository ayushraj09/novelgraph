"""
Stage 7: Evidence Assembly

Once a hypothesis is approved in Stage 6, this stage pulls the exact
supporting triplets and their source document/chunk provenance, so the
final output can show citations instead of free text.

Requires TRIPLET_EMBEDDING=true to have been set BEFORE Stage 2's cognify()
ran - otherwise the Triplet_text collection this retriever relies on won't
exist.

FIXED: previously returned the raw `results` list from cognee.search()
untouched - i.e. a list of {'dataset_id': UUID(...), 'search_result': [...]}
dicts, which main.py then printed verbatim as "Evidence: [{...}]". Now
returns a plain list[str] of evidence bullets via
search_utils.extract_search_results(), so callers can render them as
"- <bullet>" lines directly.
"""

from typing import List
import cognee
from cognee import SearchType
from .search_utils import extract_search_results


async def get_evidence(method_name: str, dataset_name: str) -> List[str]:
    """Returns a list of evidence strings (not a single blob and not the
    raw Cognee response), ready to render as report bullets."""
    query = f"{method_name} {dataset_name} evidence"
    results = await cognee.search(
        query_type=SearchType.TRIPLET_COMPLETION,
        query_text=query,
    )
    return extract_search_results(results)


if __name__ == "__main__":
    import asyncio
    for bullet in asyncio.run(get_evidence("Method A", "Dataset Z")):
        print("-", bullet)
