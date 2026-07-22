"""
Stage 4: Multi-Hop Hypothesis Generation

For each novel (method, dataset) pair from Stage 3, asks Cognee to reason
across the graph and propose a hypothesis. GRAPH_COMPLETION_COT performs its
own retrieve-validate-follow-up loop internally, which is what covers
multi-hop Method -> Task -> Result reasoning.

FIXED: previously returned `result[0]` (the raw dataset dict cognee.search()
returns) instead of the answer text. Now uses search_utils.extract_search_text().

WIRING: this stage's output was previously computed but never actually used
anywhere in the pipeline (main.py only imports novelty/temporal/agents/
evidence) - Stage 4 was dead code. It's now consumed by agents.py's
Generator as an initial "seed" candidate on round 0, so Cognee's own
multi-hop COT retrieval and the Generator/Critic verification loop work
together instead of the COT output being computed and discarded.
"""

import cognee
from cognee import SearchType
from .search_utils import extract_search_text


async def generate_hypothesis(method_name: str, dataset_name: str) -> str:
    prompt = (
        f"Method '{method_name}' and Dataset '{dataset_name}' have no direct connection "
        f"in the graph but share related context. Propose one concise, evidence-backed "
        f"hypothesis for applying '{method_name}' to '{dataset_name}'. "
        f"Cite the specific triplet or node names you rely on."
    )
    result = await cognee.search(
        query_type=SearchType.GRAPH_COMPLETION_COT,
        query_text=prompt,
    )
    return extract_search_text(result)


if __name__ == "__main__":
    import asyncio
    print(asyncio.run(generate_hypothesis("Method A", "Dataset Z")))
