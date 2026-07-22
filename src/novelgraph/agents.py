"""
Stage 6: Multi-Agent Refinement Loop

Two LangGraph agents share one Cognee session: a Generator that proposes a
hypothesis and cites evidence, and a Critic that checks every cited claim
against the graph using the same session's search tool. If the Critic finds
an unsupported claim, it sends feedback back to the Generator. This loop
runs at most twice.

Uses the real, published "cognee-integration-langgraph" package
(get_sessionized_cognee_tools), confirmed against the official README and
docs.cognee.ai/integrations/langgraph-integration.

Note on the model: rather than guessing a "provider:model" string for
init_chat_model (which is version-sensitive), this uses an explicit
ChatOpenAI instance pinned to gpt-4.1-mini. Reads OPENAI_API_KEY from
the environment automatically via python-dotenv + langchain-openai.

WIRING FIX: Stage 4 (hypothesis.py's GRAPH_COMPLETION_COT) previously ran
but was never actually consumed anywhere - main.py doesn't import it, so it
was dead code. run_refinement() now calls generate_hypothesis() once up
front and passes its output to the Generator as a "seed" candidate on round
0. This means Cognee's own multi-hop chain-of-thought retrieval proposes
the starting hypothesis, and the Generator/Critic loop's job becomes
verifying and refining that seed (citing real nodes/triplets) rather than
generating one from scratch. If the seed call fails for any reason (e.g.
COT search unsupported in a given Cognee version), we fall back to the
original from-scratch behavior rather than crashing the whole run.

Set SKIP_COT_SEED=true in the environment to skip the Stage 4 call
entirely and go back to pure from-scratch generation (useful for cost
control while debugging Stage 6 in isolation).

PROMPT FIX (both agents): the original Generator/Critic prompts said
"cite the specific graph node or triplet names you rely on" / "verify
every cited node or triplet... actually exists" without ever defining
what counts as a valid citation. In practice this let the Generator cite
paraphrased fragments of a node's *description* text as if they were
node names themselves (e.g. "Node: prior preservation with EyePACS/APTOS"
- not a real node in the graph, just wording lifted from a description) -
and the Critic then "verified" these using GRAPH_COMPLETION-style search,
which returns fluent supporting prose for almost any plausible-sounding
query rather than actually checking node identity. This let ungrounded
citations get APPROVED, defeating the whole point of the two-agent check.

Both prompts now explicitly define a valid citation as an EXACT node/edge
name as it appears in the graph, not a paraphrase or summary fragment,
and the Critic is explicitly told to reject citations that aren't exact
identities even if the surrounding claim sounds plausible.
"""

import os
from typing import TypedDict
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from cognee_integration_langgraph import get_sessionized_cognee_tools

from .hypothesis import generate_hypothesis

MODEL = ChatOpenAI(model="gpt-4.1-mini", temperature=0)


class RefinementState(TypedDict):
    method_name: str
    dataset_name: str
    hypothesis: str
    feedback: str
    seed_hypothesis: str
    rounds: int
    approved: bool


def build_agents(session_id: str):
    add_tool, search_tool = get_sessionized_cognee_tools(session_id)

    generator = create_react_agent(
        MODEL,
        tools=[search_tool],
        prompt=(
            "Propose one hypothesis about the given method and dataset. "
            "Every claim you cite must reference an EXACT node or "
            "relationship name as it literally appears in the graph - use "
            "the search tool to look it up first. Do NOT cite a paraphrase, "
            "summary, or fragment of a node's description text as if it "
            "were itself a node (e.g. do not invent a citation like 'Node: "
            "prior preservation with X' unless a node with that exact name "
            "exists). If you cannot find an exact node/relationship to "
            "support a claim, drop that claim rather than citing something "
            "approximate. If a seed candidate hypothesis is provided, "
            "verify it against the graph using the search tool and refine "
            "it so every citation is an exact, verifiable node/relationship "
            "name. If feedback is provided instead, revise the hypothesis "
            "to address it, fixing any citation the feedback flagged as "
            "not exact."
        ),
    )

    critic = create_react_agent(
        MODEL,
        tools=[search_tool],
        prompt=(
            "Verify every cited claim in the hypothesis against the graph "
            "using the search tool. A citation is only valid if it names an "
            "EXACT node or relationship as it literally appears in the "
            "graph - not a paraphrase, summary, or fragment of a node's "
            "description text. If a citation names something that is not "
            "itself a real, exact node/edge identity (e.g. a constructed "
            "phrase like 'prior preservation with X' that isn't a graph "
            "node), treat that citation as UNSUPPORTED, even if the "
            "surrounding claim sounds plausible or is corroborated by "
            "fluent search results. Reply 'APPROVED' only if every "
            "citation resolves to a real, exact node or relationship. "
            "Otherwise reply 'REJECTED: <reason>', listing each citation "
            "that failed and why."
        ),
    )

    return generator, critic


def _build_generator_message(state: RefinementState) -> str:
    parts = [f"Method: {state['method_name']}, Dataset: {state['dataset_name']}."]

    if state["rounds"] == 0 and state.get("seed_hypothesis"):
        parts.append(
            "A candidate hypothesis was already generated via multi-hop graph "
            f"reasoning: \"{state['seed_hypothesis']}\". Verify this against the "
            "graph and refine it, citing only exact nodes/relationships it "
            "relies on."
        )
    else:
        parts.append(f"Prior feedback: {state.get('feedback', 'none')}")

    return " ".join(parts)


async def generator_node(state: RefinementState, generator) -> RefinementState:
    msg = _build_generator_message(state)
    response = await generator.ainvoke({"messages": [HumanMessage(content=msg)]})
    state["hypothesis"] = response["messages"][-1].content
    return state


async def critic_node(state: RefinementState, critic) -> RefinementState:
    response = await critic.ainvoke({"messages": [HumanMessage(content=state["hypothesis"])]})
    verdict = response["messages"][-1].content
    state["approved"] = verdict.strip().upper().startswith("APPROVED")
    state["feedback"] = "" if state["approved"] else verdict
    state["rounds"] += 1
    return state


def should_continue(state: RefinementState) -> str:
    if state["approved"] or state["rounds"] >= 2:
        return "end"
    return "retry"


async def run_refinement(method_name: str, dataset_name: str, session_id: str) -> RefinementState:
    generator, critic = build_agents(session_id)

    seed_hypothesis = ""
    skip_seed = os.environ.get("SKIP_COT_SEED", "false").lower() == "true"
    if not skip_seed:
        try:
            seed_hypothesis = await generate_hypothesis(method_name, dataset_name)
        except Exception as exc:
            # Non-fatal: fall back to from-scratch generation if Stage 4's
            # COT search fails for any reason (e.g. unsupported in this
            # Cognee version, or the graph is too sparse for it).
            print(f"  [warn] Stage 4 COT seed failed for {method_name}/{dataset_name}: {exc}")
            seed_hypothesis = ""

    state: RefinementState = {
        "method_name": method_name,
        "dataset_name": dataset_name,
        "hypothesis": "",
        "feedback": "",
        "seed_hypothesis": seed_hypothesis,
        "rounds": 0,
        "approved": False,
    }
    while True:
        state = await generator_node(state, generator)
        state = await critic_node(state, critic)
        if should_continue(state) == "end":
            break
    return state


if __name__ == "__main__":
    import asyncio
    result = asyncio.run(run_refinement("Method A", "Dataset Z", session_id="hackathon-run-1"))
    print(result)