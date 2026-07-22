"""
Interactive terminal REPL for chatting with the ingested knowledge graph.

The actual Q&A logic (memory handling, paper-inventory shortcut, graph
querying) lives in `novelgraph.chat`; this script is just the terminal
loop around it, so the same logic is reused by the Streamlit app.

Requires that ingestion has already run at least once (see
`scripts/main.py`), so there is a graph to query.

Run with:
    uv run scripts/chat_cli.py
"""

import asyncio
from collections import deque

from dotenv import load_dotenv

from novelgraph.chat import MAX_HISTORY_TURNS, _compact_text, ask

load_dotenv()


async def chat() -> None:
    print("Chat with your knowledge graph. Type 'exit' or 'quit' to stop.\n")
    history: deque = deque(maxlen=MAX_HISTORY_TURNS)

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit"):
            break

        answer = await ask(query, history)
        print(f"\nCognee: {answer}\n")
        history.append((query, _compact_text(answer)))


if __name__ == "__main__":
    asyncio.run(chat())
