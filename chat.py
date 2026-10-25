#!/usr/bin/env python3
"""Simple text chat for development and demonstrations."""

from __future__ import annotations

import argparse

from src.agent import ZentroMartAgent
from src.config import get_settings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Path to extracted dataset")
    parser.add_argument("--lexical-only", action="store_true")
    args = parser.parse_args()

    agent = ZentroMartAgent(
        get_settings(args.dataset), use_embeddings=not args.lexical_only
    )
    print("ZentroMart test chat. Type 'quit' to stop.\n")
    while True:
        message = input("Customer: ").strip()
        if message.casefold() in {"quit", "exit"}:
            break
        result = agent.respond(message)
        print(f"Agent: {result.reply}")
        print("Sources:", ", ".join(source["source_id"] for source in result.sources))
        if result.tool_calls:
            print("Tools:", ", ".join(call["name"] for call in result.tool_calls))
        print()


if __name__ == "__main__":
    main()

