from __future__ import annotations

import argparse
import sys

from lxp_agent.agent import DEFAULT_MODEL, LXPAgent, MissingGroqApiKeyError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lxp-agent",
        description="Run an AI agent powered by L.X.P. and Groq.",
    )
    parser.add_argument("prompt", help="Question or task for the agent.")
    parser.add_argument(
        "--scan",
        action="append",
        default=[],
        help="Directory or file to map into Zero-Hop Context. Can be repeated.",
    )
    parser.add_argument(
        "--url",
        action="append",
        default=[],
        help="HTTP(S) resource to register in the Hyper-Schema. Can be repeated.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Groq model identifier.")
    parser.add_argument("--max-tokens", type=int, default=900)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    agent = LXPAgent(model=args.model)
    try:
        answer = agent.ask(
            args.prompt,
            scan_paths=args.scan,
            urls=args.url,
            max_tokens=args.max_tokens,
        )
    except MissingGroqApiKeyError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(answer.content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
