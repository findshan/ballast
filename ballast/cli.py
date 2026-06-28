"""Command-line entry point: point ballast at a prompt log, get a gate verdict.

Input is a JSONL file where each line is the full prompt your harness sent on one
turn, in order. A line may be either:

  - a bare JSON string:           "system...\\nuser: hi"
  - an object with a prompt field: {"prompt": "system...", "turn": 3}

The latter lets you dump a richer log and tell ballast which field holds the
on-the-wire prompt via ``--field`` (default: ``prompt``).

Exit code is 0 when the gate passes and 1 when it fails, so this drops straight
into CI as a regression check.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from .prefix import analyze_prefix_stability, gate
from .scorecard import render


def load_prompts(text: str, field: str = "prompt") -> list[str]:
    """Parse JSONL text into an ordered list of per-turn prompt strings."""
    prompts: list[str] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"line {lineno}: invalid JSON ({e})") from e
        if isinstance(obj, str):
            prompts.append(obj)
        elif isinstance(obj, dict):
            if field not in obj:
                raise ValueError(f"line {lineno}: object has no '{field}' field")
            prompts.append(str(obj[field]))
        else:
            raise ValueError(f"line {lineno}: expected string or object, got {type(obj).__name__}")
    return prompts


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ballast",
        description="Measure prompt prefix-cache locality and gate on regressions.",
    )
    parser.add_argument("log", help="Path to a JSONL prompt log (one turn per line), or - for stdin")
    parser.add_argument("--field", default="prompt", help="Object field holding the prompt (default: prompt)")
    parser.add_argument(
        "--min-reuse", type=float, default=0.70,
        help="Minimum mean prefix reuse over post-baseline turns (default: 0.70)",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Require every turn to be append-only (any earlier-prefix mutation fails)",
    )
    args = parser.parse_args(argv)

    try:
        text = sys.stdin.read() if args.log == "-" else open(args.log, encoding="utf-8").read()
        prompts = load_prompts(text, field=args.field)
    except (OSError, ValueError) as e:
        print(f"ballast: {e}", file=sys.stderr)
        return 2

    if not prompts:
        print("ballast: no prompts found in log", file=sys.stderr)
        return 2

    report = analyze_prefix_stability(prompts)
    verdict = gate(report, min_mean_reuse=args.min_reuse, require_clean_append=args.strict)
    print(render(report, verdict))
    return 0 if verdict.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
