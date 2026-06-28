# ballast

**A cache-locality measurement + regression gate for agent harnesses.**

Modern LLM APIs (Anthropic, DeepSeek, OpenAI) bill a *byte-identical* prompt prefix at a steep discount on a cache hit, and recompute everything from the first differing byte onward. Agent loops are structurally cache-friendly — long context, short append per turn — so a healthy loop's turn _N+1_ should share almost its entire prefix with turn _N_.

In practice harnesses quietly throw this away. The usual culprits, in rough order of how often they bite:

1. **Context compaction** — when the thread grows past a budget, the older messages get rewritten into a summary. That replaces the bytes right after the system prompt, so the next turn is a near-total cache miss. Caching and compaction are in direct tension, and most harnesses compact without measuring the cost.
2. **Non-deterministic tool / MCP ordering** — tool definitions sit *ahead* of the messages, so iterating a map in random order, or an MCP server reconnecting and re-emitting its tools, reshuffles the prefix and busts everything downstream.
3. **Volatile content in the system prompt** — an injected timestamp, session id, or per-turn memory block placed before the static content makes everything after it uncacheable on every single call.

Any of these shifts the prefix, and the cost shows up on the bill, not in any test. **ballast measures it and fails CI when it regresses.**

It is deliberately **harness- and provider-agnostic**: it analyzes the bytes you actually sent, not a particular framework's objects, so it works on a prompt log dumped from CodeWhale, nanobot, LangChain, or anything else.

## What the field says

This isn't hypothetical — prefix caching is now the dominant cost lever for agents, and keeping the prefix byte-stable is treated as a first-class engineering concern:

- **DeepSeek** was the first major API to cache context on disk, pricing a cache hit at **\$0.014 / M tokens — up to a 90% discount** — and is explicit about the bar: *"Only requests with identical prefixes (starting from the 0th token) will be considered duplicates."* That byte-for-byte, from-token-0 rule is exactly what ballast measures. ([DeepSeek API](https://api-docs.deepseek.com/news/news0802))
- **Anthropic's Claude Code team** titled their engineering lessons post *"Prompt caching is everything,"* and treat a cache-hit-rate drop as an incident: *"we run alerts on our prompt cache hit rate and declare SEVs if they're too low."* ballast is that alert — made portable and runnable in CI against any harness, not just your own. ([Anthropic](https://claude.com/blog/lessons-from-building-claude-code-prompt-caching-is-everything))
- Research has started measuring the same failure mode head-on: *"Don't Break the Cache: An Evaluation of Prompt Caching for Long-Horizon Agentic Tasks."* ([arXiv 2601.06007](https://arxiv.org/abs/2601.06007))

## Install

```bash
pip install -e .
```

## Use it as a CLI

Dump the full prompt your harness sent each turn, in order, to a JSONL file — one turn per line, either a bare JSON string or an object with a `prompt` field:

```jsonl
{"prompt": "SYSTEM: ...\nTOOLS: ...\n\nturn 1: ...", "turn": 1}
{"prompt": "SYSTEM: ...\nTOOLS: ...\n\nturn 1: ...\nturn 2: ...", "turn": 2}
```

```bash
ballast prompts.jsonl              # scorecard + gate verdict; exit 0 pass / 1 fail
ballast prompts.jsonl --strict     # require every turn to be append-only
ballast prompts.jsonl --min-reuse 0.85
cat prompts.jsonl | ballast -      # read from stdin
```

Example output on a real-shaped trace where the agent compacted its history into a summary at turn 5 (the most common prefix-buster):

```
turn   prompt_B   cached_B    reuse  note
   0         61          0     -    baseline
   1        126         61     48%  append
   2        191        126     66%  append
   3        256        191     75%  append
   4        321        256     80%  append
   5        153         62     41%  MUTATED @62     ← compaction rewrote the prefix
   6        218        153     70%  append
   7        283        218     77%  append

mean reuse (turns>0): 65%   █████████████·······
gate: FAIL — mean prefix reuse 65% < required 70%. worst: turn 5 reused 41%

worst turn 5 (41% reused):
  prev: 'U1: ...request...\nA1: ...reply...\nTOOL1: ...long'
  cur : '[SUMMARY of turns 1-4: user asked to refactor mo'
```

The divergence preview points at the exact bytes that broke the cache — here it shows the live conversation being replaced by a `[SUMMARY ...]` block right after the system prompt — so you can see *what* reset the prefix rather than guessing.

`ballast` returns a non-zero exit code on gate failure, so it drops straight into CI as a regression check on a recorded prompt trace.

Three runnable traces live in [`examples/`](examples/), each a different culprit: `clean_append.jsonl` (append-only growth → PASS), `compaction_bust.jsonl` (summary rewrite at turn 5 → FAIL), and `timestamp_drift.jsonl` (a per-turn timestamp in the system prefix → FAIL). Try `ballast examples/compaction_bust.jsonl`.

## Use it as a library

```python
from ballast import analyze_prefix_stability, gate

report = analyze_prefix_stability(prompts)   # prompts: list[str | bytes]
print(report.mean_reused_ratio, report.worst_turn)

verdict = gate(report, min_mean_reuse=0.70, require_clean_append=False)
assert verdict.ok, verdict.reason
```

- `analyze_prefix_stability(prompts)` → `CacheLocalityReport` with per-turn `TurnStability` (cached bytes = longest common prefix with the previous turn, divergence offset, preview) plus aggregates (`mean_reused_ratio`, `worst_turn`, total recomputed bytes).
- `gate(report, ...)` → `GateVerdict(ok, reason)`. `min_mean_reuse` mirrors the "healthy agent workload stays cache-resident" rule of thumb; `require_clean_append=True` is strict mode — any mutation of an earlier prefix fails.

## Why "ballast"

Ballast keeps a ship stable and low in the water. This keeps your prompt prefix stable and your token bill low.

## Development

```bash
pip install -e ".[dev]"
ruff check ballast tests
pytest -q
```

MIT licensed.
