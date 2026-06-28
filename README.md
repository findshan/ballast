# ballast

**A cache-locality measurement + regression gate for agent harnesses.**

Modern LLM APIs (Anthropic, DeepSeek, OpenAI) bill a *byte-identical* prompt prefix at a steep discount on a cache hit, and recompute everything from the first differing byte onward. Agent loops are structurally cache-friendly — long context, short append per turn — so a healthy loop's turn _N+1_ should share almost its entire prefix with turn _N_.

In practice harnesses quietly throw this away: a reordered tool block, an injected timestamp, a rebuilt system prompt, or a tokenizer drift between "agentic loop" and "session reload" shifts the prefix and busts the cache. The cost shows up on the bill, not in any test. **ballast measures it and fails CI when it regresses.**

It is deliberately **harness- and provider-agnostic**: it analyzes the bytes you actually sent, not a particular framework's objects, so it works on a prompt log dumped from CodeWhale, nanobot, LangChain, or anything else.

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

Example output on a log where a per-turn timestamp was injected into the system prefix:

```
turn   prompt_B   cached_B    reuse  note
   0        114          0     -    baseline
   1        126         60     48%  MUTATED @60
   2        138         60     43%  MUTATED @60
   ...
mean reuse (turns>0): 42%   ████████············
gate: FAIL — mean prefix reuse 42% < required 70%. worst: turn 4 reused 37%
```

The divergence preview points at the exact bytes that broke the cache, so you can find the culprit (here: `now=...14:04` vs `14:05` inside the system prompt) rather than guessing.

`ballast` returns a non-zero exit code on gate failure, so it drops straight into CI as a regression check on a recorded prompt trace.

Two runnable traces live in [`examples/`](examples/): `clean_append.jsonl` (append-only growth → PASS) and `timestamp_drift.jsonl` (a per-turn timestamp injected into the system prefix → FAIL). Try `ballast examples/timestamp_drift.jsonl`.

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
