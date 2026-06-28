"""Prefix-stability analysis for agent prompt sequences.

The economic premise (see README): DeepSeek/Anthropic bill a *byte-identical* prompt
prefix at a steep discount on a cache hit, and recompute everything from the first
differing byte onward. Agent loops are structurally cache-friendly (long context,
short append), so a healthy loop's turn N+1 should share almost its entire prefix
with turn N. When it doesn't, some context mutation -- a reordered tool block, an
injected timestamp, a rebuilt system prompt -- silently pushed tokens out of cache.

This module is pure and dependency-free: give it the sequence of prompts your harness
actually sent (each turn's full prompt as bytes/str), and it tells you, per turn, how
much of the prefix was reusable, where it first diverged from the previous turn, and a
preview of what changed there. `gate()` turns that into a pass/fail for CI.

It is deliberately harness- and provider-agnostic: it analyzes bytes, not a particular
framework's objects, so it works on a JSONL log from CodeWhale, LangChain, or anything
that can dump the prompts it sent.
"""

from __future__ import annotations

from dataclasses import dataclass


def longest_common_prefix_len(a: bytes, b: bytes) -> int:
    """Number of leading bytes `a` and `b` share.

    This is exactly the boundary a byte-prefix KV cache cares about: bytes
    `[0, lcp)` can be reused on a cache hit; `[lcp, len)` must be recomputed.
    """
    limit = min(len(a), len(b))
    i = 0
    while i < limit and a[i] == b[i]:
        i += 1
    return i


def _preview(prev: bytes, cur: bytes, offset: int, window: int = 48) -> str:
    """Human-readable snippet of where `cur` first diverges from `prev`.

    Shows a little context before the divergence (the last bytes that *did* match)
    plus the first diverging bytes on each side, so a reader can recognize the
    culprit (e.g. a reordered tool name or an injected timestamp).
    """
    start = max(0, offset - window)
    matched = prev[start:offset].decode("utf-8", "replace")
    prev_tail = prev[offset : offset + window].decode("utf-8", "replace")
    cur_tail = cur[offset : offset + window].decode("utf-8", "replace")
    return (
        f"...{matched!r} then\n"
        f"  prev: {prev_tail!r}\n"
        f"  cur : {cur_tail!r}"
    )


@dataclass(frozen=True)
class TurnStability:
    """Prefix reuse for one turn relative to the previous turn."""

    index: int
    prompt_bytes: int
    cached_bytes: int  # bytes reusable from the previous turn's prefix (the LCP)
    # offset where this turn first diverged from the previous one; None for turn 0,
    # and equal to the previous turn's length for a clean append-only growth.
    first_divergence_offset: int | None
    divergence_preview: str | None

    @property
    def reused_ratio(self) -> float:
        """Fraction of this turn's prompt that a byte-prefix cache could reuse."""
        if self.prompt_bytes == 0:
            return 1.0
        return self.cached_bytes / self.prompt_bytes

    @property
    def recomputed_bytes(self) -> int:
        return self.prompt_bytes - self.cached_bytes

    @property
    def is_clean_append(self) -> bool:
        """True if this turn only *appended* to the previous prompt (ideal case):
        the entire previous prefix was preserved, nothing earlier mutated."""
        return self.first_divergence_offset == self._prev_len

    # set by the analyzer so `is_clean_append` can compare against the prev length
    _prev_len: int = -1


@dataclass(frozen=True)
class CacheLocalityReport:
    """Aggregate prefix-stability across a whole turn sequence."""

    turns: list[TurnStability]

    @property
    def total_prompt_bytes(self) -> int:
        return sum(t.prompt_bytes for t in self.turns)

    @property
    def total_cached_bytes(self) -> int:
        return sum(t.cached_bytes for t in self.turns)

    @property
    def total_recomputed_bytes(self) -> int:
        return sum(t.recomputed_bytes for t in self.turns)

    @property
    def mean_reused_ratio(self) -> float:
        """Mean per-turn reuse over turns 1..N (turn 0 has no prior prefix to reuse)."""
        scored = [t.reused_ratio for t in self.turns if t.index > 0]
        if not scored:
            return 0.0
        return sum(scored) / len(scored)

    @property
    def worst_turn(self) -> TurnStability | None:
        """The turn that reused the least of its prefix (the prime cache-bust suspect)."""
        scored = [t for t in self.turns if t.index > 0]
        if not scored:
            return None
        return min(scored, key=lambda t: t.reused_ratio)


def analyze_prefix_stability(prompts: list[str | bytes]) -> CacheLocalityReport:
    """Analyze a sequence of full per-turn prompts for prefix (cache) stability.

    `prompts[i]` is the entire prompt your harness sent on turn i (system + messages
    + tools, already serialized the way it goes on the wire). Returns a per-turn +
    aggregate report. Turn 0 is the baseline (nothing to reuse yet).
    """
    blobs = [p.encode("utf-8") if isinstance(p, str) else p for p in prompts]
    turns: list[TurnStability] = []
    for i, cur in enumerate(blobs):
        if i == 0:
            turns.append(
                TurnStability(
                    index=0,
                    prompt_bytes=len(cur),
                    cached_bytes=0,
                    first_divergence_offset=None,
                    divergence_preview=None,
                    _prev_len=-1,
                )
            )
            continue
        prev = blobs[i - 1]
        lcp = longest_common_prefix_len(prev, cur)
        # A clean append keeps the whole previous prompt as prefix (lcp == len(prev)).
        # Anything less means the prefix mutated before that point -> cache bust.
        preview = None if lcp == len(prev) else _preview(prev, cur, lcp)
        turns.append(
            TurnStability(
                index=i,
                prompt_bytes=len(cur),
                cached_bytes=lcp,
                first_divergence_offset=lcp,
                divergence_preview=preview,
                _prev_len=len(prev),
            )
        )
    return CacheLocalityReport(turns=turns)


@dataclass(frozen=True)
class GateVerdict:
    ok: bool
    reason: str


def gate(
    report: CacheLocalityReport,
    *,
    min_mean_reuse: float = 0.70,
    require_clean_append: bool = False,
) -> GateVerdict:
    """Turn a report into a CI pass/fail.

    `min_mean_reuse` mirrors the "healthy agent workload hits 70%+ after warmup"
    rule of thumb. `require_clean_append=True` is the strict mode: every turn must
    only append (any earlier-prefix mutation fails), which is what a cache-maximal
    harness should guarantee.
    """
    if require_clean_append:
        mutated = [t for t in report.turns if t.index > 0 and not t.is_clean_append]
        if mutated:
            first = mutated[0]
            return GateVerdict(
                ok=False,
                reason=(
                    f"prefix mutated on {len(mutated)} turn(s); first at turn "
                    f"{first.index} (offset {first.first_divergence_offset}):\n"
                    f"{first.divergence_preview}"
                ),
            )
    mean = report.mean_reused_ratio
    if mean < min_mean_reuse:
        worst = report.worst_turn
        worst_msg = (
            f" worst: turn {worst.index} reused {worst.reused_ratio:.0%}"
            if worst is not None
            else ""
        )
        return GateVerdict(
            ok=False,
            reason=(
                f"mean prefix reuse {mean:.0%} < required {min_mean_reuse:.0%}.{worst_msg}"
            ),
        )
    return GateVerdict(ok=True, reason=f"mean prefix reuse {mean:.0%} >= {min_mean_reuse:.0%}")
