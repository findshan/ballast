"""Human-readable rendering of a CacheLocalityReport + gate verdict.

Kept separate from `prefix.py` (the pure analysis) so the analysis stays free of
any presentation concerns. This is what the CLI prints; it is plain text so it
reads fine in a terminal, a CI log, or a PR comment.
"""

from __future__ import annotations

from .prefix import CacheLocalityReport, GateVerdict


def _bar(ratio: float, width: int = 20) -> str:
    filled = max(0, min(width, round(ratio * width)))
    return "█" * filled + "·" * (width - filled)


def format_scorecard(report: CacheLocalityReport) -> str:
    """Render the per-turn + aggregate prefix-reuse table."""
    lines = ["turn   prompt_B   cached_B    reuse  note"]
    for t in report.turns:
        if t.index == 0:
            note = "baseline"
        elif t.is_clean_append:
            note = "append"
        else:
            note = f"MUTATED @{t.first_divergence_offset}"
        reuse = "  -  " if t.index == 0 else f"{t.reused_ratio:5.0%}"
        lines.append(
            f"{t.index:>4}   {t.prompt_bytes:>8}   {t.cached_bytes:>8}   {reuse}  {note}"
        )

    lines.append("")
    lines.append(
        f"mean reuse (turns>0): {report.mean_reused_ratio:.0%}   "
        f"{_bar(report.mean_reused_ratio)}"
    )
    lines.append(
        f"total: {report.total_prompt_bytes} prompt B, "
        f"{report.total_recomputed_bytes} recomputed B"
    )
    worst = report.worst_turn
    if worst is not None and worst.divergence_preview is not None:
        lines.append("")
        lines.append(f"worst turn {worst.index} ({worst.reused_ratio:.0%} reused):")
        lines.append(worst.divergence_preview)
    return "\n".join(lines)


def render(report: CacheLocalityReport, verdict: GateVerdict) -> str:
    """Full CLI output: the scorecard followed by the gate verdict line."""
    status = "PASS" if verdict.ok else "FAIL"
    return f"{format_scorecard(report)}\n\ngate: {status} — {verdict.reason}"
