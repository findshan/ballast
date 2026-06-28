"""ballast -- a cache-locality measurement + regression gate for agent harnesses.

P1 surface: prefix-stability analysis. Give it the prompts your harness sent and it
reports how much of each turn's prefix a byte-prefix KV cache could reuse, where it
diverged, and a CI pass/fail gate.
"""

from .prefix import (
    CacheLocalityReport,
    GateVerdict,
    TurnStability,
    analyze_prefix_stability,
    gate,
    longest_common_prefix_len,
)

__all__ = [
    "CacheLocalityReport",
    "GateVerdict",
    "TurnStability",
    "analyze_prefix_stability",
    "gate",
    "longest_common_prefix_len",
]

__version__ = "0.0.1"
