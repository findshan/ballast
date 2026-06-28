"""Tests for ballast.prefix — prefix-stability analysis + CI gate."""

from ballast.prefix import (
    CacheLocalityReport,
    analyze_prefix_stability,
    gate,
    longest_common_prefix_len,
)

# --- longest_common_prefix_len ----------------------------------------------


def test_lcp_identical():
    assert longest_common_prefix_len(b"abcdef", b"abcdef") == 6


def test_lcp_partial():
    assert longest_common_prefix_len(b"abcdef", b"abXdef") == 2


def test_lcp_append_is_full_prev():
    # cur extends prev: the whole of the shorter is the shared prefix
    assert longest_common_prefix_len(b"abc", b"abcdef") == 3


def test_lcp_empty():
    assert longest_common_prefix_len(b"", b"abc") == 0
    assert longest_common_prefix_len(b"abc", b"") == 0


# --- analyze_prefix_stability -----------------------------------------------


def test_turn_zero_is_baseline():
    report = analyze_prefix_stability(["hello"])
    t0 = report.turns[0]
    assert t0.index == 0
    assert t0.cached_bytes == 0
    assert t0.first_divergence_offset is None
    assert t0.divergence_preview is None
    # turn 0 has nothing to reuse, but reused_ratio is defined; mean ignores it
    assert report.mean_reused_ratio == 0.0


def test_clean_append_reuses_whole_previous_prefix():
    report = analyze_prefix_stability(["abc", "abcdef"])
    t1 = report.turns[1]
    assert t1.cached_bytes == 3  # whole previous prompt reused
    assert t1.is_clean_append is True
    assert t1.divergence_preview is None  # nothing mutated -> no preview
    assert t1.reused_ratio == 3 / 6
    assert t1.recomputed_bytes == 3


def test_mutation_breaks_prefix_and_sets_preview():
    report = analyze_prefix_stability(["abcdef", "abXdef"])
    t1 = report.turns[1]
    assert t1.cached_bytes == 2
    assert t1.first_divergence_offset == 2
    assert t1.is_clean_append is False
    assert t1.divergence_preview is not None
    assert "prev" in t1.divergence_preview and "cur" in t1.divergence_preview


def test_str_and_bytes_inputs_equivalent():
    a = analyze_prefix_stability(["abc", "abcd"])
    b = analyze_prefix_stability([b"abc", b"abcd"])
    assert a.turns[1].cached_bytes == b.turns[1].cached_bytes == 3


def test_aggregates_and_worst_turn():
    # turn 1: clean append (high reuse); turn 2: early mutation (low reuse)
    report = analyze_prefix_stability(["aaaa", "aaaabbbb", "aXaabbbbcccc"])
    assert report.total_prompt_bytes == 4 + 8 + 12
    assert report.total_cached_bytes == 0 + 4 + 1  # t0=0, t1=lcp(aaaa)=4, t2=lcp=1
    assert report.total_recomputed_bytes == report.total_prompt_bytes - report.total_cached_bytes
    worst = report.worst_turn
    assert worst is not None and worst.index == 2  # the mutated turn is worst


def test_mean_reuse_only_scores_post_baseline_turns():
    report = analyze_prefix_stability(["abc", "abcdef"])  # only turn 1 scored
    assert report.mean_reused_ratio == report.turns[1].reused_ratio


def test_worst_turn_none_for_single_turn():
    report = analyze_prefix_stability(["abc"])
    assert report.worst_turn is None


# --- gate -------------------------------------------------------------------


def _report_with_mean(prompts):
    return analyze_prefix_stability(prompts)


def test_gate_passes_on_high_reuse():
    # near-clean appends -> high mean reuse
    report = _report_with_mean(["a" * 100, "a" * 100 + "b" * 5, "a" * 100 + "b" * 10])
    verdict = gate(report, min_mean_reuse=0.70)
    assert verdict.ok is True
    assert ">=" in verdict.reason


def test_gate_fails_below_threshold():
    # each turn rewrites from the start -> near-zero reuse
    report = _report_with_mean(["xabc", "yabc", "zabc"])
    verdict = gate(report, min_mean_reuse=0.70)
    assert verdict.ok is False
    assert "<" in verdict.reason
    assert "worst" in verdict.reason


def test_gate_strict_clean_append_catches_mutation():
    report = _report_with_mean(["abcdef", "abXdef"])  # mutated, not append
    verdict = gate(report, require_clean_append=True)
    assert verdict.ok is False
    assert "mutated" in verdict.reason
    assert "turn 1" in verdict.reason


def test_gate_strict_passes_on_pure_appends():
    report = _report_with_mean(["abc", "abcde", "abcdefg"])  # all clean appends
    verdict = gate(report, require_clean_append=True, min_mean_reuse=0.0)
    assert verdict.ok is True


def test_gate_empty_report_does_not_crash():
    report = CacheLocalityReport(turns=[])
    verdict = gate(report, min_mean_reuse=0.70)
    # no scored turns -> mean 0.0 -> fails the default threshold cleanly
    assert verdict.ok is False
