"""Tests for ballast.cli (JSONL loader, exit codes) and scorecard rendering."""

import json

import pytest

from ballast.cli import load_prompts, main
from ballast.prefix import analyze_prefix_stability, gate
from ballast.scorecard import format_scorecard, render

# --- load_prompts ------------------------------------------------------------


def test_load_bare_strings():
    text = "\n".join(json.dumps(s) for s in ["abc", "abcdef"])
    assert load_prompts(text) == ["abc", "abcdef"]


def test_load_objects_with_field():
    text = "\n".join(json.dumps({"prompt": s, "turn": i}) for i, s in enumerate(["a", "ab"]))
    assert load_prompts(text) == ["a", "ab"]


def test_load_custom_field():
    text = json.dumps({"wire": "hello"})
    assert load_prompts(text, field="wire") == ["hello"]


def test_load_skips_blank_lines():
    text = '"a"\n\n  \n"ab"\n'
    assert load_prompts(text) == ["a", "ab"]


def test_load_missing_field_errors():
    with pytest.raises(ValueError, match="no 'prompt' field"):
        load_prompts(json.dumps({"other": "x"}))


def test_load_invalid_json_errors():
    with pytest.raises(ValueError, match="invalid JSON"):
        load_prompts("{not json}")


def test_load_wrong_type_errors():
    with pytest.raises(ValueError, match="expected string or object"):
        load_prompts("[1, 2, 3]")


# --- main / exit codes -------------------------------------------------------


def _write(tmp_path, prompts):
    p = tmp_path / "log.jsonl"
    p.write_text("\n".join(json.dumps(s) for s in prompts), encoding="utf-8")
    return str(p)


def test_main_pass_exit_zero(tmp_path, capsys):
    log = _write(tmp_path, ["a" * 100, "a" * 100 + "b" * 3, "a" * 100 + "b" * 6])
    rc = main([log, "--min-reuse", "0.70"])
    assert rc == 0
    assert "gate: PASS" in capsys.readouterr().out


def test_main_fail_exit_one(tmp_path, capsys):
    log = _write(tmp_path, ["xabc", "yabc", "zabc"])  # rewrites from byte 0
    rc = main([log, "--min-reuse", "0.70"])
    assert rc == 1
    assert "gate: FAIL" in capsys.readouterr().out


def test_main_strict_flag(tmp_path, capsys):
    log = _write(tmp_path, ["abcdef", "abXdef"])  # mutated, not append
    rc = main([log, "--strict"])
    assert rc == 1
    assert "MUTATED" in capsys.readouterr().out


def test_main_missing_file_exit_two(tmp_path, capsys):
    rc = main([str(tmp_path / "nope.jsonl")])
    assert rc == 2
    assert "ballast:" in capsys.readouterr().err


def test_main_empty_log_exit_two(tmp_path, capsys):
    log = tmp_path / "empty.jsonl"
    log.write_text("\n\n", encoding="utf-8")
    rc = main([str(log)])
    assert rc == 2
    assert "no prompts" in capsys.readouterr().err


# --- scorecard rendering -----------------------------------------------------


def test_scorecard_marks_baseline_append_and_mutation():
    report = analyze_prefix_stability(["abcdef", "abcdefgh", "abXdefgh"])
    out = format_scorecard(report)
    assert "baseline" in out
    assert "append" in out
    assert "MUTATED" in out
    assert "mean reuse" in out


def test_render_includes_gate_line():
    report = analyze_prefix_stability(["abc", "abcdef"])
    verdict = gate(report, min_mean_reuse=0.0)
    out = render(report, verdict)
    assert "gate: PASS" in out
