"""Tests for the pure feature-extraction helpers in detection/features.py
(shared by the RandomForest and XGBoost pipelines). Deterministic, no-I/O
functions, so a handful of known input/output cases catches a broken
heuristic before it silently degrades a trained model.
"""
import pytest

from detection import features as f


def test_calculate_entropy_of_uniform_two_symbol_text():
    assert f.calculate_entropy("aabb") == pytest.approx(1.0)


def test_calculate_long_lines_share():
    text = "short\n" + ("x" * 1500) + "\nshort2"
    assert f.calculate_long_lines(text) == pytest.approx(1 / 3)


def test_calculate_long_lines_no_long_lines():
    assert f.calculate_long_lines("short\nlines\nonly") == 0


def test_calculate_avg_line_length():
    assert f.calculate_avg_line_length("ab\nabcd") == pytest.approx(3.0)


def test_eval_frequency():
    text = "eval(x); eval(y);"
    assert f.eval_frequency(text, len(text)) == pytest.approx(2 / len(text))


def test_eval_frequency_zero_chars():
    assert f.eval_frequency("", 0) == 0


def test_analyze_strings_counts_only_long_strings():
    text = "'" + ("a" * 60) + "' \"short\""
    assert f.analyze_strings(text) == 1


def test_keyword_frequency():
    text = "var x = 1; function foo() { return x; }"
    assert f.keyword_frequency(text) == pytest.approx(3 / 7)


def test_variable_uniqueness_with_repeats():
    assert f.variable_uniqueness("var a; a = 1; var b;") == pytest.approx(2 / 3)


def test_detect_token_frequency():
    text = "eval(x); setTimeout(y, 1); eval(z);"
    assert f.detect_token_frequency(text) == 3


def test_calculate_escape_sequence_density():
    text = "\\x41\\x42"
    assert f.calculate_escape_sequence_density(text) == pytest.approx(0.25)


def test_calculate_escape_sequence_density_empty_text():
    assert f.calculate_escape_sequence_density("") == 0


def test_calculate_english_word_proportion():
    # features.py backs this with the full nltk word corpus, so "quick" also
    # counts here - unlike naural_network.py's small hand-picked word set.
    assert f.calculate_english_word_proportion("the quick xyzzy") == pytest.approx(2 / 3)


def test_calculate_compression_ratio_empty_text():
    assert f.calculate_compression_ratio("") == 0


def test_calculate_compression_ratio_repetitive_text_compresses_well():
    ratio = f.calculate_compression_ratio("a" * 1000)
    assert 0 < ratio < 0.1


def test_control_flow_complexity():
    text = "if (a) { for (;;) { while(b) {} } } else {}"
    assert f.control_flow_complexity(text) == 4


def test_is_only_code_blocks_true_for_plain_json():
    assert f.is_only_code_blocks('{"a": 1, "b": 2}') is True


def test_is_only_code_blocks_false_for_executable_code():
    assert f.is_only_code_blocks('eval("x")') is False
