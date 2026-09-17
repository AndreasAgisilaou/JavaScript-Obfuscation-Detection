"""Tests for the pure feature-extraction helpers in
detection/naural_network.py. These are deterministic, no-I/O functions, so a
handful of known input/output cases is enough to catch a broken heuristic
(e.g. a mis-typed kwarg or a flipped comparison) before it silently degrades
a trained model.
"""
import base64

import pytest

from detection import naural_network as nn


def test_calculate_entropy_of_uniform_two_symbol_text():
    assert nn.calculate_entropy("aabb") == pytest.approx(1.0)


def test_calculate_entropy_of_single_repeated_char_is_zero():
    assert nn.calculate_entropy("aaaa") == pytest.approx(0.0)


def test_calculate_comment_density():
    text = "// a comment\ncode();\n/* another */\nmore_code();"
    assert nn.calculate_comment_density(text) == pytest.approx(0.5)


def test_calculate_comment_density_empty_text():
    assert nn.calculate_comment_density("") == 0


def test_calculate_avg_line_length():
    assert nn.calculate_avg_line_length("ab\nabcd") == pytest.approx(3.0)


def test_calculate_longest_line():
    assert nn.calculate_longest_line("ab\nabcdef\nabc") == 6


def test_keyword_frequency():
    text = "var x = 1; function foo() { return x; }"
    assert nn.keyword_frequency(text) == pytest.approx(3 / 7)


def test_keyword_frequency_no_words():
    assert nn.keyword_frequency("") == 0


def test_variable_uniqueness_all_unique():
    assert nn.variable_uniqueness("var a; var b; var c;") == pytest.approx(1.0)


def test_variable_uniqueness_with_repeats():
    assert nn.variable_uniqueness("var a; a = 1; var b;") == pytest.approx(2 / 3)


def test_detect_token_frequency():
    text = "eval(x); setTimeout(y, 1); eval(z);"
    assert nn.detect_token_frequency(text) == 3


def test_calculate_numeric_density():
    assert nn.calculate_numeric_density("a 1 b 22 c") == pytest.approx(2 / 5)


def test_calculate_numeric_density_no_tokens():
    assert nn.calculate_numeric_density("") == 0


def test_calculate_english_word_proportion():
    # Only "the" is in the module's small hand-picked word set, not "quick"
    # or "xyzzy" - this differs from features.py's nltk-backed word list.
    assert nn.calculate_english_word_proportion("the quick xyzzy") == pytest.approx(1 / 3)


def test_calculate_compression_ratio_empty_text():
    assert nn.calculate_compression_ratio("") == 0


def test_calculate_compression_ratio_repetitive_text_compresses_well():
    ratio = nn.calculate_compression_ratio("a" * 1000)
    assert 0 < ratio < 0.1


def test_calculate_hex_literal_density():
    assert nn.calculate_hex_literal_density("0xFF 0x1A") == pytest.approx(2 / 9)


def test_calculate_hex_literal_density_empty_text():
    assert nn.calculate_hex_literal_density("") == 0


def test_detect_encoding_patterns_counts_hex_and_unicode_escapes():
    text = "\\x41\\x42 \\u0041\\u0042"
    assert nn.detect_encoding_patterns(text) == 4


def test_calculate_function_call_density():
    assert nn.calculate_function_call_density("foo(); bar(1, 2); baz") == pytest.approx(2 / 5)


def test_looks_like_base64_true_for_valid_base64_with_digit():
    candidate = base64.b64encode(b"0123456789obfuscated-payload-data").decode()
    assert nn._looks_like_base64(candidate) is True


def test_looks_like_base64_false_for_plain_word_without_digit_or_symbol():
    assert nn._looks_like_base64("HelloWorldThisIsJustEnglishText") is False


def test_looks_like_base64_false_for_short_invalid_base64():
    assert nn._looks_like_base64("12345") is False


def test_build_model_compiles_for_thirteen_features():
    model = nn.build_model(input_dim=len(nn.FEATURE_COLUMNS))
    assert model.input_shape == (None, len(nn.FEATURE_COLUMNS))
    assert model.output_shape == (None, 2)
