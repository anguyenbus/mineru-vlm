"""Tests for quality_gate.py two-layer quality gate."""

from __future__ import annotations

import unittest.mock as mock

import pytest

from hybrid_doc_parser.models import ElementRecord, ElementType


def make_element(text: str, etype: ElementType = ElementType.text) -> ElementRecord:
    """Construct a minimal ElementRecord for testing.

    Args:
        text: The text content of the element.
        etype: The element type; defaults to ElementType.text.

    Returns:
        A minimal ElementRecord with fixed test values.
    """
    return ElementRecord(
        element_id="test-id",
        type=etype,
        text=text,
        bbox=[0.0, 0.0, 100.0, 100.0],
        page_idx=0,
    )


def test_measure_text_quality_clean():
    """Test that clean English text passes all quality signals."""
    from hybrid_doc_parser.quality_gate import _measure_text_quality

    signals = _measure_text_quality("The quick brown fox jumps over the lazy dog")
    assert signals.passes


def test_measure_text_quality_garbled():
    """Test that garbled mixed alpha+digit tokens fail garbled_token_ratio."""
    from hybrid_doc_parser.quality_gate import _measure_text_quality

    # Garbled: mixed alpha+digit tokens
    garbled = " ".join(f"a{i}" for i in range(30))
    signals = _measure_text_quality(garbled)
    assert not signals.passes
    assert "garbled_token_ratio" in signals.failing_signals


def test_evaluate_page_layer1_low_coverage():
    """Test Layer 1 promotes to VLM when extracted token coverage is too low."""
    from hybrid_doc_parser.quality_gate import evaluate_page

    # 10 tokens extracted vs 100 in PDF text layer → coverage = 0.10 < 0.30
    element = make_element("word " * 10)
    decision = evaluate_page(0, [element], pdf_text_layer_tokens=100)
    assert decision.action == "promote_to_vlm"
    assert decision.layer == 1


def test_evaluate_page_layer2_repeated_chars():
    """Test Layer 2 promotes to VLM when text contains a long repeated char run."""
    from hybrid_doc_parser.quality_gate import evaluate_page

    # Repeated char run: "aaaaaaa" → fails MAX_REPEATED_CHAR_RUN=6
    element = make_element("aaaaaaaaaa " * 5)
    # Skip layer 1 by passing None
    decision = evaluate_page(0, [element], pdf_text_layer_tokens=None)
    assert decision.action == "promote_to_vlm"
    assert decision.layer == 2


def test_evaluate_page_empty_elements_returns_keep():
    """Test that an empty element list returns a keep decision."""
    from hybrid_doc_parser.quality_gate import evaluate_page

    decision = evaluate_page(0, [], pdf_text_layer_tokens=None)
    assert decision.action == "keep"


def test_evaluate_page_never_raises_on_internal_error():
    """Test that evaluate_page returns a keep decision with gate_error reason on internal failure."""
    from hybrid_doc_parser.quality_gate import evaluate_page

    with mock.patch(
        "hybrid_doc_parser.quality_gate._measure_text_quality",
        side_effect=RuntimeError("boom"),
    ):
        element = make_element("hello world this is a test")
        decision = evaluate_page(0, [element], pdf_text_layer_tokens=None)
        assert decision.action == "keep"
        assert "gate_error" in (decision.reason or "")


@pytest.mark.parametrize(
    "text,signal_name",
    [
        ("\x00" * 100, "ascii_printable_ratio"),  # non-printable chars → low ASCII ratio
        ("a " * 100, "mean_word_length"),  # single-char words → mean word length too short
    ],
)
def test_evaluate_page_layer2_parametrized(text: str, signal_name: str):
    """Test that individual Layer 2 heuristic breaches trigger promote_to_vlm."""
    from hybrid_doc_parser.quality_gate import evaluate_page

    element = make_element(text)
    decision = evaluate_page(0, [element], pdf_text_layer_tokens=None)
    assert decision.action == "promote_to_vlm"
    assert decision.layer == 2


def test_is_content_token():
    """Test _is_content_token classification for various token types."""
    from hybrid_doc_parser.quality_gate import _is_content_token

    assert _is_content_token("hello") is True
    assert _is_content_token("42") is True
    assert _is_content_token("$1,234.00") is True  # stripped digits → isdigit
    assert _is_content_token("a1b") is False  # mixed alpha+digit
