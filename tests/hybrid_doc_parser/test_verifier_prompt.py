"""Tests for verifier.py input-preparation helpers (Task Group 3).

Covers page serialization (no bbox), the versioned per-element-verdict prompt,
and full-page rendering under ``_PDFIUM_LOCK`` via ``render.render_page``.
Golden prompt-string assertions are intentionally avoided (too brittle).
"""
from __future__ import annotations

import unittest.mock as mock

from hybrid_doc_parser import verifier
from hybrid_doc_parser.models import ElementRecord, ElementType, VerifierConfig
from hybrid_doc_parser.verifier import (
    PROMPT_VERSION,
    _render_full_page,
    _serialize_page_elements,
    build_verifier_prompt,
)


def _element(element_id: str, page_idx: int, etype: ElementType, text: str):
    return ElementRecord(
        element_id=element_id,
        type=etype,
        text=text,
        bbox=[10.0, 20.0, 30.0, 40.0],
        page_idx=page_idx,
    )


def test_serialize_emits_id_index_type_text_and_no_bbox():
    elements = [
        _element("p0-e0", 0, ElementType.heading, "Title"),
        _element("p0-e1", 0, ElementType.text, "Body paragraph"),
        _element("p1-e0", 1, ElementType.text, "Other page"),
    ]

    out = _serialize_page_elements(elements, page_idx=0)

    # Only this page's elements, with id + ordinal index + type + text.
    assert "p0-e0" in out
    assert "p0-e1" in out
    assert "p1-e0" not in out
    assert "[0]" in out and "[1]" in out
    assert "heading" in out and "Title" in out
    # No bbox coordinates anywhere in the serialization.
    for coord in ("10.0", "20.0", "30.0", "40.0"):
        assert coord not in out


def test_serialize_empty_page_marker():
    out = _serialize_page_elements([], page_idx=5)
    assert out == "(no elements)"


def test_prompt_requests_per_element_verdict_with_channels():
    elements = [_element("p3-e7", 3, ElementType.table, "cell")]

    prompt = build_verifier_prompt(elements, page_idx=3)

    lowered = prompt.lower()
    assert "per-element" in lowered or "per element" in lowered
    # NOT a "list only disagreements" prompt.
    assert "list only disagreements" in lowered
    assert "do not list only disagreements" in lowered
    # Missing/extra channels present.
    assert "missing_elements" in prompt
    assert "extra_elements" in prompt
    # References by element_id and the serialized element appears.
    assert "element_id" in prompt
    assert "p3-e7" in prompt


def test_prompt_carries_stable_prompt_version():
    prompt = build_verifier_prompt([], page_idx=0)
    assert PROMPT_VERSION == "v1"
    assert f"prompt_version: {PROMPT_VERSION}" in prompt
    # page_idx marker so fakes/caches can key on it.
    assert "page_idx: 0" in prompt


def test_render_full_page_uses_render_page_under_lock():
    config = VerifierConfig(backend="fake", render_dpi=200)

    with (
        mock.patch.object(
            verifier.render, "render_page", return_value=b"PNGDATA"
        ) as mock_render,
        mock.patch.object(
            verifier.render, "render_region"
        ) as mock_region,
        mock.patch.object(verifier, "_PDFIUM_LOCK") as mock_lock,
    ):
        result = _render_full_page("/tmp/doc.pdf", page_idx=4, config=config)

    assert result == b"PNGDATA"
    # The lock context manager was acquired around the render call.
    mock_lock.__enter__.assert_called_once()
    mock_lock.__exit__.assert_called_once()
    # render_page called with the configured dpi; render_region NEVER used.
    mock_render.assert_called_once_with("/tmp/doc.pdf", 4, dpi=200)
    mock_region.assert_not_called()
