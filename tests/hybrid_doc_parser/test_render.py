"""Tests for render.py rasterizer utilities."""
from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def test_clamp_scale_no_clamp():
    from hybrid_doc_parser.render import _clamp_scale_to_budget

    scale, clamped = _clamp_scale_to_budget(595.0, 842.0, 2.0, 40_000_000)
    assert not clamped
    assert scale == 2.0


def test_clamp_scale_with_clamp():
    from hybrid_doc_parser.render import _clamp_scale_to_budget

    scale, clamped = _clamp_scale_to_budget(595.0, 842.0, 10.0, 40_000_000)
    assert clamped
    assert scale < 10.0
    assert scale > 0


def test_clamp_scale_zero_width():
    from hybrid_doc_parser.render import _clamp_scale_to_budget

    # Must not raise
    scale, clamped = _clamp_scale_to_budget(0.0, 842.0, 2.0, 40_000_000)
    assert scale == 2.0
    assert not clamped


def test_clamp_scale_zero_max_pixels():
    """_clamp_scale_to_budget with max_pixels=0 passes through unchanged."""
    from hybrid_doc_parser.render import _clamp_scale_to_budget

    scale, clamped = _clamp_scale_to_budget(595.0, 842.0, 2.0, 0)
    assert scale == 2.0
    assert not clamped


def test_text_layer_tokens_digital():
    from hybrid_doc_parser.render import text_layer_tokens

    pdf = FIXTURES / "digital_simple.pdf"
    result = text_layer_tokens(pdf)
    assert isinstance(result, dict)
    # digital_simple.pdf has at least 1 page with some text
    assert len(result) >= 1
    assert all(isinstance(v, int) and v >= 0 for v in result.values())


def test_text_layer_tokens_scanned():
    from hybrid_doc_parser.render import text_layer_tokens

    pdf = FIXTURES / "scanned.pdf"
    result = text_layer_tokens(pdf)
    # Scanned PDF should have a dict (possibly all zeros)
    assert isinstance(result, dict)
    assert all(v >= 0 for v in result.values())


def test_text_layer_tokens_nonexistent():
    from hybrid_doc_parser.render import text_layer_tokens

    result = text_layer_tokens(Path("/nonexistent/file.pdf"))
    assert result == {}


def test_render_region_zero_width_raises():
    from hybrid_doc_parser.render import render_region

    pdf = FIXTURES / "digital_simple.pdf"
    with pytest.raises((ValueError, Exception)):
        render_region(pdf, 0, [0.0, 0.0, 0.0, 100.0])


def test_render_page_returns_png_bytes():
    """render_page() returns non-empty PNG bytes for a valid page."""
    from hybrid_doc_parser.render import render_page

    pdf = FIXTURES / "digital_simple.pdf"
    result = render_page(pdf, 0)
    # PNG files start with the PNG signature bytes
    assert isinstance(result, bytes)
    assert len(result) > 0
    assert result[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_page_clamped_dpi(monkeypatch):
    """render_page() applies scale clamping when DPI would exceed the MP budget."""
    from hybrid_doc_parser.render import render_page

    # NOTE: Set a tiny MP budget so that even a small page triggers clamping.
    monkeypatch.setenv("PARSER_MAX_RENDER_MP", "0.01")
    pdf = FIXTURES / "digital_simple.pdf"
    result = render_page(pdf, 0, dpi=1440)
    assert isinstance(result, bytes)
    assert len(result) > 0


def test_render_region_valid_bbox_returns_png():
    """render_region() with a valid in-page bbox returns PNG bytes."""
    from hybrid_doc_parser.render import render_region

    pdf = FIXTURES / "digital_simple.pdf"
    # A small region near the centre of a standard A4 page (595 x 842 pts)
    bbox = [100.0, 300.0, 400.0, 500.0]
    result = render_region(pdf, 0, bbox)
    assert isinstance(result, bytes)
    assert len(result) > 0
    assert result[:8] == b"\x89PNG\r\n\x1a\n"
