"""Tests for modal_processors.py image/table/equation enrichment."""
from __future__ import annotations

import unittest.mock as mock

import pytest

from hybrid_doc_parser.models import ElementType


def make_block(block_type: str, text: str = "", page_idx: int = 0) -> dict:
    return {
        "type": block_type,
        "text": text,
        "page_idx": page_idx,
        "img_path": None,
    }


def make_mock_vlm(description: str = "test description") -> mock.MagicMock:
    """Return a mock VLMClient whose call() returns {"description": description}."""
    vlm = mock.MagicMock()
    vlm.call.return_value = {"description": description}
    return vlm


def make_mock_config(context_window: int = 3, max_context_tokens: int = 512) -> mock.MagicMock:
    """Return a minimal mock EnrichmentConfig."""
    cfg = mock.MagicMock()
    cfg.context_window = context_window
    cfg.max_context_tokens = max_context_tokens
    return cfg


# ---------------------------------------------------------------------------
# process() tests (already existed)
# ---------------------------------------------------------------------------

def test_image_processor_returns_description():
    from hybrid_doc_parser.modal_processors import ImageModalProcessor
    block = make_block("image", text="")
    vlm = make_mock_vlm("A bar chart showing sales data")
    processor = ImageModalProcessor(vlm, content_list=[block], context_window=2)
    result = processor.process(block, block_index=0, image_bytes=None)
    assert result == "A bar chart showing sales data"
    assert vlm.call.called


def test_table_processor_returns_description():
    from hybrid_doc_parser.modal_processors import TableModalProcessor
    html = "<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>"
    block = make_block("table", text=html)
    vlm = make_mock_vlm("A two-column table with headers A and B")
    processor = TableModalProcessor(vlm, content_list=[block], context_window=2)
    result = processor.process(block, block_index=0, image_bytes=None)
    assert result == "A two-column table with headers A and B"


def test_equation_processor_returns_description():
    from hybrid_doc_parser.modal_processors import EquationModalProcessor
    block = make_block("interline_equation", text=r"E = mc^2")
    vlm = make_mock_vlm("Einstein's mass-energy equivalence formula")
    processor = EquationModalProcessor(vlm, content_list=[block], context_window=2)
    result = processor.process(block, block_index=0, image_bytes=None)
    assert result == "Einstein's mass-energy equivalence formula"


def test_processor_returns_empty_on_vlm_error():
    from hybrid_doc_parser.modal_processors import ImageModalProcessor
    block = make_block("image")
    vlm = mock.MagicMock()
    vlm.call.return_value = {"error": "timeout"}
    processor = ImageModalProcessor(vlm, content_list=[block], context_window=2)
    result = processor.process(block, block_index=0, image_bytes=None)
    assert result == ""


def test_processor_returns_empty_on_exception():
    from hybrid_doc_parser.modal_processors import ImageModalProcessor
    block = make_block("image")
    vlm = mock.MagicMock()
    vlm.call.side_effect = RuntimeError("VLM unavailable")
    processor = ImageModalProcessor(vlm, content_list=[block], context_window=2)
    result = processor.process(block, block_index=0, image_bytes=None)
    assert result == ""


def test_context_injected_in_prompt():
    from hybrid_doc_parser.modal_processors import ImageModalProcessor
    content_list = [
        {"type": "text", "text": "surrounding paragraph", "page_idx": 0},
        {"type": "image", "text": "", "page_idx": 0},
    ]
    vlm = make_mock_vlm("description with context")
    processor = ImageModalProcessor(vlm, content_list=content_list, context_window=3)
    block = content_list[1]
    result = processor.process(block, block_index=1, image_bytes=None)
    # Verify the VLM was called with a prompt that contains the surrounding context
    call_args = vlm.call.call_args
    prompt = call_args.args[1] if call_args.args else call_args.kwargs.get("prompt", "")
    assert "surrounding paragraph" in prompt


# ---------------------------------------------------------------------------
# enrich() tests — ImageModalProcessor
# ---------------------------------------------------------------------------

def test_image_enrich_returns_description():
    """enrich() on ImageModalProcessor produces a VLM description."""
    from hybrid_doc_parser.modal_processors import ImageModalProcessor
    block = make_block("image", text="")
    vlm = make_mock_vlm("enriched image description")
    config = make_mock_config()
    content_list = [block]
    processor = ImageModalProcessor(vlm, content_list=content_list, context_window=2)
    result = processor.enrich(block, block_index=0, content_list=content_list, vlm_client=vlm, config=config)
    assert result == "enriched image description"


def test_image_enrich_returns_empty_on_vlm_error():
    """enrich() returns '' when the VLM returns an error key."""
    from hybrid_doc_parser.modal_processors import ImageModalProcessor
    block = make_block("image")
    vlm = mock.MagicMock()
    vlm.call.return_value = {"error": "service unavailable"}
    config = make_mock_config()
    content_list = [block]
    processor = ImageModalProcessor(vlm, content_list=content_list, context_window=2)
    result = processor.enrich(block, block_index=0, content_list=content_list, vlm_client=vlm, config=config)
    assert result == ""


def test_image_enrich_returns_empty_on_exception():
    """enrich() returns '' when VLMClient.call raises."""
    from hybrid_doc_parser.modal_processors import ImageModalProcessor
    block = make_block("image")
    vlm = mock.MagicMock()
    vlm.call.side_effect = ConnectionError("network failure")
    config = make_mock_config()
    content_list = [block]
    processor = ImageModalProcessor(vlm, content_list=content_list, context_window=2)
    result = processor.enrich(block, block_index=0, content_list=content_list, vlm_client=vlm, config=config)
    assert result == ""


# ---------------------------------------------------------------------------
# enrich() tests — TableModalProcessor
# ---------------------------------------------------------------------------

def test_table_enrich_returns_description():
    """enrich() on TableModalProcessor uses HTML content in the prompt."""
    from hybrid_doc_parser.modal_processors import TableModalProcessor
    html = "<table><tr><th>X</th></tr><tr><td>1</td></tr></table>"
    block = {"type": "table", "text": "", "html": html, "page_idx": 0}
    vlm = make_mock_vlm("table about X values")
    config = make_mock_config()
    content_list = [block]
    processor = TableModalProcessor(vlm, content_list=content_list, context_window=2)
    result = processor.enrich(block, block_index=0, content_list=content_list, vlm_client=vlm, config=config)
    assert result == "table about X values"


def test_table_enrich_falls_back_to_text_when_no_html():
    """enrich() uses block text when no HTML field is present."""
    from hybrid_doc_parser.modal_processors import TableModalProcessor
    block = {"type": "table", "text": "plain table body", "page_idx": 0}
    vlm = make_mock_vlm("plain table description")
    config = make_mock_config()
    content_list = [block]
    processor = TableModalProcessor(vlm, content_list=content_list, context_window=2)
    result = processor.enrich(block, block_index=0, content_list=content_list, vlm_client=vlm, config=config)
    assert result == "plain table description"
    call_args = vlm.call.call_args
    prompt = call_args.args[1] if call_args.args else call_args.kwargs.get("prompt", "")
    assert "plain table body" in prompt


def test_table_enrich_returns_empty_on_vlm_error():
    """enrich() returns '' when the VLM returns an error."""
    from hybrid_doc_parser.modal_processors import TableModalProcessor
    block = make_block("table", text="some table")
    vlm = mock.MagicMock()
    vlm.call.return_value = {"error": "timeout"}
    config = make_mock_config()
    content_list = [block]
    processor = TableModalProcessor(vlm, content_list=content_list, context_window=2)
    result = processor.enrich(block, block_index=0, content_list=content_list, vlm_client=vlm, config=config)
    assert result == ""


def test_table_enrich_returns_empty_on_exception():
    """enrich() returns '' when VLMClient.call raises."""
    from hybrid_doc_parser.modal_processors import TableModalProcessor
    block = make_block("table")
    vlm = mock.MagicMock()
    vlm.call.side_effect = RuntimeError("boom")
    config = make_mock_config()
    content_list = [block]
    processor = TableModalProcessor(vlm, content_list=content_list, context_window=2)
    result = processor.enrich(block, block_index=0, content_list=content_list, vlm_client=vlm, config=config)
    assert result == ""


# ---------------------------------------------------------------------------
# enrich() tests — EquationModalProcessor
# ---------------------------------------------------------------------------

def test_equation_enrich_returns_description():
    """enrich() on EquationModalProcessor reads LaTeX from content field."""
    from hybrid_doc_parser.modal_processors import EquationModalProcessor
    block = {"type": "interline_equation", "content": r"\frac{a}{b}", "page_idx": 0}
    vlm = make_mock_vlm("fraction a over b")
    config = make_mock_config()
    content_list = [block]
    processor = EquationModalProcessor(vlm, content_list=content_list, context_window=2)
    result = processor.enrich(block, block_index=0, content_list=content_list, vlm_client=vlm, config=config)
    assert result == "fraction a over b"


def test_equation_enrich_returns_empty_on_vlm_error():
    """enrich() returns '' when the VLM returns an error key."""
    from hybrid_doc_parser.modal_processors import EquationModalProcessor
    block = make_block("interline_equation", text=r"\alpha + \beta")
    vlm = mock.MagicMock()
    vlm.call.return_value = {"error": "not_found"}
    config = make_mock_config()
    content_list = [block]
    processor = EquationModalProcessor(vlm, content_list=content_list, context_window=2)
    result = processor.enrich(block, block_index=0, content_list=content_list, vlm_client=vlm, config=config)
    assert result == ""


def test_equation_enrich_returns_empty_on_exception():
    """enrich() returns '' when VLMClient.call raises."""
    from hybrid_doc_parser.modal_processors import EquationModalProcessor
    block = make_block("interline_equation", text=r"x^2")
    vlm = mock.MagicMock()
    vlm.call.side_effect = ValueError("bad input")
    config = make_mock_config()
    content_list = [block]
    processor = EquationModalProcessor(vlm, content_list=content_list, context_window=2)
    result = processor.enrich(block, block_index=0, content_list=content_list, vlm_client=vlm, config=config)
    assert result == ""


# ---------------------------------------------------------------------------
# GenericModalProcessor tests
# ---------------------------------------------------------------------------

def test_generic_processor_returns_description():
    """GenericModalProcessor.process() returns a VLM description."""
    from hybrid_doc_parser.modal_processors import GenericModalProcessor
    block = make_block("unknown", text="some content")
    vlm = make_mock_vlm("generic element description")
    processor = GenericModalProcessor(vlm, content_list=[block], context_window=2)
    result = processor.process(block, block_index=0, image_bytes=None)
    assert result == "generic element description"


def test_generic_processor_returns_empty_on_vlm_error():
    """GenericModalProcessor.process() returns '' when VLM returns an error."""
    from hybrid_doc_parser.modal_processors import GenericModalProcessor
    block = make_block("unknown", text="content")
    vlm = mock.MagicMock()
    vlm.call.return_value = {"error": "unavailable"}
    processor = GenericModalProcessor(vlm, content_list=[block], context_window=2)
    result = processor.process(block, block_index=0, image_bytes=None)
    assert result == ""


def test_generic_processor_returns_empty_on_exception():
    """GenericModalProcessor.process() returns '' when VLMClient raises."""
    from hybrid_doc_parser.modal_processors import GenericModalProcessor
    block = make_block("unknown")
    vlm = mock.MagicMock()
    vlm.call.side_effect = RuntimeError("crash")
    processor = GenericModalProcessor(vlm, content_list=[block], context_window=2)
    result = processor.process(block, block_index=0, image_bytes=None)
    assert result == ""


def test_generic_enrich_returns_description():
    """GenericModalProcessor.enrich() delegates to VLM correctly."""
    from hybrid_doc_parser.modal_processors import GenericModalProcessor
    block = make_block("unknown", text="some generic content")
    vlm = make_mock_vlm("enriched generic description")
    config = make_mock_config()
    content_list = [block]
    processor = GenericModalProcessor(vlm, content_list=content_list, context_window=2)
    result = processor.enrich(block, block_index=0, content_list=content_list, vlm_client=vlm, config=config)
    assert result == "enriched generic description"


def test_generic_enrich_returns_empty_on_vlm_error():
    """GenericModalProcessor.enrich() returns '' when VLM returns error key."""
    from hybrid_doc_parser.modal_processors import GenericModalProcessor
    block = make_block("unknown")
    vlm = mock.MagicMock()
    vlm.call.return_value = {"error": "error"}
    config = make_mock_config()
    content_list = [block]
    processor = GenericModalProcessor(vlm, content_list=content_list, context_window=2)
    result = processor.enrich(block, block_index=0, content_list=content_list, vlm_client=vlm, config=config)
    assert result == ""


def test_generic_enrich_returns_empty_on_exception():
    """GenericModalProcessor.enrich() returns '' when VLMClient raises."""
    from hybrid_doc_parser.modal_processors import GenericModalProcessor
    block = make_block("unknown")
    vlm = mock.MagicMock()
    vlm.call.side_effect = RuntimeError("network error")
    config = make_mock_config()
    content_list = [block]
    processor = GenericModalProcessor(vlm, content_list=content_list, context_window=2)
    result = processor.enrich(block, block_index=0, content_list=content_list, vlm_client=vlm, config=config)
    assert result == ""
