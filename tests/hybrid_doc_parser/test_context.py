"""Tests for context.py ContextExtractor."""

from __future__ import annotations

import pytest


def make_item(block_type: str, text: str, page_idx: int = 0) -> dict:
    """Build a minimal MinerU content_list block dict for testing."""
    return {"type": block_type, "text": text, "page_idx": page_idx}


def test_extract_for_block_excludes_images():
    from hybrid_doc_parser.context import ContextExtractor

    content_list = [make_item("text", f"paragraph {i}") for i in range(10)]
    # Replace index 3 with an image (should be excluded from context)
    content_list[3] = make_item("image", "image data")
    extractor = ContextExtractor(content_list, window=2)
    # Extract context for index 5 (image at 3 should be excluded from window [3,4,6,7])
    result = extractor.extract_for_block(5)
    assert "image data" not in result
    # Text paragraphs adjacent to 5 should be included
    assert "paragraph" in result


def test_extract_for_block_at_start():
    from hybrid_doc_parser.context import ContextExtractor

    content_list = [make_item("text", f"block {i}") for i in range(5)]
    extractor = ContextExtractor(content_list, window=3)
    # block index 0: window goes from max(0, -3)=0 to min(5, 4)=4, excluding index 0
    result = extractor.extract_for_block(0)
    # Should not raise; should include blocks 1, 2, 3
    assert isinstance(result, str)
    assert "block 1" in result


def test_extract_for_block_truncates():
    from hybrid_doc_parser.context import ContextExtractor

    content_list = [
        make_item("text", "target"),  # index 0 = target
        make_item("text", " ".join(["word"] * 100)),  # index 1 = very long
    ]
    extractor = ContextExtractor(content_list, window=1, max_tokens=5)
    result = extractor.extract_for_block(0)
    assert result.endswith("...")
    word_count = len(result.replace("...", "").split())
    assert word_count <= 5


def test_extract_for_page_adjacent_pages():
    from hybrid_doc_parser.context import ContextExtractor

    content_list = [
        make_item("text", "page0 content", page_idx=0),
        make_item("text", "page1 content", page_idx=1),
        make_item("text", "page2 content", page_idx=2),
        make_item("text", "page3 content", page_idx=3),
        make_item("image", "page1 image", page_idx=1),  # image excluded
    ]
    extractor = ContextExtractor(content_list)
    result = extractor.extract_for_page(1)
    # Should include pages 0, 1, 2 but not 3
    assert "page0 content" in result
    assert "page1 content" in result
    assert "page2 content" in result
    assert "page3 content" not in result
    # Image excluded
    assert "page1 image" not in result


def test_extract_methods_return_empty_on_error():
    from hybrid_doc_parser.context import ContextExtractor

    # Pass non-iterable content_list to trigger error
    extractor = ContextExtractor("not a list")  # type: ignore[arg-type]
    assert extractor.extract_for_block(0) == ""
    assert extractor.extract_for_page(0) == ""


def test_context_extractor_uses_slots():
    from hybrid_doc_parser.context import ContextExtractor

    extractor = ContextExtractor([])
    assert hasattr(extractor, "__slots__")
    with pytest.raises(AttributeError):
        extractor.nonexistent_attribute = 1  # type: ignore[attr-defined]
