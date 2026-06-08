"""Tests for markdown.py canonical Markdown renderer."""
from __future__ import annotations

import pytest

from hybrid_doc_parser.models import (
    ElementRecord,
    ElementType,
    EnrichmentConfig,
    PageRecord,
    ParserOutput,
    WarningRecord,
)


def make_element(
    etype: ElementType,
    text: str = "",
    description: str = "",
    is_enriched: bool = False,
    page_idx: int = 0,
) -> ElementRecord:
    """Construct a minimal ElementRecord for test purposes.

    Args:
        etype: The semantic element type.
        text: Raw text content for the element.
        description: VLM-generated description (enriched elements only).
        is_enriched: Whether this element has been VLM-enriched.
        page_idx: Zero-indexed page number.

    Returns:
        A minimal ElementRecord suitable for use in tests.
    """
    return ElementRecord(
        element_id="test-id",
        type=etype,
        text=text,
        description=description,
        is_enriched=is_enriched,
        page_idx=page_idx,
        bbox=[],
    )


def make_output(elements: list[ElementRecord]) -> ParserOutput:
    """Construct a minimal ParserOutput wrapping the given elements.

    Args:
        elements: List of ElementRecord objects to include.

    Returns:
        A minimal ParserOutput for use in render_markdown tests.
    """
    return ParserOutput(
        file_path="/tmp/test.pdf",
        file_sha256="a" * 64,
        page_count=1,
        pages=[
            PageRecord(
                page_idx=0,
                quality_decision="keep",
                element_count=len(elements),
                vlm_used=False,
            )
        ],
        elements=elements,
        warnings=[],
        enrichment_config=EnrichmentConfig(),
    )


def test_escape_cell() -> None:
    """Test that _escape_cell escapes pipe characters and removes newlines."""
    from hybrid_doc_parser.markdown import _escape_cell

    result = _escape_cell("hello|world\nfoo")
    assert "\\|" in result
    assert "\n" not in result
    assert "foo" in result


def test_table_html_to_markdown_valid() -> None:
    """Test that _table_html_to_markdown converts a valid HTML table correctly."""
    from hybrid_doc_parser.markdown import _table_html_to_markdown

    html = "<table><tr><th>Name</th><th>Age</th></tr><tr><td>Alice</td><td>30</td></tr></table>"
    result = _table_html_to_markdown(html)
    assert "| Name | Age |" in result
    assert "---" in result
    assert "| Alice | 30 |" in result


def test_table_html_to_markdown_invalid() -> None:
    """Test that _table_html_to_markdown returns empty string for non-table HTML."""
    from hybrid_doc_parser.markdown import _table_html_to_markdown

    result = _table_html_to_markdown("<notatable>garbage</notatable>")
    assert result == ""


def test_furniture_filtered() -> None:
    """Test that header, footer, and page_number elements are excluded from output."""
    from hybrid_doc_parser.markdown import render_markdown

    elements = [
        make_element(ElementType.header, text="Chapter 1"),
        make_element(ElementType.footer, text="Page 1"),
        make_element(ElementType.page_number, text="1"),
        make_element(ElementType.text, text="Real content"),
    ]
    output = make_output(elements)
    result = render_markdown(output)
    assert "Chapter 1" not in result
    assert "Page 1" not in result
    assert "Real content" in result


def test_heading_renders_with_hash() -> None:
    """Test that heading elements with leading '#' are emitted as-is."""
    from hybrid_doc_parser.markdown import render_markdown

    elements = [make_element(ElementType.heading, text="## My Heading")]
    output = make_output(elements)
    result = render_markdown(output)
    assert "## My Heading" in result


def test_heading_without_hash_wrapped_as_h1() -> None:
    """Test that a heading element without a '#' prefix is wrapped as h1."""
    from hybrid_doc_parser.markdown import _render_element

    element = make_element(ElementType.heading, text="Plain Heading")
    result = _render_element(element)
    assert result == "# Plain Heading"


def test_heading_empty_text_returns_empty() -> None:
    """Test that an empty heading element renders as empty string."""
    from hybrid_doc_parser.markdown import _render_element

    element = make_element(ElementType.heading, text="")
    result = _render_element(element)
    assert result == ""


def test_list_item_renders_with_dash() -> None:
    """Test that list_item elements render with a leading dash."""
    from hybrid_doc_parser.markdown import _render_element

    element = make_element(ElementType.list_item, text="First item")
    result = _render_element(element)
    assert result == "- First item"


def test_list_item_empty_text_returns_empty() -> None:
    """Test that an empty list_item element renders as empty string."""
    from hybrid_doc_parser.markdown import _render_element

    element = make_element(ElementType.list_item, text="")
    result = _render_element(element)
    assert result == ""


def test_caption_element_renders_text() -> None:
    """Test that caption elements render their text directly."""
    from hybrid_doc_parser.markdown import _render_element

    element = make_element(ElementType.caption, text="Figure 1: A caption")
    result = _render_element(element)
    assert result == "Figure 1: A caption"


def test_equation_renders_latex_block() -> None:
    """Test that equation elements are wrapped in $$ delimiters."""
    from hybrid_doc_parser.markdown import _render_element

    element = make_element(ElementType.equation, text=r"E = mc^2")
    result = _render_element(element)
    assert result.startswith("$$")
    assert r"E = mc^2" in result
    assert result.endswith("$$")


def test_equation_empty_text_returns_empty() -> None:
    """Test that an empty equation element renders as empty string."""
    from hybrid_doc_parser.markdown import _render_element

    element = make_element(ElementType.equation, text="")
    result = _render_element(element)
    assert result == ""


def test_table_enriched_fallback_blockquote() -> None:
    """Table with no parseable HTML but enriched description renders as blockquote."""
    from hybrid_doc_parser.markdown import _render_element

    # Non-HTML text for table means _table_html_to_markdown returns ""
    element = make_element(
        ElementType.table,
        text="plain table text no html",
        description="A table about sales",
        is_enriched=True,
    )
    result = _render_element(element)
    assert "> A table about sales" in result


def test_table_no_html_no_enrichment_returns_text() -> None:
    """Table with no HTML and no enrichment falls back to raw text."""
    from hybrid_doc_parser.markdown import _render_element

    element = make_element(
        ElementType.table,
        text="raw table fallback",
        description="",
        is_enriched=False,
    )
    result = _render_element(element)
    assert result == "raw table fallback"


def test_unknown_element_type_returns_empty() -> None:
    """Unknown element types render as empty string."""
    from hybrid_doc_parser.markdown import _render_element

    element = make_element(ElementType.unknown, text="some unknown block")
    result = _render_element(element)
    assert result == ""


def test_image_enriched_renders_blockquote() -> None:
    """Test that enriched images render as blockquotes and non-enriched images are skipped."""
    from hybrid_doc_parser.markdown import render_markdown

    elements = [
        make_element(ElementType.image, text="", description="a diagram", is_enriched=True),
        make_element(ElementType.image, text="", description="", is_enriched=False),
    ]
    output = make_output(elements)
    result = render_markdown(output)
    assert "> a diagram" in result
    # NOTE: Only one blockquote fragment from the enriched image.
    assert result.count("> ") == 1


def test_output_ends_with_newline() -> None:
    """Test that render_markdown output ends with exactly one newline and has no leading whitespace."""
    from hybrid_doc_parser.markdown import render_markdown

    elements = [make_element(ElementType.text, text="hello")]
    output = make_output(elements)
    result = render_markdown(output)
    assert result.endswith("\n")
    assert not result.startswith("\n")
    assert not result.endswith("\n\n")


def test_render_markdown_skips_exception_elements() -> None:
    """render_markdown continues rendering when _render_element raises for one element."""
    import unittest.mock as mock
    from hybrid_doc_parser.markdown import render_markdown

    elements = [
        make_element(ElementType.text, text="good content"),
    ]
    output = make_output(elements)

    # NOTE: Patch _render_element to raise on the first call, succeed on the next.
    call_count = {"n": 0}
    original = __import__("hybrid_doc_parser.markdown", fromlist=["_render_element"])._render_element

    def raising_render(element):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("simulated render error")
        return original(element)

    with mock.patch("hybrid_doc_parser.markdown._render_element", side_effect=raising_render):
        result = render_markdown(output)
    # The element raised an exception so it is skipped; result is just the newline
    assert isinstance(result, str)
    assert result.endswith("\n")
