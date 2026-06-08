"""Canonical Markdown renderer for ParserOutput.

Converts a ParserOutput to RAG-ready Markdown, dropping page furniture
(headers, footers, page numbers) and formatting enriched elements.

# NOTE: Heading level is expected to be encoded in ElementRecord.text as
# leading '#' characters by parser.py. The renderer emits heading text as-is.
"""

from __future__ import annotations

from html.parser import HTMLParser
from typing import Any, Final

from hybrid_doc_parser.models import ElementType, ParserOutput

_FURNITURE: Final[frozenset[ElementType]] = frozenset(
    {
        ElementType.header,
        ElementType.footer,
        ElementType.page_number,
    }
)


def _escape_cell(text: str) -> str:
    """Escape pipe characters and replace newlines for Markdown table cells.

    Args:
        text: Raw cell text.

    Returns:
        Escaped cell text safe for Markdown table formatting.
    """
    return str(text).replace("|", "\\|").replace("\n", " ").strip()


class _TableParser(HTMLParser):
    """Internal HTML parser for extracting table rows and cells."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._current_row: list[str] = []
        self._current_cell: list[str] = []
        self._in_cell: bool = False
        self._has_table: bool = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Handle opening HTML tags, setting parser state for table elements.

        Args:
            tag: The HTML tag name.
            attrs: List of (name, value) attribute pairs.
        """
        if tag == "table":
            self._has_table = True
        elif tag == "tr":
            self._current_row = []
        elif tag in {"td", "th"}:
            self._current_cell = []
            self._in_cell = True

    def handle_endtag(self, tag: str) -> None:
        """Handle closing HTML tags, finalising cells and rows.

        Args:
            tag: The HTML tag name.
        """
        if tag in {"td", "th"}:
            self._current_row.append("".join(self._current_cell).strip())
            self._current_cell = []
            self._in_cell = False
        elif tag == "tr":
            if self._current_row:
                self.rows.append(self._current_row)
                self._current_row = []

    def handle_data(self, data: str) -> None:
        """Accumulate text data within active table cells.

        Args:
            data: Text content from the parser.
        """
        if self._in_cell:
            self._current_cell.append(data)


def _table_html_to_markdown(html: str) -> str:
    """Parse an HTML table string and render it as GitHub-flavored Markdown.

    Args:
        html: HTML string containing a <table> element.

    Returns:
        Markdown table string, or '' if no table found or parsing fails.
    """
    try:
        parser = _TableParser()
        parser.feed(html)
        if not parser._has_table or not parser.rows:
            return ""
        lines: list[str] = []
        for i, row in enumerate(parser.rows):
            cells = [_escape_cell(c) for c in row]
            lines.append("| " + " | ".join(cells) + " |")
            if i == 0:
                lines.append("| " + " | ".join(["---"] * len(cells)) + " |")
        return "\n".join(lines)
    except Exception:
        return ""


def _render_element(element: Any) -> str:
    """Render a single ElementRecord to a Markdown fragment.

    Args:
        element: ElementRecord to render.

    Returns:
        Rendered Markdown string, or '' if nothing to render.
    """
    etype = element.type
    text = (element.text or "").strip()

    if etype == ElementType.heading:
        # NOTE: Heading level is pre-encoded as '#' prefix by parser.py.
        # If text already starts with '#', emit it verbatim; otherwise wrap as h1.
        if text.startswith("#"):
            return text
        return f"# {text}" if text else ""

    if etype in {ElementType.text, ElementType.caption}:
        return text

    if etype == ElementType.list_item:
        return f"- {text}" if text else ""

    if etype == ElementType.table:
        md = _table_html_to_markdown(text)
        if md:
            return md
        if element.is_enriched and element.description:
            return f"> {element.description}"
        return text

    if etype == ElementType.equation:
        return f"$$\n{text}\n$$" if text else ""

    if etype == ElementType.image:
        if element.is_enriched and element.description:
            return f"> {element.description}"
        return ""

    return ""


def render_markdown(parser_output: ParserOutput) -> str:
    """Render a ParserOutput to RAG-ready Markdown.

    Drops page furniture (header, footer, page_number). Formats each element
    by type. Joins non-empty blocks with double newlines.

    Args:
        parser_output: Validated ParserOutput to render.

    Returns:
        Markdown string ending with exactly one newline character.
    """
    blocks: list[str] = []

    for element in parser_output.elements:
        if element.type in _FURNITURE:
            continue
        try:
            rendered = _render_element(element)
        except Exception:
            continue
        if rendered:
            blocks.append(rendered)

    result = "\n\n".join(b for b in blocks if b)
    return result.strip() + "\n"
