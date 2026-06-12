"""End-to-end integration tests for the hybrid-doc-parser pipeline."""

from __future__ import annotations

import asyncio
import unittest.mock as mock
from pathlib import Path

from hybrid_doc_parser.models import ElementType, EnrichmentConfig

FIXTURES = Path(__file__).parent / "fixtures"


def _fake_content_list(page_count: int = 1, include_table: bool = False) -> list[dict]:
    blocks = []
    for i in range(page_count):
        blocks.append(
            {
                "type": "title",
                "text": f"Section {i + 1}",
                "page_idx": i,
                "page_size": [595.0, 842.0],
                "bbox": [50.0, 750.0, 545.0, 780.0],
            }
        )
        blocks.append(
            {
                "type": "text",
                "text": "This is a realistic paragraph with enough words to satisfy the quality gate heuristics.",
                "page_idx": i,
                "page_size": [595.0, 842.0],
                "bbox": [50.0, 700.0, 545.0, 740.0],
            }
        )
        if include_table:
            blocks.append(
                {
                    "type": "table",
                    "text": "<table><tr><th>Col A</th><th>Col B</th></tr><tr><td>1</td><td>2</td></tr></table>",
                    "page_idx": i,
                    "page_size": [595.0, 842.0],
                    "bbox": [50.0, 600.0, 545.0, 680.0],
                }
            )
    return blocks


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


def test_parse_digital_simple_no_enrichment(monkeypatch, tmp_path):
    """parse() on a digital PDF returns elements and no fatal warnings."""
    monkeypatch.setenv("HYBRID_DOC_PARSER_CACHE_DIR", str(tmp_path))
    from hybrid_doc_parser import parse

    fake_cl = _fake_content_list(1)
    with mock.patch("hybrid_doc_parser.parser._run_mineru", return_value=fake_cl):
        result = parse(FIXTURES / "digital_simple.pdf", EnrichmentConfig())

    assert result.page_count == 1
    assert len(result.elements) >= 1
    assert result.schema_version == "1.1"
    # No critical (non-escalation) warnings
    fatal = [w for w in result.warnings if w.code not in {"quality_gate_escalation"}]
    assert fatal == []


def test_parse_multi_page(monkeypatch, tmp_path):
    """parse() on a multi-page document returns correct page_count and per-page records."""
    monkeypatch.setenv("HYBRID_DOC_PARSER_CACHE_DIR", str(tmp_path))
    from hybrid_doc_parser import parse

    fake_cl = _fake_content_list(3)
    with mock.patch("hybrid_doc_parser.parser._run_mineru", return_value=fake_cl):
        result = parse(FIXTURES / "digital_simple.pdf", EnrichmentConfig())

    assert result.page_count == 3
    assert len(result.pages) == 3
    page_indices = {p.page_idx for p in result.pages}
    assert page_indices == {0, 1, 2}


def test_parse_to_markdown_pipeline(monkeypatch, tmp_path):
    """Full parse -> render_markdown pipeline produces valid Markdown."""
    monkeypatch.setenv("HYBRID_DOC_PARSER_CACHE_DIR", str(tmp_path))
    from hybrid_doc_parser import parse, render_markdown

    fake_cl = _fake_content_list(1)
    with mock.patch("hybrid_doc_parser.parser._run_mineru", return_value=fake_cl):
        result = parse(FIXTURES / "digital_simple.pdf", EnrichmentConfig())

    md = render_markdown(result)
    assert isinstance(md, str)
    assert md.endswith("\n")
    assert len(md) > 0
    # Heading and paragraph present
    assert "#" in md or "Section" in md


def test_parse_with_table_renders_markdown_table(monkeypatch, tmp_path):
    """Table elements render as Markdown tables in the output."""
    monkeypatch.setenv("HYBRID_DOC_PARSER_CACHE_DIR", str(tmp_path))
    from hybrid_doc_parser import parse, render_markdown

    fake_cl = _fake_content_list(1, include_table=True)
    with mock.patch("hybrid_doc_parser.parser._run_mineru", return_value=fake_cl):
        result = parse(FIXTURES / "digital_simple.pdf", EnrichmentConfig())

    assert any(e.type == ElementType.table for e in result.elements)
    md = render_markdown(result)
    assert "| Col A |" in md
    assert "---" in md


def test_parse_furniture_filtered_from_markdown(monkeypatch, tmp_path):
    """Header/footer/page_number elements are excluded from render_markdown output."""
    monkeypatch.setenv("HYBRID_DOC_PARSER_CACHE_DIR", str(tmp_path))
    from hybrid_doc_parser import parse, render_markdown

    fake_cl = [
        {
            "type": "header",
            "text": "Document Title Header",
            "page_idx": 0,
            "page_size": [595.0, 842.0],
            "bbox": [0, 800, 595, 840],
        },
        {
            "type": "footer",
            "text": "Confidential",
            "page_idx": 0,
            "page_size": [595.0, 842.0],
            "bbox": [0, 0, 595, 40],
        },
        {
            "type": "page_number",
            "text": "1",
            "page_idx": 0,
            "page_size": [595.0, 842.0],
            "bbox": [280, 0, 315, 20],
        },
        {
            "type": "text",
            "text": "Real content paragraph here.",
            "page_idx": 0,
            "page_size": [595.0, 842.0],
            "bbox": [50, 600, 545, 640],
        },
    ]
    with mock.patch("hybrid_doc_parser.parser._run_mineru", return_value=fake_cl):
        result = parse(FIXTURES / "digital_simple.pdf", EnrichmentConfig())

    md = render_markdown(result)
    assert "Document Title Header" not in md
    assert "Confidential" not in md
    assert "Real content paragraph here." in md


def test_parse_batch_all_fixtures(monkeypatch, tmp_path):
    """parse_batch() processes all fixture PDFs without raising."""
    monkeypatch.setenv("HYBRID_DOC_PARSER_CACHE_DIR", str(tmp_path))
    from hybrid_doc_parser import parse_batch

    pdfs = [
        FIXTURES / "digital_simple.pdf",
        FIXTURES / "scanned.pdf",
        FIXTURES / "mixed.pdf",
        FIXTURES / "equation_heavy.pdf",
    ]
    fake_cl = _fake_content_list(1)
    with mock.patch("hybrid_doc_parser.parser._run_mineru", return_value=fake_cl):
        results = asyncio.run(parse_batch(pdfs, EnrichmentConfig()))

    assert len(results) == 4
    file_paths = {r.file_path for r in results}
    assert all(str(p) in file_paths for p in pdfs)


def test_parse_never_raises_on_corrupt_content_list(monkeypatch, tmp_path):
    """parse() never raises even when MinerU returns malformed output."""
    monkeypatch.setenv("HYBRID_DOC_PARSER_CACHE_DIR", str(tmp_path))
    from hybrid_doc_parser import parse

    # Return a list with badly-formed blocks
    bad_cl = [
        {"type": None, "text": 12345},  # bad types
        {"missing_type_key": True},  # no type key
        None,  # None item
    ]
    with mock.patch("hybrid_doc_parser.parser._run_mineru", return_value=bad_cl):
        result = parse(FIXTURES / "digital_simple.pdf", EnrichmentConfig())

    assert result is not None
    assert isinstance(result.file_path, str)
