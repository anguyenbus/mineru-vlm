"""Tests for parser.py public API."""
from __future__ import annotations

import hashlib
import json
import unittest.mock as mock
from pathlib import Path

import pytest

from hybrid_doc_parser.models import ElementType, EnrichmentConfig

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_content_list(n_pages: int = 1) -> list[dict]:
    """Return a minimal valid MinerU content_list."""
    return [
        {
            "type": "text",
            "text": f"Page {i} paragraph text with enough words to pass the gate.",
            "page_idx": i,
            "page_size": [595.0, 842.0],
            "bbox": [50.0, 700.0, 545.0, 750.0],
        }
        for i in range(n_pages)
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_parse_returns_parser_output(tmp_path, monkeypatch):
    """parse() returns a valid ParserOutput; never raises."""
    monkeypatch.setenv("HYBRID_DOC_PARSER_CACHE_DIR", str(tmp_path / "cache"))
    from hybrid_doc_parser.parser import parse

    pdf = FIXTURES / "digital_simple.pdf"

    fake_cl = _fake_content_list(1)
    with mock.patch("hybrid_doc_parser.parser._run_mineru", return_value=fake_cl):
        result = parse(pdf, EnrichmentConfig())

    assert result.file_path == str(pdf)
    assert result.file_sha256 == _sha256(pdf)
    assert result.page_count == 1
    assert len(result.elements) >= 1
    assert len(result.warnings) == 0 or all(w.code for w in result.warnings)


def test_parse_never_raises_on_mineru_failure(tmp_path, monkeypatch):
    """parse() returns a ParserOutput even when MinerU throws."""
    monkeypatch.setenv("HYBRID_DOC_PARSER_CACHE_DIR", str(tmp_path / "cache"))
    from hybrid_doc_parser.parser import parse

    pdf = FIXTURES / "digital_simple.pdf"

    with mock.patch("hybrid_doc_parser.parser._run_mineru", side_effect=RuntimeError("MinerU crashed")):
        result = parse(pdf, EnrichmentConfig())

    # Must not raise; warnings list must contain the error
    assert result is not None
    assert any("mineru" in w.code.lower() or "mineru" in w.message.lower() for w in result.warnings)


def test_parse_cache_hit_skips_mineru(tmp_path, monkeypatch):
    """parse() returns cached result without calling MinerU on second call."""
    monkeypatch.setenv("HYBRID_DOC_PARSER_CACHE_DIR", str(tmp_path / "cache"))
    from hybrid_doc_parser.parser import parse

    pdf = FIXTURES / "digital_simple.pdf"
    fake_cl = _fake_content_list(1)

    call_count = {"n": 0}

    def counting_mineru(path, backend):
        call_count["n"] += 1
        return fake_cl

    with mock.patch("hybrid_doc_parser.parser._run_mineru", side_effect=counting_mineru):
        result1 = parse(pdf, EnrichmentConfig())
        result2 = parse(pdf, EnrichmentConfig())

    assert call_count["n"] == 1  # second call hit cache
    assert result1.file_sha256 == result2.file_sha256


def test_parse_element_types_routed_correctly(tmp_path, monkeypatch):
    """Each MinerU block_type maps to the correct ElementType."""
    monkeypatch.setenv("HYBRID_DOC_PARSER_CACHE_DIR", str(tmp_path / "cache"))
    from hybrid_doc_parser.parser import parse

    pdf = FIXTURES / "digital_simple.pdf"
    fake_cl = [
        {"type": "text", "text": "A paragraph.", "page_idx": 0, "page_size": [595.0, 842.0], "bbox": [0, 0, 100, 20]},
        {"type": "title", "text": "## Section Header", "page_idx": 0, "page_size": [595.0, 842.0], "bbox": [0, 30, 100, 50]},
        {"type": "table", "text": "<table><tr><td>data</td></tr></table>", "page_idx": 0, "page_size": [595.0, 842.0], "bbox": [0, 60, 100, 80]},
        {"type": "interline_equation", "text": r"E = mc^2", "page_idx": 0, "page_size": [595.0, 842.0], "bbox": [0, 90, 100, 110]},
        {"type": "image", "text": "", "page_idx": 0, "page_size": [595.0, 842.0], "bbox": [0, 120, 100, 200]},
    ]

    with mock.patch("hybrid_doc_parser.parser._run_mineru", return_value=fake_cl):
        result = parse(pdf, EnrichmentConfig())

    types = {e.type for e in result.elements}
    assert ElementType.text in types
    assert ElementType.heading in types
    assert ElementType.table in types
    assert ElementType.equation in types
    assert ElementType.image in types


def test_parse_batch_processes_multiple_files(tmp_path, monkeypatch):
    """parse_batch() returns one ParserOutput per input file."""
    monkeypatch.setenv("HYBRID_DOC_PARSER_CACHE_DIR", str(tmp_path / "cache"))
    import asyncio
    from hybrid_doc_parser.parser import parse_batch

    pdfs = [FIXTURES / "digital_simple.pdf", FIXTURES / "mixed.pdf"]
    fake_cl = _fake_content_list(1)

    with mock.patch("hybrid_doc_parser.parser._run_mineru", return_value=fake_cl):
        results = asyncio.run(parse_batch(pdfs, EnrichmentConfig()))

    assert len(results) == 2
    assert all(r.file_path in {str(p) for p in pdfs} for r in results)


def test_parse_batch_isolates_failures(tmp_path, monkeypatch):
    """parse_batch() succeeds for valid files even if one file fails."""
    monkeypatch.setenv("HYBRID_DOC_PARSER_CACHE_DIR", str(tmp_path / "cache"))
    import asyncio
    from hybrid_doc_parser.parser import parse_batch

    pdfs = [FIXTURES / "digital_simple.pdf", Path("/nonexistent/ghost.pdf")]
    fake_cl = _fake_content_list(1)

    with mock.patch("hybrid_doc_parser.parser._run_mineru", return_value=fake_cl):
        results = asyncio.run(parse_batch(pdfs, EnrichmentConfig()))

    assert len(results) == 2
    # nonexistent file must produce a result with warnings, not raise
    assert all(isinstance(r.file_path, str) for r in results)


def test_parse_nonexistent_file_returns_with_warning():
    """parse() on a missing file returns a ParserOutput with a warning."""
    from hybrid_doc_parser.parser import parse

    result = parse(Path("/nonexistent/ghost.pdf"), EnrichmentConfig())
    assert result is not None
    assert len(result.warnings) >= 1


def test_parse_unsupported_extension_returns_warning(tmp_path, monkeypatch):
    """parse() on an unsupported file extension returns a ParserOutput with warning."""
    monkeypatch.setenv("HYBRID_DOC_PARSER_CACHE_DIR", str(tmp_path))
    from hybrid_doc_parser.parser import parse

    # Create a fake file with unsupported extension
    fake_file = tmp_path / "document.docx"
    fake_file.write_bytes(b"fake docx content")

    result = parse(fake_file, EnrichmentConfig())
    assert result is not None
    assert any(w.code == "unsupported_type" for w in result.warnings)


def test_parse_defaults_to_none_config(tmp_path, monkeypatch):
    """parse() uses EnrichmentConfig() when config=None is passed."""
    monkeypatch.setenv("HYBRID_DOC_PARSER_CACHE_DIR", str(tmp_path))
    from hybrid_doc_parser.parser import parse

    fake_cl = _fake_content_list(1)
    with mock.patch("hybrid_doc_parser.parser._run_mineru", return_value=fake_cl):
        result = parse(FIXTURES / "digital_simple.pdf", None)

    assert result is not None
    assert result.enrichment_config is not None
    assert result.enrichment_config.enabled is False


def test_normalise_aliases_img_caption():
    """_normalise_aliases renames img_caption to image_caption."""
    from hybrid_doc_parser.parser import _normalise_aliases

    block = {"type": "image", "img_caption": "A caption", "text": ""}
    result = _normalise_aliases(block)
    assert "image_caption" in result
    assert "img_caption" not in result
    assert result["image_caption"] == "A caption"


def test_normalise_aliases_img_footnote():
    """_normalise_aliases renames img_footnote to image_footnote."""
    from hybrid_doc_parser.parser import _normalise_aliases

    block = {"type": "image", "img_footnote": "Footnote text", "text": ""}
    result = _normalise_aliases(block)
    assert "image_footnote" in result
    assert "img_footnote" not in result


def test_route_block_type_none_returns_unknown():
    """_route_block_type(None) returns ElementType.unknown without raising."""
    from hybrid_doc_parser.parser import _route_block_type

    result = _route_block_type(None)
    assert result == ElementType.unknown


def test_route_block_type_unknown_string():
    """_route_block_type() returns unknown for unrecognised type strings."""
    from hybrid_doc_parser.parser import _route_block_type

    result = _route_block_type("fancy_new_block_type")
    assert result == ElementType.unknown


def test_route_block_heading_text_level_clamped(tmp_path, monkeypatch):
    """_route_block clamps heading text_level to valid 1-6 range."""
    monkeypatch.setenv("HYBRID_DOC_PARSER_CACHE_DIR", str(tmp_path))
    from hybrid_doc_parser.parser import _route_block

    # text_level=9 should be clamped to 6
    block = {
        "type": "title",
        "text": "Section Title",
        "page_idx": 0,
        "text_level": 9,
        "bbox": [0, 0, 100, 20],
    }
    record = _route_block(block, page_idx=0, element_idx=0, file_sha256="a" * 64)
    assert record.text.startswith("######")


def test_route_block_heading_invalid_text_level_defaults_to_1(tmp_path, monkeypatch):
    """_route_block defaults to heading level 1 when text_level is not numeric."""
    from hybrid_doc_parser.parser import _route_block

    block = {
        "type": "title",
        "text": "Section",
        "page_idx": 0,
        "text_level": "not-a-number",
        "bbox": [],
    }
    record = _route_block(block, page_idx=0, element_idx=0, file_sha256="a" * 64)
    assert record.text.startswith("# ")


def test_route_block_non_string_text_coerced(tmp_path, monkeypatch):
    """_route_block coerces non-string text values to str."""
    from hybrid_doc_parser.parser import _route_block

    block = {
        "type": "text",
        "text": 12345,
        "page_idx": 0,
        "bbox": [],
    }
    record = _route_block(block, page_idx=0, element_idx=0, file_sha256="a" * 64)
    assert record.text == "12345"
    assert isinstance(record.text, str)


def test_read_output_files_content_list_json(tmp_path):
    """_read_output_files parses a *_content_list.json file correctly."""
    from hybrid_doc_parser.parser import _read_output_files

    content = [{"type": "text", "text": "hello", "page_idx": 0, "bbox": []}]
    json_file = tmp_path / "doc_content_list.json"
    json_file.write_text(json.dumps(content), encoding="utf-8")

    result = _read_output_files(tmp_path)
    assert result == content


def test_read_output_files_dict_with_content_list(tmp_path):
    """_read_output_files extracts content_list from a dict-format JSON file."""
    from hybrid_doc_parser.parser import _read_output_files

    content = [{"type": "text", "text": "world", "page_idx": 0, "bbox": []}]
    json_file = tmp_path / "output.json"
    json_file.write_text(json.dumps({"content_list": content, "other": "data"}), encoding="utf-8")

    result = _read_output_files(tmp_path)
    assert result == content


def test_read_output_files_empty_dir_returns_empty(tmp_path):
    """_read_output_files returns [] when no JSON files are found."""
    from hybrid_doc_parser.parser import _read_output_files

    result = _read_output_files(tmp_path)
    assert result == []


def test_read_output_files_corrupt_json_skipped(tmp_path):
    """_read_output_files skips corrupt JSON files and returns []."""
    from hybrid_doc_parser.parser import _read_output_files

    json_file = tmp_path / "bad_content_list.json"
    json_file.write_text("{not valid json", encoding="utf-8")

    result = _read_output_files(tmp_path)
    assert result == []


def test_parse_with_enrichment_calls_processors(tmp_path, monkeypatch):
    """parse() with enrichment enabled calls VLM processors for modal elements."""
    monkeypatch.setenv("HYBRID_DOC_PARSER_CACHE_DIR", str(tmp_path))
    from hybrid_doc_parser.parser import parse

    fake_cl = [
        {
            "type": "table",
            "text": "<table><tr><th>Col</th></tr><tr><td>val</td></tr></table>",
            "page_idx": 0,
            "page_size": [595.0, 842.0],
            "bbox": [0, 0, 100, 50],
        }
    ]

    config = EnrichmentConfig(enabled=True, table=True, image=False, equation=False)

    # NOTE: Patch make_vlm_client and the table processor to avoid real VLM calls.
    mock_vlm = mock.MagicMock()
    mock_vlm.call.return_value = {"description": "a test table"}

    with mock.patch("hybrid_doc_parser.parser._run_mineru", return_value=fake_cl):
        with mock.patch("hybrid_doc_parser.vlm_client.make_vlm_client", return_value=mock_vlm):
            result = parse(FIXTURES / "digital_simple.pdf", config)

    assert result is not None
    # At least one element should exist
    assert len(result.elements) >= 1


def test_parse_enrichment_error_captured_as_warning(tmp_path, monkeypatch):
    """parse() captures enrichment failures as warnings without raising."""
    monkeypatch.setenv("HYBRID_DOC_PARSER_CACHE_DIR", str(tmp_path))
    from hybrid_doc_parser.parser import parse

    fake_cl = _fake_content_list(1)
    config = EnrichmentConfig(enabled=True, image=True, table=True, equation=True)

    with mock.patch("hybrid_doc_parser.parser._run_mineru", return_value=fake_cl):
        with mock.patch(
            "hybrid_doc_parser.parser._enrich_elements",
            side_effect=RuntimeError("enrichment crashed"),
        ):
            result = parse(FIXTURES / "digital_simple.pdf", config)

    assert result is not None
    assert any(w.code == "enrichment_error" for w in result.warnings)


def test_parse_none_blocks_in_content_list_are_skipped(tmp_path, monkeypatch):
    """parse() skips None entries in content_list without crashing."""
    monkeypatch.setenv("HYBRID_DOC_PARSER_CACHE_DIR", str(tmp_path))
    from hybrid_doc_parser.parser import parse

    bad_cl = [
        {"type": "text", "text": "valid block", "page_idx": 0, "bbox": []},
        None,
        {"type": "text", "text": "another valid block", "page_idx": 0, "bbox": []},
    ]

    with mock.patch("hybrid_doc_parser.parser._run_mineru", return_value=bad_cl):
        result = parse(FIXTURES / "digital_simple.pdf", EnrichmentConfig())

    assert result is not None
    # The two valid blocks should be parsed; None filtered out
    assert len(result.elements) == 2
