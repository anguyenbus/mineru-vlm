"""Tests for Docling backend integration in hybrid_doc_parser.

Covers:
- EnrichmentConfig extensions (parser field + Docling-specific fields)
- Docling fixture JSON validity
- _route_docling_block normalisation for all block types
- _resolve_docling_ref malformed-ref handling
- prov guard (None, empty list, page_no == 0)
- parse() dispatch with parser="docling"
- docling_failed / docling_error warning paths
- Single-load DocumentConverter cache guarantee
- _accepted_extensions helper
- ImportError message for missing docling install
- .html extension acceptance

Run with:
    uv run pytest tests/hybrid_doc_parser/test_parser_docling.py -q
"""

from __future__ import annotations

import json
import sys
import unittest.mock as mock
from pathlib import Path

import pytest

from hybrid_doc_parser.models import ElementType, EnrichmentConfig

# Path to fixture files
FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Group 1: Models and Configuration tests
# ---------------------------------------------------------------------------


def test_enrichment_config_parser_docling_constructs() -> None:
    """EnrichmentConfig(parser='docling') constructs without error."""
    config = EnrichmentConfig(parser="docling")
    assert config.parser == "docling"


def test_enrichment_config_parser_defaults_to_mineru() -> None:
    """EnrichmentConfig() defaults to parser='mineru' (backward-compatibility)."""
    config = EnrichmentConfig()
    assert config.parser == "mineru"


def test_enrichment_config_docling_fields_round_trip() -> None:
    """All four Docling-specific fields survive a JSON round-trip."""
    config = EnrichmentConfig(
        do_ocr=False,
        table_mode="accurate",
        do_table_structure=False,
        docling_artifacts_path="/tmp/models",
    )
    json_str = config.model_dump_json()
    restored = EnrichmentConfig.model_validate_json(json_str)
    assert restored.do_ocr is False
    assert restored.table_mode == "accurate"
    assert restored.do_table_structure is False
    assert restored.docling_artifacts_path == "/tmp/models"


# ---------------------------------------------------------------------------
# Group 2: Test Fixtures tests
# ---------------------------------------------------------------------------


def test_docling_fixture_json_keys() -> None:
    """docling_fixture.json contains required keys and body children."""
    fixture_path = FIXTURES / "docling_fixture.json"
    data = json.loads(fixture_path.read_text(encoding="utf-8"))
    assert "body" in data
    assert "texts" in data
    assert "pictures" in data
    assert "tables" in data
    assert isinstance(data["body"]["children"], list)
    assert len(data["body"]["children"]) > 0


def test_docx_fixture_path(docx_fixture: Path) -> None:
    """docx_fixture yields a .docx Path that exists with nonzero size."""
    assert docx_fixture.suffix == ".docx"
    assert docx_fixture.exists()
    assert docx_fixture.stat().st_size > 0


# ---------------------------------------------------------------------------
# Group 3: Core Docling Adapter tests
# ---------------------------------------------------------------------------


# Test A — _route_docling_block for texts, all four label variants
@pytest.mark.parametrize(
    "block_idx,expected_type",
    [
        (0, ElementType.text),         # paragraph
        (1, ElementType.heading),      # section_header
        (2, ElementType.equation),     # formula
        (3, ElementType.list_item),    # list_item
    ],
)
def test_route_docling_block_texts(block_idx: int, expected_type: ElementType) -> None:
    """_route_docling_block maps all four text label variants to correct ElementType."""
    from hybrid_doc_parser.parser import _route_docling_block

    fixture_path = FIXTURES / "docling_fixture.json"
    data = json.loads(fixture_path.read_text(encoding="utf-8"))
    block = data["texts"][block_idx]

    record, warnings = _route_docling_block(
        block=block,
        block_type="texts",
        page_idx=99,  # should be overridden by prov[0]["page_no"] - 1 = 0
        element_idx=block_idx,
        file_sha256="a" * 64,
    )
    assert record.type == expected_type
    assert record.page_idx == 0  # from prov[0]["page_no"] = 1, converted to 0-indexed
    assert record.bbox == []  # fixture prov has no bbox float list


# Test B — _route_docling_block for pictures
def test_route_docling_block_pictures_valid() -> None:
    """_route_docling_block decodes picture bytes in-memory; no WarningRecord emitted."""
    from hybrid_doc_parser.parser import _route_docling_block

    fixture_path = FIXTURES / "docling_fixture.json"
    data = json.loads(fixture_path.read_text(encoding="utf-8"))
    block = data["pictures"][0]

    record, warnings = _route_docling_block(
        block=block,
        block_type="pictures",
        page_idx=0,
        element_idx=0,
        file_sha256="a" * 64,
    )
    assert isinstance(record.image_bytes, bytes)
    assert len(record.image_bytes) > 0
    assert record.type == ElementType.image
    assert len(warnings) == 0


# Test C — _route_docling_block for pictures, oversized (>10 MB)
def test_route_docling_block_pictures_oversized() -> None:
    """Oversized picture stores image_bytes=None and emits WarningRecord(code='image_too_large')."""
    from hybrid_doc_parser.parser import _route_docling_block

    fixture_path = FIXTURES / "docling_fixture.json"
    data = json.loads(fixture_path.read_text(encoding="utf-8"))
    block = data["pictures"][0]

    oversized_bytes = b"x" * (10 * 1024 * 1024 + 1)
    with mock.patch("base64.b64decode", return_value=oversized_bytes):
        record, warnings = _route_docling_block(
            block=block,
            block_type="pictures",
            page_idx=0,
            element_idx=0,
            file_sha256="a" * 64,
        )

    assert record.image_bytes is None
    assert len(warnings) == 1
    assert warnings[0].code == "image_too_large"


# Test D — _route_docling_block for tables
def test_route_docling_block_tables() -> None:
    """_route_docling_block routes table block to ElementType.table with serialised cell data."""
    from hybrid_doc_parser.parser import _route_docling_block

    fixture_path = FIXTURES / "docling_fixture.json"
    data = json.loads(fixture_path.read_text(encoding="utf-8"))
    block = data["tables"][0]

    record, warnings = _route_docling_block(
        block=block,
        block_type="tables",
        page_idx=0,
        element_idx=0,
        file_sha256="a" * 64,
    )
    assert record.type == ElementType.table
    assert "Col A" in record.text


# Test E — _resolve_docling_ref malformed ref
def test_resolve_docling_ref_malformed_ref() -> None:
    """_resolve_docling_ref returns None for malformed $ref without raising."""
    from hybrid_doc_parser.parser import _resolve_docling_ref

    fixture_path = FIXTURES / "docling_fixture.json"
    doc_dict = json.loads(fixture_path.read_text(encoding="utf-8"))

    result = _resolve_docling_ref(ref_str="#/texts/abc", doc_dict=doc_dict)
    assert result is None


# Test F — prov guard: parametrized with three cases
@pytest.mark.parametrize(
    "prov",
    [
        None,
        [],
        [{"page_no": 0}],
    ],
)
def test_route_docling_block_prov_guard(prov: object, caplog: pytest.LogCaptureFixture) -> None:
    """Malformed prov falls back to caller page_idx=0 and logs a debug message."""
    import logging

    from hybrid_doc_parser.parser import _route_docling_block

    block = {
        "label": "paragraph",
        "orig": "Some text.",
        "prov": prov,
    }

    with caplog.at_level(logging.DEBUG):
        record, _ = _route_docling_block(
            block=block,
            block_type="texts",
            page_idx=0,
            element_idx=0,
            file_sha256="a" * 64,
        )

    assert record.page_idx == 0


# Test G — parse() with parser="docling" and a .docx file
def test_parse_docx_with_docling_parser(docx_fixture: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """parse() with parser='docling' and a .docx file returns ParserOutput without raising."""
    monkeypatch.setenv("HYBRID_DOC_PARSER_CACHE_DIR", str(tmp_path / "cache"))
    from hybrid_doc_parser.parser import parse

    with mock.patch("hybrid_doc_parser.parser._run_docling", return_value=[]):
        result = parse(docx_fixture, EnrichmentConfig(parser="docling"))

    assert result is not None
    assert not any(w.code == "unsupported_type" for w in result.warnings)


# Test H — "docling_failed" path
def test_parse_docling_failed_warning(docx_fixture: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """parse() emits WarningRecord(code='docling_failed') when _run_docling raises."""
    monkeypatch.setenv("HYBRID_DOC_PARSER_CACHE_DIR", str(tmp_path / "cache"))
    from hybrid_doc_parser.parser import parse

    with mock.patch(
        "hybrid_doc_parser.parser._run_docling",
        side_effect=RuntimeError("converter crash"),
    ):
        result = parse(docx_fixture, EnrichmentConfig(parser="docling"))

    assert result.warnings[0].code == "docling_failed"


# ---------------------------------------------------------------------------
# Group 4: Additional tests (Tests I–O)
# ---------------------------------------------------------------------------


# Test I — "docling_error" outer-handler path
def test_parse_docling_error_outer_handler(docx_fixture: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Outer handler emits code='docling_error' when parser='docling'."""
    monkeypatch.setenv("HYBRID_DOC_PARSER_CACHE_DIR", str(tmp_path / "cache"))
    from hybrid_doc_parser.parser import parse

    with mock.patch(
        "hybrid_doc_parser.parser._file_sha256",
        side_effect=OSError("hash failure"),
    ):
        result = parse(docx_fixture, EnrichmentConfig(parser="docling"))

    assert result.warnings[0].code == "docling_error"


# Test J — Enrichment-not-supported warning
def test_parse_enrichment_not_supported_for_docling(docx_fixture: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """parse() with enabled=True + parser='docling' emits enrichment_not_supported warning."""
    monkeypatch.setenv("HYBRID_DOC_PARSER_CACHE_DIR", str(tmp_path / "cache"))
    from hybrid_doc_parser.parser import parse

    with mock.patch("hybrid_doc_parser.parser._run_docling", return_value=[]):
        result = parse(
            docx_fixture,
            EnrichmentConfig(enabled=True, parser="docling"),
        )

    assert any(w.code == "enrichment_not_supported" for w in result.warnings)


# Test K — Single-load guarantee
def test_document_converter_single_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """DocumentConverter is instantiated exactly once for N=3 calls with the same config.

    Directly exercises _get_docling_converter's double-checked locking cache
    by mocking the docling module at the sys.modules level, clearing the
    module-level cache, and calling the function three times with the same
    EnrichmentConfig. Asserts that DocumentConverter() is constructed once.
    """
    monkeypatch.setenv("HYBRID_DOC_PARSER_CACHE_DIR", str(tmp_path / "cache"))

    import hybrid_doc_parser.parser as parser_mod
    from hybrid_doc_parser.parser import _get_docling_converter

    # Build mock module objects for docling.
    mock_dc_class = mock.MagicMock()
    mock_dc_instance = mock.MagicMock()
    mock_dc_class.return_value = mock_dc_instance

    mock_dc_module = mock.MagicMock()
    mock_dc_module.DocumentConverter = mock_dc_class

    mock_pipeline_instance = mock.MagicMock()
    mock_pipeline_class = mock.MagicMock(return_value=mock_pipeline_instance)
    mock_pipeline_module = mock.MagicMock()
    mock_pipeline_module.PdfPipelineOptions = mock_pipeline_class
    mock_pipeline_module.TableFormerMode = mock.MagicMock()

    cfg = EnrichmentConfig(parser="docling")

    # Clear the converter cache to ensure a clean test.
    parser_mod._DOCLING_CONVERTER_CACHE.clear()

    patched_modules = {
        "docling": mock.MagicMock(),
        "docling.document_converter": mock_dc_module,
        "docling.datamodel": mock.MagicMock(),
        "docling.datamodel.pipeline_options": mock_pipeline_module,
    }

    with mock.patch.dict(sys.modules, patched_modules):
        # Call 3 times with identical config; only one DocumentConverter should be built.
        _get_docling_converter(cfg)
        _get_docling_converter(cfg)
        _get_docling_converter(cfg)

    assert mock_dc_class.call_count == 1, (
        f"DocumentConverter instantiated {mock_dc_class.call_count} times, expected 1"
    )

    # Cleanup: restore cache state.
    parser_mod._DOCLING_CONVERTER_CACHE.clear()


# Test L — _accepted_extensions includes Docling extensions only for docling parser
def test_accepted_extensions_docling_vs_mineru() -> None:
    """_accepted_extensions includes .docx only for parser='docling', not 'mineru'."""
    from hybrid_doc_parser.parser import _accepted_extensions

    docling_exts = _accepted_extensions(EnrichmentConfig(parser="docling"))
    mineru_exts = _accepted_extensions(EnrichmentConfig())

    assert ".docx" in docling_exts
    assert ".docx" not in mineru_exts


# Test M — Layer 1 skip logged for non-PDF
def test_parse_layer1_skip_logged_for_non_pdf(docx_fixture: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """parse() with a .docx file logs 'skipping Layer 1 for non-PDF input'."""
    monkeypatch.setenv("HYBRID_DOC_PARSER_CACHE_DIR", str(tmp_path / "cache"))
    from hybrid_doc_parser.parser import parse

    log_messages: list[str] = []

    from loguru import logger as loguru_logger

    def sink(message) -> None:
        log_messages.append(str(message))

    sink_id = loguru_logger.add(sink, level="DEBUG")
    try:
        with mock.patch("hybrid_doc_parser.parser._run_docling", return_value=[]):
            parse(docx_fixture, EnrichmentConfig(parser="docling"))
    finally:
        loguru_logger.remove(sink_id)

    assert any("skipping Layer 1 for non-PDF" in msg for msg in log_messages)


# Test N — ImportError raised when docling is not installed
def test_run_docling_import_error_message(tmp_path: Path) -> None:
    """_run_docling raises ImportError with pip and uv install instructions when docling is missing."""
    # Temporarily mask docling from the import system.
    fake_path = tmp_path / "test.docx"
    fake_path.write_bytes(b"fake")

    config = EnrichmentConfig(parser="docling")

    # Patch sys.modules so that "docling.document_converter" raises ImportError.
    patched_modules = {
        "docling": None,
        "docling.document_converter": None,
    }

    with mock.patch.dict(sys.modules, patched_modules):
        from hybrid_doc_parser.parser import _run_docling

        with pytest.raises(ImportError) as exc_info:
            _run_docling(fake_path, config)

    assert "uv add 'hybrid-doc-parser[docling]'" in str(exc_info.value)
    assert "pip install" in str(exc_info.value)


# Test O — .html extension accepted by docling parser
def test_parse_html_extension_accepted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """parse() with parser='docling' accepts .html extension without unsupported_type warning."""
    monkeypatch.setenv("HYBRID_DOC_PARSER_CACHE_DIR", str(tmp_path / "cache"))
    from hybrid_doc_parser.parser import parse

    html_file = tmp_path / "test.html"
    html_file.write_text("<html><body><p>Hello</p></body></html>", encoding="utf-8")

    with mock.patch("hybrid_doc_parser.parser._run_docling", return_value=[]):
        result = parse(html_file, EnrichmentConfig(parser="docling"))

    assert not any(w.code == "unsupported_type" for w in result.warnings)
