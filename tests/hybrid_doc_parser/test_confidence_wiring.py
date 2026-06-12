"""Tests for the MinerU confidence wiring into ParserOutput (roadmap item 22).

Group 3 covers the 4-row decision table inside ``_build_parser_output`` (the
single integration point that calls ``extract_confidence``). Group 4 adds the
cache round-trip and a real-fixture end-to-end test.

All tests drive the parse flow at the ``_build_parser_output`` /
``_run_mineru`` mock seam; no real MinerU run.
"""

from __future__ import annotations

import json
import unittest.mock as mock
from pathlib import Path

from hybrid_doc_parser.models import EnrichmentConfig

FIXTURES = Path(__file__).parent / "fixtures"


def _make_pdf(tmp_path: Path, name: str = "doc.pdf") -> Path:
    """Write a tiny fake PDF and return its path."""
    p = tmp_path / name
    p.write_bytes(b"%PDF-1.4 fake content")
    return p


def _build(file_path: Path, config: EnrichmentConfig, middle_json):
    """Call _build_parser_output with no elements; isolate the confidence table."""
    from hybrid_doc_parser.parser import _build_parser_output

    return _build_parser_output(
        file_path,
        sha256="a" * 64,
        elements=[],
        content_list=[],
        config=config,
        middle_json=middle_json,
    )


# ---------------------------------------------------------------------------
# Group 3: the 4-row confidence/warning decision table
# ---------------------------------------------------------------------------


def test_row_a_docling_no_confidence_no_warning(tmp_path, monkeypatch):
    """(a) parser != 'mineru' -> confidence is None, no confidence_unavailable warning."""
    monkeypatch.setenv("HYBRID_DOC_PARSER_CACHE_DIR", str(tmp_path / "cache"))
    pdf = _make_pdf(tmp_path)
    # Even with a usable middle_json, docling never aggregates confidence.
    middle = {"pdf_info": [{"page_idx": 0, "para_blocks": [{"score": 0.9}]}]}
    out = _build(pdf, EnrichmentConfig(parser="docling"), middle)
    assert out.confidence is None
    assert [w for w in out.warnings if w.code == "confidence_unavailable"] == []


def test_row_b_mineru_absent_middle_json_silent(tmp_path, monkeypatch):
    """(b) parser == 'mineru', middle_json is None -> confidence None, NO warning."""
    monkeypatch.setenv("HYBRID_DOC_PARSER_CACHE_DIR", str(tmp_path / "cache"))
    pdf = _make_pdf(tmp_path)
    out = _build(pdf, EnrichmentConfig(parser="mineru"), None)
    assert out.confidence is None
    assert [w for w in out.warnings if w.code == "confidence_unavailable"] == []


def test_row_c_mineru_unusable_middle_json_warns(tmp_path, monkeypatch):
    """(c) parser == 'mineru', present-but-unusable -> confidence None + one warning."""
    monkeypatch.setenv("HYBRID_DOC_PARSER_CACHE_DIR", str(tmp_path / "cache"))
    pdf = _make_pdf(tmp_path)
    from hybrid_doc_parser.parser import _BATCH_FAILURE_CODES

    for unusable in ({}, {"pdf_info": []}):
        out = _build(pdf, EnrichmentConfig(parser="mineru"), unusable)
        assert out.confidence is None
        warns = [w for w in out.warnings if w.code == "confidence_unavailable"]
        assert len(warns) == 1
        # Must NEVER mark the item as a batch failure.
        assert "confidence_unavailable" not in _BATCH_FAILURE_CODES
        assert all(w.code not in _BATCH_FAILURE_CODES for w in warns)


def test_row_d_mineru_usable_middle_json_populates(tmp_path, monkeypatch):
    """(d) parser == 'mineru', usable middle_json -> confidence set, source_path set, no warning."""
    monkeypatch.setenv("HYBRID_DOC_PARSER_CACHE_DIR", str(tmp_path / "cache"))
    pdf = _make_pdf(tmp_path)
    middle = {
        "_backend": "pipeline",
        "pdf_info": [
            {
                "page_idx": 0,
                "para_blocks": [
                    {
                        "score": 0.95,
                        "lines": [{"spans": [{"score": 0.92}]}],
                    }
                ],
            }
        ],
    }
    out = _build(pdf, EnrichmentConfig(parser="mineru"), middle)
    assert out.confidence is not None
    assert out.confidence.total_pages == 1
    assert out.confidence.source_path == str(pdf)
    assert out.confidence.backend == "pipeline"
    assert [w for w in out.warnings if w.code == "confidence_unavailable"] == []


def test_row_c_not_a_batch_failure_via_parse_batch(tmp_path, monkeypatch):
    """confidence_unavailable on the batch path must not mark the item failed."""
    import asyncio

    monkeypatch.setenv("HYBRID_DOC_PARSER_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("MINERU_BATCH_SIZE", "8")
    from hybrid_doc_parser.parser import _BATCH_FAILURE_CODES, parse_batch

    pdf = FIXTURES / "digital_simple.pdf"

    def _chunk(chunk_paths, name_map, backend):
        # Real content_list (so the file is not a do_parse miss) + unusable middle_json.
        from test_integration import _fake_content_list  # noqa: PLC0415

        return {p: (_fake_content_list(1), {"pdf_info": []}) for p in chunk_paths}

    with mock.patch("hybrid_doc_parser.parser._run_mineru_batch_chunk", side_effect=_chunk):
        results = asyncio.run(parse_batch([pdf], EnrichmentConfig(parser="mineru")))

    assert len(results) == 1
    out = results[0]
    assert out.confidence is None
    warns = [w for w in out.warnings if w.code == "confidence_unavailable"]
    assert len(warns) == 1
    # Not a batch failure.
    assert not any(w.code in _BATCH_FAILURE_CODES for w in out.warnings)


# ---------------------------------------------------------------------------
# Group 4: cache round-trip + real-fixture end-to-end
# ---------------------------------------------------------------------------


def test_cache_round_trip_preserves_confidence(tmp_path, monkeypatch):
    """Parse -> cache.put -> cache.get preserves the nested confidence deep-equal."""
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("HYBRID_DOC_PARSER_CACHE_DIR", str(cache_dir))
    from hybrid_doc_parser import cache

    pdf = _make_pdf(tmp_path)
    config = EnrichmentConfig(parser="mineru")
    middle = {
        "_backend": "pipeline",
        "pdf_info": [
            {
                "page_idx": 0,
                "para_blocks": [{"score": 0.88, "lines": [{"spans": [{"score": 0.81}]}]}],
            }
        ],
    }
    out = _build(pdf, config, middle)
    assert out.confidence is not None

    # _build_parser_output already wrote to cache; read it back with the SAME config.
    cached = cache.get(pdf, config)
    assert cached is not None
    assert cached.confidence is not None
    # Deep-equal, including source_path and nested per-page aggregates.
    assert cached.confidence == out.confidence
    assert cached.confidence.source_path == str(pdf)
    assert cached.confidence.pages == out.confidence.pages


def test_real_fixture_end_to_end_via_parse(tmp_path, monkeypatch):
    """Monkeypatch the runner to return the real _middle.json fixture; parse() populates confidence."""
    monkeypatch.setenv("HYBRID_DOC_PARSER_CACHE_DIR", str(tmp_path / "cache"))
    from hybrid_doc_parser import parse
    from test_integration import _fake_content_list  # noqa: PLC0415

    middle = json.loads((FIXTURES / "confidence_middle.json").read_text(encoding="utf-8"))
    content_list = _fake_content_list(1)

    pdf = FIXTURES / "digital_simple.pdf"
    with mock.patch(
        "hybrid_doc_parser.parser._run_mineru",
        return_value=(content_list, middle),
    ):
        result = parse(pdf, EnrichmentConfig(parser="mineru"))

    assert result.confidence is not None
    assert result.confidence.total_pages > 0
    assert result.confidence.backend == "pipeline"
    assert result.confidence.source_path == str(pdf)
    assert [w for w in result.warnings if w.code == "confidence_unavailable"] == []

    # The populated confidence survives a full model_dump_json round-trip.
    from hybrid_doc_parser.models import ParserOutput

    restored = ParserOutput.model_validate_json(result.model_dump_json())
    assert restored.confidence == result.confidence
