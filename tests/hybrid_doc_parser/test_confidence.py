"""Tests for hybrid_doc_parser.confidence — the confidence aggregation core.

All tests use synthetic ``_middle.json`` dicts (fast, no MinerU run). They
assert the load-bearing correctness rules of ``extract_confidence``:

- SIMPLE (``lines`` -> ``spans``) and HIERARCHICAL (``blocks`` -> sub-block ->
  ``lines`` -> ``spans``) block traversal,
- the score rules (``block["score"] is None`` skipped, scoreless span skipped,
  ``span["score"] == 0.0`` kept, pure-image block has no span score),
- headline source = ``para_blocks`` else ``preproc_blocks`` (never both),
- ``None`` mean/min sentinels when a dimension is empty,
- page flagging on either headline mean below threshold,
- the never-raises contract on malformed/empty/non-pipeline input.

Run in isolation:
    uv run --no-sync pytest tests/hybrid_doc_parser/test_confidence.py -v
"""

from __future__ import annotations

import json

from hybrid_doc_parser.confidence import (
    DEFAULT_LOW_CONFIDENCE_THRESHOLD,
    DocumentConfidence,
    PageConfidence,
    extract_confidence,
    extract_confidence_from_path,
)

# ---------------------------------------------------------------------------
# Synthetic-block builders
# ---------------------------------------------------------------------------


def _span(score: float | None = 0.9, with_key: bool = True) -> dict:
    """A span dict; omit the ``"score"`` key entirely when with_key is False."""
    if not with_key:
        return {"type": "text", "content": "x"}
    return {"type": "text", "content": "x", "score": score}


def _line(*spans: dict) -> dict:
    return {"spans": list(spans)}


def _simple_block(block_score: float | None, *span_scores: float | None) -> dict:
    """A SIMPLE block: ``lines`` -> ``spans``."""
    return {
        "type": "text",
        "score": block_score,
        "lines": [_line(*(_span(s) for s in span_scores))],
    }


def _hierarchical_block(block_score: float | None, *span_scores: float | None) -> dict:
    """A HIERARCHICAL block: ``blocks`` -> sub-block -> ``lines`` -> ``spans``."""
    return {
        "type": "image",
        "score": block_score,
        "blocks": [
            {
                "type": "image_body",
                "lines": [_line(*(_span(s) for s in span_scores))],
            }
        ],
    }


def _page(page_idx: int, **collections: object) -> dict:
    page: dict = {"page_idx": page_idx}
    page.update(collections)
    return page


def _middle(*pages: dict, version: str = "v1", backend: str = "pipeline") -> dict:
    return {
        "pdf_info": list(pages),
        "_version_name": version,
        "_backend": backend,
    }


# ---------------------------------------------------------------------------
# 3.1 Foundation tests: models + score rules + iterators
# ---------------------------------------------------------------------------


def test_simple_block_lines_spans_traversed() -> None:
    """A SIMPLE block's spans are counted via lines -> spans."""
    doc = extract_confidence(_middle(_page(0, para_blocks=[_simple_block(0.8, 0.9, 0.7)])))
    page = doc.pages[0]
    assert page.block_count == 1
    assert page.span_count == 2
    assert page.mean_span_score == 0.8
    assert page.min_span_score == 0.7


def test_hierarchical_block_subblock_spans_traversed() -> None:
    """A HIERARCHICAL block's spans are reached via blocks -> sub-block -> spans."""
    doc = extract_confidence(_middle(_page(0, para_blocks=[_hierarchical_block(0.6, 0.5, 0.95)])))
    page = doc.pages[0]
    assert page.block_count == 1
    assert page.span_count == 2
    assert page.min_span_score == 0.5


def test_block_score_none_skipped_from_block_dimension() -> None:
    """A block with ``score is None`` is excluded from the block dimension."""
    doc = extract_confidence(
        _middle(
            _page(
                0,
                para_blocks=[
                    _simple_block(None, 0.9),
                    _simple_block(0.8, 0.85),
                ],
            )
        )
    )
    page = doc.pages[0]
    # Only the scored block counts; the None-scored one still gives its span.
    assert page.block_count == 1
    assert page.mean_block_score == 0.8
    assert page.span_count == 2


def test_scoreless_span_skipped_and_zero_span_kept() -> None:
    """A span without a ``"score"`` key is skipped; a ``0.0`` span is kept."""
    block = {
        "type": "text",
        "score": 0.9,
        "lines": [
            {"spans": [_span(with_key=False), _span(0.0), _span(0.8)]},
        ],
    }
    doc = extract_confidence(_middle(_page(0, para_blocks=[block])))
    page = doc.pages[0]
    # scoreless span dropped -> 2 kept; the 0.0 span is one of them.
    assert page.span_count == 2
    assert page.min_span_score == 0.0
    assert page.low_confidence_spans == 1  # only the 0.0 span is below 0.70


def test_pure_image_block_contributes_no_span_score() -> None:
    """A pure-image block (no OCR spans) adds 0 to span_count, not low-conf."""
    image_block = {"type": "image", "score": 0.4, "blocks": []}
    doc = extract_confidence(_middle(_page(0, para_blocks=[image_block])))
    page = doc.pages[0]
    assert page.span_count == 0
    assert page.mean_span_score is None
    assert page.low_confidence_spans == 0
    # Block dimension still sees the 0.4 block score.
    assert page.block_count == 1
    assert page.low_confidence_blocks == 1


def test_para_blocks_preferred_no_double_count() -> None:
    """When both para_blocks and preproc_blocks exist, only para_blocks count."""
    blocks = [_simple_block(0.8, 0.9), _simple_block(0.6, 0.7)]
    # Identical counts in both collections — counting both would double everything.
    doc = extract_confidence(_middle(_page(0, para_blocks=blocks, preproc_blocks=blocks)))
    page = doc.pages[0]
    assert page.block_count == 2  # NOT 4
    assert page.span_count == 2  # NOT 4


def test_preproc_fallback_when_para_blocks_empty() -> None:
    """Falls back to preproc_blocks when para_blocks is absent or empty."""
    blocks = [_simple_block(0.8, 0.9)]
    doc_empty = extract_confidence(_middle(_page(0, para_blocks=[], preproc_blocks=blocks)))
    doc_absent = extract_confidence(_middle(_page(0, preproc_blocks=blocks)))
    assert doc_empty.pages[0].block_count == 1
    assert doc_absent.pages[0].block_count == 1


def test_models_are_frozen_and_default_threshold() -> None:
    """Models are frozen Pydantic v2 and the default threshold is 0.70."""
    assert DEFAULT_LOW_CONFIDENCE_THRESHOLD == 0.70
    doc = extract_confidence(_middle(_page(0, para_blocks=[_simple_block(0.8, 0.9)])))
    assert isinstance(doc, DocumentConfidence)
    assert isinstance(doc.pages[0], PageConfidence)
    try:
        doc.pages[0].page_idx = 5  # type: ignore[misc]
        raised = False
    except Exception:
        raised = True
    assert raised


# ---------------------------------------------------------------------------
# 3.6 Synthetic-dict matrix
# ---------------------------------------------------------------------------


def test_discarded_bucket_separate_from_headline() -> None:
    """discarded_blocks populate only discarded_* fields, never the headline."""
    doc = extract_confidence(
        _middle(
            _page(
                0,
                para_blocks=[_simple_block(0.9, 0.95)],
                discarded_blocks=[_simple_block(0.2, 0.1)],
            )
        )
    )
    page = doc.pages[0]
    # Headline sees only the para block.
    assert page.block_count == 1
    assert page.mean_block_score == 0.9
    assert page.low_confidence_blocks == 0
    # Discarded bucket parallel set.
    assert page.discarded_block_count == 1
    assert page.discarded_mean_block_score == 0.2
    assert page.discarded_low_confidence_blocks == 1
    # The bad discarded scores must NOT flag the page.
    assert page.flagged is False


def test_empty_page_yields_none_sentinels_not_zero() -> None:
    """An empty page has count==0 and None mean/min (not 0.0)."""
    doc = extract_confidence(_middle(_page(0)))
    page = doc.pages[0]
    assert page.block_count == 0
    assert page.span_count == 0
    assert page.mean_block_score is None
    assert page.min_block_score is None
    assert page.mean_span_score is None
    assert page.min_span_score is None
    assert page.flagged is False


def test_page_flagged_when_headline_mean_below_threshold() -> None:
    """A page flags when either headline mean is below the threshold."""
    # block mean 0.5 below 0.70 -> flagged; span mean 0.9 above.
    doc = extract_confidence(_middle(_page(0, para_blocks=[_simple_block(0.5, 0.9)])))
    page = doc.pages[0]
    assert page.mean_block_score == 0.5
    assert page.flagged is True


def test_multi_page_document_aggregates_and_pages_flagged_order() -> None:
    """Document means/mins skip None dims; pages_flagged is ordered by page_idx."""
    doc = extract_confidence(
        _middle(
            _page(0, para_blocks=[_simple_block(0.9, 0.9)]),  # not flagged
            _page(1, para_blocks=[_simple_block(0.5, 0.5)]),  # flagged (both low)
            _page(2),  # empty -> None dims, contributes nothing
            _page(3, para_blocks=[_simple_block(0.6, 0.95)]),  # flagged (block low)
            version="ver2",
            backend="pipeline",
        )
    )
    assert doc.total_pages == 4
    # block scores present: 0.9, 0.5, 0.6 -> mean 0.6666..., min 0.5
    assert doc.overall_min_block_score == 0.5
    assert abs(doc.overall_mean_block_score - (0.9 + 0.5 + 0.6) / 3) < 1e-9
    # span scores present: 0.9, 0.5, 0.95
    assert doc.overall_min_span_score == 0.5
    assert doc.pages_flagged == [1, 3]
    assert doc.version_name == "ver2"
    assert doc.backend == "pipeline"
    assert doc.source_path is None


def test_document_counts_summed_across_pages() -> None:
    """Document-wide low_confidence counts are summed across pages."""
    doc = extract_confidence(
        _middle(
            _page(0, para_blocks=[_simple_block(0.5, 0.4)]),
            _page(1, para_blocks=[_simple_block(0.6, 0.65)]),
        )
    )
    assert doc.low_confidence_blocks == 2
    assert doc.low_confidence_spans == 2


def test_model_dump_json_round_trip() -> None:
    """A DocumentConfidence round-trips through model_dump_json()."""
    doc = extract_confidence(_middle(_page(0, para_blocks=[_simple_block(0.8, 0.9)])))
    payload = doc.model_dump_json()
    restored = DocumentConfidence.model_validate_json(payload)
    assert restored == doc


def test_malformed_middle_json_degrades_without_raising() -> None:
    """Missing pdf_info / wrong types degrade to empty/None aggregates."""
    # missing pdf_info
    d1 = extract_confidence({"_version_name": "v", "_backend": "pipeline"})
    assert d1.total_pages == 0
    assert d1.pages == []
    assert d1.overall_mean_block_score is None
    assert d1.pages_flagged == []
    assert d1.version_name == "v"
    assert d1.backend == "pipeline"

    # pdf_info not a list
    d2 = extract_confidence({"pdf_info": "nope"})
    assert d2.total_pages == 0
    assert d2.pages == []

    # blocks not lists / scores non-numeric — must not raise
    bad_page = {
        "page_idx": 0,
        "para_blocks": [
            {"type": "text", "score": "high", "lines": "broken"},
            {"type": "text", "score": 0.8, "lines": [{"spans": "nope"}]},
        ],
    }
    d3 = extract_confidence({"pdf_info": [bad_page]})
    assert d3.total_pages == 1
    # the non-numeric score block is skipped; the 0.8 block is kept.
    assert d3.pages[0].block_count == 1
    assert d3.pages[0].span_count == 0


def test_non_pipeline_empty_input_degrades() -> None:
    """Empty / non-pipeline input gives a well-formed empty DocumentConfidence."""
    d_empty = extract_confidence({})
    assert d_empty.total_pages == 0
    assert d_empty.pages == []
    assert d_empty.overall_mean_block_score is None
    assert d_empty.overall_min_span_score is None
    assert d_empty.low_confidence_blocks == 0
    assert d_empty.pages_flagged == []
    assert d_empty.version_name is None
    assert d_empty.backend is None


def test_extract_confidence_from_path_reads_and_degrades(tmp_path) -> None:
    """from_path parses a real JSON file and degrades (no raise) on bad input."""
    good = tmp_path / "good_middle.json"
    good.write_text(
        json.dumps(_middle(_page(0, para_blocks=[_simple_block(0.8, 0.9)]))),
        encoding="utf-8",
    )
    doc = extract_confidence_from_path(good)
    assert doc.total_pages == 1
    assert doc.pages[0].block_count == 1

    missing = tmp_path / "nope_middle.json"
    d_missing = extract_confidence_from_path(missing)
    assert d_missing.total_pages == 0

    corrupt = tmp_path / "corrupt_middle.json"
    corrupt.write_text("{not json", encoding="utf-8")
    d_corrupt = extract_confidence_from_path(corrupt)
    assert d_corrupt.total_pages == 0
