"""Real-input + live integration tests for the confidence aggregation core.

These complement the synthetic-dict matrix in ``test_confidence.py`` (which
covers the correctness rules) by exercising the core against *real* MinerU
``_middle.json`` structure rather than hand-built dicts:

- A FAST test over a committed real ``_middle.json`` fixture (generated once by
  running the MinerU pipeline on the tiny ``digital_simple.pdf`` and committed
  under ``fixtures/``). It asserts ``extract_confidence_from_path`` yields a
  well-formed :class:`DocumentConfidence` with sane aggregates and round-trips
  through ``model_dump_json()``.
- A LIVE schema-drift canary gated by ``MINERU_CONFIDENCE_INTEGRATION=1`` and
  marked ``@pytest.mark.slow``. When enabled it runs MinerU end-to-end through
  the Group-2 capture path and asserts the freshly captured ``_middle.json``
  still parses into a :class:`DocumentConfidence` — the guard against MinerU
  ``>=3.2.3`` schema drift. It is skipped by default so the fast suite never
  invokes MinerU.

Run the fast test in isolation::

    uv run --no-sync pytest tests/hybrid_doc_parser/test_confidence_integration.py -v

Run the live canary explicitly::

    MINERU_CONFIDENCE_INTEGRATION=1 uv run --no-sync pytest \
        tests/hybrid_doc_parser/test_confidence_integration.py -m slow -v
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
from hybrid_doc_parser.confidence import (
    DocumentConfidence,
    extract_confidence,
    extract_confidence_from_path,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"
REAL_MIDDLE_JSON = FIXTURES_DIR / "confidence_middle.json"

# Env-gating for the live MinerU canary; mirrors the suite's marker convention.
_INTEGRATION_ENABLED = os.environ.get("MINERU_CONFIDENCE_INTEGRATION") == "1"


def _assert_mean_in_unit_or_none(value: float | None) -> None:
    """A mean/min aggregate is either ``None`` (no data) or within ``[0, 1]``."""
    assert value is None or 0.0 <= value <= 1.0


# ---------------------------------------------------------------------------
# Fast real-input test (committed fixture)
# ---------------------------------------------------------------------------


def test_extract_confidence_from_real_fixture() -> None:
    """A committed real ``_middle.json`` yields well-formed, sane aggregates."""
    assert REAL_MIDDLE_JSON.is_file(), (
        f"missing committed fixture {REAL_MIDDLE_JSON}; regenerate it by running "
        "MinerU on tests/hybrid_doc_parser/fixtures/digital_simple.pdf"
    )

    doc = extract_confidence_from_path(REAL_MIDDLE_JSON)

    assert isinstance(doc, DocumentConfidence)
    assert doc.total_pages > 0
    assert len(doc.pages) == doc.total_pages
    # The pipeline dump records its provenance.
    assert doc.backend == "pipeline"
    assert isinstance(doc.version_name, str) and doc.version_name

    # At least one page carries real block scores from layout detection.
    assert any(p.block_count > 0 for p in doc.pages)

    # Overall aggregates are either None (no data) or genuine probabilities.
    for value in (
        doc.overall_mean_block_score,
        doc.overall_min_block_score,
        doc.overall_mean_span_score,
        doc.overall_min_span_score,
    ):
        _assert_mean_in_unit_or_none(value)

    # Per-page aggregates obey the same invariant.
    for page in doc.pages:
        _assert_mean_in_unit_or_none(page.mean_block_score)
        _assert_mean_in_unit_or_none(page.min_block_score)
        _assert_mean_in_unit_or_none(page.mean_span_score)
        _assert_mean_in_unit_or_none(page.min_span_score)

    # Pydantic round-trip on the real-input result.
    restored = DocumentConfidence.model_validate_json(doc.model_dump_json())
    assert restored == doc


# ---------------------------------------------------------------------------
# Live schema-drift canary (MINERU_CONFIDENCE_INTEGRATION=1, @slow)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(
    not _INTEGRATION_ENABLED,
    reason="set MINERU_CONFIDENCE_INTEGRATION=1 to run the live MinerU canary",
)
def test_live_mineru_middle_json_parses_into_confidence() -> None:
    """Run MinerU end-to-end and assert the captured middle_json still parses.

    Schema-drift canary: exercises the Group-2 capture path
    (``_read_middle_json_for_stem`` over a real ``do_parse`` dump) and feeds the
    captured dict to :func:`extract_confidence`. A MinerU ``>=3.2.3`` change to
    the ``_middle.json`` shape that broke aggregation would surface here.
    """
    from hybrid_doc_parser.parser import _read_middle_json_for_stem  # noqa: PLC0415
    from mineru.cli.common import do_parse, read_fn  # noqa: PLC0415

    pdf = FIXTURES_DIR / "digital_simple.pdf"
    pdf_bytes = read_fn(pdf)
    name = pdf.stem

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        do_parse(
            output_dir=str(out_dir),
            pdf_file_names=[name],
            pdf_bytes_list=[pdf_bytes],
            p_lang_list=["en"],
            backend="pipeline",
            parse_method="auto",
            f_draw_layout_bbox=False,
            f_draw_span_bbox=False,
            f_dump_md=False,
            f_dump_middle_json=True,
            f_dump_model_output=False,
            f_dump_orig_pdf=False,
            f_dump_content_list=True,
        )
        middle_json = _read_middle_json_for_stem(out_dir, name)

    assert middle_json is not None, "captured _middle.json was not read back"
    doc = extract_confidence(middle_json)
    assert isinstance(doc, DocumentConfidence)
    assert doc.total_pages > 0
    assert doc.backend == "pipeline"
