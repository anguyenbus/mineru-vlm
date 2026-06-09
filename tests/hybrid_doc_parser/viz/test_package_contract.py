"""Group 2 package-contract tests (TDD-first, capped at the 2–8 budget).

These lock the four contract points the package refactor must honor:

1. ``to_canonical`` per-mille (MinerU): ÷1000, NO Y-flip.
2. ``to_canonical`` Docling bottom-left points: ÷page points, flip Y.
3. ``doc_from_parser_output`` consumes the Pydantic ``ParserOutput`` (not dict
   keys) and links at ELEMENT granularity — each span carries the source
   ``element_id`` and the canonical bbox.
4. ``schema_version`` mismatch is warned/refused, never mis-mapped.

Edge cases are deferred to Group 4.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hybrid_doc_parser.models import ParserOutput
from hybrid_doc_parser.viz.coords import to_canonical
from hybrid_doc_parser.viz.normalize import doc_from_parser_output

FIXTURES = Path(__file__).resolve().parent / "fixtures"
US_LETTER = (612.0, 792.0)


def _load(name: str) -> ParserOutput:
    return ParserOutput.model_validate_json((FIXTURES / name).read_text())


# --- 1. to_canonical: MinerU per-mille (÷1000, no Y-flip) -------------------


def test_to_canonical_permille_divides_by_1000_no_flip() -> None:
    """MinerU per-mille [250,100,750,200] -> (0.25,0.10,0.75,0.20), no flip."""
    box = to_canonical([250, 100, 750, 200], "mineru")
    assert box == pytest.approx((0.25, 0.10, 0.75, 0.20))
    # No Y-flip: a small top-left y stays a small canonical y.
    assert box[1] < box[3]


# --- 2. to_canonical: Docling bottom-left points + Y-flip -------------------


def test_to_canonical_docling_bottomleft_points_flips_y() -> None:
    """A bottom-of-page Docling bbox maps to a large (near-1) canonical y."""
    box = to_canonical([0.0, 0.0, 612.0, 79.2], "docling", *US_LETTER)
    assert box is not None
    assert box[1] == pytest.approx(0.90, abs=1e-4)
    assert box[3] == pytest.approx(1.0, abs=1e-4)


# --- 3. doc_from_parser_output: Pydantic access + element-granularity link --


def test_doc_from_parser_output_links_at_element_granularity() -> None:
    """Spans carry the source element_id + canonical bbox via Pydantic access."""
    po = _load("us_letter.mineru.json")
    doc = doc_from_parser_output(po, "mineru")

    assert doc.backend == "mineru"
    assert len(doc.spans) == len(po.elements)

    first = doc.spans[0]
    # Element-granularity link: the source element_id is preserved verbatim.
    assert first.element_id == po.elements[0].element_id
    # MinerU per-mille [250,100,750,200] -> canonical heading band.
    assert first.bbox == pytest.approx((0.25, 0.10, 0.75, 0.20))
    # Type came from the Pydantic enum, not a dict key.
    assert first.type == po.elements[0].type.value


def test_doc_from_parser_output_docling_applies_y_flip() -> None:
    """Docling path uses points + page size and applies the Y-flip per element."""
    po = _load("us_letter.docling.json")
    doc = doc_from_parser_output(po, "docling", page_sizes_pt=[US_LETTER])
    first = doc.spans[0]
    assert first.element_id == po.elements[0].element_id
    assert first.bbox == pytest.approx((0.10, 0.08, 0.90, 0.15), abs=1e-4)


# --- 4. schema_version mismatch: warn/refuse, never mis-map -----------------


def test_schema_version_mismatch_is_refused(recwarn) -> None:
    """A stale schema_version yields no drawn spans and emits a warning."""
    po = _load("us_letter.mineru.json").model_copy(update={"schema_version": "9.9"})
    doc = doc_from_parser_output(po, "mineru")
    # Refuse to map: no element spans are produced for a mismatched schema.
    assert all(s.element_id != po.elements[0].element_id for s in doc.spans)
    assert any("schema" in str(w.message).lower() for w in recwarn.list)
