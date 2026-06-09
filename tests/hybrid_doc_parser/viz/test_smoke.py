"""``smoke`` (view.md §9, under the ``slow`` marker): the ONE test that touches
the real ``pypdfium2`` render path end-to-end.

It rasterizes a tiny real fixture PDF (``us_letter.pdf`` from Group 1) through the
actual ``viz.render`` pipeline, maps the saved ``ParserOutput`` goldens, and
asserts every canonical box falls inside ``[0, 1]`` and the expected span count
comes through. The whole-pipeline render check that the fast offline suite cannot
do.

Marked ``slow`` so the fast CI suite (golden / snapshot / isolation) stays OFFLINE
— no PDF rendering there. Run it with ``-m slow``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hybrid_doc_parser.models import ParserOutput
from hybrid_doc_parser.viz.normalize import doc_from_parser_output

FIXTURES = Path(__file__).resolve().parent / "fixtures"

pytestmark = pytest.mark.slow


def _load(name: str) -> ParserOutput:
    return ParserOutput.model_validate_json((FIXTURES / name).read_text())


@pytest.fixture(scope="module")
def rendered():
    """Rasterize us_letter.pdf via the real pypdfium2 path (the slow step)."""
    pdfium = pytest.importorskip("pypdfium2")
    pytest.importorskip("PIL")
    assert pdfium  # used implicitly by render_pages
    from hybrid_doc_parser.viz.render import render_pages

    imgs, sizes = render_pages(FIXTURES / "us_letter.pdf")
    return imgs, sizes


def test_render_produces_one_letter_page(rendered) -> None:
    """The fixture renders to a single 612x792 (US-Letter, points) page."""
    imgs, sizes = rendered
    assert len(imgs) == 1
    assert all(isinstance(b, str) and b for b in imgs)  # base64 PNG strings
    (w, h), = sizes
    assert (round(w), round(h)) == (612, 792)


def test_mineru_smoke_all_boxes_within_unit_square(rendered) -> None:
    """All MinerU canonical boxes land inside [0,1]; expected span count comes through."""
    po = _load("us_letter.mineru.json")
    doc = doc_from_parser_output(po, "mineru")  # per-mille needs no page size
    assert len(doc.spans) == len(po.elements) == 3
    boxed = [s for s in doc.spans if s.bbox is not None]
    assert len(boxed) == 3  # every element in the fixture has usable geometry
    for s in boxed:
        x0, y0, x1, y1 = s.bbox
        assert 0.0 <= x0 < x1 <= 1.0, s.bbox
        assert 0.0 <= y0 < y1 <= 1.0, s.bbox


def test_docling_smoke_all_boxes_within_unit_square(rendered) -> None:
    """Docling boxes use the rendered page size + Y-flip and stay inside [0,1]."""
    _, sizes = rendered
    po = _load("us_letter.docling.json")
    doc = doc_from_parser_output(po, "docling", page_sizes_pt=sizes)
    assert len(doc.spans) == len(po.elements) == 2
    boxed = [s for s in doc.spans if s.bbox is not None]
    assert len(boxed) == 2
    for s in boxed:
        x0, y0, x1, y1 = s.bbox
        assert 0.0 <= x0 < x1 <= 1.0, s.bbox
        assert 0.0 <= y0 < y1 <= 1.0, s.bbox
