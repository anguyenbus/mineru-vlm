"""Group 4 edge-case matrix (view.md §7): the viewer degrades, never crashes.

These mirror the library's "``parse()`` never raises" contract: every row of the
edge matrix must produce a report (or a clear in-pane note), NOT a traceback.
Capped at 10 strategic tests covering the highest-risk rows; pure-layer only (no
pypdfium2), so they run in the fast suite.

The render-path rows (0-page / encrypted / corrupt PDF) are exercised through the
pure ``select_pages`` page-selection helper plus the CLI's graceful-render
fallback, neither of which needs a real rasterizer.
"""

from __future__ import annotations

import html as _html
import warnings

from hybrid_doc_parser.models import (
    EnrichmentConfig,
    ElementRecord,
    ElementType,
    ParserOutput,
)
from hybrid_doc_parser.viz.coords import to_canonical
from hybrid_doc_parser.viz.html import build_html
from hybrid_doc_parser.viz.model import Doc, Span
from hybrid_doc_parser.viz.normalize import doc_from_parser_output, doc_from_pdfplumber
from hybrid_doc_parser.viz.render import select_pages

_CFG = EnrichmentConfig()


def _po(elements: list[ElementRecord], schema_version: str = "1.1") -> ParserOutput:
    return ParserOutput(
        schema_version=schema_version,
        file_path="x.pdf",
        file_sha256="a" * 64,
        page_count=1,
        pages=[],
        elements=elements,
        warnings=[],
        enrichment_config=_CFG,
    )


def _el(**kw) -> ElementRecord:
    base = dict(
        element_id="e0",
        type=ElementType.text,
        text="t",
        description="",
        bbox=[10.0, 10.0, 100.0, 50.0],
        page_idx=0,
    )
    base.update(kw)
    return ElementRecord(**base)


# --- Row: empty elements -> empty tab, no error -----------------------------


def test_empty_elements_yields_empty_doc_and_renders() -> None:
    doc = doc_from_parser_output(_po([]), "mineru")
    assert doc.spans == []
    out = build_html(["B64"], {"mineru": doc}, "t")
    assert "panel-mineru" in out  # the tab still renders, no exception


# --- Row: bbox == [] (no geometry) -> listed + labelled, not drawn ----------


def test_no_geometry_span_listed_labelled_not_drawn() -> None:
    doc = doc_from_parser_output(_po([_el(bbox=[])]), "mineru")
    assert doc.spans[0].bbox is None  # to_canonical([]) -> None
    out = build_html(["B64"], {"mineru": doc}, "t")
    assert "no geometry" in out  # labelled in the text pane
    # Not drawn: no overlay box div carries this element id.
    assert 'class="box" data-eid="e0"' not in out


# --- Row: page_idx out of range -> skip box + visible note ------------------


def test_page_idx_out_of_range_is_noted_not_drawn() -> None:
    # One page rendered, element claims page 7.
    span = Span("e0", page=7, type="text", text="orphan", bbox=(0.1, 0.1, 0.4, 0.3))
    out = build_html(["B64"], {"mineru": Doc("mineru", [span])}, "t")
    # No box drawn (only page 0 exists), but the span is visibly noted.
    assert 'class="box" data-eid="e0"' not in out
    assert "out of range" in out.lower()


# --- Row: NaN / inf / negative -> rejected -> boxless span ------------------


def test_pathological_coords_render_span_without_box() -> None:
    # NaN / inf / negative are genuinely rejected by to_canonical (-> None). An
    # INVERTED box (x1<x0) is instead RECOVERED by sorting per the locked Group 1
    # transform, so it is asserted separately, not as a rejection.
    rejected = {
        "nan": [float("nan"), 0.0, 100.0, 50.0],
        "inf": [0.0, 0.0, float("inf"), 50.0],
        "negative": [-5.0, -5.0, 100.0, 50.0],
    }
    for tag, bb in rejected.items():
        assert to_canonical(bb, "mineru") is None, tag
    # Inverted is recovered (sorted), not rejected -> still a usable box.
    assert to_canonical([100.0, 50.0, 10.0, 10.0], "mineru") is not None

    # A genuinely-rejected box flows through to a boxless, labelled text span.
    doc = doc_from_parser_output(_po([_el(bbox=rejected["nan"])]), "mineru")
    assert doc.spans[0].bbox is None
    out = build_html(["B64"], {"mineru": doc}, "t")
    assert "no geometry" in out  # boxless spans fall back to the no-geometry label


# --- Row: image input -> pixels as units, no points math --------------------


def test_image_input_treats_pixels_as_units() -> None:
    # A 400x300 image; a box covering the right half / lower third.
    box = to_canonical([200, 200, 400, 300], "image", 400, 300)
    assert box is not None
    x0, y0, x1, y1 = box
    assert (x0, y0, x1, y1) == (0.5, 200 / 300, 1.0, 1.0)  # pure pixel ratio, no flip


# --- Row: schema_version mismatch -> warn/refuse ----------------------------


def test_schema_mismatch_warns_and_refuses_geometry() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        doc = doc_from_parser_output(_po([_el()], schema_version="9.9"), "mineru")
    assert any("schema" in str(w.message).lower() for w in caught)
    # Refused: the only span is the warning, never the mis-mapped element.
    assert all(s.element_id != "e0" for s in doc.spans)


# --- Row: HTML in document text/description -> escaped, inert ----------------


def test_document_html_is_escaped_and_inert() -> None:
    payload = "<script>alert(1)</script><img src=x onerror=y>"
    doc = doc_from_parser_output(_po([_el(text=payload)]), "mineru")
    out = build_html(["B64"], {"mineru": doc}, payload)
    assert "<script>alert(1)</script>" not in out  # never emitted raw
    assert _html.escape(payload) in out  # present only as inert escaped text


# --- Row: pdfplumber absent/failed -> graceful in-pane note (not a crash) ---


def test_pdfplumber_failure_degrades_to_note() -> None:
    # A non-existent path makes pdfplumber.open raise (or the dep is absent);
    # either way it must degrade to a single warning span, never propagate.
    doc = doc_from_pdfplumber("/no/such/file/at/all.pdf")
    assert doc.backend == "pdfplumber"
    assert len(doc.spans) == 1
    assert doc.spans[0].bbox is None
    assert doc.spans[0].type == "warning"


# --- Row: 0-page / large PDF -> page selection + truncation note (no raster) -


def test_select_pages_caps_and_notes_truncation() -> None:
    idx, note = select_pages(total_pages=120, max_pages=50, pages_spec=None)
    assert idx == list(range(50))
    assert note is not None and "50" in note and "120" in note  # visible truncation note

    # Under the cap: no note.
    idx, note = select_pages(total_pages=3, max_pages=50, pages_spec=None)
    assert idx == [0, 1, 2]
    assert note is None

    # 0-page document: empty selection, no crash, no spurious note.
    idx, note = select_pages(total_pages=0, max_pages=50, pages_spec=None)
    assert idx == []
    assert note is None


def test_select_pages_range_selection() -> None:
    # --pages 3-10 (1-based inclusive) over a 20-page doc -> 0-based 2..9.
    idx, note = select_pages(total_pages=20, max_pages=50, pages_spec="3-10")
    assert idx == [2, 3, 4, 5, 6, 7, 8, 9]
    assert note is None
    # A range exceeding the doc is clamped, not an error.
    idx, _ = select_pages(total_pages=5, max_pages=50, pages_spec="3-99")
    assert idx == [2, 3, 4]


# --- regression: non-prefix page subset must not alias page <-> position -----


def test_page_subset_draws_boxes_on_correct_absolute_page():
    """Non-prefix selection draws each span over its ABSOLUTE page image.

    Regression for the silent-lie bug: with one image rendered for absolute
    page 2, a span on page 2 is drawn while a span on page 0 (not rendered) is
    not drawn and is labelled as not rendered in the text pane.
    """
    page2 = Span(element_id="e-pg2", page=2, type="text", text="on page two",
                 bbox=(0.1, 0.1, 0.4, 0.2))
    page0 = Span(element_id="e-pg0", page=0, type="text", text="on page zero",
                 bbox=(0.1, 0.1, 0.4, 0.2))
    doc = Doc(backend="mineru", spans=[page2, page0])

    # pages_b64 has ONE image whose absolute index is 2.
    out = build_html(["FAKEB64"], {"mineru": doc}, page_indices=[2])

    # The page-2 box is drawn; the page-0 box is not.
    assert 'class="box" data-eid="e-pg2"' in out
    assert 'class="box" data-eid="e-pg0"' not in out
    # Page 0 is reported as not rendered (degrade, don't silently drop).
    assert "page 0 — not rendered" in out
    # And page-0's span is still listed in the text pane.
    assert out.count('data-eid="e-pg0"') >= 1


def test_default_page_indices_unchanged_for_prefix():
    """Omitting page_indices keeps the legacy leading-pages mapping.

    Position == absolute index, so existing full-document reports are unaffected.
    """
    s0 = Span("e0", 0, "text", "p0", (0.1, 0.1, 0.3, 0.2))
    s1 = Span("e1", 1, "text", "p1", (0.1, 0.1, 0.3, 0.2))
    out = build_html(["IMG0", "IMG1"], {"mineru": Doc("mineru", [s0, s1])})
    assert 'class="box" data-eid="e0"' in out
    assert 'class="box" data-eid="e1"' in out
