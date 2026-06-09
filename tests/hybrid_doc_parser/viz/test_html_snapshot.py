"""``html_snapshot`` (view.md §9): the ``--selftest`` / ``build_html`` output is
STRUCTURALLY stable against the ``demo.html`` baseline.

This guards rendering regressions WITHOUT a brittle full-string match: it asserts
the structural invariants the layout depends on — the tab set
(mineru/docling/pdfplumber), the per-backend overlay-box counts, the span counts,
and the no-geometry handling (listed in the text pane, NOT drawn). A cosmetic CSS
or whitespace edit must not fail this test; a dropped tab, a mis-counted box, or a
no-geometry span that leaks into a drawn ``.box`` must.

The baseline is the committed ``demo.html`` (the prototype's ``--selftest``
output). The selftest emitter is reproduced here through ``build_html`` so the
test stays OFFLINE — no PDF rendering, no pypdfium2/pdfplumber.
"""

from __future__ import annotations

import collections
import re
from pathlib import Path

import pytest

from hybrid_doc_parser.viz.html import build_html
from hybrid_doc_parser.viz.model import Doc, Span


def _find_demo_html() -> Path | None:
    """Locate the committed demo.html by walking up from this test file."""
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "demo.html"
        if candidate.exists():
            return candidate
    return None


DEMO_HTML = _find_demo_html()

# The locked structural baseline (matches demo.html). Docling drops one box (it
# "missed the table") and adds a no-geometry span, so it has 3 boxes / 4 spans.
EXPECTED_TABS = ["docling", "mineru", "pdfplumber"]
EXPECTED_BOX_COUNTS = {"mineru": 4, "docling": 3, "pdfplumber": 4}
EXPECTED_SPAN_COUNTS = {"mineru": 4, "docling": 4, "pdfplumber": 4}
EXPECTED_NO_GEOMETRY = 1


# ---------------------------------------------------------------------------
# Structural helpers — count the load-bearing markers, not the whole string.
# ---------------------------------------------------------------------------


def _tabs(html: str) -> list[str]:
    return sorted(set(re.findall(r'<button class="tab[^"]*" data-tab="([^"]+)"', html)))


def _panels(html: str) -> list[str]:
    return sorted(set(re.findall(r'id="panel-([^"]+)"', html)))


def _box_counts(html: str) -> dict[str, int]:
    eids = re.findall(r'class="box" data-eid="([^"]+)"', html)
    return dict(collections.Counter(e.rsplit("-", 1)[0] for e in eids))


def _span_counts(html: str) -> dict[str, int]:
    eids = re.findall(r'class="span(?: nogeo)?" data-eid="([^"]+)"', html)
    return dict(collections.Counter(e.rsplit("-", 1)[0] for e in eids))


def _selftest_docs() -> dict[str, Doc]:
    """Reproduce the ``--selftest`` doc set (mirrors scripts/parse_report.py).

    Built from canonical 0..1 boxes directly so no rasterizer is needed.
    """
    raw = [
        ("heading", "DEMO PAGE", (0.07, 0.05, 0.93, 0.11)),
        ("text", "Intro paragraph.", (0.07, 0.15, 0.93, 0.29)),
        ("text", "Left column body.", (0.07, 0.33, 0.48, 0.64)),
        ("table", "Right column table.", (0.52, 0.33, 0.93, 0.64)),
    ]

    def spans(backend: str, drop_one: bool, no_geo: bool) -> list[Span]:
        rows = raw[:-1] if drop_one else raw
        out = [
            Span(f"{backend}-{i}", 0, t, txt, bb) for i, (t, txt, bb) in enumerate(rows)
        ]
        if no_geo:  # a no-geometry unit, like a DOCX element
            out.append(Span(f"{backend}-x", 0, "text", "No bbox (docx-style).", None))
        return out

    return {
        "mineru": Doc("mineru", spans("mineru", drop_one=False, no_geo=False)),
        "docling": Doc("docling", spans("docling", drop_one=True, no_geo=True)),
        "pdfplumber": Doc("pdfplumber", spans("pdfplumber", drop_one=False, no_geo=False)),
    }


# ---------------------------------------------------------------------------
# The baseline file exists and is itself structurally what we assert against.
# ---------------------------------------------------------------------------


def test_demo_html_baseline_is_structurally_what_we_lock() -> None:
    """The committed demo.html baseline carries the locked structure."""
    assert DEMO_HTML is not None and DEMO_HTML.exists(), "missing snapshot baseline demo.html"
    html = DEMO_HTML.read_text()
    assert _tabs(html) == EXPECTED_TABS
    assert _panels(html) == EXPECTED_TABS
    assert _box_counts(html) == EXPECTED_BOX_COUNTS
    assert _span_counts(html) == EXPECTED_SPAN_COUNTS
    assert html.count("no geometry") == EXPECTED_NO_GEOMETRY


# ---------------------------------------------------------------------------
# build_html output is structurally stable vs the baseline.
# ---------------------------------------------------------------------------


def test_build_html_tab_set_matches_baseline() -> None:
    """Same three tabs (and panels), in the locked backend order."""
    html = build_html(["B64"], _selftest_docs(), "Self-test")
    assert _tabs(html) == EXPECTED_TABS
    assert _panels(html) == EXPECTED_TABS
    # Tab order in the source string is the fixed mineru/docling/pdfplumber order.
    order = re.findall(r'data-tab="([^"]+)"', html)
    assert order == ["mineru", "docling", "pdfplumber"]


def test_build_html_per_backend_box_counts_match_baseline() -> None:
    """Per-backend overlay-box counts are stable (docling drops one)."""
    html = build_html(["B64"], _selftest_docs(), "Self-test")
    assert _box_counts(html) == EXPECTED_BOX_COUNTS


def test_build_html_per_backend_span_counts_match_baseline() -> None:
    """Per-backend text-pane span counts are stable."""
    html = build_html(["B64"], _selftest_docs(), "Self-test")
    assert _span_counts(html) == EXPECTED_SPAN_COUNTS


def test_build_html_no_geometry_listed_not_drawn() -> None:
    """The no-geometry span is listed + labelled, but never a drawn box."""
    html = build_html(["B64"], _selftest_docs(), "Self-test")
    # Exactly one "no geometry" label, on docling's bbox-less element.
    assert html.count("no geometry") == EXPECTED_NO_GEOMETRY
    # It is a (non-drawn) text-pane span...
    assert 'class="span nogeo" data-eid="docling-x"' in html
    # ...and it must NOT appear as a drawn overlay box.
    assert 'class="box" data-eid="docling-x"' not in html


def test_build_html_box_count_never_exceeds_span_count() -> None:
    """Invariant: a backend never draws more boxes than it lists spans."""
    html = build_html(["B64"], _selftest_docs(), "Self-test")
    boxes, spans = _box_counts(html), _span_counts(html)
    for backend in EXPECTED_TABS:
        assert boxes[backend] <= spans[backend], backend


@pytest.mark.parametrize("backend", EXPECTED_TABS)
def test_build_html_emits_a_panel_per_backend(backend: str) -> None:
    """Every backend present in ``docs`` gets its own panel + tab button."""
    html = build_html(["B64"], _selftest_docs(), "Self-test")
    assert f'id="panel-{backend}"' in html
    assert f'data-tab="{backend}"' in html
