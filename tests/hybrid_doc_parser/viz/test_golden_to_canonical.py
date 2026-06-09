"""Golden ``to_canonical`` tests — the lie-catcher (Task Group 1.5).

Each test asserts the EXACT canonical (0..1, top-left) rectangle a raw
per-backend bbox maps to, locking the verified ``COORD`` table. These goldens
are reused by Group 3's ``golden_to_canonical`` suite.

Inputs are SYNTHESIZED (see ``fixtures/make_golden_json.py``): MinerU raw is
per-mille top-left; Docling raw is PDF points bottom-left (stored ``[l,b,r,t]``).
``import mineru``/``import docling`` are unavailable here, so real-parse is not
used; the offline goldens are exact and the only remaining step is a live
visual-eyeball confirmation on a real document.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hybrid_doc_parser.models import ParserOutput
from hybrid_doc_parser.viz.coords import to_canonical

FIXTURES = Path(__file__).resolve().parent / "fixtures"

# US-Letter / A4 page sizes in PDF points (for the points-unit backends).
US_LETTER = (612.0, 792.0)
A4 = (595.28, 841.89)


def _approx(box, expected, tol: float = 1e-6) -> None:
    assert box is not None, "to_canonical unexpectedly rejected the box"
    assert box == pytest.approx(expected, abs=tol), f"{box} != {expected}"


# ---------------------------------------------------------------------------
# Required case 1: MinerU per-mille — ÷1000, NO Y-flip, page size irrelevant.
# ---------------------------------------------------------------------------


def test_mineru_permille_divides_by_1000_no_flip() -> None:
    """MinerU per-mille [250,100,750,200] -> exact (0.25,0.10,0.75,0.20)."""
    _approx(
        to_canonical([250, 100, 750, 200], "mineru"),
        (0.25, 0.10, 0.75, 0.20),
    )


def test_mineru_permille_ignores_page_size() -> None:
    """Per-mille is page-relative: same canonical box for any page dims."""
    a = to_canonical([250, 100, 750, 200], "mineru", *US_LETTER)
    b = to_canonical([250, 100, 750, 200], "mineru", *A4)
    c = to_canonical([250, 100, 750, 200], "mineru")
    assert a == b == c == pytest.approx((0.25, 0.10, 0.75, 0.20))


# ---------------------------------------------------------------------------
# Required case 2: Docling bottom-left points — ÷page points, flip Y (1 - y).
# ---------------------------------------------------------------------------


def test_docling_bottomleft_points_flips_y() -> None:
    """Docling [l,b,r,t] on US-Letter maps to the top-left heading band."""
    _approx(
        to_canonical([61.2, 673.2, 550.8, 728.64], "docling", *US_LETTER),
        (0.10, 0.08, 0.90, 0.15),
        tol=1e-4,
    )


def test_docling_flip_is_actually_applied() -> None:
    """A bottom-of-page bbox (small bottom-left y) maps to a large top-left y."""
    box = to_canonical([0.0, 0.0, 612.0, 79.2], "docling", *US_LETTER)
    assert box is not None
    # y spans 0..79.2 pt from the bottom -> canonical 0.90..1.0 from the top.
    assert box[1] == pytest.approx(0.90, abs=1e-4)
    assert box[3] == pytest.approx(1.0, abs=1e-4)


# ---------------------------------------------------------------------------
# Required case 3: A4 / non-Letter — exercise non-Letter page dims.
# ---------------------------------------------------------------------------


def test_a4_mineru_permille_unaffected_by_a4_dims() -> None:
    """MinerU per-mille is unchanged by A4 page dims."""
    _approx(to_canonical([250, 100, 750, 200], "mineru", *A4), (0.25, 0.10, 0.75, 0.20))


def test_a4_docling_bottomleft_points() -> None:
    """Docling bottom-left points map correctly against non-Letter A4 dims."""
    _approx(
        to_canonical([59.528, 715.6065, 535.752, 774.5388], "docling", *A4),
        (0.10, 0.08, 0.90, 0.15),
        tol=1e-4,
    )


# ---------------------------------------------------------------------------
# Required case 4: image input — pixels are the unit, no points math.
# ---------------------------------------------------------------------------


def test_image_input_pixels_as_units() -> None:
    """Image backend divides by pixel dims (400x300), top-left, no points math."""
    _approx(
        to_canonical([40, 24, 360, 45], "image", 400.0, 300.0),
        (0.10, 0.08, 0.90, 0.15),
        tol=1e-4,
    )


def test_image_input_ref_dims_override_page_dims() -> None:
    """ref_w/ref_h take precedence over page_w/page_h for the pixels unit."""
    _approx(
        to_canonical([40, 24, 360, 45], "image", 1.0, 1.0, ref_w=400.0, ref_h=300.0),
        (0.10, 0.08, 0.90, 0.15),
        tol=1e-4,
    )


# ---------------------------------------------------------------------------
# /Rotate 90 — per-mille is rotation-agnostic (the page-relative property).
# ---------------------------------------------------------------------------


def test_rotate90_mineru_permille_unchanged() -> None:
    """MinerU per-mille against a post-rotation page needs no extra math."""
    _approx(to_canonical([250, 100, 750, 200], "mineru"), (0.25, 0.10, 0.75, 0.20))


# ---------------------------------------------------------------------------
# Reject path: never crash, return None on unusable boxes.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bbox",
    [
        None,
        [],
        [1, 2, 3],  # wrong arity
        [float("nan"), 0, 1, 1],
        [0, float("inf"), 1, 1],
        [-1, 0, 1, 1],  # negative
        [500, 100, 500, 200],  # degenerate width (zero after ÷1000)
    ],
)
def test_rejects_unusable_boxes(bbox) -> None:
    """Unusable boxes return None (never raise), mirroring parse()-never-raises."""
    assert to_canonical(bbox, "mineru", *US_LETTER) is None


def test_clamps_into_unit_range() -> None:
    """Out-of-page per-mille values clamp into [0,1], never exceeding it."""
    box = to_canonical([0, 0, 1200, 1200], "mineru")
    assert box == pytest.approx((0.0, 0.0, 1.0, 1.0))


# ---------------------------------------------------------------------------
# The synthesized golden JSON fixtures load via Pydantic and match the goldens.
# Confirms the offline inputs are real ParserOutput and the raw bboxes are the
# ones the assertions above lock (ties the fixtures to the goldens).
# ---------------------------------------------------------------------------


def _load(name: str) -> ParserOutput:
    return ParserOutput.model_validate_json((FIXTURES / name).read_text())


def test_golden_json_mineru_first_element_matches() -> None:
    """The synthesized MinerU JSON loads via Pydantic and matches the golden."""
    po = _load("us_letter.mineru.json")
    assert po.schema_version == "1.0"
    raw = po.elements[0].bbox
    _approx(to_canonical(raw, "mineru"), (0.25, 0.10, 0.75, 0.20))


def test_golden_json_docling_first_element_matches() -> None:
    """The synthesized Docling JSON loads via Pydantic and matches the golden."""
    po = _load("us_letter.docling.json")
    raw = po.elements[0].bbox
    _approx(to_canonical(raw, "docling", *US_LETTER), (0.10, 0.08, 0.90, 0.15), tol=1e-4)


def test_golden_json_a4_docling_matches() -> None:
    """The synthesized A4 Docling JSON loads via Pydantic and matches the golden."""
    po = _load("a4.docling.json")
    raw = po.elements[0].bbox
    _approx(to_canonical(raw, "docling", *A4), (0.10, 0.08, 0.90, 0.15), tol=1e-4)
