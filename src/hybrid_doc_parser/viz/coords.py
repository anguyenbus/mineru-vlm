"""Coordinate calibration — the one load-bearing assumption of the viewer.

This module maps a raw per-backend bbox into a single *canonical* coordinate
space: normalized ``0..1``, top-left origin. The HTML layer then positions
overlay boxes with CSS percentages over the page image, so the overlay is
resolution-independent and never does DPI/pixel math. The ONLY thing that must
be correct is the per-backend origin/unit convention encoded in ``COORD``.

VERIFIED per-backend conventions (Task Group 1 calibration gate)
----------------------------------------------------------------
The conventions below were verified by reading the parser/backend source, not
guessed:

* ``mineru`` — ``{origin: "topleft", unit: "permille"}``.
  The library consumes ``*_content_list.json``. MinerU's
  ``mineru/backend/pipeline/pipeline_middle_json_mkcontent.py::_build_bbox``
  emits every coord as ``int(coord * 1000 / page_dim)`` — i.e. **per-mille
  (0..1000), top-left origin**. Normalize by **dividing by 1000, with NO
  Y-flip**. Page size is NOT needed for MinerU.

* ``docling`` — ``{origin: "bottomleft", unit: "points"}``.
  ``parser.py`` (~line 727) stores the Docling prov bbox ``{l, t, r, b}`` (a
  BOTTOMLEFT coord origin) as the list ``[l, b, r, t]`` in **PDF points**.
  Normalize by dividing by the page points and **flipping Y** (``1 - y``).

* ``pdfplumber`` — ``{origin: "topleft", unit: "points"}``.
  pdfplumber reports ``top``/``bottom`` from the page top in PDF points. This
  is the known-good *reference tab* used to calibrate the others.

* ``image`` — ``{origin: "topleft", unit: "pixels"}``.
  An image input has no point system; pixels ARE the unit. Pass the image's
  pixel dimensions as ``page_w``/``page_h`` (or ``ref_w``/``ref_h``); no points
  math is applied.

This OVERTURNS two stale claims, both flagged as known issues and NOT fixed in
this spec:

* the prototype ``COORD`` had MinerU as ``topleft/points`` — wrong; it pins
  every box against the page edge.
* the ``models.py`` ``ElementRecord.bbox`` docstring says "PDF points with
  bottom-left origin" — wrong for MinerU (it is per-mille top-left).

Rotation / CropBox / image-input notes
---------------------------------------
* ``/Rotate``: MinerU already normalizes against the rotated page dimension
  (per-mille is page-relative), so a rotated page needs no extra handling at
  the ``permille`` unit. For ``points`` backends, the renderer must report the
  page size of the box the parser actually used (post-rotation); the transform
  itself stays page-relative.
* CropBox != MediaBox: normalize against the box the parser used; the renderer
  is responsible for reporting that page size. The transform here is purely a
  ratio against the supplied ``page_w``/``page_h``.
* image input: pixels are the unit (``image`` backend / ``pixels`` unit) — pass
  the image pixel dimensions; no points math.

All functions are pure and never raise: an unusable bbox yields ``None`` (the
caller renders the span in the text pane without a box), mirroring the
library's "``parse()`` never raises" contract.
"""

from __future__ import annotations

import math
from typing import Final

# A degenerate box thinner than this (in normalized units) is rejected.
_MIN_EXTENT: Final[float] = 1e-4

# Per-backend origin/unit convention.
#   origin: "topleft" | "bottomleft"
#   unit:   "permille" | "points" | "pixels" | "normalized"
COORD: Final[dict[str, dict[str, str]]] = {
    # MinerU content_list.json: per-mille (0..1000), top-left. ÷1000, no flip.
    "mineru": {"origin": "topleft", "unit": "permille"},
    # PaddleOCR PP-StructureV3 returns pixel boxes; the backend pre-normalises
    # them to per-mille (0..1000) top-left at parse time (see _run_paddleocr),
    # so the viewer treats it exactly like MinerU: ÷1000, no flip, no page size.
    "paddleocr": {"origin": "topleft", "unit": "permille"},
    # Docling prov bbox stored [l, b, r, t]: PDF points, bottom-left. ÷points,
    # flip Y.
    "docling": {"origin": "bottomleft", "unit": "points"},
    # pdfplumber line boxes: PDF points, top-left. The known-good reference tab.
    "pdfplumber": {"origin": "topleft", "unit": "points"},
    # image input: pixels are the unit, top-left. No points math.
    "image": {"origin": "topleft", "unit": "pixels"},
}

# Fallback convention for an unknown backend: the safest reference convention.
_DEFAULT_SPEC: Final[dict[str, str]] = {"origin": "topleft", "unit": "points"}

CanonicalBox = tuple[float, float, float, float]


def to_canonical(
    bbox: object,
    backend: str,
    page_w: float | None = None,
    page_h: float | None = None,
    ref_w: float | None = None,
    ref_h: float | None = None,
) -> CanonicalBox | None:
    """Map a raw per-backend bbox into canonical ``0..1`` top-left coords.

    Args:
        bbox: Raw ``[x0, y0, x1, y1]`` in the backend's native space.
        backend: One of ``COORD``'s keys (``mineru``/``docling``/
            ``pdfplumber``/``image``); unknown backends fall back to
            topleft/points.
        page_w: Page width in the unit's denominator space (PDF points for
            ``points``, pixels for ``pixels``). Not required for ``permille``
            or ``normalized``.
        page_h: Page height, as ``page_w``.
        ref_w: Reference image width in pixels, used when ``unit == "pixels"``
            and it differs from ``page_w``.
        ref_h: Reference image height in pixels, as ``ref_w``.

    Returns:
        A 4-tuple ``(x0, y0, x1, y1)`` of floats in ``[0, 1]`` with top-left
        origin, or ``None`` if the bbox is missing, malformed, non-finite,
        negative, inverted, or degenerate after clamping.
    """
    # --- shape check ---------------------------------------------------------
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return None

    # --- reject NaN / inf ----------------------------------------------------
    if not all(math.isfinite(v) for v in (x0, y0, x1, y1)):
        return None

    # --- reject negative raw coords (a real bbox is never negative) ----------
    if min(x0, y0, x1, y1) < 0.0:
        return None

    spec = COORD.get(backend, _DEFAULT_SPEC)
    unit = spec["unit"]

    # --- choose denominators per unit ---------------------------------------
    if unit == "permille":
        denom_x = denom_y = 1000.0
    elif unit == "normalized":
        denom_x = denom_y = 1.0
    elif unit == "pixels":
        denom_x = ref_w if ref_w else page_w
        denom_y = ref_h if ref_h else page_h
    else:  # points
        denom_x, denom_y = page_w, page_h

    if not denom_x or not denom_y:
        return None
    if not (math.isfinite(denom_x) and math.isfinite(denom_y)):
        return None

    nx0, nx1 = x0 / denom_x, x1 / denom_x
    ny0, ny1 = y0 / denom_y, y1 / denom_y

    # --- Y-flip for bottom-left origins -------------------------------------
    if spec["origin"] == "bottomleft":
        ny0, ny1 = 1.0 - ny1, 1.0 - ny0

    # --- order, clamp, reject degenerate ------------------------------------
    nx0, nx1 = sorted((nx0, nx1))
    ny0, ny1 = sorted((ny0, ny1))
    nx0, ny0 = max(0.0, nx0), max(0.0, ny0)
    nx1, ny1 = min(1.0, nx1), min(1.0, ny1)
    if nx1 - nx0 <= _MIN_EXTENT or ny1 - ny0 <= _MIN_EXTENT:
        return None
    return (nx0, ny0, nx1, ny1)
