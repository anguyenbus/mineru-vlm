"""PDF page rasterization and text-layer utilities using pypdfium2.

All pypdfium2 imports are deferred inside function bodies so this module is
importable even when pypdfium2 is not installed (e.g. during unit tests that
mock the PDF layer).

Coordinate system note:
    PDF uses a bottom-left origin measured in points (1/72 inch).
    pypdfium2 renders images with a top-left origin measured in pixels.
    ``render_region`` converts between the two systems; see inline comments.

# NOTE: Coordinate conversion verified visually on 2026-05-28
#       with digital_simple.pdf (reference render.py).
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Final

from loguru import logger

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_DEFAULT_DPI: Final[int] = 144
"""Default render resolution in dots per inch."""

_DEFAULT_MAX_RENDER_MP: Final[float] = 40.0
"""Default megapixel ceiling for any rasterized page.

pypdfium2 allocates the bitmap itself (PIL MAX_IMAGE_PIXELS does NOT cover a
buffer wrapped by .to_pil()), so we clamp the render scale BEFORE rendering.
Default 40 MP leaves Letter/A4/A1 untouched while downscaling large-format
pages (A0 @ 144 DPI ~= 320 MP). Overridable via PARSER_MAX_RENDER_MP env var.
"""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _clamp_scale_to_budget(
    width_pts: float,
    height_pts: float,
    scale: float,
    max_pixels: int,
) -> tuple[float, bool]:
    """Compute a scale factor that keeps rendered pixel count within budget.

    Pure function — performs no allocation and is safe to call with
    adversarially large page dimensions. Degenerate inputs (non-positive area
    or budget) pass the scale through unchanged rather than dividing by zero.

    Args:
        width_pts: Page width in PDF points.
        height_pts: Page height in PDF points.
        scale: Requested scale factor.
        max_pixels: Maximum allowed pixel count (width * height).

    Returns:
        Tuple of (effective_scale, was_clamped).
    """
    area_pts = width_pts * height_pts
    if area_pts <= 0 or max_pixels <= 0:
        return (scale, False)
    projected = area_pts * scale * scale
    if projected <= max_pixels:
        return (scale, False)
    # NOTE: clamped = scale * sqrt(max_pixels / projected)
    clamped = scale * (max_pixels / projected) ** 0.5
    return (clamped, True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_page(pdf_path: Path, page_no: int, dpi: int = _DEFAULT_DPI) -> bytes:
    """Render a single PDF page to PNG bytes.

    Reads ``PARSER_RENDER_DPI`` from the environment when the caller passes
    the default DPI sentinel. Clamps the render scale to stay within the
    ``PARSER_MAX_RENDER_MP`` megapixel budget before allocating the bitmap.

    Args:
        pdf_path: Path to the PDF file.
        page_no: Zero-based page index.
        dpi: Render resolution in dots per inch. When equal to the default
            sentinel (144) the value is overridden by the ``PARSER_RENDER_DPI``
            environment variable if set.

    Returns:
        PNG-encoded image bytes of the full page.

    Raises:
        RuntimeError: If the PDF cannot be opened or the page rendered.
    """
    import pypdfium2  # noqa: PLC0415

    effective_dpi = int(os.environ.get("PARSER_RENDER_DPI", str(dpi)))
    scale = effective_dpi / 72.0
    max_mp = float(os.environ.get("PARSER_MAX_RENDER_MP", str(_DEFAULT_MAX_RENDER_MP)))
    max_pixels = int(max_mp * 1_000_000)

    pdf = None
    try:
        pdf = pypdfium2.PdfDocument(str(pdf_path))
        page = pdf[page_no]
        width_pts = float(page.get_width())
        height_pts = float(page.get_height())
        effective_scale, was_clamped = _clamp_scale_to_budget(
            width_pts, height_pts, scale, max_pixels
        )
        if was_clamped:
            logger.warning(
                "[render] page {} scale clamped from {:.2f} to {:.2f} to stay within {} MP budget",
                page_no,
                scale,
                effective_scale,
                max_mp,
            )
        pil_image = page.render(scale=effective_scale).to_pil()
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        return buf.getvalue()
    finally:
        if pdf is not None:
            pdf.close()


def render_region(
    pdf_path: Path,
    page_no: int,
    bbox: list[float],
    dpi: int = _DEFAULT_DPI,
) -> bytes:
    """Render a bounding-box region of a PDF page to PNG bytes.

    Converts from PDF coordinate space (bottom-left origin, points) to image
    coordinate space (top-left origin, pixels) using the clamped scale so that
    the crop coordinates remain correct even when the page is downscaled.

    Coordinate conversion (PDF bottom-left origin -> image top-left origin):

    .. code-block:: text

        img_left   = x0 * scale
        img_right  = x1 * scale
        img_top    = (page_height_pts - y1) * scale  # PDF y1 (top of box)
        img_bottom = (page_height_pts - y0) * scale  # PDF y0 (bottom of box)

    Args:
        pdf_path: Path to the PDF file.
        page_no: Zero-based page index.
        bbox: Region bounding box ``[x0, y0, x1, y1]`` in PDF points
            (bottom-left origin).
        dpi: Render resolution. When equal to the default sentinel (144) the
            value is overridden by ``PARSER_RENDER_DPI`` if set.

    Returns:
        PNG-encoded image bytes of the cropped region.

    Raises:
        ValueError: If the bbox results in a zero or negative area after
            coordinate conversion and clamping to rendered image bounds.
    """
    import pypdfium2  # noqa: PLC0415

    effective_dpi = int(os.environ.get("PARSER_RENDER_DPI", str(dpi)))
    scale = effective_dpi / 72.0
    max_mp = float(os.environ.get("PARSER_MAX_RENDER_MP", str(_DEFAULT_MAX_RENDER_MP)))
    max_pixels = int(max_mp * 1_000_000)

    pdf = None
    try:
        pdf = pypdfium2.PdfDocument(str(pdf_path))
        page = pdf[page_no]
        width_pts = float(page.get_width())
        height_pts = float(page.get_height())
        # NOTE: Clamp BEFORE rendering; reuse the clamped scale for coordinate
        # conversion so crop coordinates stay aligned with the rendered pixels.
        effective_scale, _ = _clamp_scale_to_budget(width_pts, height_pts, scale, max_pixels)
        pil_image = page.render(scale=effective_scale).to_pil()
        rendered_width, rendered_height = pil_image.size
    finally:
        if pdf is not None:
            pdf.close()

    # NOTE: PDF uses bottom-left origin; image uses top-left origin.
    # Convert PDF bbox coords to image pixel coords using the reference formula.
    img_left = bbox[0] * effective_scale
    img_right = bbox[2] * effective_scale
    img_top = (height_pts - bbox[3]) * effective_scale
    img_bottom = (height_pts - bbox[1]) * effective_scale

    # Clamp to rendered image bounds so partial off-page bboxes still work.
    img_left = max(0.0, min(img_left, rendered_width))
    img_right = max(0.0, min(img_right, rendered_width))
    img_top = max(0.0, min(img_top, rendered_height))
    img_bottom = max(0.0, min(img_bottom, rendered_height))

    if img_right <= img_left or img_bottom <= img_top:
        raise ValueError(
            f"Zero or negative area bbox after coordinate transform: "
            f"left={img_left}, right={img_right}, top={img_top}, bottom={img_bottom}"
        )

    cropped = pil_image.crop((int(img_left), int(img_top), int(img_right), int(img_bottom)))
    buf = io.BytesIO()
    cropped.save(buf, format="PNG")
    return buf.getvalue()


def text_layer_tokens(pdf_path: Path) -> dict[int, int]:
    """Extract token counts from the embedded text layer of each page.

    Used by the quality gate's Layer 1 coverage check to detect pages where
    the parser under-extracted text relative to the embedded PDF text layer.
    Pages with no embedded text layer (scanned/image-only PDFs) return 0
    tokens per page rather than raising.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Mapping from zero-based page index to whitespace-split token count.
        Returns ``{}`` on any error (missing file, corrupt PDF, or if pypdfium2
        is not installed).
    """
    try:
        import pypdfium2  # noqa: PLC0415

        pdf = None
        try:
            pdf = pypdfium2.PdfDocument(str(pdf_path))
            counts: dict[int, int] = {}
            for i in range(len(pdf)):
                try:
                    text = pdf[i].get_textpage().get_text_range()
                except Exception:  # noqa: BLE001
                    text = ""
                counts[i] = len(text.split())
            return counts
        finally:
            if pdf is not None:
                pdf.close()
    except Exception as exc:
        logger.debug("[render] text_layer_tokens failed for {}: {}", pdf_path, exc)
        return {}
