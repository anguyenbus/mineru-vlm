"""Page rasterization for the viewer — base64 PNGs + page sizes in points.

All ``pypdfium2`` / Pillow imports are deferred inside function bodies (mirroring
:mod:`hybrid_doc_parser.render`), so this module imports cleanly without the
``viz`` optional deps installed. The heavy raster paths are exercised only under
the pytest ``slow`` marker.

The renderer reports each page's size in PDF points so the points-unit backends
(Docling, pdfplumber) can normalize against the box the parser actually used.
For a ``/Rotate``-d page, pypdfium2 already reports the post-rotation size, so the
canonical transform needs no extra rotation math.

Page SELECTION / TRUNCATION (``select_pages``) is pure — it decides which page
indices to embed (and whether to show a truncation note) from page counts alone,
so the page-cap behavior is unit-testable without a rasterizer. The raster path
(``safe_render_pages``) mirrors the library's "never raises" contract: a 0-page,
encrypted, or corrupt PDF degrades to a clear message, not a traceback.
"""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from typing import Final

# Reuse the library's default render resolution (overridable via env).
_DEFAULT_DPI: Final[int] = 144
"""Default render resolution in DPI; matches ``hybrid_doc_parser.render``."""

# Default page cap per report. base64-embedding every page bloats the HTML, so a
# big document is truncated (with a VISIBLE in-report note) unless --max-pages or
# --pages overrides it. See view.md §7 "large PDF" and R2.
DEFAULT_MAX_PAGES: Final[int] = 50
"""Default maximum pages embedded in one report before truncation."""


def select_pages(
    total_pages: int,
    max_pages: int = DEFAULT_MAX_PAGES,
    pages_spec: str | None = None,
) -> tuple[list[int], str | None]:
    """Decide which page indices to render and whether truncation occurred.

    This is PURE (no pypdfium2): the page selection / truncation decision is made
    on page indices alone, so it is unit-testable without a rasterizer. The CLI
    feeds the returned indices to :func:`render_pages` and surfaces the note in
    the report banner.

    Args:
        total_pages: Number of pages the document actually has.
        max_pages: Hard cap on embedded pages (default
            :data:`DEFAULT_MAX_PAGES`). A non-positive value disables the cap.
        pages_spec: Optional selection of the form ``"N-M"`` (1-based inclusive
            range), a single ``"N"``, or a comma list ``"1,3,5"``. ``None`` or an
            unparseable spec selects every page. An out-of-document range is
            clamped, never an error.

    Returns:
        ``(indices, note)`` where ``indices`` is the 0-based page list to render
        (ascending, de-duplicated) and ``note`` is a human-readable truncation
        message when the cap dropped pages, else ``None``.
    """
    total = max(0, int(total_pages))
    if total == 0:
        return [], None

    if pages_spec:
        wanted = _parse_pages_spec(pages_spec, total)
    else:
        wanted = list(range(total))

    note: str | None = None
    if max_pages and max_pages > 0 and len(wanted) > max_pages:
        dropped = len(wanted)
        wanted = wanted[:max_pages]
        note = (
            f"Truncated to the first {max_pages} of {dropped} selected pages "
            f"(document has {total}). Raise --max-pages or use --pages N-M to see more."
        )
    return wanted, note


def _parse_pages_spec(spec: str, total: int) -> list[int]:
    """Parse a ``--pages`` spec into clamped, de-duplicated 0-based indices.

    Accepts ``N-M`` ranges, single ``N`` pages, and comma-separated lists; all
    1-based and inclusive. Unparseable tokens are skipped, an empty result falls
    back to every page, and out-of-document indices are dropped (never raised).
    """
    picked: list[int] = []
    seen: set[int] = set()
    for token in spec.replace(" ", "").split(","):
        if not token:
            continue
        try:
            if "-" in token:
                lo_s, hi_s = token.split("-", 1)
                lo, hi = int(lo_s), int(hi_s)
            else:
                lo = hi = int(token)
        except ValueError:
            continue
        for one_based in range(lo, hi + 1):
            zero = one_based - 1
            if 0 <= zero < total and zero not in seen:
                seen.add(zero)
                picked.append(zero)
    picked.sort()
    return picked or list(range(total))


def render_pages(
    pdf_path: Path, dpi: int = _DEFAULT_DPI, page_indices: list[int] | None = None
) -> tuple[list[str], list[tuple[float, float]]]:
    """Rasterize selected pages of a PDF to base64 PNGs and report page sizes.

    Reads ``PARSER_RENDER_DPI`` from the environment when the caller passes the
    default sentinel, matching the library's convention.

    Args:
        pdf_path: Path to the source PDF.
        dpi: Render resolution in DPI. When equal to the default sentinel (144)
            the value is overridden by ``PARSER_RENDER_DPI`` if set.
        page_indices: 0-based pages to render (from :func:`select_pages`). When
            ``None``, every page is rendered.

    Returns:
        Tuple of ``(images_b64, sizes_pt)`` where ``images_b64`` is one base64
        PNG string per rendered page and ``sizes_pt`` is the matching
        ``(width_pt, height_pt)`` per page (post-rotation, as reported by
        pypdfium2). Order follows ``page_indices``.
    """
    import pypdfium2 as pdfium  # noqa: PLC0415

    effective_dpi = int(os.environ.get("PARSER_RENDER_DPI", str(dpi)))
    scale = effective_dpi / 72.0

    images_b64: list[str] = []
    sizes_pt: list[tuple[float, float]] = []
    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        indices = range(len(pdf)) if page_indices is None else page_indices
        for i in indices:
            if not 0 <= i < len(pdf):
                continue
            page = pdf[i]
            w, h = page.get_size()
            sizes_pt.append((float(w), float(h)))
            pil = page.render(scale=scale).to_pil().convert("RGB")
            images_b64.append(_png_b64(pil))
    finally:
        pdf.close()
    return images_b64, sizes_pt


def safe_render_pages(
    pdf_path: Path, dpi: int = _DEFAULT_DPI, page_indices: list[int] | None = None
) -> tuple[list[str], list[tuple[float, float]], str | None]:
    """Render pages, degrading a 0-page / encrypted / corrupt PDF to a message.

    Mirrors the library's "never raises" contract: any failure (missing
    pypdfium2, an encrypted or corrupt PDF, a zero-page document) returns an
    empty render plus a clear human-readable note instead of a traceback.

    Args:
        pdf_path: Path to the source PDF.
        dpi: Render resolution in DPI.
        page_indices: 0-based pages to render, or ``None`` for all.

    Returns:
        ``(images_b64, sizes_pt, error_note)`` — ``error_note`` is ``None`` on
        success, otherwise a clear message describing the render failure.
    """
    try:
        imgs, sizes = render_pages(pdf_path, dpi, page_indices)
    except Exception as exc:  # noqa: BLE001 — corrupt/encrypted PDF degrades gracefully
        return [], [], f"Could not render {Path(pdf_path).name}: {exc}"
    if not imgs:
        return (
            [],
            [],
            f"No pages rendered from {Path(pdf_path).name} "
            f"(empty, zero-page, or unselectable page range).",
        )
    return imgs, sizes, None


def _png_b64(pil_img) -> str:
    """Encode a PIL image to a base64 PNG string (ASCII)."""
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")
