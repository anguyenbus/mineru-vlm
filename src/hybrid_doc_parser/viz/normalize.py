"""Normalizers: heterogeneous sources -> :class:`Doc`. None of these may raise.

Two source shapes feed the viewer:

* ``doc_from_parser_output`` — consumes the Pydantic ``ParserOutput`` /
  ``ElementRecord`` from :mod:`hybrid_doc_parser.models` via attribute access
  (NOT stringly-typed dict keys, so a schema field rename is a hard error, not a
  silently empty panel). Linking is at ELEMENT granularity via
  ``ElementRecord.element_id`` + ``bbox`` — never via the lossy
  ``render_markdown()`` string, which drops furniture and reorders content.

* ``doc_from_pdfplumber`` — the live, highlightable baseline. pdfplumber is an
  optional dep imported inside the function; an import/parse failure degrades to
  an in-pane note and never crashes (mirroring the library's "``parse()`` never
  raises" contract).

The ``schema_version`` of a ``ParserOutput`` is checked against the library's
``SCHEMA_VERSION``. On mismatch the normalizer WARNS and REFUSES to map element
geometry (it returns a single warning span) rather than mis-rendering a stale
JSON against a transform it was not produced for.
"""

from __future__ import annotations

import warnings

from hybrid_doc_parser.models import SCHEMA_VERSION, ParserOutput
from hybrid_doc_parser.viz.coords import to_canonical
from hybrid_doc_parser.viz.model import Doc, Span

# Default page size (points) when the renderer could not supply one. Only used
# by the points-unit backends; MinerU per-mille ignores page size entirely.
_FALLBACK_PAGE_PT = (1.0, 1.0)


def _page_size(sizes, page: int) -> tuple[float, float] | None:
    """Look up a page's ``(w, h)`` by ABSOLUTE page index.

    Accepts either a mapping keyed by absolute ``page_idx`` (the correct form
    when a non-prefix page subset was rendered) or a positional list (legacy
    form, valid only when image position equals page index). Returns ``None``
    when the page was not rendered.
    """
    if isinstance(sizes, dict):
        return sizes.get(page)
    if 0 <= page < len(sizes):
        return sizes[page]
    return None


def doc_from_parser_output(
    parser_output: ParserOutput,
    backend: str,
    page_sizes_pt: list[tuple[float, float]] | dict[int, tuple[float, float]] | None = None,
) -> Doc:
    """Map a Pydantic ``ParserOutput`` into a :class:`Doc` for one backend.

    Args:
        parser_output: The library's ``ParserOutput`` (already validated). Field
            access is by attribute, not dict key.
        backend: Backend key driving the coordinate transform (``mineru`` =
            per-mille top-left, page size not needed; ``docling`` = points
            bottom-left with Y-flip, page size required).
        page_sizes_pt: Per-page ``(width_pt, height_pt)`` from the renderer,
            keyed by ABSOLUTE ``page_idx`` (a dict) or a positional list (legacy;
            only correct when image position equals page index). Required for the
            ``points`` backends; safely ignored by MinerU per-mille.

    Returns:
        A :class:`Doc`. On a ``schema_version`` mismatch the doc contains a
        single warning span and NO element geometry (refuse, never mis-map).
    """
    doc = Doc(backend=backend)
    sizes = page_sizes_pt if page_sizes_pt is not None else []

    if parser_output.schema_version != SCHEMA_VERSION:
        msg = (
            f"schema_version mismatch: report JSON is "
            f"{parser_output.schema_version!r}, viewer expects {SCHEMA_VERSION!r}. "
            f"Refusing to map geometry (would mis-render). Re-export with a "
            f"matching parser version."
        )
        warnings.warn(msg, stacklevel=2)
        doc.spans.append(
            Span(
                element_id=f"{backend}-schema-mismatch",
                page=0,
                type="warning",
                text=msg,
                bbox=None,
            )
        )
        return doc

    for el in parser_output.elements:
        page = el.page_idx
        if backend in ("mineru", "paddleocr", "mineru25pro"):
            # Per-mille is page-relative; page size is not needed. PaddleOCR and
            # MinerU2.5-Pro pre-normalise their boxes to per-mille at parse time.
            canon = to_canonical(el.bbox, backend)
        else:
            w, h = _page_size(sizes, page) or _FALLBACK_PAGE_PT
            canon = to_canonical(el.bbox, backend, w, h)

        text = (el.text or "").strip() or (el.description or "").strip()
        doc.spans.append(
            Span(
                element_id=el.element_id,
                page=page,
                type=el.type.value,
                text=text,
                bbox=canon,
            )
        )
    return doc


def doc_from_pdfplumber(pdf_path) -> Doc:
    """Build the highlightable pdfplumber baseline (line boxes, MIT-licensed).

    pdfplumber is an optional ``viz`` dep imported here so the module stays
    pure-importable. Any import or parse failure degrades to an in-pane warning
    span; this function never raises.

    Args:
        pdf_path: Path to the source PDF.

    Returns:
        A :class:`Doc` with backend ``pdfplumber``; line spans on success, or a
        single warning span on missing-dep/parse failure.
    """
    doc = Doc(backend="pdfplumber")
    try:
        import pdfplumber  # noqa: PLC0415
    except Exception:  # noqa: BLE001 — optional dep absent is a degrade, not a crash
        doc.spans.append(
            Span(
                element_id="pdfplumber-missing",
                page=0,
                type="warning",
                text="pdfplumber not installed — `pip install pdfplumber`",
                bbox=None,
            )
        )
        return doc

    try:
        line_total = 0
        page_total = 0
        with pdfplumber.open(str(pdf_path)) as pdf:
            for pg, page in enumerate(pdf.pages):
                page_total += 1
                w, h = float(page.width), float(page.height)
                try:
                    lines = page.extract_text_lines()
                except Exception:  # noqa: BLE001
                    lines = []
                line_total += len(lines)
                for j, ln in enumerate(lines):
                    canon = to_canonical(
                        (ln["x0"], ln["top"], ln["x1"], ln["bottom"]),
                        "pdfplumber",
                        w,
                        h,
                    )
                    doc.spans.append(
                        Span(
                            element_id=f"pdfplumber-{pg}-{j}",
                            page=pg,
                            type="text",
                            text=ln.get("text", ""),
                            bbox=canon,
                        )
                    )
        # NOTE: pdfplumber needs an embedded text layer. A scanned/image-only PDF
        # yields zero lines; say so in-pane rather than render a silently blank tab
        # (and flag that the known-good reference tab is unavailable for this doc).
        if page_total and line_total == 0:
            doc.spans.append(
                Span(
                    element_id="pdfplumber-no-text-layer",
                    page=0,
                    type="warning",
                    text=(
                        "pdfplumber found no embedded text layer (scanned/image PDF). "
                        "The reference tab is unavailable for this document; MinerU and "
                        "Docling boxes below come from OCR and have no text-layer anchor "
                        "to be visually calibrated against."
                    ),
                    bbox=None,
                )
            )
    except Exception as exc:  # noqa: BLE001 — corrupt/encrypted PDF degrades gracefully
        doc.spans.append(
            Span(
                element_id="pdfplumber-error",
                page=0,
                type="warning",
                text=f"pdfplumber failed: {exc}",
                bbox=None,
            )
        )
    return doc
