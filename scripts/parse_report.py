"""Thin CLI shim for the Parse Report viewer — no business logic lives here.

All real work is delegated to :mod:`hybrid_doc_parser.viz`. This shim only parses
flags, loads JSON / renders pages, and writes the HTML out. It mirrors the
library's "never raises" contract: a missing pdfplumber, a 0-page / encrypted /
corrupt PDF, or a stale ``schema_version`` degrade to an in-report note, not a
traceback.

Usage
-----
Validate the UI with synthetic data (Pillow only, no PDF/parser deps)::

    python scripts/parse_report.py --selftest

Real data — saved ``ParserOutput.model_dump_json()`` (parse once, iterate fast)::

    python scripts/parse_report.py source.pdf --mineru mineru.json --docling docling.json

The pdfplumber tab is computed from ``source.pdf`` automatically when pdfplumber
is installed. Both ``--mineru`` and ``--docling`` are OPTIONAL — one backend plus
the pdfplumber baseline is a valid report. Reports default to ``./parse_reports/``
(``.gitignore``d, since a report embeds document content). Save a ParserOutput
JSON with::

    Path("x.json").write_text(parse(p).model_dump_json())

Flags
-----
``--dpi``        render resolution (default 144, matching ``PARSER_RENDER_DPI``).
``--max-pages``  page cap before truncation (default 50); a VISIBLE in-report
                 note appears when the document exceeds it.
``--pages``      ``N-M`` range / single page / comma list (1-based, inclusive).
``-o/--out``     output file; default ``./parse_reports/<source>.parse_report.html``.
``--selftest``   emit a synthetic demo report and exit.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hybrid_doc_parser.models import ParserOutput
from hybrid_doc_parser.viz.html import build_html
from hybrid_doc_parser.viz.model import Doc, Span
from hybrid_doc_parser.viz.normalize import doc_from_parser_output, doc_from_pdfplumber
from hybrid_doc_parser.viz.render import DEFAULT_MAX_PAGES

# Default output directory for generated reports. .gitignore'd: a report embeds
# document content and is as sensitive as its source.
DEFAULT_OUT_DIR = Path("parse_reports")


def _selftest(out: Path) -> None:
    """Synthetic data: prove the linking UX with no PDF and no parser deps."""
    from PIL import Image, ImageDraw  # noqa: PLC0415

    from hybrid_doc_parser.viz.coords import to_canonical
    from hybrid_doc_parser.viz.render import _png_b64

    img = Image.new("RGB", (850, 1100), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([60, 60, 790, 120], outline="#888")
    d.text((70, 80), "DEMO PAGE — synthetic", fill="#333")
    d.rectangle([60, 160, 790, 320], outline="#888")
    d.rectangle([60, 360, 410, 700], outline="#888")
    d.rectangle([440, 360, 790, 700], outline="#888")
    pages = [_png_b64(img)]

    def spans(backend: str, drop_one: bool = False) -> list[Span]:
        raw = [
            ("heading", "DEMO PAGE — synthetic", (60, 60, 790, 120)),
            ("text", "Intro paragraph spanning the full column width.", (60, 160, 790, 320)),
            ("text", "Left column body text block.", (60, 360, 410, 700)),
            ("table", "Right column — a table here.", (440, 360, 790, 700)),
        ]
        if drop_one:  # simulate a backend that missed the table
            raw = raw[:-1]
        out_spans = [
            Span(f"{backend}-{i}", 0, t, txt, to_canonical(bb, "pdfplumber", 850, 1100))
            for i, (t, txt, bb) in enumerate(raw)
        ]
        if backend == "docling":  # a no-geometry unit, like docx
            out_spans.append(Span(f"{backend}-x", 0, "text", "Element with no bbox (docx-style).", None))
        return out_spans

    docs = {
        "mineru": Doc("mineru", spans("mineru")),
        "docling": Doc("docling", spans("docling", drop_one=True)),
        "pdfplumber": Doc("pdfplumber", spans("pdfplumber")),
    }
    out.write_text(build_html(pages, docs, "Self-test"))
    print(f"wrote {out}")


def _default_out_path(pdf_path: Path) -> Path:
    """Default report path under ``./parse_reports/`` (created if missing)."""
    DEFAULT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_OUT_DIR / f"{pdf_path.stem}.parse_report.html"


def _load_parser_output(path: str | None, backend: str, notes: list[str]) -> ParserOutput | None:
    """Load a saved ``ParserOutput`` JSON, degrading any failure to a note.

    Mirrors the viewer's "never crash" contract: a missing file, unreadable
    bytes, or JSON that fails validation appends a clear note and returns
    ``None`` (the backend's tab is simply omitted) instead of raising.
    """
    if not path:
        return None
    try:
        return ParserOutput.model_validate_json(Path(path).read_text())
    except Exception as exc:  # noqa: BLE001 — bad/missing JSON degrades, never crashes
        notes.append(f"Could not load --{backend} JSON {path!r}: {exc}")
        return None


def main(argv: list[str] | None = None) -> int:
    """Parse flags and emit the report. Returns a process exit code."""
    ap = argparse.ArgumentParser(description="Parse Report viewer.")
    ap.add_argument("pdf", nargs="?", help="source PDF")
    ap.add_argument("--mineru", help="ParserOutput JSON (mineru backend); optional")
    ap.add_argument("--docling", help="ParserOutput JSON (docling backend); optional")
    ap.add_argument("--dpi", type=int, default=144, help="render DPI (default 144)")
    ap.add_argument(
        "--max-pages",
        type=int,
        default=DEFAULT_MAX_PAGES,
        help=f"page cap before truncation (default {DEFAULT_MAX_PAGES})",
    )
    ap.add_argument("--pages", help="page selection, e.g. 3-10 (1-based, inclusive)")
    ap.add_argument("-o", "--out", help="output HTML (default ./parse_reports/<src>.parse_report.html)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)

    if args.selftest:
        out = Path(args.out) if args.out else _default_out_path(Path("selftest"))
        out.parent.mkdir(parents=True, exist_ok=True)
        _selftest(out)
        return 0

    if not args.pdf:
        ap.error("provide a source PDF (or use --selftest)")
    pdf_path = Path(args.pdf)
    out = Path(args.out) if args.out else _default_out_path(pdf_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Deferred import: render needs pypdfium2 (a viz optional dep). The page count
    # is read first so selection/truncation is decided before rasterization.
    from hybrid_doc_parser.viz.render import safe_render_pages, select_pages

    notes: list[str] = []
    total_pages = _page_count(pdf_path)
    indices, trunc_note = select_pages(total_pages, args.max_pages, args.pages)
    if trunc_note:
        notes.append(trunc_note)

    pages_b64, sizes_pt, render_err = safe_render_pages(pdf_path, args.dpi, indices)
    if render_err:
        notes.append(render_err)

    # Key page sizes by ABSOLUTE page index so points-unit backends normalize
    # against the right page even when a non-prefix subset (e.g. --pages 3-5)
    # was rendered. ``indices`` is positionally aligned with ``sizes_pt``.
    # strict=False: select_pages only ever returns in-range indices, so the two
    # stay aligned; should a defensive render skip ever shorten sizes_pt, zip
    # stops at the shorter rather than raising (preserve never-crash).
    size_by_idx = dict(zip(indices, sizes_pt, strict=False))

    docs: dict[str, Doc] = {}
    mineru_po = _load_parser_output(args.mineru, "mineru", notes)
    if mineru_po is not None:
        docs["mineru"] = doc_from_parser_output(mineru_po, "mineru", size_by_idx)
    docling_po = _load_parser_output(args.docling, "docling", notes)
    if docling_po is not None:
        docs["docling"] = doc_from_parser_output(docling_po, "docling", size_by_idx)
    docs["pdfplumber"] = doc_from_pdfplumber(pdf_path)  # always-available baseline

    out.write_text(build_html(pages_b64, docs, pdf_path.name, notes=notes, page_indices=indices))
    print(f"wrote {out}  ({len(pages_b64)} pages, backends: {', '.join(docs)})")
    return 0


def _page_count(pdf_path: Path) -> int:
    """Page count of a PDF, degrading to 0 (never raising) on any failure."""
    try:
        import pypdfium2 as pdfium  # noqa: PLC0415

        pdf = pdfium.PdfDocument(str(pdf_path))
        try:
            return len(pdf)
        finally:
            pdf.close()
    except Exception:  # noqa: BLE001 — missing dep / corrupt PDF degrades to 0
        return 0


if __name__ == "__main__":
    sys.exit(main())
