"""
Generate doc_bench prediction files from hybrid_doc_parser outputs.

Sets up benchmark directory structure and parses each bundled fixture PDF/image,
converting ParserOutput to doc_bench's parser_output.schema.json format and
writing <doc_id>.json files to the predictions/ directory.

Handles ato_bench, dp_bench, and omnidocbench bundled fixtures.

Usage:
    CUDA_VISIBLE_DEVICES="" uv run python scripts/generate_predictions.py
"""

import json
import os
import sys
import argparse
from pathlib import Path

# NOTE: We no longer force CPU here — the benchmark runs each backend on the GPU
# (MinerU pipeline, MinerU2.5-Pro vLLM) when one is present. To force CPU (e.g.
# for Docling or a GPU-less host) set CUDA_VISIBLE_DEVICES="" in the environment
# before invoking this script; it is respected as-is.

# Resolve project root so this script can be run from any working directory.
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import doc_bench as _doc_bench_pkg  # noqa: E402  (after sys.path mutation)

from hybrid_doc_parser import ElementType, ParserOutput, parse  # noqa: E402
from hybrid_doc_parser.models import EnrichmentConfig  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FIXTURES_DIR: Path = Path(_doc_bench_pkg.__file__).parent / "fixtures"
PREDICTIONS_DIR: Path = PROJECT_ROOT / "predictions"
BENCHMARK_DIR: Path = PROJECT_ROOT / "benchmark"
PARSER_VERSION = "0.1.0"

# Backend used for every parse() call in this run. Set from --parser in main();
# defaults to the library default (MinerU pipeline).
PARSER_CONFIG: EnrichmentConfig = EnrichmentConfig()

# Map our ElementType values to doc_bench element types.
_TYPE_MAP: dict[ElementType, str] = {
    ElementType.text: "paragraph",
    ElementType.heading: "heading",
    ElementType.table: "table",
    ElementType.image: "figure",
    ElementType.equation: "equation",
    ElementType.list_item: "list_item",
    ElementType.caption: "caption",
    ElementType.header: "header",
    ElementType.footer: "footer",
    ElementType.page_number: "page_number",
    ElementType.unknown: "paragraph",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_page_dimensions(pdf_path: Path) -> list[dict]:
    """Return per-page width/height in PDF points via pypdfium2."""
    try:
        import pypdfium2 as pdfium

        doc = pdfium.PdfDocument(str(pdf_path))
        pages = []
        for i in range(len(doc)):
            pg = doc[i]
            pages.append(
                {
                    "page_index": i,
                    "width": float(pg.get_width()),
                    "height": float(pg.get_height()),
                    "rotation": 0,
                }
            )
        doc.close()
        return pages
    except Exception as exc:
        print(f"    [WARN] pypdfium2 page dims failed: {exc}")
        return []


def _parse_markdown_table_to_content(text: str) -> dict | None:
    """
    Parse a GFM markdown table into TableContent format.

    Returns None if the text is not a recognisable markdown table.
    """
    lines = [ln.strip() for ln in text.strip().splitlines()]
    # Keep only actual data rows — skip separator lines (only |, -, space).
    data_rows = [ln for ln in lines if ln and not all(c in "-| :" for c in ln)]
    if not data_rows:
        return None

    parsed: list[list[str]] = []
    for row_line in data_rows:
        cells = [c.strip() for c in row_line.strip("|").split("|")]
        parsed.append(cells)

    n_cols = max(len(r) for r in parsed) if parsed else 0
    cells: list[dict] = []
    for row_idx, row in enumerate(parsed):
        for col_idx in range(n_cols):
            cell_text = row[col_idx] if col_idx < len(row) else ""
            cells.append(
                {
                    "row": row_idx,
                    "col": col_idx,
                    "text": cell_text,
                    "is_header": row_idx == 0,
                }
            )

    return {
        "kind": "table",
        "rows": len(parsed),
        "cols": n_cols,
        "header_rows": 1,
        "cells": cells,
    }


def _build_doc_bench_element(rec, char_offset: int) -> tuple[dict, int]:
    """
    Convert one ElementRecord to a doc_bench element dict.

    Returns (element_dict, updated_char_offset).
    """
    db_type = _TYPE_MAP.get(rec.type, "paragraph")
    raw_text: str = rec.text or ""

    # Decode heading level from # prefix added by _route_block.
    heading_level = 1
    display_text = raw_text
    if db_type == "heading" and raw_text.startswith("#"):
        stripped = raw_text.lstrip("#")
        heading_level = len(raw_text) - len(stripped)
        display_text = stripped.strip()

    char_start = char_offset
    char_end = char_offset + len(display_text)

    # Build type-specific content block.
    if db_type == "table":
        table_content = _parse_markdown_table_to_content(display_text)
        content: dict = table_content if table_content else {"kind": "text"}
    elif db_type == "figure":
        # Prefer VLM description when the element was enriched.
        alt = rec.description if rec.is_enriched and rec.description else display_text
        content = {"kind": "figure", "alt_text": alt}
    else:
        # paragraph, heading, list_item, caption, header, footer,
        # page_number, equation, unknown → TextContent.
        content = {"kind": "text"}

    elem: dict = {
        "element_id": rec.element_id,
        "type": db_type,
        "page_index": rec.page_idx,
        "char_span": [char_start, char_end],
        "text": display_text,
        "content": content,
    }

    if db_type == "heading":
        elem["level"] = heading_level

    if rec.bbox and len(rec.bbox) == 4:
        elem["bbox"] = {
            "x0": rec.bbox[0],
            "y0": rec.bbox[1],
            "x1": rec.bbox[2],
            "y1": rec.bbox[3],
        }

    # Advance offset past this element's text plus a space separator.
    return elem, char_end + 1


def _get_image_dimensions(image_path: Path) -> tuple[float, float]:
    """Return (width, height) in pixels for an image file via Pillow."""
    try:
        from PIL import Image

        with Image.open(image_path) as img:
            return float(img.width), float(img.height)
    except Exception:
        return 1700.0, 2200.0  # OmniDocBench typical scan size


def convert_to_doc_bench_schema(
    output: ParserOutput,
    doc_id: str,
    source_path: Path,
    page_dims_override: list[dict] | None = None,
    mime_type: str = "application/pdf",
) -> dict:
    """Convert a ParserOutput into a doc_bench schema-conformant dict."""
    if page_dims_override is not None:
        page_dims = page_dims_override
    else:
        page_dims = _get_page_dimensions(source_path)
    if not page_dims:
        page_dims = [
            {"page_index": i, "width": 595.0, "height": 842.0, "rotation": 0}
            for i in range(output.page_count)
        ]

    elements: list[dict] = []
    char_offset = 0
    for rec in output.elements:
        elem, char_offset = _build_doc_bench_element(rec, char_offset)
        elements.append(elem)

    warnings = [{"code": w.code, "message": w.message} for w in output.warnings]

    return {
        "schema_version": "1.0.0",
        "parser_version": PARSER_VERSION,
        "source": {
            "doc_id": doc_id,
            "filename": source_path.name,
            "mime_type": mime_type,
            "sha256": output.file_sha256,
            "page_count": output.page_count,
            "language": "en",
        },
        "pages": page_dims,
        "elements": elements,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Benchmark directory setup
# ---------------------------------------------------------------------------


def setup_dp_bench_dir() -> Path:
    """
    Synthesise the standard dp_bench directory layout expected by load_dp_bench.

    Creates benchmark/dp_bench/reference.json (mapping pdf_name → gold_elements)
    and benchmark/dp_bench/pdfs/ (symlinks to bundled fixture PDFs).

    Returns:
        Path to benchmark/dp_bench/.
    """
    dp_fixtures = FIXTURES_DIR / "dp_bench"
    dp_bench_dir = BENCHMARK_DIR / "dp_bench"
    pdfs_dir = dp_bench_dir / "pdfs"
    pdfs_dir.mkdir(parents=True, exist_ok=True)

    # Build reference.json from individual gold JSON files.
    reference: dict = {}
    for gold_path in sorted(dp_fixtures.glob("*.json")):
        pdf_name = gold_path.stem + ".pdf"
        gold_data = json.loads(gold_path.read_text())
        reference[pdf_name] = gold_data

        # Symlink PDF into pdfs/ if not already present.
        pdf_src = dp_fixtures / pdf_name
        pdf_dst = pdfs_dir / pdf_name
        if pdf_src.exists() and not pdf_dst.exists():
            pdf_dst.symlink_to(pdf_src.resolve())

    ref_path = dp_bench_dir / "reference.json"
    ref_path.write_text(json.dumps(reference, indent=2))
    print(f"  dp_bench layout written to: {dp_bench_dir}")
    print(f"    reference.json: {len(reference)} documents")
    return dp_bench_dir


# ---------------------------------------------------------------------------
# Prediction generation
# ---------------------------------------------------------------------------


def process_file(
    doc_id: str,
    source_path: Path,
    page_dims_override: list[dict] | None = None,
    mime_type: str = "application/pdf",
) -> None:
    """Parse one PDF or image, convert to doc_bench schema, write predictions/<doc_id>.json."""
    print(f"  Parsing {source_path.name} (parser={PARSER_CONFIG.parser}) ...")
    output = parse(source_path, PARSER_CONFIG)

    n_warnings = len(output.warnings)
    n_elements = len(output.elements)
    if n_warnings:
        for w in output.warnings:
            print(f"    [WARN] {w.code}: {w.message}")

    db_output = convert_to_doc_bench_schema(
        output,
        doc_id,
        source_path,
        page_dims_override=page_dims_override,
        mime_type=mime_type,
    )

    pred_path = PREDICTIONS_DIR / f"{doc_id}.json"
    pred_path.write_text(json.dumps(db_output, indent=2))
    print(
        f"  -> {pred_path.name}  "
        f"pages={output.page_count}  elements={n_elements}  warnings={n_warnings}"
    )


def setup_omnidocbench_dir() -> Path:
    """
    Synthesise an OmniDocBench.json from bundled individual page JSON fixtures.

    The load_omnidocbench() function expects a directory containing OmniDocBench.json
    (a list of page dicts). The bundled fixtures ship as individual page files.
    We combine them into the expected format.

    Returns:
        Path to benchmark/omnidocbench/ (containing OmniDocBench.json).
    """
    omni_fixtures = FIXTURES_DIR / "omnidocbench"
    omni_dir = BENCHMARK_DIR / "omnidocbench"
    omni_dir.mkdir(parents=True, exist_ok=True)

    pages: list[dict] = []
    for page_json in sorted(omni_fixtures.glob("*.json")):
        pages.append(json.loads(page_json.read_text()))

    omni_json_path = omni_dir / "OmniDocBench.json"
    omni_json_path.write_text(json.dumps(pages, indent=2))
    print(f"  omnidocbench OmniDocBench.json written: {len(pages)} pages")
    return omni_dir


def main() -> None:
    """Entry point: set up benchmark layout and generate all predictions."""
    global PARSER_CONFIG, PREDICTIONS_DIR

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--parser",
        choices=["mineru", "docling", "paddleocr", "mineru25pro"],
        default="mineru",
        help="parsing backend to benchmark (default: mineru)",
    )
    ap.add_argument(
        "--predictions-dir",
        type=Path,
        default=None,
        help="output directory for <doc_id>.json predictions (default: ./predictions)",
    )
    args = ap.parse_args()

    PARSER_CONFIG = EnrichmentConfig(parser=args.parser)
    if args.predictions_dir is not None:
        PREDICTIONS_DIR = args.predictions_dir
    print(f"[generate_predictions] backend={args.parser}  predictions_dir={PREDICTIONS_DIR}")

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Set up dp_bench benchmark directory --------------------------------
    print("[setup] Building dp_bench benchmark directory ...")
    setup_dp_bench_dir()

    # --- Set up omnidocbench benchmark directory ----------------------------
    print("[setup] Building omnidocbench benchmark directory ...")
    setup_omnidocbench_dir()

    # --- ATO-Bench ----------------------------------------------------------
    print("\n[ato_bench] Generating predictions ...")
    ato_pdf = FIXTURES_DIR / "ato_bench" / "1371-6.1997.pdf"
    if ato_pdf.exists():
        process_file("1371-6.1997", ato_pdf)
    else:
        print(f"  [ERROR] PDF not found: {ato_pdf}")

    # --- DP-Bench -----------------------------------------------------------
    print("\n[dp_bench] Generating predictions ...")
    dp_fixtures = FIXTURES_DIR / "dp_bench"
    for pdf_path in sorted(dp_fixtures.glob("*.pdf")):
        process_file(pdf_path.stem, pdf_path)

    # --- OmniDocBench -------------------------------------------------------
    print("\n[omnidocbench] Generating predictions ...")
    omni_fixtures = FIXTURES_DIR / "omnidocbench"
    for page_json_path in sorted(omni_fixtures.glob("*.json")):
        page_data = json.loads(page_json_path.read_text())
        page_info = page_data.get("page_info", {})
        image_filename = page_info.get("image_path", "")
        doc_id = Path(image_filename).stem
        if not doc_id:
            print(f"  [SKIP] no image_path in {page_json_path.name}")
            continue

        # Find the bundled image (png or jpg).
        image_path: Path | None = None
        for ext in (".png", ".jpg", ".jpeg"):
            candidate = omni_fixtures / (doc_id + ext)
            if candidate.exists():
                image_path = candidate
                break

        if image_path is None:
            print(f"  [SKIP] image not found for {doc_id}")
            continue

        # Page dimensions come from page_info (already in pixels).
        w = float(page_info.get("width", 1700))
        h = float(page_info.get("height", 2200))
        page_dims = [{"page_index": 0, "width": w, "height": h, "rotation": 0}]

        suffix = image_path.suffix.lower()
        mime = "image/png" if suffix == ".png" else "image/jpeg"

        process_file(doc_id, image_path, page_dims_override=page_dims, mime_type=mime)


if __name__ == "__main__":
    main()
