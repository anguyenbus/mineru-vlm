"""Parse a source document with MinerU and Docling and save each ParserOutput JSON.

Used by `make report-json` to produce the saved `ParserOutput.model_dump_json()`
files the Parse Report viewer consumes, so the viewer can be re-run instantly
without touching the GPU again.

Usage:
    python scripts/save_parser_json.py SRC [--mineru-out m.json] [--docling-out d.json]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hybrid_doc_parser import parse
from hybrid_doc_parser.models import EnrichmentConfig


def main(argv: list[str] | None = None) -> int:
    """Parse ``src`` with both backends and write each ParserOutput JSON."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", help="source document path")
    ap.add_argument("--mineru-out", default="m.json")
    ap.add_argument("--docling-out", default="d.json")
    ap.add_argument(
        "--paddleocr-out",
        default=None,
        help="if set, also parse with PaddleOCR-VL and write here",
    )
    args = ap.parse_args(argv)

    src = Path(args.src)
    if not src.exists():
        print(f"[save_parser_json] source not found: {src}", file=sys.stderr)
        return 1

    # NOTE: the parse cache now folds the EnrichmentConfig into its key, so the
    # MinerU and Docling parses below cache independently (no cross-backend hit).
    print(f"[save_parser_json] parsing {src} with MinerU ...")
    mineru_out = parse(src)
    Path(args.mineru_out).write_text(mineru_out.model_dump_json())
    print(
        f"[save_parser_json]   -> {args.mineru_out} "
        f"(pages={mineru_out.page_count}, elements={len(mineru_out.elements)}, "
        f"warnings={[w.code for w in mineru_out.warnings]})"
    )

    print(f"[save_parser_json] parsing {src} with Docling ...")
    docling_out = parse(src, EnrichmentConfig(parser="docling"))
    Path(args.docling_out).write_text(docling_out.model_dump_json())
    print(
        f"[save_parser_json]   -> {args.docling_out} "
        f"(pages={docling_out.page_count}, elements={len(docling_out.elements)}, "
        f"warnings={[w.code for w in docling_out.warnings]})"
    )

    if args.paddleocr_out:
        print(f"[save_parser_json] parsing {src} with PaddleOCR-VL ...")
        paddle_out = parse(src, EnrichmentConfig(parser="paddleocr"))
        Path(args.paddleocr_out).write_text(paddle_out.model_dump_json())
        print(
            f"[save_parser_json]   -> {args.paddleocr_out} "
            f"(pages={paddle_out.page_count}, elements={len(paddle_out.elements)}, "
            f"warnings={[w.code for w in paddle_out.warnings]})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
