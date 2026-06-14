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
    ap.add_argument(
        "--mineru25pro-out",
        default=None,
        help="if set, also parse with MinerU2.5-Pro (vLLM) and write here",
    )
    ap.add_argument(
        "--only",
        choices=["mineru", "docling", "paddleocr", "mineru25pro"],
        default=None,
        help="run a single backend only (avoids loading multiple GPU models at once)",
    )
    args = ap.parse_args(argv)

    src = Path(args.src)
    if not src.exists():
        print(f"[save_parser_json] source not found: {src}", file=sys.stderr)
        return 1

    # NOTE: the parse cache now folds the EnrichmentConfig into its key, so each
    # backend below caches independently (no cross-backend hit). ``--only`` runs a
    # single backend per process so GPU-resident backends (MinerU pipeline, the
    # MinerU2.5-Pro vLLM engine) never co-load and contend for VRAM.
    def _run(name: str, out_path: str, cfg: EnrichmentConfig | None) -> None:
        print(f"[save_parser_json] parsing {src} with {name} ...")
        out = parse(src) if cfg is None else parse(src, cfg)
        Path(out_path).write_text(out.model_dump_json())
        print(
            f"[save_parser_json]   -> {out_path} "
            f"(pages={out.page_count}, elements={len(out.elements)}, "
            f"warnings={[w.code for w in out.warnings]})"
        )

    want = args.only

    if want in (None, "mineru"):
        _run("MinerU", args.mineru_out, None)
    if want in (None, "docling"):
        _run("Docling", args.docling_out, EnrichmentConfig(parser="docling"))
    if (want == "paddleocr") or (want is None and args.paddleocr_out):
        _run("PaddleOCR-VL", args.paddleocr_out or "p.json", EnrichmentConfig(parser="paddleocr"))
    if (want == "mineru25pro") or (want is None and args.mineru25pro_out):
        _run(
            "MinerU2.5-Pro",
            args.mineru25pro_out or "m25pro.json",
            EnrichmentConfig(parser="mineru25pro"),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
