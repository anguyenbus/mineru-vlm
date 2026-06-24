"""Parse a source document with MinerU and Docling and save each ParserOutput JSON.

Used by `make report-json` to produce the saved `ParserOutput.model_dump_json()`
files the Parse Report viewer consumes, so the viewer can be re-run instantly
without touching the GPU again.

Usage:
    python scripts/save_parser_json.py SRC [--mineru-out m.json] [--docling-out d.json]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from hybrid_doc_parser import parse
from hybrid_doc_parser.models import EnrichmentConfig

# Unlimited-OCR runs in its OWN venv (transformers 4.57.1, separate from the
# project env). We shell out to it for inference, then build the ParserOutput in
# this (project) process. Override the interpreter with UOCR_PYTHON if needed.
_UOCR_PYTHON = os.environ.get("UOCR_PYTHON", "/workspace/uocr-venv/bin/python")


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
        "--unlimited-out",
        default=None,
        help="if set, also parse with Unlimited-OCR (separate venv) and write here",
    )
    ap.add_argument(
        "--only",
        choices=["mineru", "docling", "paddleocr", "mineru25pro", "unlimited"],
        default=None,
        help="run a single backend only (avoids loading multiple GPU models at once)",
    )
    ap.add_argument("--dpi", type=int, default=300, help="rasterisation DPI for Unlimited-OCR")
    ap.add_argument("--max-length", type=int, default=6144, help="Unlimited-OCR per-page token cap")
    ap.add_argument("--base-size", type=int, default=1024, help="Unlimited-OCR global thumbnail size")
    ap.add_argument(
        "--image-size",
        type=int,
        default=640,
        help=(
            "Unlimited-OCR per-tile crop size. 640 is the trained single-image value; "
            "768/1024 OOM/cuBLAS-fail on a 15GB GPU. Raise --dpi for small-text recall."
        ),
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

    def _run_unlimited(out_path: str) -> None:
        # Inference in the Unlimited-OCR venv -> intermediate raw JSON; convert here.
        # Keep the raw next to the output so converter fixes can re-run WITHOUT
        # re-touching the GPU (re-convert via uocr_convert.build_parser_output).
        print(f"[save_parser_json] parsing {src} with Unlimited-OCR ({_UOCR_PYTHON}) ...")
        scripts_dir = Path(__file__).resolve().parent
        raw_path = str(Path(out_path).with_suffix(".raw.json"))
        subprocess.run(
            [
                _UOCR_PYTHON,
                str(scripts_dir / "run_unlimited_ocr.py"),
                "--pdf", str(src),
                "--out", raw_path,
                "--dpi", str(args.dpi),
                "--max-length", str(args.max_length),
                "--base-size", str(args.base_size),
                "--image-size", str(args.image_size),
            ],
            check=True,
        )
        from uocr_convert import build_parser_output  # noqa: PLC0415

        out = build_parser_output(json.loads(Path(raw_path).read_text()))
        Path(out_path).write_text(out.model_dump_json())
        print(
            f"[save_parser_json]   -> {out_path} "
            f"(pages={out.page_count}, elements={len(out.elements)})"
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
    if (want == "unlimited") or (want is None and args.unlimited_out):
        _run_unlimited(args.unlimited_out or "uocr.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
