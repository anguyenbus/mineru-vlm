"""Run baidu/Unlimited-OCR on a PDF (GPU) and emit an intermediate JSON.

This runs in the dedicated Unlimited-OCR venv (transformers 4.57.1), which is
SEPARATE from the project venv. It does inference only and writes raw per-page
grounded text; the project venv (scripts/uocr_convert.py) turns that into a
``ParserOutput`` JSON the parse-report viewer consumes.

Intermediate JSON shape:
    {"file_path", "file_sha256", "page_count", "elapsed_s",
     "pages": [{"page_index": int, "raw_text": str}]}

Usage (uocr venv):
    /workspace/uocr-venv/bin/python scripts/run_unlimited_ocr.py \
        --pdf file.pdf --out report/<stem>/uocr_raw.json [--dpi 200 --max-length 4096]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path

MODEL_NAME = "baidu/Unlimited-OCR"
PROMPT = "<image>document parsing."


def pdf_to_images(pdf_path: Path, dpi: int) -> list[str]:
    import fitz  # PyMuPDF

    doc = fitz.open(str(pdf_path))
    tmp_dir = Path(tempfile.mkdtemp(prefix="uocr_pages_"))
    zoom = dpi / 72.0
    paths = []
    for i, page in enumerate(doc):
        out = tmp_dir / f"page_{i + 1:04d}.png"
        page.get_pixmap(matrix=fitz.Matrix(zoom, zoom)).save(str(out))
        paths.append(str(out))
    doc.close()
    return paths


def load_model():
    import torch
    from transformers import AutoModel, AutoTokenizer

    print(f"[uocr] loading {MODEL_NAME} (bf16, cuda) ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = (
        AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True, torch_dtype=torch.bfloat16)
        .eval()
        .cuda()
    )
    return model, tok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdf", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--dpi", type=int, default=300, help="raster DPI; 300 resolves fine print")
    ap.add_argument("--max-length", type=int, default=6144)
    ap.add_argument("--base-size", type=int, default=1024, help="global thumbnail resolution")
    ap.add_argument(
        "--image-size",
        type=int,
        default=640,
        help=(
            "per-tile crop resolution. 640 is the value the single-image vision "
            "encoder is trained for; 768/1024 fail with CUBLAS_STATUS_EXECUTION_FAILED "
            "/ OOM on a 15GB GPU. Raise input --dpi (not this) for small-text recall."
        ),
    )
    args = ap.parse_args()

    pdf = args.pdf.resolve()
    if not pdf.exists():
        print(f"[uocr] ERROR: pdf not found: {pdf}", file=sys.stderr)
        return 1

    images = pdf_to_images(pdf, args.dpi)
    print(f"[uocr] rasterised {pdf.name} @ {args.dpi} dpi -> {len(images)} pages", flush=True)

    model, tok = load_model()
    work = Path(tempfile.mkdtemp(prefix="uocr_out_"))

    pages = []
    t0 = time.perf_counter()
    for idx, img in enumerate(images):
        ts = time.perf_counter()
        out = model.infer(
            tok,
            prompt=PROMPT,
            image_file=img,
            output_path=str(work / f"p{idx}"),
            base_size=args.base_size,
            image_size=args.image_size,
            crop_mode=True,
            max_length=args.max_length,
            no_repeat_ngram_size=35,
            ngram_window=128,
            eval_mode=True,
        )
        text = (out or "").strip()
        print(f"[uocr]   page {idx + 1}/{len(images)}: {len(text)} chars in {time.perf_counter() - ts:.1f}s", flush=True)
        pages.append({"page_index": idx, "raw_text": text})
    elapsed = time.perf_counter() - t0

    payload = {
        "file_path": str(pdf),
        "file_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
        "page_count": len(images),
        "elapsed_s": round(elapsed, 2),
        "pages": pages,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload), encoding="utf-8")
    print(f"[uocr] DONE {elapsed:.1f}s ({elapsed / len(images):.1f}s/page) -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
