"""Run MinerU2.5-Pro-1.2B (vLLM) on a PDF and emit an intermediate JSON.

Runs in an ISOLATED venv (vllm + mineru-vl-utils[vllm]), separate from the
project venv, so the project's pinned torch is never disturbed. Does inference
only; the project venv (scripts/m25pro_convert.py) turns the raw blocks into a
``ParserOutput`` JSON.

Intermediate JSON shape:
    {"file_path", "file_sha256", "page_count", "elapsed_s",
     "pages": [{"page_index": int, "blocks": [{"type","content","bbox"}]}]}

Usage (m25 venv):
    /workspace/m25-venv/bin/python scripts/run_mineru25pro.py \
        --pdf file.pdf --out report/<stem>/m25pro_raw.json [--dpi 144]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

_DEFAULT_MODEL = "opendatalab/MinerU2.5-Pro-2604-1.2B"


def render_pages(pdf: Path, dpi: int):
    import pypdfium2 as pdfium

    scale = dpi / 72.0
    pdf_doc = pdfium.PdfDocument(str(pdf))
    images = []
    try:
        for page in pdf_doc:
            bmp = page.render(scale=scale)
            images.append(bmp.to_pil().convert("RGB"))
            bmp.close()
            page.close()
    finally:
        pdf_doc.close()
    return images


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdf", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--dpi", type=int, default=144)
    ap.add_argument("--gpu-mem-util", type=float, default=float(os.environ.get("MINERU25PRO_GPU_MEM_UTIL", "0.7")))
    ap.add_argument("--dtype", default="bfloat16", help="vLLM dtype (bfloat16/float16/auto)")
    ap.add_argument(
        "--enforce-eager",
        action="store_true",
        help="disable CUDA graphs (rules out graph-capture garbage-output issues)",
    )
    ap.add_argument(
        "--backend",
        choices=["transformers", "vllm"],
        default="transformers",
        help="inference backend. 'transformers' = pure HF (reliable); 'vllm' = faster but "
        "vLLM 0.21 emits garbage (!!!!) for this VLM on this stack.",
    )
    args = ap.parse_args()

    pdf = args.pdf.resolve()
    if not pdf.exists():
        print(f"[m25pro] ERROR: pdf not found: {pdf}", file=sys.stderr)
        return 1

    model = os.environ.get("MINERU25PRO_MODEL", _DEFAULT_MODEL)
    print(f"[m25pro] rendering {pdf.name} @ {args.dpi} dpi ...", flush=True)
    images = render_pages(pdf, args.dpi)
    print(f"[m25pro] {len(images)} pages; loading vLLM engine ({model}) ...", flush=True)

    from mineru_vl_utils import MinerUClient

    if args.backend == "vllm":
        from vllm import LLM

        llm_kwargs = {"model": model, "gpu_memory_utilization": args.gpu_mem_util, "dtype": args.dtype}
        if args.enforce_eager:
            llm_kwargs["enforce_eager"] = True
        print(f"[m25pro] vLLM kwargs: {llm_kwargs}", flush=True)
        client = MinerUClient(backend="vllm-engine", vllm_llm=LLM(**llm_kwargs))
    else:
        print(f"[m25pro] transformers backend (HF), model_path={model}", flush=True)
        client = MinerUClient(backend="transformers", model_path=model)

    t0 = time.perf_counter()
    per_page_blocks = client.batch_two_step_extract(images)
    elapsed = time.perf_counter() - t0

    pages = []
    total_blocks = 0
    for page_idx, blocks in enumerate(per_page_blocks):
        out_blocks = []
        for blk in blocks or []:
            bbox = getattr(blk, "bbox", None)
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                try:
                    bbox = [float(v) for v in bbox]
                except (TypeError, ValueError):
                    bbox = None
            else:
                bbox = None
            content = getattr(blk, "content", "") or ""
            out_blocks.append(
                {
                    "type": str(getattr(blk, "type", "") or "").lower(),
                    "content": content if isinstance(content, str) else str(content),
                    "bbox": bbox,
                }
            )
        total_blocks += len(out_blocks)
        pages.append({"page_index": page_idx, "blocks": out_blocks})

    payload = {
        "file_path": str(pdf),
        "file_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
        "page_count": len(images),
        "elapsed_s": round(elapsed, 2),
        "pages": pages,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload), encoding="utf-8")
    print(
        f"[m25pro] DONE {elapsed:.1f}s ({elapsed / max(len(images), 1):.1f}s/page), "
        f"{total_blocks} blocks -> {args.out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
