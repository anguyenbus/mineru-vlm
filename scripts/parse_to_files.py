"""Parse one or more documents with MinerU and write RAG-ready output files.

Unlike ``scripts/save_parser_json.py`` (which feeds the dual-backend Parse
Report viewer), this script is the single-backend "give me the artifacts I feed
downstream" path. It parses every input via ``parse_batch`` — which drives
MinerU's BATCHED ``do_parse`` (one inference window per chunk, models stay
resident) rather than re-running per file — and, for each document, writes three
files into ``<out-dir>/<stem>/`` (default base ``report/``):

    <stem>.md              - RAG-ready Markdown (``render_markdown``); text only,
                             no confidence scores by design.
    <stem>.json            - the full ``ParserOutput`` JSON, which INCLUDES the
                             ``confidence`` block (MinerU pipeline backend).
    <stem>.confidence.json - a compact, per-page average-confidence summary
                             extracted from ``ParserOutput.confidence``.

Usage:
    python scripts/parse_to_files.py SRC [SRC ...] [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from hybrid_doc_parser import parse_batch
from hybrid_doc_parser.markdown import render_markdown
from hybrid_doc_parser.models import ParserOutput


def _confidence_summary(out: ParserOutput) -> dict:
    """Extract a compact per-page average-confidence summary from ``out``.

    Confidence is only produced by the MinerU pipeline backend; for Docling (or
    when capture degraded) ``out.confidence`` is ``None`` and ``available`` is
    ``False``.
    """
    c = out.confidence
    if c is None:
        return {"available": False, "backend": out.enrichment_config.parser, "pages": []}
    return {
        "available": True,
        "source_path": c.source_path,
        "backend": c.backend,
        "version_name": c.version_name,
        "total_pages": c.total_pages,
        "overall_mean_block_score": c.overall_mean_block_score,
        "overall_mean_span_score": c.overall_mean_span_score,
        "pages_flagged": c.pages_flagged,
        "pages": [
            {
                "page_idx": p.page_idx,
                "mean_block_score": p.mean_block_score,
                "mean_span_score": p.mean_span_score,
                "flagged": p.flagged,
            }
            for p in c.pages
        ],
    }


def _write_outputs(src: Path, out: ParserOutput, base: Path) -> Path:
    """Write the markdown, full JSON, and confidence summary for one document."""
    out_dir = base / src.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{src.stem}.md").write_text(render_markdown(out))
    (out_dir / f"{src.stem}.json").write_text(out.model_dump_json())
    (out_dir / f"{src.stem}.confidence.json").write_text(
        json.dumps(_confidence_summary(out), indent=2)
    )
    return out_dir


def main(argv: list[str] | None = None) -> int:
    """Batch-parse the inputs and write per-document RAG artifacts."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", nargs="+", help="one or more source document paths")
    ap.add_argument(
        "--out-dir",
        default="report",
        help="base output dir; each file writes to <out-dir>/<stem>/ (default: report)",
    )
    args = ap.parse_args(argv)

    srcs = [Path(s) for s in args.src]
    for missing in [s for s in srcs if not s.exists()]:
        print(f"[parse_to_files] source not found (skipped): {missing}", file=sys.stderr)
    srcs = [s for s in srcs if s.exists()]
    if not srcs:
        print("[parse_to_files] no existing source files to parse", file=sys.stderr)
        return 1

    base = Path(args.out_dir)
    print(f"[parse_to_files] parsing {len(srcs)} document(s) with MinerU (batched) ...")
    # parse_batch returns one ParserOutput per input, IN INPUT ORDER.
    outputs = asyncio.run(parse_batch(srcs))

    for src, out in zip(srcs, outputs, strict=True):
        out_dir = _write_outputs(src, out, base)
        print(
            f"[parse_to_files] {src.name} -> {out_dir}/  "
            f"(pages={out.page_count}, elements={len(out.elements)}, "
            f"confidence={'present' if out.confidence is not None else 'none'}, "
            f"warnings={[w.code for w in out.warnings]})"
        )
    print(f"[parse_to_files] wrote <stem>.md / <stem>.json / <stem>.confidence.json for {len(srcs)} file(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
