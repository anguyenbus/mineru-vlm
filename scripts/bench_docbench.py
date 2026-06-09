"""Benchmark parse_batch() over the 11 doc-bench fixture documents on real MinerU.

Collects the 6 PDFs + 5 image pages bundled in doc_bench/fixtures, runs them
through parse_batch() with a fresh cache (nothing cached), and reports total
wall-clock time, the real do_parse call count, and per-document results.

Run: PYTHONPATH=src MINERU_DEVICE_MODE=cuda .venv/bin/python scripts/bench_docbench.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import time
from pathlib import Path

import doc_bench
import mineru.cli.common as mineru_common

from hybrid_doc_parser import parser as P
from hybrid_doc_parser.models import EnrichmentConfig

FIXTURES = Path(doc_bench.__file__).parent / "fixtures"
EXTS = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}


def collect_docs() -> list[Path]:
    docs = sorted(p for p in FIXTURES.rglob("*") if p.suffix.lower() in EXTS)
    return docs


# Count real do_parse calls.
_calls = {"n": 0}
_orig = mineru_common.do_parse


def _counting_do_parse(*a, **k):
    _calls["n"] += 1
    return _orig(*a, **k)


mineru_common.do_parse = _counting_do_parse


async def run(docs: list[Path], cfg: EnrichmentConfig):
    return await P.parse_batch(docs, cfg)


def main() -> int:
    docs = collect_docs()
    print(f"[bench] found {len(docs)} documents under {FIXTURES}")
    for d in docs:
        print(f"[bench]   {d.relative_to(FIXTURES)}  ({d.stat().st_size // 1024} KiB)")

    # Fresh cache dir so nothing is a cache hit.
    os.environ["HYBRID_DOC_PARSER_CACHE_DIR"] = tempfile.mkdtemp(prefix="bench_cache_")
    batch_size = os.environ.get("MINERU_BATCH_SIZE", "8")
    device = os.environ.get("MINERU_DEVICE_MODE", "(default)")
    print(
        f"\n[bench] device={device}  MINERU_BATCH_SIZE={batch_size}  "
        f"expected do_parse calls = ceil({len(docs)}/{batch_size})"
    )

    which = os.environ.get("BENCH_PARSER", "mineru")
    print(f"[bench] parser={which}")
    t0 = time.perf_counter()
    results = asyncio.run(run(docs, EnrichmentConfig(parser=which)))
    elapsed = time.perf_counter() - t0

    print("\n[bench] === RESULTS ===")
    total_pages = total_elems = 0
    failed = 0
    for d, r in zip(docs, results, strict=True):
        codes = [w.code for w in r.warnings]
        total_pages += r.page_count
        total_elems += len(r.elements)
        if any(
            c in ("mineru_failed", "mineru_error", "docling_failed", "docling_error") for c in codes
        ):
            failed += 1
        print(
            f"[bench]   {d.relative_to(FIXTURES)!s:60s} "
            f"pages={r.page_count:<3} elements={len(r.elements):<4} warnings={codes}"
        )

    print(f"\n[bench] documents:        {len(docs)}")
    print(f"[bench] real do_parse:    {_calls['n']} calls (per-file design would be {len(docs)})")
    print(f"[bench] total pages:      {total_pages}")
    print(f"[bench] total elements:   {total_elems}")
    print(f"[bench] failed docs:      {failed}")
    print(f"[bench] WALL-CLOCK TIME:  {elapsed:.1f} s  ({elapsed / len(docs):.2f} s/doc)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
