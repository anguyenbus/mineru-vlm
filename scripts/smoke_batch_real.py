"""Real end-to-end smoke for the MinerU batch fast path.

Runs parse_batch() over the fixture PDFs with REAL MinerU (no mocks) and asserts
that N files funnel through exactly ONE do_parse call (chunk size default 8),
that output order matches input, and that elements were extracted.

Run: .venv/bin/python scripts/smoke_batch_real.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import mineru.cli.common as mineru_common

from hybrid_doc_parser import parser as P
from hybrid_doc_parser.models import EnrichmentConfig

FIXTURES = Path("tests/hybrid_doc_parser/fixtures")
PDFS = [FIXTURES / "digital_simple.pdf", FIXTURES / "mixed.pdf"]

# Count REAL do_parse invocations by wrapping the module attribute the lazy
# import inside _run_mineru_batch_chunk resolves at call time.
_calls = {"n": 0}
_orig_do_parse = mineru_common.do_parse


def _counting_do_parse(*args, **kwargs):
    _calls["n"] += 1
    return _orig_do_parse(*args, **kwargs)


mineru_common.do_parse = _counting_do_parse


def main() -> int:
    for p in PDFS:
        assert p.exists(), f"missing fixture: {p}"

    # Fresh cache dir so nothing is a cache hit.
    import os
    import tempfile

    os.environ["HYBRID_DOC_PARSER_CACHE_DIR"] = tempfile.mkdtemp(prefix="smoke_cache_")

    results = asyncio.run(parse_batch_wrapper())

    print(f"\n[smoke] real do_parse calls = {_calls['n']}")
    print(f"[smoke] results returned     = {len(results)}")
    for r, p in zip(results, PDFS, strict=True):
        codes = [w.code for w in r.warnings]
        print(
            f"[smoke]   {p.name}: pages={r.page_count} elements={len(r.elements)} warnings={codes}"
        )

    ok = True
    if _calls["n"] != 1:
        print(f"[smoke] FAIL: expected exactly 1 do_parse call, got {_calls['n']}")
        ok = False
    if len(results) != len(PDFS):
        print("[smoke] FAIL: result count mismatch")
        ok = False
    if [r.file_path for r in results] != [str(p) for p in PDFS]:
        print("[smoke] FAIL: output order does not match input order")
        ok = False
    if not any(len(r.elements) > 0 for r in results):
        print("[smoke] FAIL: no elements extracted from any file")
        ok = False

    print("\n[smoke] RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


async def parse_batch_wrapper():
    return await P.parse_batch(PDFS, EnrichmentConfig(parser="mineru"))


if __name__ == "__main__":
    sys.exit(main())
