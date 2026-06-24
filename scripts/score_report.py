"""Score saved ParserOutput JSONs against the doc_bench ato_bench gold and write a md report.

For each backend ParserOutput JSON under report/<stem>/ it converts to the
doc_bench schema, renders markdown, and computes NED / TEDS / TEDS-S via
``doc_bench.runners.run_parsing_eval._grade`` (same path used for every backend,
so the numbers are directly comparable).

Usage:
    python scripts/score_report.py --pdf 1371-6.1997.pdf --doc-id 1371-6.1997 \
        --report-dir report/1371-6.1997 --out report/1371-6.1997/score_report.md
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from doc_bench.metrics.parsing.markdown_converter import parser_output_to_markdown  # noqa: E402
from doc_bench.metrics.parsing.ned import ned_score  # noqa: E402
from doc_bench.metrics.parsing.table_teds import (  # noqa: E402
    TEDSEvaluator,
    _extract_tables_from_markdown,
    _markdown_table_to_html,
)
from doc_bench.runners.run_parsing_eval import _strip_equations, load_dataset  # noqa: E402

from generate_predictions import convert_to_doc_bench_schema  # noqa: E402
from hybrid_doc_parser.models import ParserOutput  # noqa: E402

import doc_bench  # noqa: E402


def _gold_json() -> dict:
    """Load the raw ato_bench gold JSON (keeps the table HTML the loader drops)."""
    from pathlib import Path as _P

    return json.loads(
        (_P(doc_bench.__file__).parent / "fixtures/ato_bench/1371-6.1997.json").read_text()
    )


def _gold_table_htmls(gold: dict) -> list[str]:
    """Gold table element bodies that are HTML <table> markup."""
    return [
        e["text"]
        for e in gold.get("elements", [])
        if e.get("type") == "table" and "<table" in (e.get("text") or "").lower()
    ]


def _pred_table_htmls(po: ParserOutput, pred_markdown: str) -> list[str]:
    """Collect a prediction's tables as HTML candidates (HTML / markdown / flat-text)."""
    cands: list[str] = []
    for e in po.elements:
        if e.type.value != "table":
            continue
        t = (e.text or "").strip()
        if not t:
            continue
        if "<table" in t.lower():
            cands.append(t)
        elif "|" in t:
            for md in _extract_tables_from_markdown(t):
                cands.append(_markdown_table_to_html(md))
        else:  # flat text -> degenerate single-cell table (so TEDS reflects no structure)
            cands.append(f"<table><tr><td>{t}</td></tr></table>")
    # also pick up any markdown pipe-tables rendered into the doc_bench markdown
    for md in _extract_tables_from_markdown(pred_markdown):
        cands.append(_markdown_table_to_html(md))
    return cands


def _best_teds(gold_html: str, cands: list[str]) -> tuple[float, float]:
    """Best (TEDS, TEDS-S) over candidate predicted tables vs the gold table."""
    if not gold_html or not cands:
        return 0.0, 0.0
    g = f"<html><body>{gold_html}</body></html>"
    best = (0.0, 0.0)
    for c in cands:
        ph = f"<html><body>{c}</body></html>"
        tc = TEDSEvaluator(structure_only=False).evaluate(ph, g) or 0.0
        if tc > best[0]:
            ts = TEDSEvaluator(structure_only=True).evaluate(ph, g) or 0.0
            best = (tc, ts)
    return best

# (backend key, json filename, display label)
BACKENDS = [
    ("unlimited", "uocr.json", "Unlimited-OCR-3B"),
    ("mineru25pro", "m25pro.json", "MinerU2.5-Pro-1.2B"),
    ("mineru", "m.json", "MinerU (pipeline)"),
    ("docling", "d.json", "Docling"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdf", required=True, type=Path)
    ap.add_argument("--doc-id", required=True)
    ap.add_argument("--report-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    gold = None
    for g in load_dataset("ato_bench", None):
        if g.doc_id == args.doc_id:
            gold = g
            break
    if gold is None:
        print(f"[score] ERROR: {args.doc_id} not in ato_bench gold", file=sys.stderr)
        return 1
    gold_chars = len(gold.text)

    # The shipped ato_bench loader sets html_tables=[], zeroing TEDS for everyone.
    # Pull the gold table HTML straight from the fixture so TEDS is real.
    gold_table_htmls = _gold_table_htmls(_gold_json())
    gold_table_html = gold_table_htmls[0] if gold_table_htmls else ""
    print(f"[score] gold has {len(gold_table_htmls)} HTML table(s)")

    rows = []
    for key, fname, label in BACKENDS:
        p = args.report_dir / fname
        if not p.exists():
            continue
        po = ParserOutput.model_validate_json(p.read_text())
        codes = [w.code for w in po.warnings]
        failed = any(c.endswith(("_failed", "_error")) for c in codes) or len(po.elements) == 0
        db = convert_to_doc_bench_schema(po, args.doc_id, args.pdf)
        md = parser_output_to_markdown(db)
        # NED: same as doc_bench _grade (gold text vs equation-stripped prediction).
        ned = ned_score(gold.text, _strip_equations(md))
        # TEDS: compare the prediction's table(s) against the gold table HTML directly.
        teds, teds_s = _best_teds(gold_table_html, _pred_table_htmls(po, md))
        rows.append(
            {
                "label": label,
                "key": key,
                "ned": ned,
                "teds": teds,
                "teds_s": teds_s,
                "elements": len(po.elements),
                "chars": len(md),
                "warnings": codes,
                "failed": failed,
            }
        )
        print(f"[score] {label}: NED={ned:.4f} TEDS={teds:.4f} TEDS-S={teds_s:.4f} elements={len(po.elements)}")

    rows.sort(key=lambda r: r["ned"], reverse=True)
    best = rows[0]["ned"] if rows else 0.0

    L = []
    L.append("# Parsing Score Report — MinerU2.5-Pro vs Unlimited-OCR vs MinerU vs Docling")
    L.append("")
    L.append(f"**Date:** {datetime.date.today().isoformat()}  ")
    L.append(f"**Document:** `{args.pdf.name}` (doc_id `{args.doc_id}`, 2 pages)  ")
    L.append("**Gold / dataset:** doc_bench `ato_bench` bundled fixture (the PDF is byte-identical to it)  ")
    L.append(f"**Gold text length:** {gold_chars} chars  ")
    L.append("**Hardware:** NVIDIA RTX A4000 (15 GB)  ")
    L.append("**Scoring:** `doc_bench` NED (normalized edit-distance similarity, ↑) via "
             "`ned_score` on the gold text, and TEDS / TEDS-S (table edit-distance, ↑) via "
             "`doc_bench`'s `TEDSEvaluator` comparing each prediction's table to the gold table "
             "HTML. All in [0,1]; identical path for every backend.")
    L.append("")
    L.append("## Scores")
    L.append("")
    L.append("| Backend | NED ↑ | TEDS ↑ | TEDS-S ↑ | Elements | Pred chars | Warnings |")
    L.append("|---|---|---|---|---|---|---|")
    for r in rows:
        ned = f"**{r['ned']:.4f}**" if r["ned"] == best and best > 0 else f"{r['ned']:.4f}"
        warn = ", ".join(r["warnings"]) or "—"
        note = " (FAILED/empty)" if r["failed"] else ""
        L.append(
            f"| {r['label']}{note} | {ned} | {r['teds']:.4f} | {r['teds_s']:.4f} "
            f"| {r['elements']} | {r['chars']} | {warn} |"
        )
    L.append("")
    L.append("Winner (highest NED) is **bold**. NED is the primary text-accuracy metric for this "
             "dense form. TEDS / TEDS-S compare each backend's reconstruction of the gold's one "
             "table (the ETP-codes table, 12×2) against the gold table HTML.")
    L.append("")
    L.append("## Notes")
    L.append("")
    L.append("- All backends parse the same 2-page PDF; scored against the same gold.")
    L.append("- **TEDS fix:** doc_bench's bundled `ato_bench` loader yields `html_tables=[]`, which "
             "forces `_grade` to report TEDS=0 for every backend even though the gold contains an "
             "HTML table. This report instead reads the gold table HTML straight from the fixture and "
             "compares each prediction's best table to it with doc_bench's own `TEDSEvaluator` — so "
             "TEDS here is real, not structurally zero.")
    L.append("- Each backend's table is taken as HTML when emitted as `<table>`, converted from a "
             "markdown pipe-table when emitted that way, or treated as a single-cell table when only "
             "flat text is produced (so structure-less output scores low, as it should).")
    L.append("- MinerU2.5-Pro-1.2B (`opendatalab/MinerU2.5-Pro-2604-1.2B`, a Qwen2-VL model) ran via "
             "the **transformers** backend (HF), NOT vLLM: on this stack vLLM 0.21 emitted garbage "
             "(`!!!!`) and transformers 5.12 broke on `Qwen2VLConfig.max_position_embeddings`; the "
             "working path is `mineru-vl-utils` + transformers 4.57 (no vLLM). ~64 s/page on the A4000.")
    L.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(L), encoding="utf-8")
    print(f"[score] report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
