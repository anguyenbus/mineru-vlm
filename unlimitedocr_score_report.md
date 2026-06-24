# Parsing Score Report — MinerU2.5-Pro vs Unlimited-OCR vs MinerU vs Docling

**Date:** 2026-06-24  
**Document:** `1371-6.1997.pdf` (doc_id `1371-6.1997`, 2 pages)  
**Gold / dataset:** doc_bench `ato_bench` bundled fixture (the PDF is byte-identical to it)  
**Gold text length:** 5218 chars  
**Scoring:** `doc_bench` NED (normalized edit-distance similarity, ↑) via `ned_score` on the gold text, and TEDS / TEDS-S (table edit-distance, ↑) via `doc_bench`'s `TEDSEvaluator` comparing each prediction's table to the gold table HTML. All in [0,1]; identical path for every backend.

## Scores

| Backend | NED ↑ | TEDS ↑ | TEDS-S ↑ | Elements | Pred chars | Warnings |
|---|---|---|---|---|---|---|
| Unlimited-OCR-3B | **0.6873** | 0.9926 | 1.0000 | 129 | 5451 | — |
| MinerU2.5-Pro-1.2B | 0.6825 | 1.0000 | 1.0000 | 133 | 5416 | — |
| Docling | 0.6809 | 0.0000 | 0.0000 | 158 | 4699 | — |
| MinerU (pipeline) | 0.4783 | 0.5960 | 0.6667 | 94 | 4609 | quality_gate_escalation |

Winner (highest NED) is **bold**. NED is the primary text-accuracy metric for this dense form. TEDS / TEDS-S compare each backend's reconstruction of the gold's one table (the ETP-codes table, 12×2) against the gold table HTML.

## Notes

- All backends parse the same 2-page PDF; scored against the same gold.
- **TEDS fix:** doc_bench's bundled `ato_bench` loader yields `html_tables=[]`, which forces `_grade` to report TEDS=0 for every backend even though the gold contains an HTML table. This report instead reads the gold table HTML straight from the fixture and compares each prediction's best table to it with doc_bench's own `TEDSEvaluator` — so TEDS here is real, not structurally zero.
- Each backend's table is taken as HTML when emitted as `<table>`, converted from a markdown pipe-table when emitted that way, or treated as a single-cell table when only flat text is produced (so structure-less output scores low, as it should).
- MinerU2.5-Pro-1.2B (`opendatalab/MinerU2.5-Pro-2604-1.2B`, a Qwen2-VL model) ran via the **transformers** backend (HF), NOT vLLM: on this stack vLLM 0.21 emitted garbage (`!!!!`) and transformers 5.12 broke on `Qwen2VLConfig.max_position_embeddings`; the working path is `mineru-vl-utils` + transformers 4.57 (no vLLM). ~64 s/page on the A4000.
