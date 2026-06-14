# 3-Way Benchmark: MinerU vs Docling vs MinerU2.5-Pro

Deterministic, CPU-graded comparison of the three parsing backends using
**doc-bench** (`doc_bench-0.1.0`) on its bundled fixture datasets. Predictions
were generated per backend on an NVIDIA RTX A5000, then graded with NED + TEDS
(no LLM judge, no API keys — same input, same score).

## How it was run

```bash
# 1. Generate predictions per backend (GPU for mineru / mineru25pro; CPU for docling)
python scripts/generate_predictions.py --parser mineru      --predictions-dir predictions_mineru
python scripts/generate_predictions.py --parser docling     --predictions-dir predictions_docling
python scripts/generate_predictions.py --parser mineru25pro --predictions-dir predictions_mineru25pro

# 2. Grade each backend on each dataset
for bk in mineru docling mineru25pro; do
  for ds in ato_bench dp_bench omnidocbench; do
    doc-bench --dataset $ds --predictions predictions_$bk --output-dir results/$bk
  done
done
```

- **Backends:** `mineru` = MinerU 3.x *pipeline* backend; `docling` = Docling;
  `mineru25pro` = `opendatalab/MinerU2.5-Pro-2604-1.2B` VLM via vLLM.
- **Datasets (bundled fixtures):** `ato_bench` (1 doc / 2 pages — the ATO tax
  form `1371-6.1997.pdf`), `dp_bench` (5 PDFs), `omnidocbench` (5 image pages).
- **Metrics:** `ned` = text similarity `1 − normalized_edit_distance`
  (**higher is better**, comparable to the OmniDocBench leaderboard).
  `teds` / `teds_s` = Tree-Edit-Distance Similarity for tables, full / structure-only
  (**higher is better**).

> ⚠️ These are **smoke-scale** samples (1/5/5 docs), not the full public
> benchmarks. They show relative behaviour, not leaderboard-final numbers. For
> full datasets use `doc-bench-download`.

## Results

| Dataset | Metric | MinerU (pipeline) | Docling | MinerU2.5-Pro |
|---|---|---|---|---|
| ato_bench (1) | NED | **0.169** | 0.154 | 0.151 |
| dp_bench (5) | NED | 0.794 | 0.813 | **0.879** |
| omnidocbench (5) | NED | **0.719** | 0.450 | 0.706 |
| omnidocbench (5) | TEDS | **0.281** | 0.050 | 0.080 |
| omnidocbench (5) | TEDS-S | **0.328** | 0.071 | 0.098 |

(TEDS is 0 for ato_bench/dp_bench because their gold has no HTML table tree to
score against — only OmniDocBench carries table-structure gold.)

## Takeaways

- **MinerU2.5-Pro wins clean text extraction** on dp_bench (NED 0.879, best),
  and is essentially tied with the MinerU pipeline on omnidocbench text (0.706
  vs 0.719) — strong for a 1.2B model.
- **MinerU pipeline wins table structure** decisively (TEDS 0.281 / TEDS-S 0.328
  vs ≤0.10 for the others), and edges out on omnidocbench text. Its dedicated
  table model still beats the end-to-end VLMs here.
- **Docling** is competitive on clean digital PDFs (dp_bench NED 0.813) but drops
  sharply on the scanned/image-heavy OmniDocBench pages (NED 0.450) and tables.
- **The ATO form is hard for everyone** (NED ~0.15–0.17): dense government-form
  layout; flat-text NED is unforgiving here.

## Performance (MinerU2.5-Pro via vLLM, RTX A5000)

- **Cold start:** ~62 s one-time per process — vLLM engine init (weight load +
  CUDA-graph capture). Paid once; the engine is a process-wide singleton.
- **Warm inference:** ~4 s/page (8 s for a 2-page doc) once the engine is loaded.
- 11 fixture docs graded after a single warm-up: ~25 s total.

> Note: pipe vLLM's stdout straight to a file, never through `... | grep | tail`
> — a full 64 KB pipe buffer blocks the EngineCore subprocess and deadlocks the
> engine at 0 % GPU (looks like a hang, isn't).

## Case study: `1371-6.1997.pdf` — why the viewer disagrees with ato_bench NED

In the Parse Report viewer, MinerU2.5-Pro is visibly the cleanest parse of this
ATO tax form — yet ato_bench ranks it **last** (NED 0.151 vs MinerU 0.169). The
NED number is misleading here, and the data shows why.

**The metric artifact.** ato_bench's gold for this document is only **814
characters** — a sparse, sampled reference, not the full form. Every backend
extracts the *whole* form (4.6k–5.4k chars). Because
`NED = 1 − editdistance / max(len(gold), len(pred))`, the denominator is the
prediction length, so **the more complete the extraction, the lower the score**,
independent of correctness:

| backend | md chars | headings | jammed words (18+ chars, no space) | NED (raw) | NED (markup stripped) | gold recall |
|---|---|---|---|---|---|---|
| MinerU pipeline | 4609 (shortest) | **0** | 0 | **0.169** | 0.176 | 0.564 |
| Docling | 4699 | 9 | **36** | 0.154 | 0.156 | 0.504 |
| MinerU2.5-Pro | 5416 (longest) | **12** | 0 | 0.151 | 0.154 | 0.528 |

The NED ranking tracks **length almost perfectly** (shortest wins); stripping
markdown markup barely moves it (≈+0.005), and gold-recall is ~0.5 for all three.
So the gap is a length/brevity artifact, not a quality difference.

**Actual text fidelity** on the one sentence that *is* in the gold (the TFN
clause):

- **MinerU2.5-Pro** — `…your TFN helps the Australian Taxation Office (ATO) to
  correctly identify your tax records.` — correct spacing, correct "ATO".
- **MinerU pipeline** — same sentence, but elsewhere emits "ATÓ" (an OCR
  diacritic slip) and detects **0 headings** (every title flattened to a plain
  paragraph → document structure lost).
- **Docling** — `Itisnotanoffencenot toquoteyourTFN.However,yourTFNhelps…` —
  **word boundaries destroyed** (36 jammed runs across the doc), which would
  wreck downstream search / RAG.

**Verdict for this document:**

1. **MinerU2.5-Pro — best.** Cleanest text, correct OCR, full semantic structure
   (12 headings, captions, page numbers, 33 form-field regions). Its only
   "penalty" is being the most complete — which the truncated gold punishes.
2. **MinerU pipeline — middle.** Clean spacing, but no heading detection and a
   minor OCR glitch.
3. **Docling — worst.** Severe word-jamming on this form.

**Lesson:** ato_bench NED is the wrong lens for a document whose gold is an
814-char stub — it rewards extracting *less*. The viewer's structural overlay
(and the fidelity check above) is the correct quality signal here, and it
confirms MinerU2.5-Pro is ahead. The benchmark's trustworthy text signals are
**dp_bench** (MinerU2.5-Pro best, 0.879) and **OmniDocBench** (fuller gold),
where MinerU2.5-Pro is top or tied.
