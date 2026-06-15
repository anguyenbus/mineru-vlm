# Document Parsing Options Analysis

**Workload:** ~21,000 images/month, OCR with **text + tables + forms** extracted to a usable (Markdown-style) structured output.
**Options evaluated:** AWS Textract · Amazon Bedrock (Claude Sonnet 4.6) · MinerU VLM (self-hosted GPU) · Docling (self-hosted)
**Prepared:** June 2026 · Pricing is US-region list; AWS ap-southeast-2 (Sydney) may run marginally higher — confirm in the AWS pricing calculator before sign-off.

---

## TL;DR / Recommendation

For a workload that needs **text + tables + forms parsed into structured Markdown**, the realistic shortlist is **Bedrock Sonnet** (lowest-ops managed option) and **MinerU VLM** (highest accuracy, lowest marginal cost if you already run GPUs).

- **Textract is the weakest fit** for this specific requirement. To extract text + tables + forms it lands on the most expensive per-page tier (~$0.065/page ≈ **$16.4k/year**), and it still does **not** emit Markdown or LaTeX — you would build a JSON→Markdown layer yourself and accept weaker table/reading-order fidelity.
- **Bedrock Sonnet** delivers full Markdown parsing at **~$0.018/page (~$4.5k/year)**, halving to **~$0.009/page (~$2.3k/year)** with Batch inference. Zero GPU ops. Strong all-rounder.
- **MinerU VLM** has the best parsing accuracy on public benchmarks and the lowest *compute* cost at this volume (the entire month is ~3 GPU-hours), but carries real engineering/ops overhead.
- **Docling** is the cheapest and easiest to run locally, but the lowest accuracy on complex tables and effectively no formula support — suitable only if documents are simple.

**Suggested path:** if staying managed/low-ops → Bedrock Sonnet (test Batch mode). If output quality is paramount and a GPU/MLOps capability exists → MinerU. Run a side-by-side accuracy bake-off on a sample of *your* documents before committing.

---

## Assumptions

- Volume: 21,000 pages/month (1 PNG = 1 page). Well below AWS volume-discount thresholds (first 1M pages), so full first-tier rates apply.
- Required output: text **and** tables **and** forms/key-values, assembled into structured Markdown for a downstream pipeline (RAG/LLM ingestion).
- Token estimate for the LLM option: ~2,000 input tokens/page (page image ≈1,800 + prompt ≈200) and ~800 output tokens/page of Markdown. Dense, table-heavy pages push output higher.

---

## Cost comparison (21,000 pages/month)

| Option | Effective $/page | Monthly | Annual | Notes |
|---|---|---|---|---|
| **Textract** — AnalyzeDocument (Forms + Tables, OCR included) | $0.065 | ~$1,365 | ~$16,380 | Each feature billed separately; Forms $50/1k + Tables $15/1k. |
| Textract — Tables only | $0.015 | ~$315 | ~$3,780 | If forms turn out not to be needed. |
| Textract — raw text only | $0.0015 | ~$32 | ~$378 | OCR only; not your use case. |
| **Bedrock Sonnet 4.6** — on-demand | ~$0.018 | ~$378 | ~$4,536 | $3/$15 per M tokens. Range ~$0.012–0.028 by page density. |
| **Bedrock Sonnet 4.6** — Batch (–50%) | ~$0.009 | ~$189 | ~$2,268 | 24-hour SLA; ideal for non-real-time batch parsing. |
| **MinerU VLM** — self-hosted GPU | compute only (~3 GPU-hrs/mo) | ~$50–730 + ops | varies + ops | Compute is trivial at this volume; cost is the GPU instance posture + engineering/ops. |
| **Docling** — self-hosted (CPU/GPU) | compute only (light) | ~$30–300 + ops | varies + ops | Runs on CPU; cheapest compute, but lowest accuracy. |

**Self-hosting cost note:** MinerU2.5 parses ~2.12 pages/sec on an A100, so 21,000 pages ≈ 2.7 hours of compute/month. If you keep a warm endpoint (e.g. an A10G-class `g5.xlarge` ~$1/hr on-demand, ~$0.30/hr spot) the dominant cost is idle instance time and operational effort, not the inference itself. Below ~hundreds of thousands of pages/month, the open-source cost advantage is mostly about avoiding per-page API fees at scale — at 21k/month the managed options are already cheap, so the self-hosting case rests on accuracy and data-residency, not raw cost.

---

## Capability & accuracy comparison

OmniDocBench is the standard public benchmark for full-document parsing (text edit distance ↓, Table TEDS ↑, Formula CDM ↑, reading order ↓).

| Option | Markdown output | Tables | Forms/KV | Formulas (LaTeX) | OmniDocBench standing |
|---|---|---|---|---|---|
| **MinerU VLM** | Native | Strong (TEDS 88.2; Pro higher) | Via layout | Strong (CDM ~88; Pro ~97) | Overall **90.67** (2.5) / **95.69** (2.5-Pro, v1.6) — SOTA |
| **Bedrock Sonnet** | Native (prompt-driven) | Strong | Strong (prompt-driven) | Good | Not on leaderboard; frontier VLMs land ~86–88 overall; rated among top table performers in independent tests |
| **Textract** | No (JSON only) | Moderate (table TEDS ~80.75, 3rd-party) | Strong (Form F1 ~88.4) | None | Not on leaderboard — doesn't emit Markdown; only table subset measurable |
| **Docling** | Native | Weak (table TEDS ~61) | Basic | Effectively none (~fails) | Overall edit ~0.589 EN / 0.909 ZH — lowest of the four |

*Key reference points:* MinerU2.5 reports text edit distance 0.047, Table TEDS 88.22, Formula CDM 88.46, reading-order edit 0.044. A third-party run put Textract at ~80.75 table TEDS / 88.4 form F1. Docling, as a lightweight pipeline tool, scores well below VLM-based parsers on complex tables and does not handle formulas.

---

## Per-option summary

### AWS Textract
**Pros:** Fully managed, no GPU/ops, mature and reliable, excellent forms/key-value extraction, strong SLA, native AWS integration (S3, Lambda, IAM), good latency for real-time.
**Cons:** No Markdown or LaTeX output (you build the conversion layer); weaker reading order on complex layouts; most expensive tier (~$0.065/page) once you need forms+tables; not designed for full-document parsing.
**Best when:** the real need is structured field extraction from business forms/invoices, not Markdown for an LLM pipeline.

### Amazon Bedrock — Claude Sonnet 4.6
**Pros:** Native Markdown parsing of text/tables/forms via prompting; no GPU ops; fully managed inside the AWS/VPC boundary; flexible (handles odd layouts, mixed content, instructions); ~$0.018/page on-demand and ~$0.009/page on Batch; strong table fidelity in independent tests.
**Cons:** Cost is output-token sensitive — dense pages cost more; potential for occasional hallucination/format drift (needs validation/guardrails); per-page cost higher than self-hosted compute at very high volume; no formal OmniDocBench number.
**Best when:** you want managed, AWS-native, flexible Markdown parsing without standing up a GPU stack — likely the best balance for this workload.

### MinerU VLM (self-hosted GPU)
**Pros:** Best public parsing accuracy (SOTA on OmniDocBench); native Markdown + LaTeX; lowest *compute* cost at this volume; full data residency / offline capability; no per-page vendor fees.
**Cons:** Requires GPU provisioning and MLOps (deployment, scaling, monitoring, updates); engineering/ops is the true cost; you own reliability and uptime.
**Best when:** output quality is paramount, you have or want GPU/MLOps capability, or data must stay on your own infrastructure.

### Docling (self-hosted)
**Pros:** Easiest/cheapest to run (CPU-capable); fast; simple Python integration; native Markdown/JSON/HTML; good for clean, simple documents.
**Cons:** Lowest accuracy on complex/nested tables; effectively no formula support; struggles on non-English and messy layouts.
**Best when:** documents are simple and structured, budget/ops must be minimal, and table/formula fidelity is not critical.

---

## Decision guidance

1. **If the documents are forms-heavy and you mainly need fields, not Markdown** → Textract is defensible despite the price; it's purpose-built for that.
2. **If you need flexible Markdown parsing with minimal ops** → Bedrock Sonnet (start with Batch mode to halve cost).
3. **If you need maximum fidelity and can run GPUs** → MinerU VLM.
4. **If documents are simple and cost/ops must be near-zero** → Docling.

**Before committing:** run all candidates on a representative sample (50–100 of your actual pages spanning your hardest table/form layouts) and score table TEDS + field accuracy + reading order on *your* data. Public benchmarks set expectations; your documents decide the winner.

> Note on "staying in AWS": if the underlying driver is AWS-native deployment, Bedrock Sonnet is the natural managed parser to weigh against MinerU. Amazon Bedrock Data Automation (flat per-document IDP pricing) is also worth a separate look as a managed alternative if you'd prefer a turnkey IDP service over prompt-driven parsing.