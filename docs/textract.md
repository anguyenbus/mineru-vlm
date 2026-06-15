# Document Parsing Options Analysis

**Workload:** ~21,000 images/month, OCR with **text + tables + forms** extracted to a usable (Markdown-style) structured output.
**Options evaluated:** AWS Textract · Amazon Bedrock (Claude Sonnet 4.6) · MinerU VLM (self-hosted GPU) · Docling (self-hosted)
**Prepared:** June 2026 · Pricing is US-region list, **verified directly against the official AWS pricing pages on 15 June 2026** (see Sources & Verification). AWS ap-southeast-2 (Sydney) may run marginally higher — confirm in the AWS pricing calculator before sign-off.

---

## TL;DR / Recommendation

For a workload that needs **text + tables + forms parsed into structured Markdown**, the realistic shortlist is **Bedrock Sonnet** (lowest-ops managed option) and **MinerU VLM** (highest accuracy, lowest marginal cost if you already run GPUs).

- **Textract is the weakest fit** for this specific requirement. To extract text + tables + forms it lands on the most expensive per-page tier (~$0.065/page ≈ **$16.4k/year**), and it still does **not** emit Markdown or LaTeX — you would build a JSON→Markdown layer yourself and accept weaker table/reading-order fidelity.
- **Bedrock Sonnet** delivers full Markdown parsing at **~$0.018/page (~$4.5k/year)**, halving to **~$0.009/page (~$2.3k/year)** with Batch inference. Zero GPU ops. Strong all-rounder.
- **MinerU VLM** has the best parsing accuracy on public benchmarks. On a 10h×5d GPU schedule its raw compute is only **~$96/mo (spot) or ~$218/mo (on-demand)** — actually the cheapest option at 21k pages — but it carries real, unpriced engineering/ops overhead, which is the true deciding factor.
- **Docling** is the cheapest and easiest to run locally, but the lowest accuracy on complex tables and effectively no formula support — suitable only if documents are simple.

**Suggested path:** if staying managed/low-ops → Bedrock Sonnet (test Batch mode). If output quality is paramount and a GPU/MLOps capability exists → MinerU. Run a side-by-side accuracy bake-off on a sample of *your* documents before committing.

---

## Assumptions

- Volume: 21,000 pages/month (1 PNG = 1 page). Well below AWS volume-discount thresholds (first 1M pages), so full first-tier rates apply.
- Required output: text **and** tables **and** forms/key-values, assembled into structured Markdown for a downstream pipeline (RAG/LLM ingestion).
- Token estimate for the LLM option: ~2,000–2,900 input tokens/page and ~750–800 output tokens/page of Markdown. **AWS's own Bedrock documentation estimates ~2,900 input + ~750 output tokens per page** for parsing pages where ~30% contain tables and ~30% contain figures — this independently validates the ~$0.018–0.020/page figure. Dense, table-heavy pages push output (and therefore cost) higher.

---

## Cost comparison (21,000 pages/month)

| Option | Effective $/page | Monthly | Annual | Notes |
|---|---|---|---|---|
| **Textract** — AnalyzeDocument (Forms + Tables, OCR included) | $0.065 | ~$1,365 | ~$16,380 | Each feature billed separately; Forms $50/1k + Tables $15/1k. |
| Textract — Tables only | $0.015 | ~$315 | ~$3,780 | If forms turn out not to be needed. |
| Textract — raw text only | $0.0015 | ~$32 | ~$378 | OCR only; not your use case. |
| **Bedrock Sonnet 4.6** — on-demand | ~$0.018 | ~$378 | ~$4,536 | $3/$15 per M tokens. Range ~$0.012–0.028 by page density. |
| **Bedrock Sonnet 4.6** — Batch (–50%) | ~$0.009 | ~$189 | ~$2,268 | 24-hour SLA; ideal for non-real-time batch parsing. |
| **Bedrock Data Automation** — Standard Output | $0.010 | ~$210 | ~$2,520 | Managed IDP; flat per-page; verified on AWS pricing page. |
| **Bedrock Data Automation** — Custom Output (≤30 fields) | $0.040 | ~$840 | ~$10,080 | Structured field extraction via Blueprints; +$0.0005/field above 30. |
| **MinerU VLM** — self-hosted GPU (10h×5d) | compute only (~6–19 hrs needed) | ~$96–218 + ops | ~$1,149–2,616 + ops | g5.xlarge run 10h/day × 5d/wk ≈ 217 hrs/mo: $96 spot / $218 on-demand. Cost is instance posture + ops, not inference. |
| **Docling** — self-hosted (CPU/GPU, 10h×5d) | compute only (light) | ~$96–218 + ops | ~$1,149–2,616 + ops | Same schedule/instance; runs even on CPU; cheapest compute, but lowest accuracy. |

**Self-hosting cost note (verified):** MinerU2.5 parses ~2.12 pages/sec on an A100 (figure from the MinerU2.5 paper; an A10G will be slower). Run on the assumed schedule of **10 hrs/day × 5 days/week ≈ 217 instance-hours/month**, an A10G-class `g5.xlarge` costs **$1.006/hr → $217.97/mo on-demand**, or **$0.4419/hr spot → $95.75/mo**. That schedule has ample headroom: 21,000 pages needs only ~6–19 hours of actual compute (at 0.3–1.0 pages/sec), so a single instance on this schedule could handle **~390,000 pages/month** before you'd need more hours or a second box. At 21k pages the raw compute is therefore cheap — **$96–218/mo** — but **engineering/ops time remains the real, unpriced cost**, and spot adds interruption risk (manageable for batch with checkpointing).

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

> Note on "staying in AWS": if the underlying driver is AWS-native deployment, Bedrock Sonnet is the natural managed parser to weigh against MinerU. **Amazon Bedrock Data Automation (BDA)** is also a strong managed alternative — its Standard Output is a verified **$0.010/page** (~$210/month here), cheaper than both Textract and on-demand Sonnet, and it's purpose-built for IDP/RAG parsing. It's worth a direct trial alongside Sonnet.

---

## Cost-optimal option by monthly volume

The economics split into two cost shapes: **per-page managed APIs** are purely variable (zero fixed cost, cost scales linearly), while **self-hosted GPU** is a near-fixed monthly cost. On the assumed schedule (10h/day × 5d/wk ≈ 217 hrs/mo), one g5.xlarge costs a flat **~$96/mo (spot) or ~$218/mo (on-demand)** and can serve up to ~390,000 pages/month before needing more hours. Variable options win at low volume; the fixed-cost self-host wins once volume amortizes the instance.

**Break-even volumes (raw cost, verified math; self-host on the 10h×5d schedule):**
- Self-host (spot, ~$96/mo) beats the cheapest managed option (Sonnet Batch, $0.009/pg) at **~10,600 pages/month**.
- Self-host (on-demand, ~$218/mo) beats Sonnet Batch at **~24,200 pages/month**.
- Self-host (spot) beats Textract Forms+Tables ($0.065/pg) at just **~1,500 pages/month** — because Textract is the most expensive per-page option.
- **Your 21,000-page workload sits right around these crossovers:** self-host spot (~$96) is already cheaper than every managed option, and self-host on-demand (~$218) is roughly level with Sonnet Batch ($189) and BDA ($210).

**Recommended option by volume** (assumes the text+tables+forms→Markdown workload; "ops" = engineering/maintenance, unpriced but real):

| Monthly volume | Cheapest on raw $/page | Practical recommendation | Why |
|---|---|---|---|
| **< 10,000** | Sonnet Batch / BDA | **Managed** — Sonnet Batch or BDA Standard | Below self-host break-even; variable cost trivial; ops never justified |
| **10,000 – 25,000** *(your 21k sits here)* | Self-host spot (~$96) edges ahead | **Managed if low-ops; self-host spot if you already run GPUs** | Raw saving vs Sonnet Batch is small (~$90/mo) and easily offset by ops + spot-interruption handling |
| **25,000 – 80,000** | Self-host (spot, then on-demand past ~24k) | **Self-host GPU, or hybrid** | Self-host now clearly cheaper; ops increasingly justified by the saving |
| **80,000 – 390,000** | Self-host (scheduled instance) | **Self-host GPU** (MinerU/Docling) | Decisive cost win; one scheduled instance still has capacity |
| **> 390,000** | Self-host (extend hours / add instances) | **Self-host GPU + hybrid burst** | Exceeds single-instance scheduled capacity; extend toward 24/7 or add boxes; route spikes to a managed API |

**Where each named option is actually optimal:**
- **Textract** — essentially *never* the cheapest for this text+tables+forms workload (most expensive per-page at $0.065). Justified only when its specific forms/key-value extraction quality is a hard requirement, and then only at low volume.
- **Bedrock Sonnet Batch / BDA Standard** — the low-to-mid volume sweet spot (roughly < 35k/mo), and the right choice for your 21k workload.
- **Self-host GPU (MinerU / Docling)** — on the 10h×5d schedule, becomes the raw-cost winner above ~10,600 pages/mo (spot) or ~24,200/mo (on-demand). Practically, the high-volume choice (> ~25k/mo) where the saving clearly outweighs GPU ops.
- **Hybrid (self-host baseline + API burst)** — optimal when you have a *high, steady baseline* plus *variable spikes*: size the GPU to the baseline (cheap fixed cost, near-zero marginal) and overflow peaks to Sonnet/BDA. Typically meaningful above ~50k/mo with bursty load, or near single-instance capacity (~390k/mo).

**Monthly cost at sample volumes** (managed = no ops; self-host = scheduled instance 10h×5d, ops excluded):

| Pages/mo | Sonnet Batch | BDA Std | Sonnet on-demand | Textract F+T | Self-host spot (10h×5d) | Self-host on-demand (10h×5d) |
|---|---|---|---|---|---|---|
| 5,000 | **$45** | $50 | $90 | $325 | $96 | $218 |
| **21,000** | $189 | $210 | $378 | $1,365 | **$96** | $218 |
| 35,000 | $315 | $350 | $630 | $2,275 | **$96** | $218 |
| 50,000 | $450 | $500 | $900 | $3,250 | **$96** | $218 |
| 100,000 | $900 | $1,000 | $1,800 | $6,500 | **$96** | $218 |
| 250,000 | $2,250 | $2,500 | $4,500 | $16,250 | **$96** | $218 |
| 500,000 † | $4,500 | $5,000 | $9,000 | $32,500 | **~$192–323 †** | ~$436–734 † |

*Bold = cheapest in that row. Self-host figures use the 10h×5d schedule (~217 hrs/mo) and exclude engineering/ops, which pushes the practical break-even above the raw ~10.6k/mo crossover. † 500k exceeds a single scheduled instance's capacity (~390k/mo at ~0.5 pg/s) — you'd extend toward 24/7 (spot ~$323 / on-demand ~$734) or add a second instance.*

---

## Cost Verification (worked calculations)

Every figure below = **verified per-unit rate × 21,000 pages/month**. Rates are sourced in the next section; arithmetic was recomputed and checked.

**Textract** (rate × 21,000):
- OCR only: $0.0015 × 21,000 = **$31.50/mo** → $378/yr
- Tables only: $0.015 × 21,000 = **$315.00/mo** → $3,780/yr
- Forms only: $0.05 × 21,000 = **$1,050.00/mo** → $12,600/yr
- **Forms + Tables: ($0.015 + $0.05) × 21,000 = $0.065 × 21,000 = $1,365.00/mo → $16,380/yr**
- Analyze Expense: $0.01 × 21,000 = **$210.00/mo** → $2,520/yr

**Bedrock Sonnet 4.6** (rates: $3.00/M input, $15.00/M output):
- Per-page @ user assumption (2,000 in + 800 out) = (2,000 × $3 + 800 × $15) / 1,000,000 = $0.006 + $0.012 = **$0.018/page**
- Per-page @ AWS doc estimate (2,900 in + 750 out) = (2,900 × $3 + 750 × $15) / 1,000,000 = $0.0087 + $0.01125 = **$0.01995/page**
- On-demand monthly: $0.018 × 21,000 = **$378.00/mo** → $4,536/yr  *(at AWS estimate: $0.01995 × 21,000 = $418.95/mo → $5,027/yr)*
- Batch (–50%): $0.009 × 21,000 = **$189.00/mo** → $2,268/yr  *(at AWS estimate: $0.009975 × 21,000 = $209.47/mo → $2,514/yr)*

**Bedrock Data Automation:**
- Standard Output: $0.010 × 21,000 = **$210.00/mo** → $2,520/yr
- Custom Output (≤30 fields): $0.040 × 21,000 = **$840.00/mo** → $10,080/yr

**Self-hosted GPU (g5.xlarge, A10G, us-east-1) — schedule: 10h/day × 5d/wk ≈ 216.7 hrs/mo:**
- Compute needed for 21k pages: 21,000 ÷ ~0.5 pages/sec ÷ 3,600 ≈ **11.7 hours/month** (≈5.8h at 1 pg/s, ≈19.4h at 0.3 pg/s) — comfortably within the 216.7 scheduled hours.
- On-demand: $1.006/hr × 216.7 hrs = **$217.97/mo** → $2,615.60/yr
- Spot: $0.4419/hr × 216.7 hrs = **$95.75/mo** → $1,148.94/yr
- Single-instance capacity on this schedule ≈ 216.7 × 3,600 × 0.5 ≈ **390,000 pages/mo**.
- *(Plus unpriced engineering/ops. The 2.12 pg/s headline is A100-measured; A10G is slower, hence the 0.3–1.0 pg/s working range above.)*

**Cost ranking at this volume (cheapest → most expensive):** Self-host spot 10h×5d ($96) < Bedrock Sonnet Batch ($189) < BDA Standard ($210) ≈ Self-host on-demand 10h×5d ($218) < Textract Tables-only ($315) < Bedrock Sonnet on-demand ($378–419) < BDA Custom ($840) < **Textract Forms+Tables ($1,365)**. Self-host figures exclude engineering/ops, which is the deciding factor at this volume given the small raw gap.

---

## Sources & Verification

Verification key: **✅ Verified (primary)** = fetched directly from the official source on 15 Jun 2026 · **◑ Primary literature** = figure from the model's own paper or official model card · **○ Third-party** = single-vendor benchmark, not independently verified.

### Pricing (all verified directly against official AWS pages, 15 Jun 2026)

| Figure | Value | Source | Status |
|---|---|---|---|
| Textract DetectDocumentText (OCR) | $0.0015/page (first 1M) | AWS Textract pricing page, "Pricing example 1" | ✅ |
| Textract Tables | $0.015/page | AWS Textract pricing page, "Pricing example 3" & "16" | ✅ |
| Textract Forms | $0.05/page | AWS Textract pricing page, "Pricing example 3" | ✅ |
| Textract Forms + Tables | $0.065/page | AWS Textract pricing page, "Pricing example 3" ($0.015 + $0.05) and "18" | ✅ |
| Textract Layout | Free when used with Tables | AWS Textract pricing page, "Pricing example 16" | ✅ |
| Textract Analyze Expense | $0.01/page (first 1M) | AWS Textract pricing page, "Pricing example 10" | ✅ |
| Volume discount threshold | After first 1M pages/month | AWS Textract pricing page, "Pricing example 2/4" | ✅ |
| Bedrock Claude Sonnet 4.6 | $3.00 / $15.00 per 1M tokens (in/out) | AWS Bedrock pricing page (Anthropic); cross-checked vs. multiple 2026 trackers | ✅ |
| Bedrock Batch discount | 50% off on-demand | AWS Bedrock pricing page (Batch note) | ✅ |
| Bedrock per-page token estimate | ~2,900 in / ~750 out per page | AWS Bedrock pricing page, "Data Automation – Pricing Example 3" | ✅ |
| Bedrock Data Automation Standard Output | $0.010/page | AWS Bedrock pricing page, "Data Automation – Pricing Example 3" | ✅ |
| Bedrock Data Automation Custom Output | $0.040/page (≤30 fields) | AWS Bedrock pricing page, "Data Automation – Pricing Example 1" | ✅ |
| EC2 g5.xlarge (A10G GPU) | $1.006/hr on-demand; $0.4419/hr spot → at 10h×5d (~217h): $218/mo OD, $96/mo spot | AWS EC2 on-demand pricing; cross-checked Vantage/DoiT (us-east-1) | ✅ |

**Primary source URLs:**
- AWS Textract pricing: https://aws.amazon.com/textract/pricing/ (page last updated 2026-05-13)
- AWS Bedrock pricing: https://aws.amazon.com/bedrock/pricing/ (page last updated 2026-06-02)
- AWS EC2 on-demand pricing: https://aws.amazon.com/ec2/pricing/on-demand/ · g5.xlarge spec/spot: https://instances.vantage.sh/aws/ec2/g5.xlarge

### Benchmark / accuracy figures

| Figure | Value | Source | Status |
|---|---|---|---|
| MinerU2.5 OmniDocBench overall | 90.67 | MinerU2.5 paper, arXiv:2509.22186, §5.1.2 | ◑ |
| MinerU2.5 text edit distance | 0.047 | MinerU2.5 paper, arXiv:2509.22186 | ◑ |
| MinerU2.5 Table TEDS / TEDS-S | 88.22 / 92.38 | MinerU2.5 paper, arXiv:2509.22186 | ◑ |
| MinerU2.5 Formula CDM | 88.46 | MinerU2.5 paper, arXiv:2509.22186 | ◑ |
| MinerU2.5 reading-order edit | 0.044 | MinerU2.5 paper, arXiv:2509.22186 | ◑ |
| MinerU2.5 throughput | 2.12 pages/sec (A100 80G) | MinerU2.5 paper, arXiv:2509.22186 §5 | ◑ |
| MinerU2.5-Pro OmniDocBench v1.6 overall | 95.69 | MinerU2.5-Pro model card (HuggingFace) & arXiv:2604.04771 | ◑ |
| Docling OmniDocBench (EN) overall edit | ~0.589 | dots.ocr paper, arXiv:2512.02498, Table 1 | ◑ / ○ |
| Docling Table TEDS (EN) | ~61.3 | dots.ocr paper, arXiv:2512.02498, Table 1 | ◑ / ○ |
| Docling formula edit | ~0.999 (effectively no support) | dots.ocr paper, arXiv:2512.02498, Table 1 | ◑ / ○ |
| Textract Table TEDS / Form F1 | ~80.75 / ~88.4 | Tensorlake benchmark (OmniDocBench v1.5 table subset, 512 imgs) | ○ |

**Primary source URLs:**
- MinerU2.5: https://arxiv.org/abs/2509.22186 · model card: https://huggingface.co/opendatalab/MinerU2.5-Pro-2604-1.2B
- OmniDocBench: https://github.com/opendatalab/OmniDocBench (CVPR 2025)
- dots.ocr (contains Docling comparison row): https://arxiv.org/abs/2512.02498
- Tensorlake Textract/Azure table benchmark: https://medium.com/tensorlake-ai/benchmarking-the-most-reliable-document-parsing-api-b8065686daff

### Verification notes & caveats

1. **Pricing is the most defensible part of this analysis** — every per-page rate above was read directly from AWS's own pricing pages, not third-party summaries.
2. **Textract is not on the official OmniDocBench leaderboard** because it does not emit Markdown (the benchmark's end-to-end pipeline expects Markdown). The only Textract OmniDocBench-style figure available is a third-party (Tensorlake) run on the table subset — treat as indicative, not authoritative. Run your own test for a defensible number.
3. **Docling figures** come from the comparison table in the dots.ocr paper; Docling has since iterated, so re-test the current version before relying on these.
4. **Bedrock Sonnet has no official OmniDocBench score** (it's a general VLM, not a registered parser). Comparable frontier VLMs (e.g. Gemini 2.5 Pro) land ~88 overall on v1.5; treat Sonnet's parsing quality as "strong general-VLM" pending your own test.
5. **Self-hosting cost ranges are estimates**, not vendor quotes — actual cost depends on instance type, utilization, spot vs. on-demand, and engineering overhead, which dominates at this volume.
6. **All figures are point-in-time (15 Jun 2026).** AWS pricing and benchmark leaderboards change frequently; re-verify before final sign-off.