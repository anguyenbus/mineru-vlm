# hybrid-doc-parser

A production-grade Python library that extracts clean, RAG-ready content from PDFs,
office documents, and images. It routes each document through a selectable parsing
backend — **MinerU** for scientific/academic PDFs and images, **Docling** for office and
HTML formats — applies a cost-aware **two-layer quality gate**, and offers optional
per-element **VLM enrichment**, all behind a single `parse()` / `parse_batch()` API.

> **Core promise:** `parse()` never raises. Every failure — missing file, unsupported
> type, engine crash, enrichment error — is returned as a structured `WarningRecord`, so
> callers always get a valid result.

---

## Contents

- [Features](#features)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Batch processing](#batch-processing)
- [Configuration (`EnrichmentConfig`)](#configuration-enrichmentconfig)
- [Output schema](#output-schema)
- [Quality gate](#quality-gate)
- [VLM enrichment](#vlm-enrichment)
- [Rendering to Markdown](#rendering-to-markdown)
- [Caching](#caching)
- [Environment variables](#environment-variables)
- [GPU / PyTorch notes](#gpu--pytorch-notes)
- [Parse Report — developer debug viewer](#parse-report--developer-debug-viewer)
- [Development](#development)
- [Further documentation](#further-documentation)

---

## Features

- **One API, two backends.** `EnrichmentConfig.parser` selects `"mineru"` (default; PDFs
  and images) or `"docling"` (PDF, DOCX, HTML). Both produce the same `ParserOutput`
  schema, so downstream code is identical regardless of which engine ran.
- **Efficient batch inference (MinerU).** `parse_batch()` groups uncached documents and
  runs them through the model in **shared inference passes** rather than one per file —
  `ceil(N / batch_size)` model runs instead of `N`. See [Batch processing](#batch-processing).
- **Two-layer quality gate.** Flags low-quality / likely-OCR-garbled pages so bad text
  doesn't silently enter your vector store.
- **Optional VLM enrichment.** Generate plain-language descriptions for images, tables,
  and equations via any OpenAI-compatible endpoint or AWS Bedrock. Fully opt-in and
  granular — with enrichment off, the library is a high-quality text extractor.
- **File-based caching.** Re-parsing an unchanged file returns instantly.
- **Canonical Markdown renderer.** `render_markdown()` produces RAG-ready Markdown.
- **Never raises.** All failures surface as `WarningRecord`s in the result.

---

## Installation

Requires **Python 3.12+**. The project uses [`uv`](https://docs.astral.sh/uv/).

```bash
# MinerU backend (default)
uv add "hybrid-doc-parser"          # or: pip install hybrid-doc-parser

# Add the Docling backend (DOCX / HTML support)
uv add "hybrid-doc-parser[docling]"
```

The MinerU pipeline (`mineru[pipeline]`) pulls in PyTorch and downloads model weights on
first use. If you intend to run on a GPU, read [GPU / PyTorch notes](#gpu--pytorch-notes)
first — the PyTorch CUDA build must match your driver.

---

## Quick start

### Parse a single document

```python
from pathlib import Path
from hybrid_doc_parser import parse, EnrichmentConfig

result = parse(Path("report.pdf"))          # MinerU backend by default

print(result.page_count, "pages")
for el in result.elements:
    print(el.type, el.text[:60])

# Failures never raise — check the warnings list instead:
for w in result.warnings:
    print(w.code, w.message)
```

### Render to Markdown

```python
from hybrid_doc_parser import parse, render_markdown

md = render_markdown(parse(Path("report.pdf")))
Path("report.md").write_text(md)
```

### Use the Docling backend (DOCX / HTML)

```python
from hybrid_doc_parser import parse, EnrichmentConfig

result = parse(Path("memo.docx"), EnrichmentConfig(parser="docling"))
```

---

## Batch processing

`parse_batch()` is the efficient path for processing many documents. It is an **async**
function returning one result per input, in input order.

```python
import asyncio
from pathlib import Path
from hybrid_doc_parser import parse_batch, EnrichmentConfig

paths = [Path(p) for p in ("a.pdf", "b.pdf", "c.png")]
results = asyncio.run(parse_batch(paths, EnrichmentConfig()))
assert len(results) == len(paths)            # one result per input, same order
```

**How it works (MinerU backend):**

1. **Classify** every path — missing / unsupported files and cache hits are resolved up
   front and never touch the model.
2. **Batch inference** — the remaining uncached files are split into chunks of
   `MINERU_BATCH_SIZE` (default **8**) and each chunk is run through **one** model pass.
   So 11 uncached files → `ceil(11 / 8)` = **2 model runs** instead of 11. The model stays
   resident across chunks, so heavy setup is paid once.
3. **Merge** results back into the original input order.

**Safety and robustness:**

- **Bounded memory.** Chunking by `MINERU_BATCH_SIZE` keeps each model pass within GPU
  memory limits. Raise it on a larger GPU for fewer passes; lower it under memory pressure.
- **GPU serialization.** A process-wide gate (`MINERU_MAX_INFLIGHT`, default `1`) prevents
  two concurrent `parse_batch()` calls from running the model on the same GPU at once.
- **Isolated fallback.** If a chunk's inference fails, only that chunk falls back to
  per-file parsing — the rest of the batch is unaffected.
- **No silent data loss.** A valid file that produces no output is reported with a
  `mineru_failed` warning, and a single per-call summary line logs
  `requested / cache_hits / parsed / empty_or_failed / fallback`.
- **Non-blocking.** Inference, file hashing, and assembly run off the event loop
  (`asyncio.to_thread`), so `parse_batch()` is safe to call inside an async service.

> The Docling backend uses a per-file path bounded by `max_concurrency` (default 4); the
> chunked batch optimization is MinerU-only.

---

## Configuration (`EnrichmentConfig`)

All fields have safe defaults; `EnrichmentConfig()` gives enrichment-disabled MinerU parsing.

| Field | Type | Default | Purpose |
|---|---|---|---|
| `parser` | `"mineru" \| "docling"` | `"mineru"` | Parsing backend. |
| `enabled` | `bool` | `False` | Master switch for VLM enrichment. |
| `image` / `table` / `equation` | `bool` | `True` | Which element types to enrich (when `enabled`). |
| `context_window` | `int` (0–20) | `3` | Surrounding paragraphs included in the VLM prompt. |
| `max_context_tokens` | `int` (64–4096) | `512` | Approx. token cap for context. |
| `vlm_backend` | `"openai_compatible" \| "bedrock"` | `"openai_compatible"` | VLM provider. |
| `do_ocr` | `bool` | `True` | Docling: enable OCR. |
| `table_mode` | `"fast" \| "accurate"` | `"fast"` | Docling: TableFormer mode. |
| `do_table_structure` | `bool` | `True` | Docling: table structure recognition (expensive). |
| `docling_artifacts_path` | `str \| None` | `None` | Docling: custom model-weights path. |

---

## Output schema

`parse()` / `parse_batch()` return validated Pydantic v2 `ParserOutput` objects.

```text
ParserOutput
├── schema_version: str            # "1.0"
├── file_path: str
├── file_sha256: str               # 64-char hex digest at parse time
├── page_count: int
├── pages: list[PageRecord]
│   ├── page_idx: int
│   ├── quality_decision: "keep" | "promote_to_vlm"
│   ├── element_count: int
│   └── vlm_used: bool
├── elements: list[ElementRecord]
│   ├── element_id: str            # stable UUID v5 (file hash + page + index)
│   ├── type: ElementType          # text/heading/table/image/equation/caption/…
│   ├── text: str
│   ├── description: str           # VLM-generated; "" if not enriched
│   ├── bbox: list[float]          # [x0,y0,x1,y1] PDF points; [] if unavailable
│   ├── page_idx: int
│   ├── is_enriched: bool
│   └── image_bytes: bytes | None  # Docling pictures only
├── warnings: list[WarningRecord]  # empty == clean parse
│   ├── page_idx: int | None
│   ├── code: str
│   └── message: str
└── enrichment_config: EnrichmentConfig
```

### Warning codes

| Code | Meaning |
|---|---|
| `file_not_found` | Input file does not exist. |
| `unsupported_type` | Extension not accepted by the selected parser. |
| `mineru_failed` | MinerU produced no usable output for this file. |
| `mineru_error` | Unhandled error while assembling the MinerU result. |
| `docling_failed` | Docling conversion raised. |
| `docling_error` | Unhandled error while assembling the Docling result. |
| `quality_gate_escalation` | A page was flagged `promote_to_vlm` by the quality gate. |
| `enrichment_error` | A VLM enrichment call failed (parse still succeeded). |
| `enrichment_not_supported` | Enrichment requested with `parser="docling"` (not yet supported). |
| `image_too_large` | A Docling picture exceeded the 10 MB cap; bytes discarded. |

---

## Quality gate

After either backend runs, each page is scored:

- **Layer 1 (PDF only):** ratio of extracted tokens to the embedded text-layer token count
  (via `pypdfium2`). Too low → `promote_to_vlm`.
- **Layer 2 (all inputs):** five text-quality heuristics — garbled-token ratio, mean word
  length, dictionary hit rate, repeated-character runs, ASCII-printable ratio. Any breach →
  `promote_to_vlm`.

The decision is recorded per page (`PageRecord.quality_decision`) and, on escalation, as a
`quality_gate_escalation` warning. The gate is advisory: it flags pages; acting on the flag
is done via VLM enrichment.

---

## VLM enrichment

With `EnrichmentConfig(enabled=True)`, image / table / equation elements are sent to a
vision-language model to produce a plain-language `description` (MinerU backend). The VLM
backend is vendor-neutral:

- **OpenAI-compatible** — set `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `VLM_MODEL_NAME`
  (works with OpenAI, vLLM, Ollama, Together, etc.).
- **AWS Bedrock** — set `AWS_REGION` and `BEDROCK_VLM_MODEL`; uses IAM role credentials.

Missing credentials degrade gracefully — the element is left un-enriched, no exception.

---

## Rendering to Markdown

`render_markdown(parser_output) -> str` produces RAG-ready Markdown: drops page furniture
(headers, footers, page numbers), emits tables as GitHub-flavored Markdown, wraps equations
in `$$…$$`, and includes enriched image/equation descriptions.

---

## Caching

Results are cached on disk, keyed on file **content hash + modification time**. Re-parsing
an unchanged file returns the cached `ParserOutput` instantly without running any engine.
Cache writes are atomic and never raise. Set the location with
`HYBRID_DOC_PARSER_CACHE_DIR` (default `~/.cache/hybrid_doc_parser`).

---

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `MINERU_BACKEND` | `pipeline` | MinerU backend identifier. |
| `MINERU_BATCH_SIZE` | `8` | Documents per `do_parse` inference pass in `parse_batch()`. Finite/bounded; bad or `<1` values fall back to 8. |
| `MINERU_MAX_INFLIGHT` | `1` | Concurrent MinerU inference passes across all callers (raise for multi-GPU). |
| `MINERU_DEVICE_MODE` | (MinerU default) | `cuda` / `cpu` device selection. |
| `CUDA_VISIBLE_DEVICES` | (unset) | Set to `""` to force CPU. |
| `PARSER_RENDER_DPI` | `144` | DPI for page/region rasterization. |
| `PARSER_MAX_RENDER_MP` | `40.0` | Max megapixels per rendered page. |
| `HYBRID_DOC_PARSER_CACHE_DIR` | `~/.cache/hybrid_doc_parser` | Cache directory. |
| `OPENAI_BASE_URL` / `OPENAI_API_KEY` / `VLM_MODEL_NAME` | — | OpenAI-compatible VLM. |
| `AWS_REGION` / `BEDROCK_VLM_MODEL` | `ap-southeast-2` / — | Bedrock VLM. |

See [`.env.example`](.env.example) for a template.

---

## GPU / PyTorch notes

MinerU runs on GPU when available. **The installed PyTorch CUDA build must be ≤ the CUDA
version your NVIDIA driver supports** (`nvidia-smi` shows it). A torch wheel built for a
*newer* CUDA than the driver supports will fail with
`CUDA initialization: the NVIDIA driver on your system is too old`.

To fix, install a matching PyTorch build rather than changing the driver, e.g. for a driver
that supports up to CUDA 12.5:

```bash
uv pip install --index-url https://download.pytorch.org/whl/cu124 \
  "torch==2.6.0" "torchvision==0.21.0"
```

To force CPU (no GPU, or a mismatched driver), set `CUDA_VISIBLE_DEVICES=""` and
`MINERU_DEVICE_MODE=cpu`.

---

## Parse Report — developer debug viewer

A developer-only debugging tool that renders each source page beside the units a backend
extracted and links them with two-way hover/click highlighting. It produces one
self-contained, offline `.html` artifact with three tabs — **MinerU / Docling / pdfplumber**
— for eyeballing *where* each backend placed each extracted element on the page. It is a
diagnostic aid, not part of the library API; the library never imports it.

It lives in the optional `viz` dependency group:

```bash
pip install -e ".[viz]"          # pulls in pypdfium2, pdfplumber, pillow
```

### Usage — the fast loop

The viewer never calls `parse()`. Parse **once**, save the `ParserOutput` JSON, then iterate
on the report as many times as you like (it's a millisecond-scale, offline operation):

```python
# One-time, on the GPU box: produce the JSON the viewer consumes.
from pathlib import Path
from hybrid_doc_parser import parse, EnrichmentConfig

Path("m.json").write_text(parse(Path("source.pdf")).model_dump_json())
Path("d.json").write_text(
    parse(Path("source.pdf"), EnrichmentConfig(parser="docling")).model_dump_json()
)
```

```bash
# Then iterate on the report, fast and offline:
python scripts/parse_report.py source.pdf --mineru m.json --docling d.json -o report.html
```

Both `--mineru` and `--docling` are **optional** — one backend plus the pdfplumber baseline
(always computed from the source PDF) is a valid report. The pdfplumber tab is produced
automatically whenever pdfplumber is installed.

Flags:

| Flag | Default | Purpose |
|---|---|---|
| `--mineru PATH` | — | A saved `ParserOutput` JSON for the MinerU backend (optional). |
| `--docling PATH` | — | A saved `ParserOutput` JSON for the Docling backend (optional). |
| `-o` / `--out PATH` | `./parse_reports/<source>.parse_report.html` | Output HTML file. |
| `--dpi N` | `144` | Page render resolution (matches `PARSER_RENDER_DPI`). |
| `--max-pages N` | `50` | Page cap before truncation; a **visible in-report note** appears when the document exceeds it (base64-embedding every page bloats the HTML). |
| `--pages N-M` | (all) | Page selection — range `3-10`, single page, or comma list; 1-based, inclusive. |
| `--selftest` | — | Emit a synthetic demo report (Pillow only, no PDF/parser deps) and exit; validates the linking UX. |

Reports default to `./parse_reports/`. To validate the tool itself without any real
document:

```bash
python scripts/parse_report.py --selftest
```

### Coordinate calibration (`COORD`) — why the boxes are trustworthy

The transform from a backend's raw bbox to the canonical 0–1 top-left overlay coordinate is
the one assumption that can make the tool *lie silently*. Each backend uses a different
convention, locked in `viz/coords.py`:

| Backend | Origin / unit | Transform to canonical 0–1 top-left |
|---|---|---|
| **MinerU** | top-left, **per-mille** (0–1000) | divide by 1000, **no Y-flip**; page size not needed |
| **Docling** | bottom-left, PDF **points** | divide by page points, **flip Y** (`1 - y`) |
| **pdfplumber** | top-left, PDF **points** | divide by page points |

**pdfplumber is the known-good REFERENCE tab.** Its render + overlay pipeline was confirmed
aligned on a real document, so it is used to calibrate MinerU and Docling: a backend's boxes
are trusted when they land on the same regions as pdfplumber's. Rotation (`/Rotate 90`) and
CropBox ≠ MediaBox are handled by normalizing against the page size the parser actually used
(per-mille is page-relative and rotation-agnostic; points backends use the renderer's
reported post-rotation size). Image inputs treat pixels as the unit — no points math. See
`agent-os/specs/2026-06-09-parse-report-viewer/planning/calibration-notes.md` for the source
proof and golden fixtures.

### What it can and cannot tell you

This viewer **shows placement and cross-backend divergence** — what each backend extracted
and *where* it sits on the page. It does **not** prove completeness: it cannot tell you what
a backend *missed*. Omission / coverage detection (e.g. flagging text-layer tokens no
backend covered) is a deferred non-goal, not implemented here. The developer is the oracle;
the tool's only job is to link extracted units to their source region accurately.

### Security — a report is as sensitive as its source

A generated report **embeds document content** — page images and extracted text — directly
in the `.html`. Treat it as exactly as sensitive as the source document. The default output
locations are `.gitignore`d (`parse_reports/` and `*.parse_report.html`); **never commit a
report of a confidential document.** All document-derived strings are HTML-escaped and
rendered inert (never raw HTML/markdown), and the tool writes one local file — no upload, no
server, no open port, no telemetry.

### Known issues (out of scope here)

Two latent library bugs were *identified* during calibration but are **NOT fixed in this
spec** — documented here so they aren't lost:

- **`models.py` `ElementRecord.bbox` docstring** says coordinates are "PDF points with
  bottom-left origin". This is **wrong for MinerU**, whose bboxes are per-mille (0–1000) with
  a top-left origin. The viewer compensates via its own `COORD` table; the docstring itself
  is unchanged here.
- **`render.py::render_region`** has a related MinerU-crop bug stemming from the same
  coordinate-convention confusion. Likewise documented as KNOWN and left for a future,
  dedicated fix.

---

## Development

```bash
# Run the test suite (mocks the MinerU/VLM boundaries — no GPU or model download needed)
uv run pytest tests/hybrid_doc_parser -q

# With coverage (project gate: >= 80%)
uv run pytest tests/hybrid_doc_parser --cov=src --cov-report=term-missing -q

# Lint & format
uv run ruff check src tests
uv run ruff format src tests
```

Test markers: `live` (needs real VLM credentials), `slow` (runs an engine end-to-end),
`asyncio`. The unit suite mocks the engine boundary, so it runs quickly and offline.

Real end-to-end validation against actual MinerU (GPU or CPU) is available via the helper
scripts under [`scripts/`](scripts/):

```bash
PYTHONPATH=src MINERU_DEVICE_MODE=cuda python scripts/smoke_batch_real.py    # quick 2-file smoke
PYTHONPATH=src MINERU_DEVICE_MODE=cuda python scripts/bench_docbench.py       # 11-doc benchmark
```

---

## Further documentation

- [`docs/parse-pipeline.md`](docs/parse-pipeline.md) — end-to-end pipeline diagram and
  step-by-step reference.
- [`docs/benchmark-results.md`](docs/benchmark-results.md) — accuracy benchmarks (NED / TEDS).
- [`docs/gpu-benchmark-report.md`](docs/gpu-benchmark-report.md) — GPU performance notes.
- [`agent-os/product/`](agent-os/product/) — product mission, roadmap, and tech stack.
