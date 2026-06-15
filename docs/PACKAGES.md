# Project Packages — Hybrid Doc Parser

This document lists the most important packages used by **hybrid-doc-parser**
(MinerU 3.x + per-element VLM enrichment hybrid document parser).

- **Python requirement:** `>=3.12`
- **Build backend:** `hatchling`
- **Lockfile:** `uv.lock` (versions below are the resolved/installed versions)

---

## 1. Core runtime dependencies

These are required to run the document-parsing pipeline (declared in
`pyproject.toml` → `[project].dependencies`).

| Package | Version | Role |
|---|---|---|
| `mineru[pipeline]` | 3.2.3 | Primary document-layout/OCR parsing engine. The backbone of the pipeline. |
| `openai` | 2.41.0 | Client for the per-element VLM enrichment (vision-language model calls). |
| `boto3` | 1.43.24 | AWS SDK — used for S3 / Bedrock-style access in the VLM client. |
| `pypdfium2` | 4.30.0 | PDF rendering to images (page rasterization for VLM input). |
| `pypdf` | 6.13.0 | PDF reading/metadata/page handling. |
| `Pillow` | 12.2.0 | Image manipulation (crop/encode element images for the VLM). |
| `pydantic` | 2.13.4 | Typed data models / validation for the parser's structured output. |
| `beartype` | 0.22.9 | Runtime type checking of function signatures. |
| `icontract` | 2.7.3 | Design-by-contract (pre/post-condition) checks. |
| `loguru` | 0.7.3 | Structured logging across the pipeline. |
| `doc-bench` | local wheel | Benchmark/evaluation harness (vendored `doc_bench-0.1.0-py3-none-any.whl`). |

## 2. Optional dependencies

### `docling` extra (alternative parsing backend)

| Package | Version | Role |
|---|---|---|
| `docling` | 2.99.0 | Alternative document parser backend (table/structure extraction). |

### `dev` extra (development & testing)

| Package | Version | Role |
|---|---|---|
| `pytest` | 9.0.3 | Test runner. |
| `pytest-cov` | — | Coverage reporting. |
| `pytest-asyncio` | — | Async test support. |
| `ruff` | 0.15.16 | Linter / import sorter. |
| `black` | 26.5.1 | Code formatter. |
| `pre-commit` | — | Git hook manager. |
| `fpdf2` | 2.8.7 | Generates synthetic PDFs for test fixtures. |
| `python-docx` | 1.2.0 | Generates/reads `.docx` test fixtures. |

---

## 3. Document-parsing package table

Packages directly involved in the **document parsing task** (core runtime +
docling backend). The "Approval Status" column is intentionally left blank.

| Software Type | Name | Python Version | Security Vulnerability | Version | Component | Approval Status | Functions |
|---|---|---|---|---|---|---|---|
| Open-source Python library | mineru[pipeline] | >=3.12 | | 3.2.3 | `parser.py` | | Core document layout analysis & OCR; produces structured elements from PDFs |
| Open-source Python library | openai | >=3.12 | | 2.41.0 | `vlm_client.py` | | VLM/LLM API client for per-element enrichment (captions, table/formula reading) |
| Open-source Python library | boto3 | >=3.12 | | 1.43.24 | `vlm_client.py` | | AWS SDK; S3 object access and Bedrock-style model backends |
| Open-source Python library | pypdfium2 | >=3.12 | | 4.30.0 | `render.py`, `parser.py` | | Rasterizes PDF pages to images for VLM input |
| Open-source Python library | pypdf | >=3.12 | | 6.13.0 | `render.py` | | PDF reading, page counting, metadata extraction |
| Open-source Python library | Pillow (PIL) | >=3.12 | | 12.2.0 | `render.py`, `models.py` | | Crops/encodes element images; image format conversion |
| Open-source Python library | pydantic | >=3.12 | | 2.13.4 | `models.py` | | Typed data models & validation for parsed document structure |
| Open-source Python library | beartype | >=3.12 | | 0.22.9 | project-wide | | Runtime type-signature enforcement |
| Open-source Python library | icontract | >=3.12 | | 2.7.3 | project-wide | | Design-by-contract pre/post-condition checks |
| Open-source Python library | loguru | >=3.12 | | 0.7.3 | project-wide | | Structured application logging |
| Open-source Python library | docling | >=3.12 | | 2.99.0 | `parser.py` (docling backend) | | Alternative document parser; table/structure extraction |
| Local wheel | doc-bench | >=3.12 | | 0.1.0 | `scripts/bench_docbench.py` | | Benchmarking/evaluation harness for parser output |

> **Python Version** reflects the project-wide requirement (`requires-python =
> ">=3.12"`); individual packages may support a wider range.
>
> **Security Vulnerability** is left blank — run an audit (e.g. `uv pip audit`
> or `pip-audit`) to populate it against the locked versions.
