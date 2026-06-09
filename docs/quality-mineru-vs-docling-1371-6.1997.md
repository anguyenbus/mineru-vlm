# Extraction Quality: MinerU vs Docling — `1371-6.1997.pdf`

A side-by-side quality read of the two parsing backends, generated from the
Parse Report viewer output (`make report 1371-6.1997.pdf`) and the saved
`ParserOutput` JSON for each backend.

> **Scope caveat — read first.** This is a **single-document, qualitative**
> comparison on one *scanned* form. It shows what each backend extracted and how
> well, not aggregate accuracy. Conclusions here do **not** generalize to
> born-digital PDFs, DOCX, or HTML — where Docling has a real structure tree to
> work from and tends to do well. The eval suite (`eval_config.yaml`), not a
> single eyeball, is what proves aggregate quality.

## Walkthrough video

A screen recording walking through the Parse Report viewer and the MinerU vs
Docling comparison for this document:

[![Parse Report walkthrough — MinerU vs Docling](https://img.youtube.com/vi/Bm3Qc_oQ6oI/hqdefault.jpg)](https://youtu.be/Bm3Qc_oQ6oI)

▶️ <https://youtu.be/Bm3Qc_oQ6oI>

## The document

- **2-page 1997 Australian income-tax return form**, A4 (595 × 849 pt).
- **Fully scanned** — `0` embedded text characters, one full-page raster image
  per page. Confirmed via pdfplumber and pypdfium2 (`text_layer_chars = 0`).
- Consequence: the **pdfplumber reference tab is empty** (no text layer to
  extract), so there is no known-good text-layer anchor for this document. Both
  backends are working purely from OCR over the page image.

## Headline numbers

| Metric | MinerU | Docling |
|---|---:|---:|
| Elements extracted | 94 | 158 |
| Text characters | 4,450 | 4,327 |
| **Tables detected** | **2** | **0** |
| Images | 19 (cropped regions) | 2 (one full-page scan per page) |
| Semantic types used | text / heading / table / image / header / footer | heading / list_item / image / **unknown** |
| Elements typed `unknown` | 0 | **126 / 158 (80%)** |
| Header/footer (furniture) tagged | 7 (5 header, 2 footer) | 0 |
| Largest box on page 2 | < 50% of page | **84% of page** (the whole scan) |
| Lost-space OCR tokens (e.g. `Areyouaresident`) | 1 | 17 |

## Dimension-by-dimension

### 1. Semantic labeling — MinerU clearly better
MinerU classifies blocks into meaningful types (`text`, `heading`, `table`,
`image`, `header`, `footer`). Docling labels **80% of its elements `unknown`**,
leaving downstream consumers without structure. For a form — where knowing what
is a field label vs a table vs furniture matters — this is a large gap.

### 2. Table detection — MinerU 2, Docling 0
MinerU detected and **structured two tables** into GFM markdown, e.g.:

```
| Sex—write X in a box  Male     Female |
| Suburb or town          State    Postcode |
| Country—if outside Australia |
```

Docling returned **no tables** for this form — the tabular field regions were
flattened into `unknown`/`list_item` text. On a form-heavy document this is the
most consequential quality difference.

### 3. OCR word segmentation — MinerU clearly better
Docling's OCR frequently **drops inter-word spaces**:

| MinerU | Docling |
|---|---|
| `Title  For example, Mr, Mrs, Ms, Miss` | `Areyouaresidentof Australia for taxpurposes?` |
| `1 July 1996 to 30 June 1997` | `Haveyouincludedanyattachments-—otherthangroup…` |

Counting long tokens that lost their spaces: **Docling 17 vs MinerU 1**. MinerU's
text is materially more usable for search, RAG chunking, and reading.

### 4. Layout granularity / localization — MinerU clearly better
MinerU performs **box-by-box layout detection**: 54 tight regions on page 2,
**none larger than half the page**. Docling represents each scanned page as
**one ~84%-page image box** with flat OCR text layered on top, so its bounding
boxes are far less useful for "click a unit → see where it came from."

### 5. Raw text volume — roughly equal
Both pulled a comparable amount of text (4,450 vs 4,327 chars), so the gap is not
about *how much* was read but about **structure, labeling, table recovery, and
word segmentation** — all of which favor MinerU here.

## Verdict (this document)

For this scanned form, **MinerU is the materially better extractor**: meaningful
semantic types, two recovered tables, clean word segmentation, furniture
tagging, and fine-grained box localization. Docling fell back to a single
full-page image plus flat, space-collapsed OCR with 80% `unknown` typing — its
structured-document strengths don't apply to a scanned image.

**This is exactly the cross-backend divergence the Parse Report viewer is built
to surface** — open `1371-6.1997.parse_report.html`, switch between the MinerU
and Docling tabs, and the difference is obvious at a glance.

## How to reproduce

```bash
make cache-clean                 # ensure a fresh parse
make report 1371-6.1997.pdf      # parses both backends, builds the linked report
# open 1371-6.1997.parse_report.html and compare the MinerU / Docling tabs
```

## What this does NOT tell you

- **Nothing about born-digital documents.** Docling is designed for PDFs with a
  real text layer, and for DOCX/HTML where it builds a genuine structure tree;
  it is not represented fairly by one scanned form.
- **No ground truth.** With no text layer, the pdfplumber reference tab is empty,
  so this is an eyeball + element-statistics comparison, not measured accuracy.
- **Not aggregate quality.** One document cannot establish which backend to
  default to across a corpus — run the eval suite for that.
