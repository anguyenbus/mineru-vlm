Implementation Guide: MinerU Confidence Score Extractor
Problem
MinerU stores OCR confidence scores internally during processing but strips them from all user-facing outputs (markdown, content list JSON). The only file that retains scores is _middle.json, but it has no per-page summary and no document-level quality signal. This module extracts those scores and presents them as a reviewable report.

What MinerU's _middle.json contains

_middle.json
└── pdf_info: [ ...one entry per page... ]
      └── page_idx, page_size
      └── preproc_blocks   ← main content blocks
      └── discarded_blocks ← headers/footers filtered out
      └── para_blocks      ← optional, added after finalization
            └── block
                  ├── score: float | null   ← layout detection confidence
                  └── lines
                        └── spans
                              └── score: float  ← OCR recognition confidence
                                                   (only present on text/formula spans)
Two block shapes exist:

Simple (text, title, list): block["lines"] → line["spans"]
Hierarchical (image, table, chart): block["blocks"] → sub_block["lines"] → line["spans"]
Project setup
Step 1 — Create pyproject.toml at the project root:


[project]
name = "atogenai"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["rich>=13.0"]

[project.optional-dependencies]
dev = ["pytest>=8.0", "ruff>=0.4", "black>=24.0"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/pdf_pipeline"]

[tool.pytest.ini_options]
testpaths = ["tests"]
Step 2 — Create the directory layout:


src/
  pdf_pipeline/
    __init__.py
    confidence.py
tests/
  __init__.py
  pdf_pipeline/
    __init__.py
    test_confidence.py
Step 3 — Install:


uv sync --all-extras --dev
Implementation steps
Step 4 — Define constants and data models (src/pdf_pipeline/confidence.py)

LOW_CONFIDENCE_THRESHOLD: Final[float] = 0.70
# NOTE: MinerU silently drops spans below 0.50 — 0.70 gives a visible warning zone above that floor.
Two @dataclass(slots=True) models:

PageConfidence — holds per-page aggregations:

block_count, mean_block_score, min_block_score, low_confidence_blocks
span_count, mean_span_score, min_span_score, low_confidence_spans
Each has a to_dict() method for JSON serialisation
DocumentConfidence — holds document-level aggregations:

source_path, backend, version, total_pages
pages: list[PageConfidence]
overall_mean_block_score, overall_mean_span_score
pages_flagged: list[int] — page indices where mean score falls below threshold
Step 5 — Implement three internal helpers
_iter_blocks(page) — yields every block from all three collections on a page:


for key in ("preproc_blocks", "discarded_blocks", "para_blocks"):
    for block in page.get(key, []):
        yield block
Include discarded_blocks — filtered low-quality blocks are still diagnostic.

_iter_spans_with_score(block) — yields span score floats, handling both block shapes:


if "blocks" in block:                          # hierarchical
    for sub_block in block["blocks"]:
        yield from _iter_spans_with_score(sub_block)
else:                                          # simple / flat
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            if "score" in span:                # skip spans with no score key
                yield float(span["score"])
_extract_page_confidence(page) — calls the two helpers above, computes aggregations:

Skip blocks where score is None — do not treat null as zero
mean_block_score = 0.0, min_block_score = 1.0 when page has no scored blocks (safe defaults)
Same sentinel defaults for span scores
Step 6 — Implement the public entry point
extract_confidence(middle_json_path: Path) → DocumentConfidence:

Read and parse the JSON file — raise ValueError on bad JSON, FileNotFoundError if missing
Check "pdf_info" key exists
Call _extract_page_confidence() for each page
Compute document-level means from pages that have at least one scored block/span
Build pages_flagged — any page where mean block score or mean span score is below threshold
Step 7 — Implement the rich display function
print_confidence_report(doc, console=None):

Print a header line with filename, backend, version, overall scores, flagged pages
Print a rich.Table — one row per page with all score columns
Colour rows: green ≥ 0.80, yellow 0.70–0.80, red < 0.70 (based on worst of block/span mean)
Score handling rules (critical)
Situation	Correct handling
block["score"] is null/None	Skip — do not count as 0
span has no "score" key	Skip — only OCR/formula spans carry scores
Page has zero scored blocks	mean=0.0, min=1.0, count=0
Hierarchical block (image/table)	Score is on the top-level block; walk block["blocks"] for spans
Tests
Write tests in tests/pdf_pipeline/test_confidence.py using synthetic dicts — no real PDF files needed.

Cover these cases:

Test	What it verifies
Simple text block	block score + flat span scores extracted correctly
Hierarchical image block	sub-block traversal finds spans
score: null on block	skipped, not counted as 0
Span without score key	skipped silently
Discarded blocks	included in per-page stats
Page flagged below threshold	appears in pages_flagged
Empty page	returns zeros/sentinels, no crash
Multi-page document (via tmp_path)	correct document-level means and flagged list
to_dict() round-trips	json.dumps(result.to_dict()) does not raise
Bad JSON file	raises ValueError
Missing file	raises FileNotFoundError
Run with:


uv run pytest tests/ -v
How to use after implementation

from pathlib import Path
from pdf_pipeline.confidence import extract_confidence, print_confidence_report
import json

result = extract_confidence(Path("output/report_middle.json"))

# Print colour-coded table to terminal
print_confidence_report(result)

# Quick check: which pages need review?
print(result.pages_flagged)   # e.g. [3, 7]

# Save machine-readable report
Path("output/report_confidence.json").write_text(
    json.dumps(result.to_dict(), indent=2)
)