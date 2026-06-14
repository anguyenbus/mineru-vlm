# Makefile for hybrid-doc-parser
#
# All targets use `uv run --no-sync` on purpose: a plain `uv run` (or `uv sync`)
# re-resolves the lockfile and reinstalls a CUDA build of torch that does NOT
# match this box's driver (555 / CUDA 12.5), breaking GPU. Keep --no-sync.

PYTHON := uv run --no-sync python
PYTEST := uv run --no-sync pytest

# Positional source file: `make report file1.pdf`.
# Pull any *.pdf/*.docx/*.html goal off the command line; fall back to SRC=...
SRC ?=
DOC := $(filter %.pdf %.PDF %.docx %.doc %.html %.htm,$(MAKECMDGOALS))
REPORT_SRC := $(if $(DOC),$(DOC),$(SRC))
# All artifacts for a run live under report/<file-stem>/ (gitignored): the two
# per-backend ParserOutput JSONs (m.json, d.json) and the linked HTML report.
REPORT_STEM := $(notdir $(basename $(REPORT_SRC)))
REPORT_DIR := report/$(REPORT_STEM)
REPORT_OUT := $(REPORT_DIR)/$(REPORT_STEM).parse_report.html

.PHONY: help test test-fast test-viz viz-install report report-clean parse

help:
	@echo "Targets:"
	@echo "  make parse file.pdf    - parse file.pdf (MinerU) -> <stem>.md, <stem>.json, <stem>.confidence.json"
	@echo "  make report file.pdf   - parse file.pdf (MinerU + Docling + MinerU2.5-Pro) and build the linked HTML report"
	@echo "  make viz-install       - install viz extra deps (pdfplumber, pillow, pypdfium2)"
	@echo "  make test              - run the full test suite"
	@echo "  make test-fast         - run tests excluding the 'slow' marker"
	@echo "  make test-viz          - run only the parse-report (viz) tests"
	@echo "  make report-clean      - remove generated report artifacts"

# --- Tests ------------------------------------------------------------------

test:
	$(PYTEST)

test-fast:
	$(PYTEST) -m "not slow"

test-viz:
	$(PYTEST) tests/hybrid_doc_parser/viz/ -q

# --- Parse Report viewer ----------------------------------------------------

# pdfplumber powers the known-good reference tab; pypdfium2 + pillow render pages.
viz-install:
	uv pip install pdfplumber pillow pypdfium2

# `make report file.pdf` — parse once per backend (GPU), then build the report.
# The pdfplumber tab is computed automatically from the source PDF.
# `make parse file.pdf [file2.pdf ...]` — MinerU parse (BATCHED via parse_batch
# for multiple files); writes RAG-ready artifacts into report/<stem>/ per file:
# <stem>.md (markdown), <stem>.json (full ParserOutput incl. the confidence
# block), and <stem>.confidence.json (per-page average confidence).
# NOTE: REPORT_SRC is passed UNQUOTED so several positional *.pdf goals forward
# as separate args (filenames with spaces should use SRC="my file.pdf").
parse:
ifeq ($(REPORT_SRC),)
	@echo "usage: make parse <file.pdf> [file2.pdf ...]   (or: make parse SRC=<file.pdf>)"; exit 2
endif
	$(PYTHON) scripts/parse_to_files.py $(REPORT_SRC)

report:
ifeq ($(REPORT_SRC),)
	@echo "usage: make report <file.pdf>   (or: make report SRC=<file.pdf>)"; exit 2
endif
	mkdir -p "$(REPORT_DIR)"
	# One backend PER PROCESS (--only): GPU-resident backends (MinerU pipeline,
	# PaddleOCR-VL, and the MinerU2.5-Pro vLLM engine) must not co-load and
	# contend for VRAM on a single device. Each parse caches independently.
	$(PYTHON) scripts/save_parser_json.py "$(REPORT_SRC)" --only mineru     --mineru-out "$(REPORT_DIR)/m.json"
	$(PYTHON) scripts/save_parser_json.py "$(REPORT_SRC)" --only docling    --docling-out "$(REPORT_DIR)/d.json"
	$(PYTHON) scripts/save_parser_json.py "$(REPORT_SRC)" --only mineru25pro --mineru25pro-out "$(REPORT_DIR)/m25pro.json"
	$(PYTHON) scripts/parse_report.py "$(REPORT_SRC)" --mineru "$(REPORT_DIR)/m.json" --docling "$(REPORT_DIR)/d.json" --mineru25pro "$(REPORT_DIR)/m25pro.json" -o "$(REPORT_OUT)"
	@echo "Open $(REPORT_OUT) in your browser (or VS Code) to review."

report-clean:
	rm -rf report
	rm -f m.json d.json *.parse_report.html

# No-op rules for the positional file goal(s) so `make report file.pdf` does not
# error with "No rule to make target 'file.pdf'". Only matches the file passed.
ifneq ($(DOC),)
$(DOC):
	@:
endif
