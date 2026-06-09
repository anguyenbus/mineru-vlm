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
REPORT_OUT := $(basename $(REPORT_SRC)).parse_report.html

.PHONY: help test test-fast test-viz viz-install report report-clean

help:
	@echo "Targets:"
	@echo "  make report file.pdf   - parse file.pdf (MinerU + Docling) and build the linked HTML report"
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
report:
ifeq ($(REPORT_SRC),)
	@echo "usage: make report <file.pdf>   (or: make report SRC=<file.pdf>)"; exit 2
endif
	$(PYTHON) scripts/save_parser_json.py "$(REPORT_SRC)"
	$(PYTHON) scripts/parse_report.py "$(REPORT_SRC)" --mineru m.json --docling d.json -o "$(REPORT_OUT)"
	@echo "Open $(REPORT_OUT) in your browser (or VS Code) to review."

report-clean:
	rm -f m.json d.json *.parse_report.html

# No-op rules for the positional file goal(s) so `make report file.pdf` does not
# error with "No rule to make target 'file.pdf'". Only matches the file passed.
ifneq ($(DOC),)
$(DOC):
	@:
endif
