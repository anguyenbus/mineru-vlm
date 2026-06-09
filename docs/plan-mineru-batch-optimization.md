# Plan: MinerU Batch Optimization (Approach A)

## Problem

`parse_batch()` currently fans out to N individual `parse()` calls via `asyncio.to_thread`.
Each call invokes `_run_mineru_inprocess()` with a single-item `pdf_bytes_list`, so N files
produce N separate `do_parse` calls and N separate GPU inference windows.

`do_parse` accepts `pdf_bytes_list: list[bytes]` — all pages from all documents are batched
together through `pipeline_doc_analyze_streaming` in a single inference window. We are not
using this capability at all today.

## Goal

One `do_parse` call for all uncached MinerU files in a `parse_batch()` invocation. The
`parse()` single-file API is **not changed**. The Docling path is **not changed**.

---

## New Functions

### 1. `_normalise_mineru_content(content_list: list[dict]) -> list[dict]`

Extracted verbatim from `parse()`. Filters non-dict blocks and normalises MinerU field aliases.
No logic change — pure extraction.

```python
def _normalise_mineru_content(content_list: list[dict]) -> list[dict]:
    valid = [b for b in content_list if isinstance(b, dict)]
    if len(valid) < len(content_list):
        logger.debug("[parser] skipped {} non-dict block(s)", len(content_list) - len(valid))
    return [_normalise_aliases(b) for b in valid]
```

### 2. `_route_mineru_content_list(content_list: list[dict], sha256: str) -> list[ElementRecord]`

Extracted verbatim from `parse()`. Groups blocks by `page_idx` and routes each via
`_route_block`. Per-block errors are logged and skipped (non-fatal). No logic change.

```python
def _route_mineru_content_list(
    content_list: list[dict], sha256: str
) -> list[ElementRecord]:
    pages_map: dict[int, list[dict]] = {}
    for block in content_list:
        pidx = int(block.get("page_idx", 0))
        pages_map.setdefault(pidx, []).append(block)
    page_count = (max(pages_map.keys()) + 1) if pages_map else 0
    elements: list[ElementRecord] = []
    element_idx = 0
    for pidx in range(page_count):
        for block in pages_map.get(pidx, []):
            try:
                elements.append(_route_block(block, pidx, element_idx, sha256))
            except Exception as exc:
                logger.debug("[parser] skipping block page={} idx={}: {}", pidx, element_idx, exc)
            element_idx += 1
    return elements
```

### 3. `_build_parser_output(file_path, sha256, elements, content_list, config) -> ParserOutput`

**Non-raising by contract.** Owns the quality gate, enrichment, assembly, and cache write.
Both `parse()` and `parse_batch()` call this instead of duplicating those steps.

**Critical constraint:** has its own `try/except Exception` outer handler that returns a
`ParserOutput` with a backend-specific error code. Never propagates.

```python
def _build_parser_output(
    file_path: Path,
    sha256: str,
    elements: list[ElementRecord],
    content_list: list[dict],   # raw MinerU blocks for enrichment context; [] for Docling
    config: EnrichmentConfig,
) -> ParserOutput:
    try:
        page_count = (max(e.page_idx for e in elements) + 1) if elements else 0
        warnings: list[WarningRecord] = []

        # Layer 1: pypdfium2 token counts — PDF only, serialised via _PDFIUM_LOCK.
        token_counts: dict[int, int] = {}
        is_non_pdf = file_path.suffix.lower() != ".pdf"
        if not is_non_pdf:
            from hybrid_doc_parser.render import text_layer_tokens
            with _PDFIUM_LOCK:
                token_counts = text_layer_tokens(file_path)
        else:
            logger.debug("[quality_gate] skipping Layer 1 for non-PDF input: {}", file_path)

        # Quality gate — per page.
        page_records: list[PageRecord] = []
        for pidx in range(page_count):
            page_elements = [e for e in elements if e.page_idx == pidx]
            pdf_tokens = None if is_non_pdf else token_counts.get(pidx)
            decision = evaluate_page(pidx, page_elements, pdf_tokens)
            vlm_used = decision.action == "promote_to_vlm"
            if vlm_used:
                warnings.append(WarningRecord(
                    page_idx=pidx,
                    code="quality_gate_escalation",
                    message=f"Page {pidx} escalated to VLM: {decision.reason}",
                ))
            page_records.append(PageRecord(
                page_idx=pidx,
                quality_decision=decision.action,
                element_count=len(page_elements),
                vlm_used=vlm_used,
            ))

        # Enrichment.
        if config.enabled and config.parser == "docling":
            warnings.append(WarningRecord(
                code="enrichment_not_supported",
                message="VLM enrichment is not yet supported for parser='docling'",
            ))
        elif config.enabled:
            try:
                elements = _enrich_elements(elements, content_list, config, file_path)
            except Exception as exc:
                logger.warning("[parser] enrichment failed for {}: {}", file_path, exc)
                warnings.append(WarningRecord(
                    code="enrichment_error",
                    message=f"Enrichment failed: {exc}",
                ))

        output = ParserOutput(
            file_path=str(file_path),
            file_sha256=sha256,
            page_count=page_count,
            pages=page_records,
            elements=elements,
            warnings=warnings,
            enrichment_config=config,
        )
        cache_mod.put(file_path, output)
        return output

    except Exception as exc:
        logger.warning("[parser] _build_parser_output failed for {}: {}", file_path, exc)
        error_code = "docling_error" if config.parser == "docling" else "mineru_error"
        return ParserOutput(
            file_path=str(file_path),
            file_sha256=sha256,
            page_count=0, pages=[], elements=[],
            warnings=[WarningRecord(code=error_code, message=f"Pipeline error: {exc}")],
            enrichment_config=config,
        )
```

### 4. `_read_content_list_for_stem(output_dir: Path, stem: str) -> list[dict]`

Targeted per-stem lookup for multi-file `do_parse` output. Replaces the "find first"
scan in `_read_output_files` for the batch path.

`do_parse` writes to: `{output_dir}/{stem}/auto/{stem}_content_list.json`

```python
def _read_content_list_for_stem(output_dir: Path, stem: str) -> list[dict]:
    # Targeted lookup first — faster and correct for multi-file output.
    targeted = list(output_dir.rglob(f"{stem}_content_list.json"))
    if targeted:
        try:
            data = json.loads(targeted[0].read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
            if isinstance(data, dict) and "content_list" in data:
                return data["content_list"]
        except Exception as exc:
            logger.debug("[parser] failed to read content_list for stem {}: {}", stem, exc)
    return []
```

### 5. `_run_mineru_batch(file_paths: list[Path], backend: str) -> dict[Path, list[dict]]`

Calls `do_parse` **once** for all paths. Returns `{path: content_list}`.
**Raises on failure** — caller is responsible for fallback.

```python
def _run_mineru_batch(
    file_paths: list[Path], backend: str = "pipeline"
) -> dict[Path, list[dict]]:
    from mineru.cli.common import do_parse, read_fn

    # NOTE: Use unique stems — two files with the same stem from different
    # directories would collide in do_parse's output directory.
    stems = [p.stem for p in file_paths]
    if len(set(stems)) < len(stems):
        raise ValueError("Duplicate file stems in batch — cannot disambiguate output files")

    bytes_list = [read_fn(p) for p in file_paths]

    # NOTE: do_parse mutates pdf_file_names and pdf_bytes_list (removes non-PDF
    # entries handled by _process_office_doc). Pass copies so our lists stay intact.
    stems_copy = list(stems)
    bytes_copy = list(bytes_list)
    langs_copy = ["en"] * len(file_paths)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        do_parse(
            output_dir=str(out_dir),
            pdf_file_names=stems_copy,
            pdf_bytes_list=bytes_copy,
            p_lang_list=langs_copy,
            backend=backend,
            parse_method="auto",
            f_draw_layout_bbox=False,
            f_draw_span_bbox=False,
            f_dump_md=False,
            f_dump_middle_json=False,
            f_dump_model_output=False,
            f_dump_orig_pdf=False,
            f_dump_content_list=True,
        )
        results: dict[Path, list[dict]] = {}
        for path, stem in zip(file_paths, stems):
            results[path] = _read_content_list_for_stem(out_dir, stem)

    return results
```

---

## Modified Functions

### `parse()` — simplified body

The outer `try/except` stays as the last-resort safety net. The body shrinks because routing,
QG, enrichment, and assembly are now in `_build_parser_output()`.

```
parse(file_path, config):
    try:
        validate existence           ← unchanged
        validate extension           ← unchanged
        compute sha256               ← unchanged
        cache check                  ← unchanged

        if parser == "docling":
            elements = _run_docling(file_path, config)   ← unchanged
            content_list = []
        else:
            raw = _run_mineru(file_path, backend)        ← unchanged
            content_list = _normalise_mineru_content(raw)  ← extracted helper
            elements = _route_mineru_content_list(content_list, sha256)  ← extracted helper

        return _build_parser_output(file_path, sha256, elements, content_list, config)
        # ↑ non-raising; handles QG + enrichment + assembly + cache write

    except Exception as exc:          ← safety net, rarely triggered
        ...
```

### `parse_batch()` — MinerU fast path added

```
parse_batch(paths, config, max_concurrency):

    if config.parser != "mineru":
        → existing asyncio.to_thread(parse, p) path   ← unchanged

    # MinerU batch path:

    STEP 1 — Classify all paths (sequentially, fast I/O):
        for each path:
            if not exists           → invalid_outputs[path] = ParserOutput(file_not_found)
            if bad extension        → invalid_outputs[path] = ParserOutput(unsupported_type)
            else:
                sha256 = _file_sha256(path)
                cached = cache_mod.get(path)
                if cached:          → cache_hits[path] = cached
                else:               → needs_parse[path] = sha256

    STEP 2 — Batch inference (one do_parse call):
        try:
            batch_results = _run_mineru_batch(list(needs_parse.keys()), backend)
        except Exception as exc:
            logger.warning("[parser] batch MinerU failed: {}; falling back to per-file", exc)
            → fallback: asyncio.to_thread(parse, p) for each p in needs_parse
            → store results in parsed_outputs
        else:
            for path, content_list in batch_results.items():
                sha256 = needs_parse[path]
                content_list = _normalise_mineru_content(content_list)
                elements = _route_mineru_content_list(content_list, sha256)
                parsed_outputs[path] = _build_parser_output(
                    path, sha256, elements, content_list, config
                )
            # NOTE: paths where batch returned empty content_list (do_parse produced
            # nothing) get a ParserOutput with code="mineru_failed".

    STEP 3 — Merge in original input order:
        return [
            invalid_outputs.get(p)
            or cache_hits.get(p)
            or parsed_outputs.get(p)
            for p in paths
        ]
```

---

## Files Changed

| File | Change |
|------|--------|
| `src/hybrid_doc_parser/parser.py` | Add `_normalise_mineru_content`, `_route_mineru_content_list`, `_build_parser_output`, `_read_content_list_for_stem`, `_run_mineru_batch`; simplify `parse()`; rewrite `parse_batch()` MinerU path |
| `tests/hybrid_doc_parser/test_parser.py` | Update existing tests that cover routing/QG/enrichment to also test via `_build_parser_output` directly |
| `tests/hybrid_doc_parser/test_parser_batch.py` | New file — batch-specific tests (see below) |

`models.py`, `quality_gate.py`, `cache.py`, `render.py`, all Docling code — **untouched**.

---

## Test Plan (TDD order)

### Group 1 — Extracted helpers (no MinerU required)

These tests can run without MinerU installed because they test pure routing and pipeline logic.

| Test | What it verifies |
|------|-----------------|
| `test_normalise_mineru_content_filters_non_dict` | Non-dict items dropped, aliases normalised |
| `test_route_mineru_content_list_page_grouping` | Blocks grouped by `page_idx`, `_route_block` called per block |
| `test_route_mineru_content_list_skips_bad_blocks` | Exception in `_route_block` skipped non-fatally |
| `test_build_parser_output_quality_gate_keep` | QG `keep` decision propagated to `PageRecord` |
| `test_build_parser_output_quality_gate_promote` | QG `promote_to_vlm` adds `quality_gate_escalation` warning |
| `test_build_parser_output_non_pdf_skips_layer1` | No pypdfium2 call for `.docx` / `.html` suffix |
| `test_build_parser_output_enrichment_not_supported_docling` | `enrichment_not_supported` warning emitted for Docling + enabled |
| `test_build_parser_output_never_raises` | Inject an exception inside the helper; assert returns `ParserOutput` with error code |
| `test_build_parser_output_writes_cache` | `cache_mod.put` called with the assembled output |

### Group 2 — `_run_mineru_batch` (mock `do_parse`)

| Test | What it verifies |
|------|-----------------|
| `test_run_mineru_batch_calls_do_parse_once` | `do_parse` called exactly once for N files |
| `test_run_mineru_batch_reads_per_stem` | Each path maps to its own `*_content_list.json` |
| `test_run_mineru_batch_raises_on_duplicate_stems` | `ValueError` when two paths share the same stem |
| `test_run_mineru_batch_raises_on_do_parse_failure` | `RuntimeError` from `do_parse` propagated |
| `test_run_mineru_batch_passes_copies_to_do_parse` | Original `file_paths` list not mutated |

### Group 3 — `parse_batch()` integration

| Test | What it verifies |
|------|-----------------|
| `test_parse_batch_mineru_output_identical_to_parse` | `parse_batch([f])` result equals `parse(f)` result |
| `test_parse_batch_calls_do_parse_once_for_n_files` | N files → 1 `do_parse` call (not N) |
| `test_parse_batch_preserves_input_order` | Output list order matches `paths` input order |
| `test_parse_batch_handles_cache_hits` | Cached paths skip `do_parse`; `do_parse` called with remaining N−M paths |
| `test_parse_batch_handles_invalid_paths` | Missing file → `file_not_found` at correct index |
| `test_parse_batch_fallback_on_batch_failure` | `_run_mineru_batch` raises → falls back to per-file `parse()` |
| `test_parse_batch_docling_uses_per_file_path` | `parser="docling"` never calls `_run_mineru_batch` |
| `test_parse_batch_mixed_cached_and_uncached` | Merge step produces correct output for mixed batch |

### Group 4 — Regression

```bash
uv run pytest tests/hybrid_doc_parser/test_parser.py tests/hybrid_doc_parser/test_models.py tests/hybrid_doc_parser/test_parser_docling.py -q
```

All pre-existing tests must remain green. Zero tolerance.

---

## Non-Obvious Constraints

**Duplicate stems.** Two files with the same stem (e.g. `invoice.pdf` from two different
directories) would collide in `do_parse`'s output directory. `_run_mineru_batch` must detect
this before calling `do_parse` and raise `ValueError` — the caller falls back to per-file.

**`do_parse` mutates its input lists.** `_process_office_doc` deletes entries it handles
(DOCX etc.) from `pdf_file_names` and `pdf_bytes_list` by index. Always pass copies.

**`_PDFIUM_LOCK` scope.** `_build_parser_output` acquires `_PDFIUM_LOCK` for
`text_layer_tokens`. In `parse_batch`, `_build_parser_output` is called per-file
sequentially after batch inference completes — the lock serialises these calls correctly
without deadlock.

**Empty content_list from batch.** `do_parse` may silently produce no output for a file
that triggers an internal error. `_run_mineru_batch` returns `[]` for that stem.
`_build_parser_output` called with `elements=[]` produces a valid `ParserOutput` with
`page_count=0` and no `mineru_failed` warning — this is silent data loss. Mitigation: in
`parse_batch`, if `content_list` is empty AND `file_path` exists AND extension is valid,
emit a `WarningRecord(code="mineru_failed", message="do_parse produced no output")`.

**Fallback atomicity.** When the batch `do_parse` call fails, the fallback re-parses all
`needs_parse` files individually. This means those files are parsed twice in the worst case.
This is acceptable — correctness over efficiency. Do not try to salvage partial batch results.

---

## What This Does NOT Change

- `parse()` public signature and never-raise contract
- `parse_batch()` public signature
- Docling engine path (any input)
- CLI subprocess fallback inside `_run_mineru()`
- `_read_output_files()` (still used by single-file `_run_mineru_inprocess`)
- All models, cache, quality gate, render, markdown modules
- Any existing test
