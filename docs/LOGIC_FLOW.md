# Logic Flow — hybrid-doc-parser

This document explains **how the code thinks**: the control flow through the
library, from a single `parse()` call to batch inference, the quality gate, and
VLM enrichment. It complements [parse-pipeline.md](parse-pipeline.md) (which is
the exhaustive per-branch reference) by focusing on the *decision logic* and the
*module relationships*.

- **Entry points:** `parse()` and `parse_batch()` in
  [`parser.py`](../src/hybrid_doc_parser/parser.py)
- **Core invariant:** neither entry point raises — failures become
  `WarningRecord`s inside a valid `ParserOutput`.

---

## 1. Module map — who calls whom

```mermaid
flowchart TD
    CALLER([Caller]) --> PARSE["parse() / parse_batch()<br/>parser.py"]

    PARSE --> CACHE["cache.py<br/>get / put (sha256+mtime key)"]
    PARSE --> MINERU["MinerU engine<br/>_run_mineru / _run_mineru_batch"]
    PARSE --> DOCLING["Docling engine<br/>_run_docling"]

    MINERU --> ROUTE["Block routing<br/>_route_block / _route_mineru_content_list"]
    DOCLING --> ROUTED["Docling routing<br/>_route_docling_block"]

    ROUTE --> GATE["quality_gate.py<br/>evaluate_page"]
    ROUTED --> GATE
    GATE --> RENDER1["render.py<br/>text_layer_tokens (PDF only)"]

    GATE --> ENRICH["_enrich_elements<br/>parser.py"]
    ENRICH --> CTX["context.py<br/>surrounding-block context"]
    ENRICH --> MODAL["modal_processors.py<br/>image / table / equation"]
    MODAL --> RENDER2["render.py<br/>render_region → PNG"]
    MODAL --> VLM["vlm_client.py<br/>OpenAI-compatible / Bedrock"]

    ENRICH --> BUILD["_build_parser_output<br/>assemble + validate"]
    BUILD --> MODELS["models.py<br/>Pydantic ParserOutput"]
    BUILD --> CACHE
    BUILD --> CALLER

    CALLER --> MD["markdown.py<br/>render_markdown (optional)"]
```

---

## 2. Single-document flow — `parse()`

The happy path and every guard, in order. Each red terminal is a valid
`ParserOutput` carrying a warning — **not** an exception.

```mermaid
flowchart TD
    START([parse file_path, config]) --> CFG{config is None?}
    CFG -- yes --> DEF[config = EnrichmentConfig defaults]
    CFG -- no --> EXIST
    DEF --> EXIST{file exists?}

    EXIST -- no --> W1([ParserOutput<br/>file_not_found]):::err
    EXIST -- yes --> EXT{extension accepted?<br/>backend-aware set}
    EXT -- no --> W2([ParserOutput<br/>unsupported_type]):::err
    EXT -- yes --> SHA[compute SHA-256 of bytes]

    SHA --> CACHE{cache hit?<br/>sha256 + mtime}
    CACHE -- hit --> HIT([return cached ParserOutput]):::ok
    CACHE -- miss --> DISPATCH{config.parser}

    DISPATCH -- docling --> DRUN[_run_docling]
    DISPATCH -- mineru / auto --> MRUN[_run_mineru]

    DRUN -- raises --> WD([ParserOutput<br/>docling_failed]):::err
    MRUN -- raises --> WM([ParserOutput<br/>mineru_failed]):::err

    DRUN -- ok --> NORM[normalise + route blocks<br/>→ ElementRecords]
    MRUN -- ok --> NORM

    NORM --> BUILD[_build_parser_output]
    BUILD --> GATE[per-page quality gate]
    GATE --> ENR{enrichment enabled<br/>AND parser=mineru?}
    ENR -- no --> ASM
    ENR -- docling --> WNS[warning:<br/>enrichment_not_supported] --> ASM
    ENR -- yes --> VLMSTEP[VLM enrichment per element] --> ASM
    ASM[assemble + validate<br/>ParserOutput] --> PUT[cache.put<br/>atomic .tmp rename]
    PUT --> RET([return ParserOutput]):::ok

    OUTER[["outer try/except<br/>last-resort net → mineru_error / docling_error"]]:::err

    classDef err fill:#fcc,stroke:#c00;
    classDef ok fill:#cfc,stroke:#090;
```

**Why two layers of try/except:** the engine-specific blocks preserve the precise
codes (`mineru_failed` / `docling_failed`); the outer block is a last-resort net
that guarantees the never-raise contract for anything unforeseen
(`mineru_error` / `docling_error`).

---

## 3. Quality gate logic — `evaluate_page()`

The gate decides, per page, whether the cheap engine output is good enough
(`keep`) or should be escalated for VLM review (`promote_to_vlm`). Two layers,
short-circuit on the first failure.

```mermaid
flowchart TD
    P([evaluate_page idx, elements, pdf_tokens]) --> L1{Layer 1: coverage<br/>pdf_tokens known<br/>AND ≥ 50?}
    L1 -- no / unknown --> L2
    L1 -- yes --> COV[coverage = extracted_tokens ÷ pdf_tokens]
    COV --> COVQ{coverage < 30%?}
    COVQ -- yes --> PROMO([promote_to_vlm<br/>layer=1]):::warn
    COVQ -- no --> L2

    L2{Layer 2: heuristics<br/>combined page text empty?} -- empty --> KEEP([keep]):::ok
    L2 -- non-empty --> SIG[measure 5 signals]
    SIG --> SIGQ{any signal fails?}
    SIGQ -- yes --> PROMO
    SIGQ -- no --> KEEP

    classDef warn fill:#fc9,stroke:#a60;
    classDef ok fill:#cfc,stroke:#090;
```

| Layer | Signal | Threshold | Catches |
|---|---|---|---|
| 1 | Coverage ratio | < 30% (PDF only, ≥ 50 text-layer tokens) | Engine missed most of the embedded text |
| 2 | Garbled token ratio | > 20% | Mixed letter+digit junk (`a1b`) |
| 2 | Mean word length | < 2.0 chars | OCR noise / fragments |
| 2 | Dictionary hit rate | < 50% | Few real words/numbers |
| 2 | Repeated char run | ≥ 7 | OCR smearing (`aaaaaaa`) |
| 2 | ASCII printable ratio | < 90% | Control/garbage characters |

Layer 1 is **PDF-only** (needs an embedded text layer; pypdfium2 counts tokens via
[`render.py`](../src/hybrid_doc_parser/render.py)). DOCX/HTML skip it. Layer 2 runs
for all input types. A blank page is a clean `keep`, never an escalation.

---

## 4. VLM enrichment logic — `_enrich_elements()`

Runs only when `config.enabled=True` **and** `config.parser="mineru"`. Each target
element is enriched independently; an error on one element is captured as a warning
and does not stop the others.

```mermaid
flowchart TD
    E([for each ElementRecord]) --> T{type & config flag}
    T -- image + config.image --> IMG[render_region → PNG bytes<br/>ImageModalProcessor]
    T -- table + config.table --> TBL[TableModalProcessor<br/>HTML body]
    T -- equation + config.equation --> EQ[EquationModalProcessor<br/>LaTeX source]
    T -- other / disabled --> SKIP[leave element as-is]

    IMG --> CTX[ContextExtractor<br/>N surrounding blocks]
    TBL --> CTX
    EQ --> CTX
    CTX --> PROMPT[build type-specific prompt + payload]
    PROMPT --> BACKEND{vlm_backend}
    BACKEND -- openai_compatible --> OAI[OpenAICompatibleClient<br/>OPENAI_BASE_URL / KEY / MODEL]
    BACKEND -- bedrock --> BR[BedrockClient<br/>boto3 invoke_model]
    OAI --> PARSE2[_robust_json_parse<br/>strip thinking tags, 4 fallbacks]
    BR --> PARSE2
    PARSE2 --> UPD[ElementRecord:<br/>description set, is_enriched=True]
    PARSE2 -- error --> WARN[warning: enrichment_error<br/>element unchanged]
```

The VLM backend is selected by `make_vlm_client(config)` in
[`vlm_client.py`](../src/hybrid_doc_parser/vlm_client.py). To self-host the model
(MinerU 2.5 VL via vLLM), use `openai_compatible` and point `OPENAI_BASE_URL` at
the vLLM server — see
[SELF_HOSTING_JUSTIFICATION.md](SELF_HOSTING_JUSTIFICATION.md).

---

## 5. Batch flow — `parse_batch()`

The batch path exists to cut MinerU inference windows from `N` (one per file) to
`ceil(N / MINERU_BATCH_SIZE)` (one per chunk). This is the focus of the
`batch_process` work.

```mermaid
flowchart TD
    B([parse_batch paths, config, max_concurrency]) --> NB{parser == mineru?}
    NB -- no, e.g. docling --> FAN[per-file fan-out<br/>asyncio.gather + semaphore]
    FAN --> ORDER

    NB -- yes --> CLS[STEP 1: classify all paths<br/>_classify_batch_paths, off-thread]
    CLS --> BUCKETS[/three buckets:<br/>invalid · cache_hits · needs_parse/]
    BUCKETS --> CHUNK[STEP 2: slice needs_parse<br/>into chunks of MINERU_BATCH_SIZE]

    CHUNK --> LOOP{for each chunk}
    LOOP --> ASM[_assemble_chunk_outputs<br/>ONE do_parse call, off-thread]
    ASM -- ok --> COLLECT[collect chunk outputs]
    ASM -- raises --> FB[per-chunk fallback:<br/>per-file parse for THIS chunk only]
    FB --> COLLECT
    COLLECT --> LOOP

    LOOP -- done --> ORDER[STEP 3: merge in input order<br/>invalid ∪ cache_hits ∪ parsed]
    ORDER --> SUM[log batch summary:<br/>requested / cache_hits / parsed / failed / fallback]
    SUM --> RET([list ParserOutput, input order]):::ok

    classDef ok fill:#cfc,stroke:#090;
```

Key properties:

- **Cache hits and invalid paths never reach inference** — they are resolved in
  STEP 1 and merged back at the end.
- **Per-chunk isolation** — if one chunk's `do_parse` raises, only that chunk
  falls back to per-file `parse()`; other chunks are unaffected.
- **Concurrency knobs are separate:** `max_concurrency` governs the per-file
  fallback and the non-MinerU path; `MINERU_BATCH_SIZE` controls chunk size and
  `MINERU_MAX_INFLIGHT` bounds concurrent MinerU inference calls (multi-GPU). All
  blocking work (sha256, inference, token counting, enrichment) is offloaded with
  `asyncio.to_thread` so the event loop never stalls.
- **Output order always equals input order**, one `ParserOutput` per input path.

---

## 6. The never-raise contract, visualised

Every box that can fail funnels into a warning, never an exception:

```mermaid
flowchart LR
    IN([input]) --> WORK[engine / gate / enrichment / cache]
    WORK -- success --> OUT([ParserOutput<br/>warnings = empty]):::ok
    WORK -- any failure --> OUTW([ParserOutput<br/>warnings = code + message]):::warn
    classDef ok fill:#cfc,stroke:#090;
    classDef warn fill:#fc9,stroke:#a60;
```

Warning codes (full table in [parse-pipeline.md](parse-pipeline.md#warning-codes)):
`file_not_found`, `unsupported_type`, `mineru_failed`, `mineru_error`,
`docling_failed`, `docling_error`, `enrichment_error`, `enrichment_not_supported`,
`quality_gate_escalation`, `image_too_large`, `cache_write_error`, `render_failed`.

---

## See also

- [parse-pipeline.md](parse-pipeline.md) — exhaustive per-branch reference + full schema
- [pdf-routing.md](pdf-routing.md) — how PDFs are routed between backends
- [PACKAGES.md](PACKAGES.md) — dependency inventory
- [SELF_HOSTING_JUSTIFICATION.md](SELF_HOSTING_JUSTIFICATION.md) — why self-host MinerU 2.5 VL
