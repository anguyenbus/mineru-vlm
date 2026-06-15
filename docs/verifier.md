# Standalone Advisory Verifier (`verify()`)

> **STATUS: DESIGN / EXPERIMENTAL — NOT YET RECOMMENDED FOR PRODUCTION USE.**
> The verifier's design, schema, and behaviour are settled and implemented, but
> the feature is **not** recommended for production until a real-model
> precision/recall measurement passes the eval gate (see
> [Production status & eval gate](#production-status--eval-gate)). The current gate
> artifact validates only the *eval harness* against a deterministic fake backend,
> not a real model.

The verifier is a **standalone, advisory second-opinion** step. After `parse()`
has produced a `ParserOutput`, the caller may run `verify()` to compare MinerU's
extraction against the source page image using a vision-language model (VLM) and
get back a JSON report of **high-confidence disagreements** plus **missing** and
**extra** elements, for human review.

It is deliberately **separate** from `parse()` and from enrichment:

- It is **never** called from inside `parse()` or `parse_batch()`.
- It **never** mutates the parse output (`ElementRecord.text`, `is_enriched`, and
  `ParserOutput` are never touched).
- All VLM/network activity lives at the **caller's edge**, where the caller owns
  the cost decision — so `parse()`'s invariants (local, deterministic,
  never-network, never-raises, cacheable) are fully preserved.

---

## Integration model: run `verify()` AFTER `parse()`

The verifier is an edge-only follow-up, not a parse phase. The caller:

1. Runs `parse()` (or `parse_batch()`) — fully local, deterministic, cacheable.
2. Optionally runs `verify(parser_output, file_path, config)` afterwards, paying
   for VLM calls only where it chooses to.

Because verification is a separate entrypoint:

- The `VerificationReport` is a **separate return value** — it is **not** a field
  on `ParserOutput` and is **not** written to the parse cache.
- A parse-cache hit **never** triggers a verification-cache read or write.
- `parse()` keeps its "no required cloud credentials" guarantee: the cloud SDKs
  the verifier needs are an **optional** extra (see below).

### Usage example

```python
from pathlib import Path
from hybrid_doc_parser import parse, verify, VerifierConfig

path = Path("scan.pdf")

# 1. Parse — local, deterministic, never-network, cacheable.
output = parse(path)

# 2. Verify — edge-only second opinion. Run AFTER parse().
config = VerifierConfig(
    enabled=True,
    backend="bedrock",                 # or "openai_compatible" / "fake"
    model="anthropic.claude-3-5-sonnet-20241022-v2:0",  # bare on-demand id
    region="ap-southeast-2",
    min_severity_to_report="high",     # precision-favoring (default)
)
report = verify(output, path, config)

# 3. Inspect the advisory report (parse output is untouched).
for page in report.pages:
    for d in page.disagreements:
        print(page.page_idx, d.element_id, d.severity, d.reason)
    for m in page.missing_elements:
        print(page.page_idx, "MISSING", m.severity, m.approx_location)

# Serialize: always nested under a top-level "verification" key.
report.model_dump_json()
```

`verify`, `VerifierConfig`, and `VerificationReport` are all exported from the
top-level `hybrid_doc_parser` package.

### Trigger policy

By default, `verify()` verifies **only** pages the quality gate flagged
(`PageRecord.quality_decision == "promote_to_vlm"`). Pages the gate decided to
`keep` are not verified. `config.force_verify_all` overrides this to verify every
page — but that is **eval/calibration only** (see
[Limitations](#limitations)).

### Supported inputs (PDF / image only; DOCX/HTML no-op)

The verifier compares the extraction against a **rendered source page image**, so
it supports only inputs that can be rasterised: **PDF and raster images**
(`.pdf`, `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, `.webp`).

For **DOCX / HTML** (or any other non-renderable input), there is no source page
image to compare against, so `verify()` **no-ops** and returns a report whose
only content is a `verification_unsupported` warning. This holds even if pages
were flagged for VLM review.

---

## The `verify()` contract

```python
def verify(
    parser_output: ParserOutput,
    file_path: Path,
    config: VerifierConfig,
    *,
    sleeper: Callable[[float], None] = time.sleep,
) -> VerificationReport
```

| Behaviour | Guarantee |
|---|---|
| **Advisory** | Never mutates `ElementRecord.text`, never sets `is_enriched`, never touches `ParserOutput`. |
| **Never-raises** | A page-level failure degrades to a `verification_failed` warning (no verdict for that page); a whole-run failure returns an empty report with `model_id`/`prompt_version` populated where known. `verify()` never raises — mirroring `parse()`. |
| **Cost guard** | `max_pages_per_doc` is a HARD per-document cap; truncated pages are logged and flagged with a `verification_truncated` warning so a partial run is never mistaken for "all clean". |
| **Concurrency** | Per-page calls are bounded by a **separate** `asyncio.Semaphore(config.max_concurrency)`, independent of `parse_batch` concurrency (Bedrock quotas are independent of local engine concurrency and Bedrock throttles hard). |
| **Resilience** | Throttling/transient errors (429/503) are retried with jittered exponential backoff up to a bounded number of attempts; on exhaustion the page degrades to `verification_failed` rather than raising. |

The `sleeper` keyword argument exists only so tests can inject a no-wait sleep; in
production you never set it.

---

## `VerifierConfig`

A frozen Pydantic model, separate from `EnrichmentConfig`. All fields have safe
defaults, and the defaults leave the verifier **OFF** (`enabled=False`) and tuned
to favour **precision over recall**.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `enabled` | `bool` | `False` | Master switch. When `False`, `verify()` performs no work. |
| `backend` | `"bedrock" \| "openai_compatible" \| "fake"` | `"bedrock"` | Which verifier backend to dispatch. `"bedrock"` is the v1 primary; `"openai_compatible"` is for local vllm/Ollama eval; `"fake"` returns canned verdicts (CI / no network). |
| `model` | `str` | `""` | Bare on-demand model identifier (no inference-profile prefix — those are SCP-blocked). |
| `region` | `str` | `""` | Cloud region for the backend (e.g. AWS region for Bedrock). |
| `force_verify_all` | `bool` | `False` | When `True`, verify **every** page, not just flagged pages. **Eval/calibration only** — exists solely to measure the quality gate's recall gap. |
| `max_concurrency` | `int` (`>= 1`) | `2` | Upper bound on in-flight verifier calls; a **separate** semaphore from `parse_batch`. |
| `render_dpi` | `int` (`72`–`600`) | `150` | DPI for the full-page render fed to the verifier (the megapixel clamp in `render_page` guards huge pages). |
| `max_pages_per_doc` | `int` (`>= 1`) | `50` | HARD per-document cap on verified pages. |
| `timeout` | `float` (`> 0`) | `60.0` | Per-call backend timeout in seconds. |
| `min_severity_to_report` | `"low" \| "medium" \| "high"` | `"high"` | Drop findings below this severity from the report. Precision-favoring. |

For the OpenAI-compatible backend, the base URL and API key are read from the
standard `OPENAI_BASE_URL` / `OPENAI_API_KEY` environment variables.

---

## Report envelope (`VerificationReport`)

`VerificationReport` always serialises with the **entire payload nested under a
top-level `verification` key**. `model_dump()` and `model_dump_json()` both emit:

```json
{
  "verification": {
    "model_id": "anthropic.claude-...",
    "prompt_version": "v1",
    "pages": [
      {
        "page_idx": 3,
        "disagreements": [
          {
            "element_id": "p3-e7",
            "type": "table",
            "severity": "high",
            "reason": "Row 4 merged two columns; values misaligned vs image.",
            "suggested_text": "...",
            "vlm_confidence": 0.86
          }
        ],
        "missing_elements": [
          {
            "severity": "medium",
            "reason": "Footnote at page bottom not extracted.",
            "approx_location": "below last paragraph"
          }
        ],
        "extra_elements": []
      }
    ],
    "warnings": [
      {
        "code": "verification_failed",
        "page_idx": 9,
        "message": "Bedrock throttled after 3 retries"
      }
    ]
  }
}
```

Deserialization is symmetric: `VerificationReport.model_validate(...)` accepts
either the wrapped `{"verification": {...}}` envelope or the bare inner payload.

Shape notes:

- **Top-level `verification`** carries `model_id`, `prompt_version`, a `pages`
  list, and a `warnings` list.
- **Each page** carries `page_idx`, `disagreements`, `missing_elements`, and
  `extra_elements`.
- A **disagreement** is keyed by `element_id` and carries `type`, `severity`,
  `reason`, `suggested_text`, and `vlm_confidence` (`0.0`–`1.0`).
- A **missing element** (MinerU false negative) and an **extra element** (MinerU
  false positive) share the same shape — `severity`, `reason`, `approx_location`
  — and have **no** `element_id`.
- **Warnings** reuse the `WarningRecord` shape (`page_idx`, `code`, `message`).
  Verifier warning codes: `verification_failed`, `verification_unsupported`,
  `verification_truncated`.

Internally the model is asked for a **per-element verdict** (to avoid pressuring
it to invent a disagreement); `verify()` then **filters** to keep only items at
or above `min_severity_to_report` and persists only the disagreement / missing /
extra signal.

---

## The optional `verifier` extra (lazy SDK imports)

The cloud SDKs the verifier may need are **not** core dependencies. Install them
via the optional `verifier` extra:

```bash
uv pip install -e ".[verifier]"   # adds boto3 (Bedrock) and openai (OpenAI-compatible)
```

`boto3` and `openai` are imported **lazily inside the verifier client methods**,
so:

- The core library stays cloud-SDK-free.
- Callers who never verify need no cloud credentials and incur no SDK import.
- The `"fake"` backend needs **no** SDKs at all (CI and the eval harness use it).

---

## Separate verification cache

The verifier has its **own** file-based cache (`verifier_cache.py`), entirely
separate from the parse cache (`cache.py`):

- **Key:** the 4-tuple `(content_hash, page_idx, model_id, prompt_version)`, where
  `content_hash` is `ParserOutput.file_sha256`, `model_id` is
  `VerifierConfig.model`, and `prompt_version` is the verifier's `PROMPT_VERSION`.
  A change to the model id **or** the prompt version produces a new key and so
  correctly busts the cached verdict.
- **Unit cached:** a single page's verdict (`PageVerification`).
- **Never piggybacks** on the parse cache: it is touched **only** from inside
  `verify()`'s per-page flow, never as a side effect of a parse-cache hit.
- **Location:** `HYBRID_DOC_PARSER_VERIFIER_CACHE_DIR`, falling back to
  `~/.cache/hybrid_doc_parser/verifier` (a separate subdirectory from the parse
  cache).
- **Non-raising:** read/write failures degrade silently (a miss / a no-op), and
  writes are atomic via a `.json.tmp` rename. Only **successful** verdicts are
  cached; a `verification_failed` page is never cached.

---

## Limitations

- **Recall is bounded by the quality gate.** In the default trigger mode, only
  pages flagged `promote_to_vlm` are verified — a defect on a page the gate
  wrongly *keeps* is **never seen**. The verifier cannot exceed the quality
  gate's recall. `force_verify_all` exists **solely** to *measure* that gap during
  eval; it is **not** a production mode (it verifies every page, multiplying cost).
- **Precision-over-recall tuning.** `min_severity_to_report` (default `"high"`)
  filters out lower-severity findings so the surfaced report is high-confidence.
  Lowering it raises recall at the cost of precision.

### Known issues

- **The verification cache key does not include `min_severity_to_report`, and it
  caches the *post-filter* verdict.** The severity filter is applied *before* the
  verdict is written to cache, but `min_severity_to_report` is **not** part of the
  cache key. So if you change the severity floor and re-run against a warm cache,
  you can get back a **stale, differently-filtered** result rather than a fresh
  one. **When sweeping severity thresholds, run with a clean verification cache**
  (point `HYBRID_DOC_PARSER_VERIFIER_CACHE_DIR` at a fresh directory, or clear it
  between runs). This is a known limitation, not intended behaviour.

---

## Production status & eval gate

The feature is documented as **DESIGN / EXPERIMENTAL — NOT YET RECOMMENDED FOR
PRODUCTION USE.**

A measured precision/recall figure is a **hard prerequisite** before any
"recommended for production" claim. The current gate artifact lives at
`agent-os/specs/2026-06-15-llm-disagreement-verifier/verifications/eval-results.md`
and reports a **deterministic fake-path** result only:

- It validates that the eval **harness** (`scripts/eval_verifier.py`) computes
  precision/recall correctly against the labeled set, using `FakeVerifierClient`'s
  known verdicts. These are **not** real-model numbers.
- It records the precision-favoring severity floor (`min_severity_to_report =
  "medium"` on the labeled set) and quantifies the quality-gate recall gap via
  `force_verify_all`.

A **real-model** precision/recall measurement via the `--live` path (Bedrock, or a
local vllm/Ollama endpoint, against real PDF fixtures) is **still required** and
must report an acceptable, precision-favoring figure before the eval gate is
satisfied. **Until that live run passes, do not treat the verifier as
production-ready.**
