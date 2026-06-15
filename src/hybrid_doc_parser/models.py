"""Pydantic v2 schema definitions for the hybrid-doc-parser library.

This module defines the complete output schema produced by ``parse()`` and
consumed by ``render_markdown()``. All models are frozen (immutable) to
prevent accidental mutation after construction.

Typical usage:

    from hybrid_doc_parser.models import ParserOutput, EnrichmentConfig
    config = EnrichmentConfig(enabled=True)
    # ... pass config to parse() ...
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

SCHEMA_VERSION: Final[str] = "1.0"
"""Current schema version for ParserOutput serialisation."""


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class ElementType(str, Enum):
    """Semantic type of a document block extracted by the MinerU engine.

    Inherits from ``str`` so members compare equal to their string values,
    e.g. ``ElementType.text == "text"`` is ``True``.
    """

    text = "text"
    heading = "heading"
    list_item = "list_item"
    image = "image"
    table = "table"
    equation = "equation"
    caption = "caption"
    header = "header"
    footer = "footer"
    page_number = "page_number"
    unknown = "unknown"


class Severity(str, Enum):
    """Severity of a verifier finding (disagreement / missing / extra element).

    Inherits from ``str`` so members compare equal to their string values,
    e.g. ``Severity.high == "high"`` is ``True``. Used by the standalone
    ``verify()`` advisory verifier to rank findings and to filter them via
    ``VerifierConfig.min_severity_to_report`` (precision-favoring).
    """

    low = "low"
    medium = "medium"
    high = "high"


# ---------------------------------------------------------------------------
# Configuration model
# ---------------------------------------------------------------------------


class EnrichmentConfig(BaseModel):
    """Configuration for the optional VLM enrichment pipeline.

    All fields have sensible defaults so callers can construct with zero
    arguments to get a safe, enrichment-disabled configuration.

    Attributes:
        enabled: Master switch; when ``False`` no VLM calls are made.
        image: Enrich image elements when ``enabled`` is ``True``.
        table: Enrich table elements when ``enabled`` is ``True``.
        equation: Enrich equation elements when ``enabled`` is ``True``.
        context_window: Number of surrounding paragraphs to include in the
            VLM prompt for context (0–20).
        max_context_tokens: Approximate token cap for context text (64–4096).
        vlm_backend: Which VLM backend to use; ``"openai_compatible"`` for
            any OpenAI-protocol server, ``"bedrock"`` for AWS Bedrock.
        parser: Which document parsing backend to use; ``"mineru"`` for the
            MinerU pipeline (default), ``"docling"`` for Docling (supports
            DOCX, HTML, and other non-PDF formats), or ``"auto"`` to select
            the backend per file at classify-time via MinerU's ``classify()``
            heuristic (scanned PDFs -> MinerU, text-layer PDFs -> Docling,
            images -> MinerU, DOCX/HTML -> Docling).
        do_ocr: Enable OCR during Docling PDF processing; default ``True``.
        table_mode: Docling TableFormer mode; ``"fast"`` (default) or
            ``"accurate"`` for higher fidelity at the cost of latency.
        do_table_structure: Enable Docling table structure recognition
            (TableFormer); default ``True``. Set to ``False`` for
            latency-sensitive workloads.
        docling_artifacts_path: Optional local path where Docling model
            artefacts are stored; ``None`` uses the Docling default cache.
    """

    model_config = ConfigDict(frozen=True)

    enabled: bool = False
    image: bool = True
    table: bool = True
    equation: bool = True
    context_window: int = Field(default=3, ge=0, le=20)
    max_context_tokens: int = Field(default=512, ge=64, le=4096)
    vlm_backend: Literal["openai_compatible", "bedrock"] = "openai_compatible"
    # NOTE: "auto" selects the backend per file at classify-time via MinerU's
    # classify() heuristic: "ocr" result -> "mineru", "txt" result -> "docling".
    # Images fall back to MinerU; DOCX/HTML fall back to Docling.
    parser: Literal["mineru", "docling", "auto"] = "mineru"

    # NOTE: Docling-specific pipeline controls; ignored when parser="mineru"
    do_ocr: bool = True
    table_mode: Literal["fast", "accurate"] = "fast"
    do_table_structure: bool = True
    docling_artifacts_path: str | None = None


class VerifierConfig(BaseModel):
    """Configuration for the standalone, advisory ``verify()`` second-opinion verifier.

    This model is deliberately SEPARATE from :class:`EnrichmentConfig`: the
    verifier is a distinct, edge-only capability (all VLM/network activity lives
    at the caller's edge, never inside ``parse()``). All fields have safe
    defaults so callers can construct with zero arguments and get a verifier
    that is OFF (``enabled=False``).

    Defaults are tuned to favor PRECISION over recall: ``min_severity_to_report``
    defaults to ``"high"`` so only high-confidence findings surface, and
    ``force_verify_all`` defaults to ``False`` so only quality-gate-flagged pages
    are verified.

    Attributes:
        enabled: Master switch; when ``False`` ``verify()`` performs no work.
        backend: Which verifier backend to dispatch; ``"bedrock"`` (v1
            primary), ``"openai_compatible"`` (local vllm/Ollama eval), or
            ``"fake"`` (CI / no-network canned verdicts).
        model: Bare on-demand model identifier (no inference-profile prefix).
        region: Cloud region for the backend (e.g. AWS region for Bedrock).
        force_verify_all: When ``True`` verify EVERY page, not just pages flagged
            ``quality_decision == "promote_to_vlm"``. Eval/calibration only —
            exists solely to measure the recall gap left by the quality gate.
        max_concurrency: Upper bound on in-flight verifier calls; a SEPARATE
            semaphore from ``parse_batch`` concurrency (Bedrock quotas are
            independent of local engine concurrency). Must be ``>= 1``.
        render_dpi: DPI for the full-page render fed to the verifier; the
            existing megapixel clamp in ``render_page`` guards huge pages.
            Bounded to a sane raster range (72–600).
        max_pages_per_doc: HARD per-document cap on verified pages; truncation
            is logged so a partially-verified document is never mistaken for
            "all clean". Must be ``>= 1``.
        timeout: Per-call timeout in seconds for the backend; must be ``> 0``.
        min_severity_to_report: Drop findings below this severity from the
            report; precision-favoring default of ``"high"``.
    """

    model_config = ConfigDict(frozen=True)

    enabled: bool = False
    backend: Literal["bedrock", "openai_compatible", "fake"] = "bedrock"
    model: str = ""
    region: str = ""
    force_verify_all: bool = False
    max_concurrency: int = Field(default=2, ge=1)
    render_dpi: int = Field(default=150, ge=72, le=600)
    max_pages_per_doc: int = Field(default=50, ge=1)
    timeout: float = Field(default=60.0, gt=0)
    min_severity_to_report: Literal["low", "medium", "high"] = "high"


# ---------------------------------------------------------------------------
# Record models
# ---------------------------------------------------------------------------


class ElementRecord(BaseModel):
    """A single semantic block extracted from a document page.

    Attributes:
        element_id: Stable UUID v5 keyed on
            ``file_sha256 + page_idx + sequential_block_index``.
        type: Semantic classification of the block.
        text: Raw extracted text or LaTeX string; never ``None``.
        description: LLM-generated plain-language description; empty string
            when the element has not been enriched.
        bbox: Bounding box ``[x0, y0, x1, y1]`` in PDF points with
            bottom-left origin; empty list when unavailable.
        page_idx: Zero-indexed page number this element belongs to.
        is_enriched: ``True`` after a successful VLM enrichment call.
        image_bytes: Raw PNG/JPEG bytes for image elements; ``None`` for all
            other element types and when enrichment is disabled.
    """

    # NOTE: ``ser_json_bytes`` / ``val_json_bytes`` = "base64" so ``image_bytes``
    # (binary PNG/JPEG, common on the Docling path) round-trips through JSON
    # serialisation. The Pydantic v2 default ("utf8") raises on binary bytes,
    # which previously caused every Docling-with-images ParserOutput to fail the
    # cache write ("invalid utf-8 sequence") and silently never cache.
    model_config = ConfigDict(
        frozen=True, ser_json_bytes="base64", val_json_bytes="base64"
    )

    element_id: str
    type: ElementType
    text: str
    description: str = ""
    bbox: list[float]
    page_idx: int
    is_enriched: bool = False
    image_bytes: bytes | None = None


class PageRecord(BaseModel):
    """Quality-gate decision and metadata for a single document page.

    Attributes:
        page_idx: Zero-indexed page number.
        quality_decision: ``"keep"`` when the page passes both gate layers;
            ``"promote_to_vlm"`` when at least one layer requests VLM review.
        element_count: Total number of ``ElementRecord`` objects on this page.
        vlm_used: ``True`` when at least one element on this page was enriched
            by a VLM call.
    """

    model_config = ConfigDict(frozen=True)

    page_idx: int
    quality_decision: Literal["keep", "promote_to_vlm"]
    element_count: int
    vlm_used: bool


class WarningRecord(BaseModel):
    """Non-fatal diagnostic message produced during parsing.

    Attributes:
        page_idx: Zero-indexed page that triggered the warning, or ``None``
            for document-level warnings (e.g. unsupported file type).
        code: Short machine-readable code identifying the warning category.
            Known codes: ``"unsupported_type"``, ``"unhandled_exception"``,
            ``"mineru_failed"``, ``"docling_failed"``, ``"vlm_failed"``,
            ``"quality_gate_error"``, ``"cache_write_error"``,
            ``"render_failed"``, ``"enrichment_not_supported"``,
            ``"image_too_large"``, ``"docling_error"``, ``"mineru_error"``,
            ``"verification_failed"`` (a per-page/run verifier failure;
            emitted by ``verify()`` when a page or the whole run fails),
            ``"verification_unsupported"`` (the verifier no-ops a DOCX/HTML
            input because no source image exists to compare against), and
            ``"verification_truncated"`` (the ``max_pages_per_doc`` cap was hit
            and some flagged pages were skipped — never mistake for "all
            clean").
        message: Human-readable description of the problem.
    """

    model_config = ConfigDict(frozen=True)

    page_idx: int | None = None
    code: str
    message: str


# ---------------------------------------------------------------------------
# Verifier report models (standalone ``verify()`` advisory output)
# ---------------------------------------------------------------------------


class Disagreement(BaseModel):
    """A verifier disagreement keyed to an existing MinerU element.

    Expresses that the verifier believes MinerU extracted an element
    incorrectly. Advisory only: it NEVER mutates ``ElementRecord.text`` and
    NEVER sets ``is_enriched``.

    Attributes:
        element_id: The ``ElementRecord.element_id`` this disagreement refers to.
        type: The element's semantic type (mirrors ``ElementRecord.type``).
        severity: Confidence-weighted severity of the disagreement.
        reason: Human-readable explanation of why the verifier disagrees.
        suggested_text: The verifier's proposed corrected text (advisory).
        vlm_confidence: The verifier's self-reported confidence, ``0.0``–``1.0``.
    """

    model_config = ConfigDict(frozen=True)

    element_id: str
    type: ElementType
    severity: Severity
    reason: str
    suggested_text: str
    vlm_confidence: float = Field(ge=0.0, le=1.0)


class MissingElement(BaseModel):
    """A verifier-detected MinerU false NEGATIVE (an element MinerU dropped).

    Has no ``element_id`` because the element does not exist in MinerU's output.

    Attributes:
        severity: Severity of the omission.
        reason: Human-readable explanation of what was missed.
        approx_location: Free-text approximate location on the page (no bbox).
    """

    model_config = ConfigDict(frozen=True)

    severity: Severity
    reason: str
    approx_location: str


class ExtraElement(BaseModel):
    """A verifier-detected MinerU false POSITIVE (an element MinerU invented).

    Shares the shape of :class:`MissingElement` and likewise has no
    ``element_id`` (the spurious element is described, not referenced).

    Attributes:
        severity: Severity of the spurious extraction.
        reason: Human-readable explanation of why it is considered spurious.
        approx_location: Free-text approximate location on the page (no bbox).
    """

    model_config = ConfigDict(frozen=True)

    severity: Severity
    reason: str
    approx_location: str


class PageVerification(BaseModel):
    """Per-page verifier verdict: disagreements plus missing/extra channels.

    Attributes:
        page_idx: Zero-indexed page number this verdict covers.
        disagreements: Findings keyed to existing MinerU elements.
        missing_elements: MinerU false negatives (no ``element_id``).
        extra_elements: MinerU false positives (no ``element_id``).
    """

    model_config = ConfigDict(frozen=True)

    page_idx: int
    disagreements: list[Disagreement] = Field(default_factory=list)
    missing_elements: list[MissingElement] = Field(default_factory=list)
    extra_elements: list[ExtraElement] = Field(default_factory=list)


class VerificationReport(BaseModel):
    """Top-level advisory output of a single ``verify()`` call.

    This report is a SEPARATE return value: it is NOT a field on
    ``ParserOutput`` and is NOT stored in the parse cache. It is advisory only
    and never mutates the parse output.

    Serialization nests the entire payload under a top-level ``verification``
    key to match the canonical report envelope verbatim (see ``spec.md``).
    ``model_dump()`` / ``model_dump_json()`` therefore both emit::

        {"verification": {"model_id": ..., "prompt_version": ..., "pages": [...],
                          "warnings": [...]}}

    Attributes:
        model_id: The verifier model identifier used for the run.
        prompt_version: The stable prompt-version string used for the run.
        pages: Per-page verdicts; empty when nothing was verified.
        warnings: Non-fatal diagnostics (reuses :class:`WarningRecord`),
            including ``verification_failed`` / ``verification_unsupported`` /
            ``verification_truncated`` codes.
    """

    model_config = ConfigDict(frozen=True)

    model_id: str
    prompt_version: str
    pages: list[PageVerification] = Field(default_factory=list)
    warnings: list[WarningRecord] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _unwrap_verification_envelope(cls, data: Any) -> Any:
        """Accept either the bare payload or the wrapped ``verification`` envelope.

        Serialization always emits ``{"verification": {...}}``; this validator
        makes deserialization symmetric by unwrapping that single-key envelope
        on input, while still accepting a bare payload dict.
        """
        if isinstance(data, dict) and set(data.keys()) == {"verification"}:
            return data["verification"]
        return data

    @model_serializer(mode="wrap")
    def _wrap_in_verification_envelope(self, handler: Any) -> dict[str, Any]:
        """Nest the serialized payload under the top-level ``verification`` key.

        ``mode="wrap"`` calls the default serializer (``handler``) to produce
        the inner dict, then wraps it so the output matches the canonical
        ``{"verification": {...}}`` envelope verbatim.
        """
        return {"verification": handler(self)}


# ---------------------------------------------------------------------------
# Top-level output model
# ---------------------------------------------------------------------------


class ParserOutput(BaseModel):
    """Complete structured output of a single ``parse()`` call.

    This is the primary data transfer object for the library. All fields are
    immutable after construction (``ConfigDict(frozen=True)``).

    Attributes:
        schema_version: Serialisation schema version; defaults to
            ``SCHEMA_VERSION`` (``"1.0"``).
        file_path: Absolute resolved path of the parsed file as a string.
        file_sha256: Full 64-character lowercase hex SHA-256 digest of the
            file bytes at parse time.
        page_count: Total number of pages in the document.
        pages: Per-page quality-gate metadata, one entry per page.
        elements: All extracted document elements in page order.
        warnings: Non-fatal diagnostics collected during parsing; an empty
            list indicates a clean parse.
        enrichment_config: The ``EnrichmentConfig`` used for this parse run.
    """

    model_config = ConfigDict(frozen=True)

    schema_version: str = SCHEMA_VERSION
    file_path: str
    file_sha256: str
    page_count: int
    pages: list[PageRecord]
    elements: list[ElementRecord]
    warnings: list[WarningRecord]
    enrichment_config: EnrichmentConfig
