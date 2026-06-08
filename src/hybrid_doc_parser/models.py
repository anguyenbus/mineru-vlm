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
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field

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
    """

    model_config = ConfigDict(frozen=True)

    enabled: bool = False
    image: bool = True
    table: bool = True
    equation: bool = True
    context_window: int = Field(default=3, ge=0, le=20)
    max_context_tokens: int = Field(default=512, ge=64, le=4096)
    vlm_backend: Literal["openai_compatible", "bedrock"] = "openai_compatible"


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

    model_config = ConfigDict(frozen=True)

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
            ``"mineru_failed"``, ``"vlm_failed"``, ``"quality_gate_error"``,
            ``"cache_write_error"``, ``"render_failed"``.
        message: Human-readable description of the problem.
    """

    model_config = ConfigDict(frozen=True)

    page_idx: int | None = None
    code: str
    message: str


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
