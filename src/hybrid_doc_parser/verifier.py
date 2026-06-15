"""Standalone advisory verifier client layer.

This module defines the :class:`VerifierClient` protocol and its concrete
backends for the standalone, advisory ``verify()`` second-opinion verifier. It is
intentionally distinct from the deprecated enrichment ``VLMClient`` in
``vlm_client.py`` (Phase 4 removes that abstraction); only the pure helper
PATTERNS (``_robust_json_parse``, ``_detect_media_type``,
``_build_bedrock_request``) are reused.

Backends:
- :class:`BedrockVerifierClient`: AWS Bedrock via boto3 (v1 primary).
- :class:`OpenAICompatibleVerifierClient`: any OpenAI-compatible endpoint
  (local vllm/Ollama eval).
- :class:`FakeVerifierClient`: canned verdicts for CI and the doc-bench eval
  harness (no network, no SDK, no AWS spend).

The cloud SDKs (``boto3``, ``openai``) are imported LAZILY inside the client
methods so the core library stays cloud-SDK-free; install them via the optional
``verifier`` extra. Every client NEVER raises: on any failure it returns a
structured ``{"error": ...}`` result that ``verify()`` maps to a warning.

This module also holds the pure input-preparation helpers used by ``verify()``:
serializing a page's MinerU elements to text (no bbox), building the versioned
verifier prompt, and rendering the full source page.
"""

from __future__ import annotations

import asyncio
import base64
import json
import random
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from loguru import logger

from hybrid_doc_parser import render
from hybrid_doc_parser import verifier_cache
from hybrid_doc_parser.models import (
    Disagreement,
    ElementRecord,
    ElementType,
    ExtraElement,
    MissingElement,
    PageVerification,
    ParserOutput,
    Severity,
    VerificationReport,
    VerifierConfig,
    WarningRecord,
)

# The pypdfium2-backed renderer is not thread-safe; reuse parser.py's lock so
# verifier rendering serializes with parse_batch() rendering in the same process.
from hybrid_doc_parser.parser import _PDFIUM_LOCK

# Re-home the shared pure helpers as verifier-layer patterns. These are pure
# functions, not the deprecated enrichment client classes, so importing them
# keeps a single robust implementation without coupling to ``VLMClient``.
from hybrid_doc_parser.vlm_client import (
    _build_bedrock_request,
    _detect_media_type,
    _robust_json_parse,
)

__all__ = [
    "verify",
    "VerifierClient",
    "BedrockVerifierClient",
    "OpenAICompatibleVerifierClient",
    "FakeVerifierClient",
    "make_verifier_client",
    "PROMPT_VERSION",
    "build_verifier_prompt",
    "_serialize_page_elements",
    "_render_full_page",
    "_build_bedrock_request",
    "_detect_media_type",
    "_robust_json_parse",
]

# Stable prompt-version string. It is recorded in the report and is part of the
# verification cache key, so any change to the prompt below MUST bump this value
# to correctly invalidate cached verdicts.
PROMPT_VERSION = "v1"


@runtime_checkable
class VerifierClient(Protocol):
    """Protocol for advisory verifier backend clients.

    Distinct from the deprecated enrichment ``VLMClient``. Implementations
    compare a rendered page image against MinerU's serialized elements and
    return a structured per-element-verdict dict. They must NEVER raise —
    returning ``{"error": ...}`` on any failure instead.
    """

    def verify_page(
        self,
        image_bytes: bytes | None,
        prompt: str,
        config: VerifierConfig,
    ) -> dict[str, Any]:
        """Verify a single page and return a structured per-element verdict.

        Args:
            image_bytes: Raw full-page render bytes, or ``None`` for text-only.
            prompt: The verifier prompt carrying the serialized MinerU elements.
            config: The active :class:`VerifierConfig` (model, region, timeout).

        Returns:
            A parsed per-element-verdict dict. Never raises; returns
            ``{"error": ...}`` on any failure.
        """
        ...


class BedrockVerifierClient:
    """Verifier client for AWS Bedrock (v1 primary backend).

    Creates a fresh ``boto3`` client per call to respect IAM role rotation and
    uses BARE on-demand model IDs only (no inference-profile prefix, which is
    SCP-blocked). ``boto3`` is imported lazily inside :meth:`verify_page`.
    Never raises; returns ``{"error": ...}`` on any failure.
    """

    __slots__ = ()

    def verify_page(
        self,
        image_bytes: bytes | None,
        prompt: str,
        config: VerifierConfig,
    ) -> dict[str, Any]:
        """Verify a page via Bedrock ``invoke_model``.

        Args:
            image_bytes: Raw full-page render bytes, or ``None`` for text-only.
            prompt: The verifier prompt carrying the serialized MinerU elements.
            config: The active :class:`VerifierConfig`; ``model`` must be a bare
                on-demand model ID and ``timeout`` bounds the call.

        Returns:
            Parsed per-element-verdict dict. Returns ``{"error": ...}`` on failure.
        """
        try:
            import boto3  # noqa: PLC0415
            from botocore.config import Config as BotoConfig  # noqa: PLC0415

            body = _build_bedrock_request(image_bytes, prompt)
            # NOTE: per-call client honours IAM role rotation; timeout bounds the
            # call so a hung Bedrock endpoint degrades to {"error": ...}.
            boto_config = BotoConfig(
                read_timeout=config.timeout,
                connect_timeout=config.timeout,
                retries={"max_attempts": 0},
            )
            runtime = boto3.client(
                "bedrock-runtime",
                region_name=config.region,
                config=boto_config,
            )
            resp = runtime.invoke_model(
                modelId=config.model,
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json",
            )
            payload = json.loads(resp["body"].read())
            raw = "".join(
                block.get("text", "")
                for block in payload.get("content", [])
                if block.get("type") == "text"
            )
            return _robust_json_parse(raw)
        except Exception as exc:
            logger.warning("[verifier] Bedrock call failed: {}", exc)
            return {"error": str(exc)}


class OpenAICompatibleVerifierClient:
    """Verifier client for any OpenAI-compatible API endpoint.

    Enables local vllm/Ollama eval behind the same protocol. ``openai`` is
    imported lazily inside :meth:`verify_page`. Never raises; returns
    ``{"error": ...}`` on any failure.
    """

    __slots__ = ()

    def verify_page(
        self,
        image_bytes: bytes | None,
        prompt: str,
        config: VerifierConfig,
    ) -> dict[str, Any]:
        """Verify a page via an OpenAI-compatible chat completion.

        Args:
            image_bytes: Raw full-page render bytes, or ``None`` for text-only.
            prompt: The verifier prompt carrying the serialized MinerU elements.
            config: The active :class:`VerifierConfig`; ``model`` and ``timeout``
                are honoured. The base URL / API key come from the standard
                ``OPENAI_BASE_URL`` / ``OPENAI_API_KEY`` environment variables.

        Returns:
            Parsed per-element-verdict dict. Returns ``{"error": ...}`` on failure.
        """
        try:
            import os  # noqa: PLC0415

            import openai  # noqa: PLC0415

            base_url = os.environ.get("OPENAI_BASE_URL", "")
            api_key = os.environ.get("OPENAI_API_KEY", "none")

            content: list[dict[str, Any]] | str
            if image_bytes is not None:
                media_type = _detect_media_type(image_bytes)
                b64 = base64.b64encode(image_bytes).decode("utf-8")
                content = [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{media_type};base64,{b64}"},
                    },
                    {"type": "text", "text": prompt},
                ]
            else:
                content = prompt

            messages: list[dict[str, Any]] = [{"role": "user", "content": content}]
            client = openai.OpenAI(
                base_url=base_url, api_key=api_key, timeout=config.timeout
            )
            response = client.chat.completions.create(
                model=config.model,
                messages=messages,  # type: ignore[arg-type]
                max_tokens=8192,
                temperature=0.0,
            )
            raw = response.choices[0].message.content or ""
            return _robust_json_parse(raw)
        except Exception as exc:
            logger.warning("[verifier] OpenAI-compatible call failed: {}", exc)
            return {"error": str(exc)}


class FakeVerifierClient:
    """In-memory verifier client returning canned verdicts (no network, no SDK).

    For CI and the doc-bench eval harness so verification can be exercised with
    no AWS spend. Canned verdicts are injected per ``page_idx`` (and optionally
    keyed further per ``element_id`` within a verdict's payload). The prompt is
    expected to contain ``"page_idx: <n>"`` so the right canned verdict is
    selected; when no canned verdict matches the page, ``default_verdict`` is
    returned.
    """

    __slots__ = ("_verdicts", "_default_verdict", "calls")

    def __init__(
        self,
        verdicts: dict[int, dict[str, Any]] | None = None,
        default_verdict: dict[str, Any] | None = None,
    ) -> None:
        """Initialise the fake with canned per-page verdicts.

        Args:
            verdicts: Mapping of ``page_idx`` to the verdict dict to return for
                that page. The verdict payload may itself key findings by
                ``element_id``.
            default_verdict: Verdict returned when no page-specific verdict
                matches; defaults to an empty per-element verdict.
        """
        self._verdicts: dict[int, dict[str, Any]] = dict(verdicts or {})
        self._default_verdict: dict[str, Any] = (
            default_verdict
            if default_verdict is not None
            else {"disagreements": [], "missing_elements": [], "extra_elements": []}
        )
        # Records (image_present, prompt, config) for each call, aiding test
        # assertions without any network or SDK involvement.
        self.calls: list[tuple[bool, str, VerifierConfig]] = []

    def verify_page(
        self,
        image_bytes: bytes | None,
        prompt: str,
        config: VerifierConfig,
    ) -> dict[str, Any]:
        """Return the canned verdict matching the page referenced in the prompt.

        Args:
            image_bytes: Ignored beyond recording its presence.
            prompt: Used to recover the ``page_idx`` (``"page_idx: <n>"``).
            config: Recorded for test assertions.

        Returns:
            The canned per-element verdict for the page, or ``default_verdict``.
            Never raises.
        """
        self.calls.append((image_bytes is not None, prompt, config))
        page_idx = self._extract_page_idx(prompt)
        if page_idx is not None and page_idx in self._verdicts:
            return self._verdicts[page_idx]
        return self._default_verdict

    @staticmethod
    def _extract_page_idx(prompt: str) -> int | None:
        """Recover a ``page_idx`` from a ``"page_idx: <n>"`` marker in the prompt.

        Args:
            prompt: The verifier prompt text.

        Returns:
            The parsed page index, or ``None`` when no marker is present.
        """
        marker = "page_idx:"
        idx = prompt.find(marker)
        if idx == -1:
            return None
        tail = prompt[idx + len(marker):].lstrip()
        digits = ""
        for ch in tail:
            if ch.isdigit():
                digits += ch
            else:
                break
        return int(digits) if digits else None


def make_verifier_client(config: VerifierConfig) -> VerifierClient:
    """Return the :class:`VerifierClient` backend selected by ``config.backend``.

    Args:
        config: The active :class:`VerifierConfig` whose ``backend`` selects the
            implementation (``"bedrock"`` | ``"openai_compatible"`` | ``"fake"``).

    Returns:
        A concrete :class:`VerifierClient`. ``"fake"`` returns a
        :class:`FakeVerifierClient`; ``"openai_compatible"`` returns an
        :class:`OpenAICompatibleVerifierClient`; any other value (including
        ``"bedrock"``) returns a :class:`BedrockVerifierClient`.
    """
    if config.backend == "fake":
        return FakeVerifierClient()
    if config.backend == "openai_compatible":
        return OpenAICompatibleVerifierClient()
    return BedrockVerifierClient()


# ---------------------------------------------------------------------------
# Input preparation: page serialization, prompt construction, full-page render
# ---------------------------------------------------------------------------


def _serialize_page_elements(
    elements: Iterable[ElementRecord], page_idx: int
) -> str:
    """Serialize one page's MinerU elements as text for the verifier prompt.

    Emits one line per element on ``page_idx`` carrying the element's
    ``element_id``, its ordinal index within the page, its ``type``, and its
    ``text``. Raw bbox coordinates are NEVER included: they waste tokens and
    invite coordinate-system hallucination, and disagreements are referenced by
    ``element_id``/index instead.

    Args:
        elements: All extracted elements (any page); filtered to ``page_idx``.
        page_idx: Zero-indexed page whose elements are serialized.

    Returns:
        A newline-joined block of ``[i] element_id=<id> type=<type>:
        <text>`` lines, or an explicit ``"(no elements)"`` marker when the page
        has none.
    """
    lines: list[str] = []
    for ordinal, element in enumerate(
        el for el in elements if el.page_idx == page_idx
    ):
        text = element.text.replace("\n", " ").strip()
        lines.append(
            f"[{ordinal}] element_id={element.element_id} "
            f"type={element.type.value}: {text}"
        )
    if not lines:
        return "(no elements)"
    return "\n".join(lines)


def build_verifier_prompt(
    elements: Iterable[ElementRecord], page_idx: int
) -> str:
    """Build the versioned verifier prompt for a single page.

    The prompt asks the model for a PER-ELEMENT verdict (not "list only
    disagreements", which pressures the model to invent one), and provides
    separate channels for elements MinerU MISSED (false negatives) and EXTRA
    elements MinerU invented (false positives). Disagreements must be referenced
    by ``element_id`` / ordinal index. Numbered visual overlays for grounding
    are deferred and intentionally not requested here.

    Args:
        elements: All extracted elements (any page); the page's subset is
            serialized into the prompt.
        page_idx: Zero-indexed page being verified. Emitted as a
            ``"page_idx: <n>"`` marker so fakes and caches can key on it.

    Returns:
        The full prompt string carrying ``PROMPT_VERSION``, the serialized
        elements, and the per-element-verdict instructions.
    """
    serialized = _serialize_page_elements(elements, page_idx)
    return (
        f"prompt_version: {PROMPT_VERSION}\n"
        f"page_idx: {page_idx}\n\n"
        "You are a meticulous second-opinion verifier. You are given an image "
        "of a single source document page and a list of the elements that an "
        "automated extractor (MinerU) produced for that page. Compare the "
        "extracted elements against what is actually visible in the image.\n\n"
        "Give a PER-ELEMENT verdict for EVERY listed element. Do NOT list only "
        "disagreements and do NOT invent a disagreement when an element is "
        "correct. Reference every disagreement by the element's element_id and "
        "ordinal index exactly as given below.\n\n"
        "Also report, in separate channels:\n"
        "- missing_elements: content visible in the image that MinerU did NOT "
        "extract (false negatives). These have no element_id; describe the "
        "approximate location instead.\n"
        "- extra_elements: extracted elements that do NOT correspond to "
        "anything in the image (false positives). These have no element_id; "
        "describe the approximate location instead.\n\n"
        "Respond with a single JSON object containing the keys "
        '"disagreements", "missing_elements", and "extra_elements". Each '
        "disagreement carries element_id, type, severity (low|medium|high), "
        "reason, suggested_text, and vlm_confidence (0.0-1.0). Each "
        "missing/extra element carries severity, reason, and approx_location.\n\n"
        "MinerU extracted elements:\n"
        f"{serialized}\n"
    )


def _render_full_page(
    file_path: Path, page_idx: int, config: VerifierConfig
) -> bytes:
    """Render the full source page to PNG bytes for the verifier.

    Renders the FULL page (no per-element crops) via :func:`render.render_page`
    at ``config.render_dpi``, under :data:`_PDFIUM_LOCK` because the underlying
    pypdfium2 library is not thread-safe. The megapixel clamp inside
    ``render_page`` bounds adversarially large pages, so no extra clamping is
    needed here.

    Args:
        file_path: Path to the source PDF/image being verified.
        page_idx: Zero-indexed page to render.
        config: The active :class:`VerifierConfig`; ``render_dpi`` caps the
            render resolution.

    Returns:
        PNG-encoded bytes of the full rendered page.
    """
    with _PDFIUM_LOCK:
        return render.render_page(file_path, page_idx, dpi=config.render_dpi)


# ---------------------------------------------------------------------------
# Standalone orchestration: verify()
# ---------------------------------------------------------------------------

# Source extensions for which a page image can be rendered and compared. Mirrors
# parser.py's _SUPPORTED_EXTENSIONS (PDF + raster image inputs). DOCX/HTML are
# intentionally absent: there is no source page image to compare against, so
# verify() no-ops them with a ``verification_unsupported`` warning.
_VERIFIABLE_EXTENSIONS: frozenset[str] = frozenset(
    {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}
)

# Ordered severity ranking used to apply ``min_severity_to_report`` (a higher
# index is a more severe finding). Findings below the configured floor are
# dropped to favor precision over recall.
_SEVERITY_RANK: dict[str, int] = {"low": 0, "medium": 1, "high": 2}

# ---------------------------------------------------------------------------
# Throttling-aware bounded retry
# ---------------------------------------------------------------------------

# Maximum number of ATTEMPTS (initial call + retries) per page on a throttling
# error. Bedrock throttles hard and its quotas are independent of local engine
# concurrency, so a small bounded retry with jitter smooths transient 429s
# without unbounded waiting. On exhaustion the page degrades to a
# ``verification_failed`` warning (never raises).
_MAX_VERIFY_ATTEMPTS = 3

# Base backoff (seconds) for the exponential schedule; attempt N waits roughly
# ``_RETRY_BASE_DELAY * 2 ** (N - 1)`` plus jitter.
_RETRY_BASE_DELAY = 0.5

# Substrings (lower-cased) that mark a throttling / rate-limit / transient error
# surfaced by a client's ``{"error": ...}`` message (or a raised exception). Only
# these are retried; any other failure degrades immediately.
_THROTTLING_MARKERS: tuple[str, ...] = (
    "throttl",  # botocore ThrottlingException
    "too many requests",
    "rate limit",
    "ratelimit",
    "429",
    "serviceunavailable",
    "service unavailable",
    "503",
    "slow down",
    "request rate",
)


def _is_throttling_error(message: str) -> bool:
    """Return whether an error ``message`` looks like a throttling/transient error.

    Clients NEVER raise — they return ``{"error": <str>}`` — so throttling is
    detected from the error text rather than an exception type. Only throttling /
    rate-limit / transient (429/503) markers are retried; all other failures
    degrade immediately to a ``verification_failed`` warning.

    Args:
        message: The error string from a client result (or a raised exception).

    Returns:
        ``True`` when the message carries a recognised throttling marker.
    """
    lowered = message.lower()
    return any(marker in lowered for marker in _THROTTLING_MARKERS)


def _retry_delay(attempt: int, rng: random.Random) -> float:
    """Compute the jittered backoff delay (seconds) before retry ``attempt``.

    Uses exponential backoff with full jitter so concurrent verifier calls do
    not retry in lockstep and re-trigger throttling. The ``rng`` is injectable so
    the jitter is deterministic-testable.

    Args:
        attempt: 1-based index of the retry about to be waited for (the first
            retry is ``attempt == 1``).
        rng: Random source for the jitter (injected in tests for determinism).

    Returns:
        A non-negative delay in seconds.
    """
    ceiling = _RETRY_BASE_DELAY * (2 ** max(attempt - 1, 0))
    return rng.uniform(0.0, ceiling)


def _meets_min_severity(severity: str, floor: str) -> bool:
    """Return whether ``severity`` is at or above the ``floor`` threshold.

    Args:
        severity: The finding's severity string (``"low"``/``"medium"``/``"high"``).
        floor: The configured ``min_severity_to_report`` threshold.

    Returns:
        ``True`` when the finding should be kept; ``False`` when it is below the
        floor (or carries an unrecognised severity, which is dropped to favor
        precision).
    """
    sev_rank = _SEVERITY_RANK.get(severity)
    floor_rank = _SEVERITY_RANK.get(floor, _SEVERITY_RANK["high"])
    if sev_rank is None:
        return False
    return sev_rank >= floor_rank


def _select_pages(
    parser_output: ParserOutput, config: VerifierConfig
) -> list[int]:
    """Select the page indices to verify per the trigger policy.

    By default only pages the quality gate flagged
    (``PageRecord.quality_decision == "promote_to_vlm"``) are selected.
    ``config.force_verify_all`` overrides this to every page (eval/calibration
    only). Selection reads from ``parser_output.pages`` (the per-page
    quality-gate records), so the page set always comes from the parse output
    rather than re-deriving page structure here.

    Args:
        parser_output: The completed parse output carrying per-page records.
        config: The active verifier config; ``force_verify_all`` widens selection.

    Returns:
        Sorted, de-duplicated zero-indexed page numbers to verify.
    """
    if config.force_verify_all:
        pages = [page.page_idx for page in parser_output.pages]
    else:
        pages = [
            page.page_idx
            for page in parser_output.pages
            if page.quality_decision == "promote_to_vlm"
        ]
    return sorted(set(pages))


def _build_element_type_index(
    parser_output: ParserOutput,
) -> dict[str, ElementType]:
    """Index element types by ``element_id`` for disagreement type backfill.

    The verifier prompt asks the model to echo the element ``type``, but the
    authoritative type is MinerU's. This index lets ``verify()`` prefer the
    parse output's type when constructing a :class:`Disagreement`.

    Args:
        parser_output: The completed parse output.

    Returns:
        Mapping of ``element_id`` to its MinerU :class:`ElementType`.
    """
    return {el.element_id: el.type for el in parser_output.elements}


def _coerce_disagreements(
    raw_items: Any,
    type_index: dict[str, ElementType],
    config: VerifierConfig,
) -> list[Disagreement]:
    """Coerce raw model disagreement dicts into filtered :class:`Disagreement`.

    Items that fail validation are skipped (never raise); items below
    ``config.min_severity_to_report`` are dropped to favor precision.

    Args:
        raw_items: The model's ``disagreements`` payload (expected list of dicts).
        type_index: ``element_id`` -> MinerU type, used to backfill ``type``.
        config: The active verifier config supplying the severity floor.

    Returns:
        The validated, severity-filtered disagreements.
    """
    out: list[Disagreement] = []
    if not isinstance(raw_items, list):
        return out
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        severity = str(item.get("severity", "")).lower()
        if not _meets_min_severity(severity, config.min_severity_to_report):
            continue
        element_id = item.get("element_id")
        if not isinstance(element_id, str) or not element_id:
            continue
        # Prefer MinerU's authoritative type; fall back to the model's claim.
        etype = type_index.get(element_id)
        if etype is None:
            try:
                etype = ElementType(str(item.get("type", "unknown")))
            except ValueError:
                etype = ElementType.unknown
        try:
            out.append(
                Disagreement(
                    element_id=element_id,
                    type=etype,
                    severity=Severity(severity),
                    reason=str(item.get("reason", "")),
                    suggested_text=str(item.get("suggested_text", "")),
                    vlm_confidence=float(item.get("vlm_confidence", 0.0)),
                )
            )
        except (ValueError, TypeError):
            # Malformed finding (e.g. confidence out of [0,1]); drop it.
            continue
    return out


def _coerce_location_items(
    raw_items: Any,
    config: VerifierConfig,
    model_cls: type[MissingElement] | type[ExtraElement],
) -> list[Any]:
    """Coerce raw ``missing_elements`` / ``extra_elements`` dicts to models.

    Both channels share the same shape (severity / reason / approx_location, no
    ``element_id``). Items that fail validation are skipped; items below the
    severity floor are dropped.

    Args:
        raw_items: The model payload for the channel (expected list of dicts).
        config: The active verifier config supplying the severity floor.
        model_cls: :class:`MissingElement` or :class:`ExtraElement`.

    Returns:
        The validated, severity-filtered channel items.
    """
    out: list[Any] = []
    if not isinstance(raw_items, list):
        return out
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        severity = str(item.get("severity", "")).lower()
        if not _meets_min_severity(severity, config.min_severity_to_report):
            continue
        try:
            out.append(
                model_cls(
                    severity=Severity(severity),
                    reason=str(item.get("reason", "")),
                    approx_location=str(item.get("approx_location", "")),
                )
            )
        except (ValueError, TypeError):
            continue
    return out


def _verify_single_page(
    parser_output: ParserOutput,
    file_path: Path,
    page_idx: int,
    config: VerifierConfig,
    client: VerifierClient,
    type_index: dict[str, ElementType],
) -> tuple[PageVerification | None, WarningRecord | None]:
    """Verify one page: render, prompt, call the client, filter the verdict.

    Mirrors the never-raises discipline of ``parse()``: any failure (a client
    ``{"error": ...}`` result or an exception) degrades to a
    ``verification_failed`` warning and yields NO verdict for the page rather
    than propagating.

    The separate verification cache (``verifier_cache``) is consulted BEFORE any
    render/client work and written only AFTER a successful verdict. It is keyed
    on the 4-tuple ``(file_sha256, page_idx, model, PROMPT_VERSION)``; a changed
    model or prompt version produces a new key and so misses the cached verdict.
    Cache read/write never raises (it degrades silently).

    Args:
        parser_output: The completed parse output (elements are read, never
            mutated).
        file_path: Path to the source PDF/image.
        page_idx: Zero-indexed page to verify.
        config: The active verifier config.
        client: The backend :class:`VerifierClient`.
        type_index: ``element_id`` -> MinerU type for disagreement backfill.

    Returns:
        ``(PageVerification, None)`` on success, or ``(None, WarningRecord)`` on
        a page-level failure. Never raises.
    """
    try:
        # Verification-cache read BEFORE any render/client work. On a hit, reuse
        # the cached page verdict and skip rendering and the (network) client
        # call entirely.
        cached = verifier_cache.get(
            parser_output.file_sha256, page_idx, config.model, PROMPT_VERSION
        )
        if cached is not None:
            return cached, None

        image_bytes = _render_full_page(file_path, page_idx, config)
        prompt = build_verifier_prompt(parser_output.elements, page_idx)
        result = client.verify_page(image_bytes, prompt, config)

        if not isinstance(result, dict) or "error" in result:
            message = (
                str(result.get("error"))
                if isinstance(result, dict)
                else "verifier client returned a non-dict result"
            )
            return None, WarningRecord(
                page_idx=page_idx,
                code="verification_failed",
                message=f"Verifier failed on page {page_idx}: {message}",
            )

        disagreements = _coerce_disagreements(
            result.get("disagreements"), type_index, config
        )
        missing = _coerce_location_items(
            result.get("missing_elements"), config, MissingElement
        )
        extra = _coerce_location_items(
            result.get("extra_elements"), config, ExtraElement
        )
        verdict = PageVerification(
            page_idx=page_idx,
            disagreements=disagreements,
            missing_elements=missing,
            extra_elements=extra,
        )
        # Write AFTER a successful verdict only. Failed / verification_failed
        # pages return above and are never cached, so a transient failure is not
        # pinned as a verdict. The write never raises.
        verifier_cache.put(
            parser_output.file_sha256,
            page_idx,
            config.model,
            PROMPT_VERSION,
            verdict,
        )
        return verdict, None
    except Exception as exc:  # noqa: BLE001
        logger.warning("[verifier] page {} failed: {}", page_idx, exc)
        return None, WarningRecord(
            page_idx=page_idx,
            code="verification_failed",
            message=f"Verifier raised on page {page_idx}: {exc}",
        )


def _verify_single_page_with_retry(
    parser_output: ParserOutput,
    file_path: Path,
    page_idx: int,
    config: VerifierConfig,
    client: VerifierClient,
    type_index: dict[str, ElementType],
    *,
    sleeper: Callable[[float], None] = time.sleep,
    rng: random.Random | None = None,
) -> tuple[PageVerification | None, WarningRecord | None]:
    """Verify one page with bounded retry + jitter on throttling errors.

    Wraps :func:`_verify_single_page`. A page that degrades to a
    ``verification_failed`` warning whose message indicates THROTTLING (a 429 /
    rate-limit / transient error) is retried up to :data:`_MAX_VERIFY_ATTEMPTS`
    times, sleeping a jittered exponential backoff between attempts. Any
    non-throttling failure is returned immediately. On retry exhaustion the page
    still degrades to the ``verification_failed`` warning — this NEVER raises.

    ``sleeper`` and ``rng`` are injectable so tests run instantly (mock sleep)
    and the jitter is deterministic.

    Args:
        parser_output: The completed parse output (read only).
        file_path: Path to the source PDF/image.
        page_idx: Zero-indexed page to verify.
        config: The active verifier config.
        client: The backend :class:`VerifierClient`.
        type_index: ``element_id`` -> MinerU type for disagreement backfill.
        sleeper: Callable invoked with the backoff delay; injected in tests.
        rng: Random source for jitter; defaults to a fresh, page-seeded
            ``random.Random`` for reproducibility.

    Returns:
        ``(PageVerification, None)`` on success, or ``(None, WarningRecord)`` on
        a page-level failure (including retry exhaustion). Never raises.
    """
    # Page-seed the default rng so jitter is reproducible per page across runs
    # without coupling concurrent pages to the same sequence.
    rng = rng if rng is not None else random.Random(page_idx)

    verdict: PageVerification | None = None
    warning: WarningRecord | None = None
    for attempt in range(1, _MAX_VERIFY_ATTEMPTS + 1):
        verdict, warning = _verify_single_page(
            parser_output, file_path, page_idx, config, client, type_index
        )
        # Success, or a non-throttling failure: return without further retries.
        if warning is None or not _is_throttling_error(warning.message):
            return verdict, warning
        # Throttling failure with attempts remaining: backoff with jitter.
        if attempt < _MAX_VERIFY_ATTEMPTS:
            delay = _retry_delay(attempt, rng)
            logger.warning(
                "[verifier] page {} throttled (attempt {}/{}); retrying in "
                "{:.2f}s",
                page_idx,
                attempt,
                _MAX_VERIFY_ATTEMPTS,
                delay,
            )
            sleeper(delay)
    # Retry exhausted: degrade to the last (throttling) warning, never raise.
    logger.warning(
        "[verifier] page {} throttled after {} attempts; degrading to "
        "verification_failed",
        page_idx,
        _MAX_VERIFY_ATTEMPTS,
    )
    return verdict, warning


async def _verify_pages_async(
    parser_output: ParserOutput,
    file_path: Path,
    selected: list[int],
    config: VerifierConfig,
    client: VerifierClient,
    type_index: dict[str, ElementType],
    *,
    sleeper: Callable[[float], None] = time.sleep,
) -> list[tuple[PageVerification | None, WarningRecord | None]]:
    """Verify the selected pages concurrently, bounded by a SEPARATE semaphore.

    A dedicated ``asyncio.Semaphore(config.max_concurrency)`` bounds the number
    of in-flight per-page verifier calls. It is INTENTIONALLY separate from
    ``parse_batch``'s semaphore: Bedrock quotas are independent of local engine
    concurrency and Bedrock throttles hard, so the verifier owns its own
    concurrency budget. Each page's (blocking) render + client call runs in a
    worker thread via :func:`asyncio.to_thread` so the event loop is never
    blocked, and is wrapped in the bounded throttling retry.

    Args:
        parser_output: The completed parse output (read only).
        file_path: Path to the source PDF/image.
        selected: The page indices to verify (already cost-guard-capped).
        config: The active verifier config; ``max_concurrency`` sizes the
            semaphore.
        client: The backend :class:`VerifierClient`.
        type_index: ``element_id`` -> MinerU type for disagreement backfill.
        sleeper: Callable invoked with the retry backoff delay; injected in
            tests so retries do not actually wait.

    Returns:
        One ``(verdict, warning)`` pair per selected page, in ``selected`` order.
    """
    semaphore = asyncio.Semaphore(config.max_concurrency)

    async def _verify_one(
        page_idx: int,
    ) -> tuple[PageVerification | None, WarningRecord | None]:
        async with semaphore:
            return await asyncio.to_thread(
                _verify_single_page_with_retry,
                parser_output,
                file_path,
                page_idx,
                config,
                client,
                type_index,
                sleeper=sleeper,
            )

    return list(await asyncio.gather(*[_verify_one(p) for p in selected]))


def verify(
    parser_output: ParserOutput,
    file_path: Path,
    config: VerifierConfig,
    *,
    sleeper: Callable[[float], None] = time.sleep,
) -> VerificationReport:
    """Run the standalone advisory verifier over a completed parse output.

    This is the edge-only second-opinion verifier. It is invoked by the caller
    AFTER ``parse()`` and is NEVER called from within ``parse()`` /
    ``parse_batch()``, preserving those functions' local/deterministic/
    never-network/cacheable invariants. The returned :class:`VerificationReport`
    is a SEPARATE value: it is never attached to ``ParserOutput`` and never
    written to the parse cache.

    Behavior:
    - Trigger: by default only pages flagged
      ``PageRecord.quality_decision == "promote_to_vlm"`` are verified;
      ``config.force_verify_all`` overrides to every page (eval/calibration).
    - Formats: PDF and raster image inputs only. A DOCX/HTML input no-ops and
      emits a ``verification_unsupported`` warning (no source image exists).
    - Cost guard: ``config.max_pages_per_doc`` is a HARD per-document cap; pages
      beyond it are skipped, logged explicitly, and flagged with a
      ``verification_truncated`` warning so truncation is never mistaken for
      "all clean".
    - Concurrency: per-page calls are bounded by a SEPARATE
      ``asyncio.Semaphore(config.max_concurrency)`` (independent of
      ``parse_batch`` concurrency; Bedrock quotas are independent of local engine
      concurrency).
    - Resilience: throttling errors are retried with jittered exponential
      backoff up to a bounded number of attempts; on exhaustion the page
      degrades to a ``verification_failed`` warning rather than raising.
    - Advisory + never-raises: this NEVER mutates ``ElementRecord.text``, NEVER
      sets ``is_enriched``, and NEVER touches ``ParserOutput``. A page-level
      failure degrades to a ``verification_failed`` warning (no verdict for that
      page); a whole-run failure returns an empty report. Mirrors ``parse()``'s
      last-resort try/except so ``verify()`` never raises.

    Args:
        parser_output: The completed parse output to second-opinion. Read only;
            its page count and input structure come from here.
        file_path: Path to the source PDF/image that produced ``parser_output``.
        config: The active :class:`VerifierConfig` (backend, model, region,
            triggers, cost guard, severity floor, concurrency).
        sleeper: Callable invoked with the throttling-retry backoff delay;
            injected in tests so retries do not actually wait. Defaults to
            :func:`time.sleep`.

    Returns:
        A :class:`VerificationReport` with ``model_id`` / ``prompt_version``
        populated where known, per-page verdicts, and any warnings. Never raises.
    """
    model_id = config.model
    try:
        suffix = file_path.suffix.lower()
        if suffix not in _VERIFIABLE_EXTENSIONS:
            # DOCX/HTML (or any non-renderable input): no-op with a warning.
            logger.info(
                "[verifier] unsupported input type {!r} for {}; no-op",
                suffix,
                file_path,
            )
            return VerificationReport(
                model_id=model_id,
                prompt_version=PROMPT_VERSION,
                pages=[],
                warnings=[
                    WarningRecord(
                        code="verification_unsupported",
                        message=(
                            f"Verification unsupported for {suffix!r} input: no "
                            "source page image to compare against."
                        ),
                    )
                ],
            )

        selected = _select_pages(parser_output, config)
        warnings: list[WarningRecord] = []

        # HARD cost guard: cap the number of pages verified per document and
        # record exactly which pages were skipped so truncation is never mistaken
        # for "all clean".
        if len(selected) > config.max_pages_per_doc:
            kept = selected[: config.max_pages_per_doc]
            skipped = selected[config.max_pages_per_doc:]
            logger.warning(
                "[verifier] max_pages_per_doc={} reached for {}; verifying {}, "
                "skipping pages {}",
                config.max_pages_per_doc,
                file_path,
                kept,
                skipped,
            )
            warnings.append(
                WarningRecord(
                    code="verification_truncated",
                    message=(
                        f"max_pages_per_doc={config.max_pages_per_doc} reached; "
                        f"verified pages {kept}, skipped pages {skipped}."
                    ),
                )
            )
            selected = kept

        client = make_verifier_client(config)
        type_index = _build_element_type_index(parser_output)

        # Per-page verification runs concurrently behind a SEPARATE semaphore
        # (config.max_concurrency), each call wrapped in the bounded throttling
        # retry. Results come back in ``selected`` order.
        results = asyncio.run(
            _verify_pages_async(
                parser_output,
                file_path,
                selected,
                config,
                client,
                type_index,
                sleeper=sleeper,
            )
        )

        pages: list[PageVerification] = []
        for verdict, warning in results:
            if verdict is not None:
                pages.append(verdict)
            if warning is not None:
                warnings.append(warning)

        return VerificationReport(
            model_id=model_id,
            prompt_version=PROMPT_VERSION,
            pages=pages,
            warnings=warnings,
        )
    except Exception as exc:  # noqa: BLE001
        # Last-resort never-raise net: whole-run failure -> empty report with the
        # model id / prompt version populated where known.
        logger.warning("[verifier] unhandled exception for {}: {}", file_path, exc)
        return VerificationReport(
            model_id=model_id,
            prompt_version=PROMPT_VERSION,
            pages=[],
            warnings=[
                WarningRecord(
                    code="verification_failed",
                    message=f"Unhandled verification error: {exc}",
                )
            ],
        )
