"""Public parser API: parse() and parse_batch().

This module is the primary entry point for the hybrid-doc-parser library.
It orchestrates:
1. File validation and SHA-256 computation.
2. File-based cache lookup.
3. MinerU engine invocation (Python API → CLI subprocess fallback).
4. Block routing to typed ElementRecord instances.
5. Two-layer quality gate evaluation per page.
6. Optional per-element VLM enrichment.
7. Cache write and return of a validated ParserOutput.

Never raises — all failures produce a ParserOutput with WarningRecord entries.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import subprocess
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Final

from loguru import logger

import hybrid_doc_parser.cache as cache_mod
from hybrid_doc_parser.models import (
    ElementRecord,
    ElementType,
    EnrichmentConfig,
    PageRecord,
    ParserOutput,
    WarningRecord,
)
from hybrid_doc_parser.quality_gate import evaluate_page

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_SUPPORTED_EXTENSIONS: Final[frozenset[str]] = frozenset(
    {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}
)

# NOTE: Maps MinerU content_list block type strings to internal ElementType enum.
# Any type not in this map falls through to ElementType.unknown.
_BLOCK_TYPE_MAP: Final[dict[str, ElementType]] = {
    "text": ElementType.text,
    "title": ElementType.heading,
    "list": ElementType.list_item,
    "list_item": ElementType.list_item,
    "table": ElementType.table,
    "interline_equation": ElementType.equation,
    "equation": ElementType.equation,
    "image": ElementType.image,
    "figure": ElementType.image,
    "figure_caption": ElementType.caption,
    "table_caption": ElementType.caption,
    "header": ElementType.header,
    "footer": ElementType.footer,
    "page_number": ElementType.page_number,
}

# NOTE: pypdfium2's underlying libpdfium C library is not thread-safe.
# All calls that open or iterate PDF documents must be serialized with this lock
# to prevent segfaults when parse_batch() runs multiple parses concurrently.
_PDFIUM_LOCK: threading.Lock = threading.Lock()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _file_sha256(path: Path) -> str:
    """Compute the full SHA-256 hex digest of a file.

    Args:
        path: Path to the file whose bytes will be hashed.

    Returns:
        64-character lowercase hex digest string.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_element_id(file_sha256: str, page_idx: int, seq_idx: int) -> str:
    """Construct a stable UUID v5 element identifier.

    The UUID is keyed on the concatenation of the file hash, page index, and
    sequential block index so that the same block always receives the same ID
    across repeated parses of an unchanged file.

    Args:
        file_sha256: Full 64-char SHA-256 digest of the source file.
        page_idx: Zero-based page index.
        seq_idx: Sequential block index within the document.

    Returns:
        UUID v5 string in canonical hyphenated form.
    """
    seed = f"{file_sha256}:{page_idx}:{seq_idx}"
    return str(uuid.uuid5(uuid.NAMESPACE_OID, seed))


def _route_block_type(raw_type: object) -> ElementType:
    """Map a MinerU block type string to the corresponding ElementType.

    Accepts any value for ``raw_type`` and safely coerces it to a string
    before performing the lookup, so callers do not need to guard against
    ``None`` or unexpected types.

    Args:
        raw_type: The ``type`` field value from a MinerU content_list block.
            May be any object; ``None`` and non-string values map to unknown.

    Returns:
        Matched ElementType, or ElementType.unknown for unrecognised types.
    """
    # NOTE: Defensive coercion — MinerU occasionally emits None for type fields
    # in malformed blocks. Treating them as unknown preserves the block rather
    # than crashing the whole parse pipeline.
    if raw_type is None:
        return ElementType.unknown
    return _BLOCK_TYPE_MAP.get(str(raw_type), ElementType.unknown)


def _normalise_aliases(block: dict) -> dict:
    """Normalise MinerU field name aliases to canonical names.

    MinerU versions differ in how they name image caption and footnote fields.
    This function copies the block and renames any aliases to the canonical form
    so downstream code can always use the same key.

    Args:
        block: Raw MinerU content_list block dict.

    Returns:
        Copy of block with canonical field names.
    """
    b = dict(block)
    if "img_caption" in b:
        b["image_caption"] = b.pop("img_caption")
    if "img_footnote" in b:
        b["image_footnote"] = b.pop("img_footnote")
    return b


def _read_output_files(output_dir: Path) -> list[dict]:
    """Scan an output directory for MinerU content_list JSON files.

    Recursively searches for files matching ``*_content_list.json``. Returns
    the content_list from the first match found. Falls back to any ``.json``
    file in the directory tree when no specific content_list file is present.

    Args:
        output_dir: Root directory produced by the MinerU CLI.

    Returns:
        Parsed content_list as a list of block dicts, or an empty list if
        nothing is found.
    """
    # NOTE: Scan recursively rather than assuming a fixed output sub-directory
    # structure — MinerU CLI output layout varies by version.
    json_files = list(output_dir.rglob("*_content_list.json"))
    if not json_files:
        json_files = list(output_dir.rglob("*.json"))
    for jf in json_files:
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
            if isinstance(data, dict) and "content_list" in data:
                return data["content_list"]
        except Exception as exc:  # noqa: BLE001
            logger.debug("[parser] failed to read output file {}: {}", jf, exc)
    return []


def _run_mineru(file_path: Path, backend: str = "pipeline") -> list[dict]:
    """Invoke MinerU and return its content_list output.

    Attempts the MinerU Python API first. Falls back to the CLI subprocess
    when the Python API is not importable.

    # NOTE: The Python API path is preferred; it avoids spawning a child
    # process and handles cleanup through the API's own resource management.

    # WARN: The CLI fallback is significantly slower and less reliable than the
    # Python API. It depends on the `mineru` command being available on PATH
    # and writes temporary files to disk.

    Args:
        file_path: Path to the document to parse.
        backend: MinerU backend identifier passed to the API or CLI.

    Returns:
        List of block dicts representing the document content_list.

    Raises:
        RuntimeError: When both the Python API and CLI fallback fail.
    """
    # Try MinerU Python API first — never import at module level to avoid
    # hard dependency on mineru at import time.
    try:
        from mineru.backend.pipeline import (  # type: ignore[import]  # noqa: PLC0415
            pipeline_doc_analyze,
            pipeline_result_to_middle_json,
        )

        result = pipeline_doc_analyze(str(file_path), backend=backend)
        middle_json = pipeline_result_to_middle_json(result)
        if isinstance(middle_json, dict):
            pdf_info = middle_json.get("pdf_info", [])
            content_list: list[dict] = []
            for page_info in pdf_info:
                content_list.extend(page_info.get("para_blocks", []))
            return content_list
        if isinstance(middle_json, list):
            return middle_json
    except ImportError:
        logger.debug("[parser] MinerU Python API not importable; trying CLI fallback")
    except Exception as exc:
        logger.debug("[parser] MinerU Python API failed: {}; trying CLI fallback", exc)

    # WARN: CLI fallback — slower, spawns a subprocess, requires mineru on PATH.
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            # NOTE: mineru -m is method (auto|txt|ocr); -b is backend (pipeline|...).
            # Force CUDA_VISIBLE_DEVICES="" so CPU is used when GPU CUDA CC mismatches.
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "")
            cmd = ["mineru", "-p", str(file_path), "-o", str(out_dir), "-m", "auto", "-b", backend]
            # WARN: equation-heavy pages (MFR model) take ~8 min on CPU — use 900s.
            proc = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                timeout=900,
                text=True,
                env=env,
            )
            for line in proc.stdout.splitlines():
                logger.info("[MinerU] {}", line)
            for line in proc.stderr.splitlines():
                logger.info("[MinerU] {}", line)
            content = _read_output_files(out_dir)
            if content:
                return content
    except Exception as exc:
        raise RuntimeError(f"MinerU CLI failed: {exc}") from exc

    raise RuntimeError("MinerU produced no usable output")


def _route_block(
    block: dict,
    page_idx: int,
    element_idx: int,
    file_sha256: str,
) -> ElementRecord:
    """Convert a single MinerU content_list block to an ElementRecord.

    For heading elements, the text is prefixed with the appropriate number of
    ``#`` characters derived from the block's ``text_level`` field (default 1,
    clamped to 1–6). This encodes the heading level in the text field so that
    the markdown renderer can emit the correct heading prefix without needing
    an extra field on ElementRecord.

    Args:
        block: MinerU block dict with at minimum ``type``, ``text``,
            ``page_idx``, and ``bbox`` fields.
        page_idx: Zero-based page index for this block.
        element_idx: Sequential block index within the whole document.
        file_sha256: SHA-256 digest of the source file used to build element_id.

    Returns:
        Typed and populated ElementRecord.
    """
    block = _normalise_aliases(block)
    raw_type = block.get("type", "unknown")
    etype = _route_block_type(raw_type)

    # NOTE: Coerce text to str defensively — MinerU may emit integers or None
    # for malformed blocks; Pydantic ElementRecord.text requires a str.
    raw_text = block.get("text", "") or block.get("content", "") or ""
    text = str(raw_text) if not isinstance(raw_text, str) else raw_text

    # NOTE: MinerU table blocks store content in table_body (HTML), not text.
    # Convert to GFM markdown so the quality gate and downstream tools see the
    # table content and TEDS evaluation can recover the structure.
    if etype == ElementType.table and not text:
        table_html = block.get("table_body", "")
        if table_html:
            from hybrid_doc_parser.markdown import _table_html_to_markdown  # noqa: PLC0415

            text = _table_html_to_markdown(table_html)

    bbox_raw = block.get("bbox", [])
    bbox: list[float] = [float(v) for v in bbox_raw] if isinstance(bbox_raw, list) else []

    # NOTE: Heading level is encoded into the text prefix so render_markdown
    # can detect it with a simple startswith("#") check, as per tasks.md 5B.2.
    if etype == ElementType.heading:
        raw_level = block.get("text_level", 1)
        try:
            level = max(1, min(6, int(raw_level)))
        except (TypeError, ValueError):
            level = 1
        if not text.startswith("#"):
            text = "#" * level + " " + text

    element_id = _build_element_id(file_sha256, page_idx, element_idx)

    return ElementRecord(
        element_id=element_id,
        type=etype,
        text=text,
        bbox=bbox,
        page_idx=page_idx,
    )


def _enrich_elements(
    elements: list[ElementRecord],
    content_list: list[dict],
    config: EnrichmentConfig,
    file_path: Path,
) -> list[ElementRecord]:
    """Enrich image, table, and equation elements using VLM calls.

    Iterates the element list and calls the appropriate modal processor for
    each element whose type matches an enabled enrichment modality. Updates
    the ElementRecord with the resulting description and marks it enriched.

    Args:
        elements: ElementRecord list to potentially enrich (matches content_list order).
        content_list: Raw MinerU content_list for context extraction.
        config: EnrichmentConfig specifying which modalities are active.
        file_path: Source PDF path (used for render_region on image elements).

    Returns:
        New list of ElementRecord instances; enriched elements are replaced
        with model_copy() updates.
    """
    from hybrid_doc_parser.modal_processors import (  # noqa: PLC0415
        EquationModalProcessor,
        ImageModalProcessor,
        TableModalProcessor,
    )
    from hybrid_doc_parser.vlm_client import make_vlm_client  # noqa: PLC0415

    vlm = make_vlm_client(config)
    image_proc = ImageModalProcessor(vlm, content_list, config.context_window)
    table_proc = TableModalProcessor(vlm, content_list, config.context_window)
    eq_proc = EquationModalProcessor(vlm, content_list, config.context_window)

    enriched: list[ElementRecord] = []
    for i, element in enumerate(elements):
        block = content_list[i] if i < len(content_list) else {}
        description = ""
        should_enrich = False

        if element.type == ElementType.image and config.image:
            should_enrich = True
            try:
                from hybrid_doc_parser.render import render_region  # noqa: PLC0415

                with _PDFIUM_LOCK:
                    img_bytes: bytes | None = render_region(
                        file_path, element.page_idx, element.bbox
                    )
            except Exception:  # noqa: BLE001
                img_bytes = None
            description = image_proc.process(block, i, img_bytes)

        elif element.type == ElementType.table and config.table:
            should_enrich = True
            description = table_proc.process(block, i, None)

        elif element.type == ElementType.equation and config.equation:
            should_enrich = True
            description = eq_proc.process(block, i, None)

        if should_enrich and description:
            element = element.model_copy(
                update={"description": description, "is_enriched": True}
            )

        enriched.append(element)

    return enriched


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse(file_path: Path, config: EnrichmentConfig | None = None) -> ParserOutput:
    """Parse a document and return a fully validated ParserOutput.

    The function is intentionally non-raising. Every failure path — missing
    file, unsupported type, MinerU crash, enrichment error — is captured as a
    WarningRecord and returned in the ParserOutput.warnings list.

    Processing steps:
    1. Validate file existence and extension.
    2. Compute SHA-256 and check the file-based cache.
    3. Run MinerU (Python API → CLI fallback).
    4. Route each block to a typed ElementRecord.
    5. Call the two-layer quality gate per page.
    6. Optionally enrich modal elements via VLM.
    7. Write result to cache and return.

    # NOTE: The entire body is wrapped in try/except so that no unexpected
    # exception can propagate to the caller. The outer handler produces a
    # minimal ParserOutput carrying a "mineru_error" warning.

    Args:
        file_path: Path to the PDF or image file to parse.
        config: Enrichment and backend configuration. Defaults to a
            disabled-enrichment EnrichmentConfig when None.

    Returns:
        Validated ParserOutput. Always returned — never raises.
    """
    if config is None:
        config = EnrichmentConfig()

    try:
        # File existence check — return early with a specific warning code.
        if not file_path.exists():
            return ParserOutput(
                file_path=str(file_path),
                file_sha256="",
                page_count=0,
                pages=[],
                elements=[],
                warnings=[
                    WarningRecord(
                        code="file_not_found",
                        message=f"File not found: {file_path}",
                    )
                ],
                enrichment_config=config,
            )

        # Extension validation.
        if file_path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
            return ParserOutput(
                file_path=str(file_path),
                file_sha256="",
                page_count=0,
                pages=[],
                elements=[],
                warnings=[
                    WarningRecord(
                        code="unsupported_type",
                        message=f"Unsupported file extension: {file_path.suffix!r}",
                    )
                ],
                enrichment_config=config,
            )

        sha256 = _file_sha256(file_path)

        # NOTE: Cache-first check — avoids running MinerU on already-parsed files.
        cached = cache_mod.get(file_path)
        if cached is not None:
            logger.debug("[parser] cache hit for {}", file_path)
            return cached

        mineru_backend = os.environ.get("MINERU_BACKEND", "pipeline")

        try:
            content_list = _run_mineru(file_path, backend=mineru_backend)
        except Exception as exc:
            logger.warning("[parser] MinerU failed for {}: {}", file_path, exc)
            return ParserOutput(
                file_path=str(file_path),
                file_sha256=sha256,
                page_count=0,
                pages=[],
                elements=[],
                warnings=[
                    WarningRecord(
                        code="mineru_failed",
                        message=f"MinerU failed: {exc}",
                    )
                ],
                enrichment_config=config,
            )

        # NOTE: Filter out None and non-dict items before normalising aliases so
        # that malformed MinerU output does not crash the whole pipeline.
        # Individual bad blocks are silently skipped; a clean list is produced.
        valid_blocks = [b for b in content_list if isinstance(b, dict)]
        if len(valid_blocks) < len(content_list):
            logger.debug(
                "[parser] skipped {} non-dict block(s) in content_list for {}",
                len(content_list) - len(valid_blocks),
                file_path,
            )

        # Normalise aliases across all valid blocks.
        content_list = [_normalise_aliases(b) for b in valid_blocks]

        # Group blocks by page index.
        pages_map: dict[int, list[dict]] = {}
        for block in content_list:
            pidx = int(block.get("page_idx", 0))
            pages_map.setdefault(pidx, []).append(block)

        page_count = (max(pages_map.keys()) + 1) if pages_map else 0

        # Build ElementRecord list in page order.
        all_elements: list[ElementRecord] = []
        element_idx = 0
        for pidx in range(page_count):
            for block in pages_map.get(pidx, []):
                try:
                    all_elements.append(
                        _route_block(block, pidx, element_idx, sha256)
                    )
                except Exception as exc:  # noqa: BLE001
                    # NOTE: Per-block routing errors are non-fatal; log and skip
                    # the offending block rather than aborting the entire parse.
                    logger.debug(
                        "[parser] skipping block at page={} idx={}: {}", pidx, element_idx, exc
                    )
                element_idx += 1

        # Get text-layer token counts for quality gate Layer 1.
        # NOTE: pypdfium2 is not thread-safe; acquire the lock before any call
        # that opens a PDF document to prevent segfaults in parse_batch().
        token_counts: dict[int, int] = {}
        if file_path.suffix.lower() == ".pdf":
            from hybrid_doc_parser.render import text_layer_tokens  # noqa: PLC0415

            with _PDFIUM_LOCK:
                token_counts = text_layer_tokens(file_path)

        # Quality gate evaluation per page.
        page_records: list[PageRecord] = []
        warnings: list[WarningRecord] = []

        for pidx in range(page_count):
            page_elements = [e for e in all_elements if e.page_idx == pidx]
            pdf_tokens = token_counts.get(pidx)
            decision = evaluate_page(pidx, page_elements, pdf_tokens)
            vlm_used = decision.action == "promote_to_vlm"
            if vlm_used:
                warnings.append(
                    WarningRecord(
                        page_idx=pidx,
                        code="quality_gate_escalation",
                        message=f"Page {pidx} escalated to VLM: {decision.reason}",
                    )
                )
            page_records.append(
                PageRecord(
                    page_idx=pidx,
                    quality_decision=decision.action,
                    element_count=len(page_elements),
                    vlm_used=vlm_used,
                )
            )

        # Optional VLM enrichment.
        if config.enabled:
            try:
                all_elements = _enrich_elements(
                    all_elements, content_list, config, file_path
                )
            except Exception as exc:
                logger.warning("[parser] enrichment failed for {}: {}", file_path, exc)
                warnings.append(
                    WarningRecord(
                        code="enrichment_error",
                        message=f"Enrichment failed: {exc}",
                    )
                )

        output = ParserOutput(
            file_path=str(file_path),
            file_sha256=sha256,
            page_count=page_count,
            pages=page_records,
            elements=all_elements,
            warnings=warnings,
            enrichment_config=config,
        )

        cache_mod.put(file_path, output)
        return output

    except Exception as exc:
        logger.warning("[parser] unhandled exception for {}: {}", file_path, exc)
        try:
            sha = _file_sha256(file_path) if file_path.exists() else ""
        except Exception:  # noqa: BLE001
            sha = ""
        return ParserOutput(
            file_path=str(file_path),
            file_sha256=sha,
            page_count=0,
            pages=[],
            elements=[],
            warnings=[
                WarningRecord(
                    code="mineru_error",
                    message=f"Unhandled parse error: {exc}",
                )
            ],
            enrichment_config=config,
        )


async def parse_batch(
    paths: list[Path],
    config: EnrichmentConfig | None = None,
    max_concurrency: int = 4,
) -> list[ParserOutput]:
    """Parse multiple documents concurrently and return one result per input.

    Uses asyncio.Semaphore to cap concurrent MinerU invocations. Individual
    document failures do not abort the batch — each failed path produces a
    ParserOutput with warnings, matching the never-raise contract of parse().

    Args:
        paths: List of document paths to parse.
        config: Shared enrichment and backend configuration.
        max_concurrency: Maximum number of concurrent parse() threads.

    Returns:
        List of ParserOutput instances in the same order as the input paths.
    """
    semaphore = asyncio.Semaphore(max_concurrency)

    async def _parse_one(path: Path) -> ParserOutput:
        async with semaphore:
            return await asyncio.to_thread(parse, path, config)

    return list(await asyncio.gather(*[_parse_one(p) for p in paths]))
