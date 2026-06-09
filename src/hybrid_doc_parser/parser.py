"""Public parser API: parse() and parse_batch().

This module is the primary entry point for the hybrid-doc-parser library.
It orchestrates:
1. File validation and SHA-256 computation.
2. File-based cache lookup.
3. MinerU or Docling engine invocation based on EnrichmentConfig.parser.
4. Block routing to typed ElementRecord instances.
5. Two-layer quality gate evaluation per page.
6. Optional per-element VLM enrichment.
7. Cache write and return of a validated ParserOutput.

Supported engines:
- ``"mineru"``: MinerU 3.x pipeline (PDFs, images). Default.
- ``"docling"``: Docling 2.x (DOCX, HTML, and all MinerU-supported formats).

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
from typing import Any, Final

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

# NOTE: Maps Docling label strings to internal ElementType; unknown labels fall
# through to ElementType.unknown.
_DOCLING_LABEL_MAP: Final[dict[str, ElementType]] = {
    "paragraph": ElementType.text,
    "section_header": ElementType.heading,
    "formula": ElementType.equation,
    "list_item": ElementType.list_item,
}

_DOCLING_EXTENSIONS: Final[frozenset[str]] = frozenset(
    {".docx", ".doc", ".html", ".htm", ".xhtml"}
)

# NOTE: pypdfium2's underlying libpdfium C library is not thread-safe.
# All calls that open or iterate PDF documents must be serialized with this lock
# to prevent segfaults when parse_batch() runs multiple parses concurrently.
_PDFIUM_LOCK: threading.Lock = threading.Lock()

# Module-level Docling DocumentConverter cache keyed on pipeline-option tuple.
# Populated lazily on first use; shared across all parse() calls in the process.
_DOCLING_CONVERTER_CACHE: dict[tuple, Any] = {}
_DOCLING_CONVERTER_CACHE_LOCK: threading.Lock = threading.Lock()


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


def _run_mineru_inprocess(file_path: Path, backend: str = "pipeline") -> list[dict]:
    """Run MinerU in-process via ``mineru.cli.common.do_parse`` and return content_list.

    This is the fast path: unlike the CLI subprocess fallback, it keeps the
    heavy detection/OCR/table/formula models resident in this process across
    calls (MinerU's pipeline ``ModelSingleton`` caches them), so model
    initialisation is paid once per process rather than once per document.

    The device (CPU/GPU) is chosen by MinerU's own ``get_device()``, which
    honours ``MINERU_DEVICE_MODE`` and ``CUDA_VISIBLE_DEVICES`` — set
    ``CUDA_VISIBLE_DEVICES=""`` to force CPU.

    Args:
        file_path: Path to the PDF or image to parse.
        backend: MinerU backend identifier (e.g. ``pipeline``).

    Returns:
        The MinerU content_list (list of block dicts), or an empty list when
        no content was produced.

    Raises:
        ImportError: When the MinerU Python API is not importable.
        Exception: Propagated from MinerU; the caller falls back to the CLI.
    """
    from mineru.cli.common import do_parse, read_fn  # noqa: PLC0415

    # read_fn returns PDF bytes for PDFs and rasterised PDF bytes for images.
    pdf_bytes = read_fn(file_path)
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        # Only the content_list dump is needed; disable the other artefacts to
        # save I/O. parse_method="auto" mirrors the CLI's `-m auto`.
        do_parse(
            output_dir=str(out_dir),
            pdf_file_names=[file_path.stem],
            pdf_bytes_list=[pdf_bytes],
            p_lang_list=["en"],
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
        return _read_output_files(out_dir)


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
    # Preferred path: drive MinerU IN-PROCESS via its public CLI helper
    # (mineru.cli.common.do_parse). This avoids spawning a fresh `mineru`
    # subprocess per document — which costs ~20-30s of service spin-up plus a
    # full model re-load every time. do_parse uses pipeline ModelSingleton, so
    # the heavy models load on the first document and are reused for every
    # subsequent document in this process.
    #
    # NOTE: imported lazily so `mineru` is not a hard import-time dependency.
    try:
        content = _run_mineru_inprocess(file_path, backend=backend)
        if content:
            return content
        logger.debug("[parser] in-process MinerU returned no content; trying CLI fallback")
    except ImportError:
        logger.debug("[parser] MinerU Python API not importable; trying CLI fallback")
    except Exception as exc:  # noqa: BLE001
        logger.debug("[parser] in-process MinerU failed: {}; trying CLI fallback", exc)

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


def _resolve_docling_ref(ref_str: str, doc_dict: dict) -> tuple[str, dict] | None:
    """Resolve a Docling ``$ref`` pointer to its target block dict.

    Docling's ``export_to_dict()`` uses JSON Pointer-style ``$ref`` strings of
    the form ``"#/<block_type>/<index>"`` (e.g. ``"#/texts/0"``,
    ``"#/pictures/1"``, ``"#/tables/2"``). This helper splits the ref string
    and returns the referenced block dict along with its type.

    Args:
        ref_str: A ``$ref`` string such as ``"#/texts/0"``.
        doc_dict: The full Docling export dict containing top-level keys
            ``"texts"``, ``"pictures"``, ``"tables"``, etc.

    Returns:
        A ``(block_type, block_dict)`` tuple when the ref is valid, or
        ``None`` when the ref is malformed or out of range (with a warning
        logged for observability).
    """
    result: tuple[str, dict] | None = None
    try:
        parts = ref_str.split("/")
        if len(parts) < 3:  # noqa: PLR2004
            raise ValueError(f"ref has fewer than 3 parts: {ref_str!r}")
        block_type = parts[1]
        index = int(parts[2])
        result = (block_type, doc_dict[block_type][index])
    except (ValueError, KeyError, IndexError):
        logger.warning("[docling] malformed or out-of-range $ref: {}", ref_str)
    return result


def _route_docling_block(
    block: dict,
    block_type: str,
    page_idx: int,
    element_idx: int,
    file_sha256: str,
) -> tuple[ElementRecord, list[WarningRecord]]:
    """Convert a single Docling block dict to an ElementRecord.

    Handles all three Docling block categories:

    - ``"texts"``: Maps label to ElementType via ``_DOCLING_LABEL_MAP``;
      extracts text from ``"orig"`` or ``"text"`` field.
    - ``"pictures"``: Decodes the ``data:<mime>;base64,<payload>`` URI
      in-memory; applies a 10 MB size cap (stores ``None`` and emits a
      ``WarningRecord`` when exceeded).
    - ``"tables"``: Serialises caption + cell data as JSON text.

    Page index is derived from ``block["prov"][0]["page_no"] - 1`` when
    present and valid; falls back to the caller-supplied ``page_idx`` with
    a debug log when ``prov`` is missing, empty, or contains ``page_no == 0``.

    Bounding boxes are extracted from ``block["bbox"]`` or
    ``prov[0]["bbox"]``; an empty list is returned and is valid (common for
    DOCX and HTML documents where no layout coordinates exist).

    Args:
        block: Docling block dict from the export dict.
        block_type: One of ``"texts"``, ``"pictures"``, ``"tables"``.
        page_idx: Fallback zero-based page index used when ``prov`` is absent
            or malformed.
        element_idx: Sequential block index within the document (used for
            element ID construction and warning messages).
        file_sha256: SHA-256 digest of the source file for element ID generation.

    Returns:
        A tuple of:
        - ``ElementRecord``: The normalised element.
        - ``list[WarningRecord]``: Any warnings emitted for this block (e.g.
          ``"image_too_large"``). Empty for text and table blocks.
    """
    # NOTE: _normalise_aliases is MinerU-specific and must NOT be called here.
    import base64  # noqa: PLC0415

    extra_warnings: list[WarningRecord] = []

    # Page index extraction with prov guard.
    prov = block.get("prov")
    if not prov:
        logger.debug(
            "[docling] malformed prov for element {}: {}; using fallback page_idx",
            element_idx,
            prov,
        )
    else:
        raw_page_no = prov[0].get("page_no", 0) if isinstance(prov[0], dict) else 0
        if raw_page_no == 0:
            logger.debug(
                "[docling] malformed prov for element {}: {}; using fallback page_idx",
                element_idx,
                prov,
            )
        else:
            page_idx = raw_page_no - 1

    # Block type routing.
    etype: ElementType
    text: str
    image_bytes: bytes | None = None

    if block_type == "texts":
        etype = _DOCLING_LABEL_MAP.get(block.get("label", ""), ElementType.unknown)
        text = str(block.get("orig", "") or block.get("text", ""))

    elif block_type == "pictures":
        etype = ElementType.image
        text = ""
        uri = block.get("image", {}).get("uri", "")
        parts = uri.split(",", maxsplit=1)
        base64_str = parts[1] if len(parts) == 2 else parts[0]  # noqa: PLR2004
        try:
            decoded_bytes = base64.b64decode(base64_str)
            if len(decoded_bytes) > 10 * 1024 * 1024:
                image_bytes = None
                extra_warnings.append(
                    WarningRecord(
                        code="image_too_large",
                        message=f"Picture block {element_idx} exceeds 10 MB cap; bytes discarded",
                    )
                )
            else:
                image_bytes = decoded_bytes
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[docling] base64 decode failed for element {}: {}", element_idx, exc
            )
            image_bytes = None

    else:
        # tables (and any unexpected block_type falls here)
        etype = ElementType.table
        text = block.get("caption", "") + "\n" + json.dumps(block.get("data", []))

    # Bounding box extraction — empty list is valid for DOCX/HTML blocks.
    # NOTE: Docling prov bbox is a dict {l, t, r, b} with BOTTOMLEFT coord origin,
    # not a list. Normalise to [l, b, r, t] (xmin, ymin, xmax, ymax).
    bbox_raw = block.get("bbox") or (block.get("prov") or [{}])[0].get("bbox")
    if isinstance(bbox_raw, dict):
        bbox = [
            float(bbox_raw.get("l", 0.0)),
            float(bbox_raw.get("b", 0.0)),
            float(bbox_raw.get("r", 0.0)),
            float(bbox_raw.get("t", 0.0)),
        ]
    elif isinstance(bbox_raw, list) and bbox_raw:
        bbox = [float(v) for v in bbox_raw]
    else:
        bbox = []

    element_id = _build_element_id(file_sha256, page_idx, element_idx)

    return (
        ElementRecord(
            element_id=element_id,
            type=etype,
            text=text,
            bbox=bbox,
            page_idx=page_idx,
            image_bytes=image_bytes,
        ),
        extra_warnings,
    )


def _get_docling_converter(config: EnrichmentConfig) -> Any:
    """Return a cached DocumentConverter instance for the given config.

    Uses double-checked locking to ensure that model weights are loaded at most
    once per unique ``(table_mode, do_ocr, do_table_structure)`` configuration
    tuple, even under concurrent ``parse_batch()`` calls.

    The pattern mirrors ``_get_converter`` from
    ``references/RAG-Anything/raganything/parser.py``: snapshot the cache dict
    outside the lock (fast path), acquire the lock, re-check, then construct
    and store if still missing.

    All ``PdfPipelineOptions`` attribute assignments are guarded with
    ``hasattr`` because Docling minor versions may add or remove attributes
    without notice.

    Args:
        config: EnrichmentConfig whose Docling-specific fields drive
            ``PdfPipelineOptions`` construction.

    Returns:
        A ``DocumentConverter`` instance (possibly newly created or retrieved
        from the module-level cache).
    """
    key = (config.table_mode, config.do_ocr, config.do_table_structure)

    # Fast path — avoid lock acquisition when already cached.
    if key in _DOCLING_CONVERTER_CACHE:
        return _DOCLING_CONVERTER_CACHE[key]

    # WARN: model load is expensive; the lock prevents duplicate initialisation
    # under parse_batch() concurrency.
    with _DOCLING_CONVERTER_CACHE_LOCK:
        if key in _DOCLING_CONVERTER_CACHE:
            return _DOCLING_CONVERTER_CACHE[key]

        from docling.document_converter import DocumentConverter  # noqa: PLC0415

        try:
            from docling.datamodel.pipeline_options import PdfPipelineOptions  # noqa: PLC0415
            from docling.datamodel.base_models import InputFormat  # noqa: PLC0415
            from docling.document_converter import PdfFormatOption  # noqa: PLC0415

            pipeline_options = PdfPipelineOptions()
            if hasattr(pipeline_options, "do_ocr"):
                pipeline_options.do_ocr = config.do_ocr
            if hasattr(pipeline_options, "do_table_structure"):
                pipeline_options.do_table_structure = config.do_table_structure
            if hasattr(pipeline_options, "table_structure_options") and hasattr(
                pipeline_options.table_structure_options, "mode"
            ):
                try:
                    from docling.datamodel.pipeline_options import TableFormerMode  # noqa: PLC0415

                    pipeline_options.table_structure_options.mode = (
                        TableFormerMode.FAST
                        if config.table_mode == "fast"
                        else TableFormerMode.ACCURATE
                    )
                except ImportError:
                    pass
            if hasattr(pipeline_options, "generate_picture_images"):
                pipeline_options.generate_picture_images = True
            if hasattr(pipeline_options, "images_scale"):
                pipeline_options.images_scale = 2.0
            if (
                hasattr(pipeline_options, "artifacts_path")
                and config.docling_artifacts_path is not None
            ):
                pipeline_options.artifacts_path = config.docling_artifacts_path

            converter = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
            )
        except (ImportError, TypeError):
            converter = DocumentConverter()

        _DOCLING_CONVERTER_CACHE[key] = converter

    return _DOCLING_CONVERTER_CACHE[key]


def _run_docling(file_path: Path, config: EnrichmentConfig) -> list[ElementRecord]:
    """Invoke Docling and return a list of pre-routed ElementRecord objects.

    Lazy-imports ``docling.document_converter.DocumentConverter`` at call time
    so that Docling model weights are not loaded when the MinerU backend is
    used. Raises ``ImportError`` with an actionable install message when
    Docling is not installed.

    # WARN: Docling's TableFormer (table_structure) is the most expensive
    # operation; set do_table_structure=False for latency-sensitive workloads.

    Processing steps:
    1. Obtain a cached converter via ``_get_docling_converter``.
    2. Convert the file and export to a dict.
    3. Walk ``body["children"]`` recursively via ``_resolve_docling_ref``.
    4. Route each leaf block to an ``ElementRecord`` via
       ``_route_docling_block``; collect warnings.
    5. Return the list of ElementRecord objects.

    Args:
        file_path: Path to the document to parse (DOCX, HTML, PDF, etc.).
        config: EnrichmentConfig with Docling-specific pipeline controls.

    Returns:
        List of ElementRecord objects in document order.

    Raises:
        ImportError: When Docling is not installed.
        RuntimeError: When Docling conversion or export raises an exception.
    """
    try:
        from docling.document_converter import DocumentConverter  # noqa: F401, PLC0415
    except ImportError as exc:
        raise ImportError(
            "Docling is not installed. Run: pip install 'hybrid-doc-parser[docling]'"
            " or uv add 'hybrid-doc-parser[docling]'"
        ) from exc

    converter = _get_docling_converter(config)

    try:
        result = converter.convert(str(file_path))
    except Exception as exc:
        raise RuntimeError(f"Docling convert failed: {exc}") from exc

    try:
        doc_dict = result.document.export_to_dict()
    except Exception as exc:
        raise RuntimeError(f"Docling export failed: {exc}") from exc

    body = doc_dict.get("body", {})
    children = body.get("children", [])

    # NOTE: Walk the body children recursively. Groups and body container nodes
    # have their own children but produce no element; only "texts", "pictures",
    # and "tables" leaf nodes are routed to ElementRecord.
    elements: list[ElementRecord] = []
    all_warnings: list[WarningRecord] = []
    element_idx = 0

    # NOTE: Default page_idx is 0; will be overridden by prov data per block.
    default_page_idx = 0
    sha256 = _file_sha256(file_path)

    def _walk(refs: list) -> None:
        """Recursively walk $ref children and route leaf blocks."""
        nonlocal element_idx
        for child in refs:
            # NOTE: Docling export_to_dict() emits children as {"$ref": "#/type/N"}
            # dicts, not bare strings. Accept both forms for robustness.
            if isinstance(child, dict):
                ref = child.get("$ref", "")
            elif isinstance(child, str):
                ref = child
            else:
                continue
            if not ref:
                continue
            resolved = _resolve_docling_ref(ref, doc_dict)
            if resolved is None:
                continue
            block_type, block = resolved
            # Skip container node types that only carry children.
            if block_type in {"groups", "body"}:
                sub_children = block.get("children", [])
                if sub_children:
                    _walk(sub_children)
                continue
            if block_type not in {"texts", "pictures", "tables"}:
                continue
            try:
                record, warnings = _route_docling_block(
                    block=block,
                    block_type=block_type,
                    page_idx=default_page_idx,
                    element_idx=element_idx,
                    file_sha256=sha256,
                )
                elements.append(record)
                all_warnings.extend(warnings)
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "[docling] skipping block type={} idx={}: {}", block_type, element_idx, exc
                )
            element_idx += 1

    _walk(children)

    # NOTE: Store warnings on elements is not possible since ElementRecord is
    # frozen. Warnings will be collected by the caller via a side channel if
    # needed. For now, log image_too_large warnings from _route_docling_block.
    for w in all_warnings:
        logger.debug("[docling] block warning: code={} msg={}", w.code, w.message)

    return elements


def _accepted_extensions(config: EnrichmentConfig) -> frozenset[str]:
    """Return the set of accepted file extensions for the given parser config.

    When ``config.parser == "docling"``, extends the base set with Docling's
    supported non-PDF formats. The module-level ``_SUPPORTED_EXTENSIONS``
    constant is never mutated.

    Args:
        config: EnrichmentConfig whose ``parser`` field controls the extension
            set.

    Returns:
        Frozenset of lowercase file extension strings (including the leading
        dot).
    """
    if config.parser == "docling":
        return _SUPPORTED_EXTENSIONS | _DOCLING_EXTENSIONS
    return _SUPPORTED_EXTENSIONS


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
    file, unsupported type, MinerU or Docling crash, enrichment error — is
    captured as a WarningRecord and returned in the ParserOutput.warnings list.

    Supported backends (controlled by ``config.parser``):

    - ``"mineru"`` (default): MinerU 3.x pipeline; accepts PDFs and images.
    - ``"docling"``: Docling 2.x; accepts DOCX, HTML, and all MinerU formats.

    Processing steps:
    1. Validate file existence and extension.
    2. Compute SHA-256 and check the file-based cache.
    3. Dispatch to MinerU or Docling based on ``config.parser``.
    4. Route each block to a typed ElementRecord.
    5. Call the two-layer quality gate per page.
    6. Optionally enrich modal elements via VLM (MinerU path only in v1).
    7. Write result to cache and return.

    # NOTE: The entire body is wrapped in try/except so that no unexpected
    # exception can propagate to the caller. The outer handler produces a
    # minimal ParserOutput carrying a backend-specific error warning code
    # (``"docling_error"`` or ``"mineru_error"``).

    Args:
        file_path: Path to the document file to parse.
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

        # Extension validation — uses the backend-aware accepted set.
        if file_path.suffix.lower() not in _accepted_extensions(config):
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

        # NOTE: Cache-first check — avoids running the engine on already-parsed files.
        cached = cache_mod.get(file_path)
        if cached is not None:
            logger.debug("[parser] cache hit for {}", file_path)
            return cached

        mineru_backend = os.environ.get("MINERU_BACKEND", "pipeline")

        # Backend dispatch: Docling or MinerU path.
        all_elements: list[ElementRecord]
        warnings: list[WarningRecord] = []
        content_list: list[dict] = []

        if config.parser == "docling":
            try:
                all_elements = _run_docling(file_path, config)
            except Exception as exc:
                logger.warning("[parser] Docling failed for {}: {}", file_path, exc)
                return ParserOutput(
                    file_path=str(file_path),
                    file_sha256=sha256,
                    page_count=0,
                    pages=[],
                    elements=[],
                    warnings=[
                        WarningRecord(
                            code="docling_failed",
                            message=f"Docling failed: {exc}",
                        )
                    ],
                    enrichment_config=config,
                )
        else:
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

            page_count_from_blocks = (max(pages_map.keys()) + 1) if pages_map else 0

            # Build ElementRecord list in page order.
            all_elements = []
            element_idx = 0
            for pidx in range(page_count_from_blocks):
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

        # Determine page count and page map from elements.
        if all_elements:
            page_count = max(e.page_idx for e in all_elements) + 1
        else:
            page_count = 0

        # Get text-layer token counts for quality gate Layer 1.
        # NOTE: pypdfium2 is not thread-safe; acquire the lock before any call
        # that opens a PDF document to prevent segfaults in parse_batch().
        token_counts: dict[int, int] = {}
        is_non_pdf = file_path.suffix.lower() != ".pdf"
        if not is_non_pdf:
            from hybrid_doc_parser.render import text_layer_tokens  # noqa: PLC0415

            with _PDFIUM_LOCK:
                token_counts = text_layer_tokens(file_path)

        # Non-PDF Layer 1 skip: log once at file level for observability.
        if is_non_pdf:
            logger.debug(
                "[quality_gate] skipping Layer 1 for non-PDF input: {}", file_path
            )

        # Quality gate evaluation per page.
        page_records: list[PageRecord] = []

        for pidx in range(page_count):
            page_elements = [e for e in all_elements if e.page_idx == pidx]

            # Layer 1: use None for non-PDF inputs (already logged above).
            if is_non_pdf:
                pdf_tokens = None
            else:
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

        # Enrichment guard: Docling backend does not support VLM enrichment in v1.
        if config.enabled and config.parser == "docling":
            warnings.append(
                WarningRecord(
                    code="enrichment_not_supported",
                    message="VLM enrichment is not yet supported for parser='docling'",
                )
            )
        elif config.enabled:
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
        error_code = "docling_error" if config.parser == "docling" else "mineru_error"
        return ParserOutput(
            file_path=str(file_path),
            file_sha256=sha,
            page_count=0,
            pages=[],
            elements=[],
            warnings=[
                WarningRecord(
                    code=error_code,
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
