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
from hybrid_doc_parser.confidence import extract_confidence
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

_DOCLING_EXTENSIONS: Final[frozenset[str]] = frozenset({".docx", ".doc", ".html", ".htm", ".xhtml"})

# NOTE: Maps PaddleOCR PP-StructureV3 ``block_label`` strings to internal
# ElementType. Verified labels emitted by PP-DocLayout (the layout model behind
# PP-StructureV3); any label not present falls through to ElementType.unknown.
_PADDLE_LABEL_MAP: Final[dict[str, ElementType]] = {
    "text": ElementType.text,
    "abstract": ElementType.text,
    "content": ElementType.text,
    "reference": ElementType.text,
    "aside_text": ElementType.text,
    "algorithm": ElementType.text,
    "doc_title": ElementType.heading,
    "paragraph_title": ElementType.heading,
    "title": ElementType.heading,
    "table": ElementType.table,
    "table_title": ElementType.caption,
    "figure_title": ElementType.caption,
    "chart_title": ElementType.caption,
    "table_caption": ElementType.caption,
    "figure_caption": ElementType.caption,
    "image": ElementType.image,
    "figure": ElementType.image,
    "chart": ElementType.image,
    "seal": ElementType.image,
    "formula": ElementType.equation,
    "formula_number": ElementType.equation,
    "header": ElementType.header,
    "header_image": ElementType.header,
    "footer": ElementType.footer,
    "footer_image": ElementType.footer,
    "footnote": ElementType.footer,
    "vision_footnote": ElementType.footer,
    "number": ElementType.page_number,
    "page_number": ElementType.page_number,
}

# NOTE: pypdfium2's underlying libpdfium C library is not thread-safe.
# All calls that open or iterate PDF documents must be serialized with this lock
# to prevent segfaults when parse_batch() runs multiple parses concurrently.
_PDFIUM_LOCK: threading.Lock = threading.Lock()

# Module-level Docling DocumentConverter cache keyed on pipeline-option tuple.
# Populated lazily on first use; shared across all parse() calls in the process.
_DOCLING_CONVERTER_CACHE: dict[tuple, Any] = {}
_DOCLING_CONVERTER_CACHE_LOCK: threading.Lock = threading.Lock()

# Module-level PP-StructureV3 pipeline, built lazily on first PaddleOCR parse and
# shared across all parse() calls (model load is expensive; reuse the instance).
_PADDLE_PIPELINE: Any = None
_PADDLE_PIPELINE_LOCK: threading.Lock = threading.Lock()

# Default chunk size for MinerU batch inference when MINERU_BATCH_SIZE is unset
# or non-parseable. Must remain finite/bounded (never 0, negative, or unbounded).
_DEFAULT_MINERU_BATCH_SIZE: Final[int] = 8

# Default number of concurrent in-process MinerU do_parse windows allowed across
# all callers. 1 == serialise GPU inference (single shared device). Tunable via
# MINERU_MAX_INFLIGHT for multi-GPU hosts.
_DEFAULT_MINERU_MAX_INFLIGHT: Final[int] = 1

# Process-wide gate serialising do_parse inference across concurrent parse_batch
# calls. Built lazily (so MINERU_MAX_INFLIGHT is read at first use, not import).
_MINERU_INFERENCE_SEMA: threading.BoundedSemaphore | None = None
_MINERU_INFERENCE_SEMA_LOCK: threading.Lock = threading.Lock()

# Warning codes that mark a per-document parse as failed (vs a clean parse or a
# soft signal like quality_gate_escalation). Used to count batch failures for
# the parse_batch() summary line.
_BATCH_FAILURE_CODES: Final[frozenset[str]] = frozenset(
    {
        "mineru_failed",
        "mineru_error",
        "docling_failed",
        "docling_error",
        "paddleocr_failed",
        "paddleocr_error",
        "file_not_found",
        "unsupported_type",
    }
)


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


def _read_content_list_for_stem(output_dir: Path, name: str) -> list[dict]:
    """Read one file's MinerU content_list by its unique synthetic name.

    Targeted analogue of ``_read_output_files`` for the batch path. Where
    ``_read_output_files`` returns the first match found, this keys on the exact
    UNIQUE SYNTHETIC name handed to ``do_parse`` (``f"{i}_{path.stem}"``), so
    same-bare-stem files from different directories never collide.

    Args:
        output_dir: Root directory produced by a chunk's ``do_parse`` call.
        name: The unique synthetic name handed to ``do_parse`` for this file.

    Returns:
        The file's content_list (list of block dicts), or ``[]`` on miss or
        read error (debug-logged).
    """
    # NOTE: ``name`` is a filename (``f"{i}_{stem}"``) and MUST NOT be fed into
    # a glob pattern — a stem containing glob metacharacters (``[ ] * ?``, all
    # legal in filenames) would be mis-parsed and silently fail to match,
    # producing spurious empty output. do_parse writes to the deterministic
    # path ``{output_dir}/{name}/auto/{name}_content_list.json``; try that
    # first, then fall back to a LITERAL recursive search (constant glob
    # pattern + exact name compare) in case the layout differs by MinerU version.
    target_filename = f"{name}_content_list.json"
    candidates: list[Path] = [output_dir / name / "auto" / target_filename]
    if not candidates[0].is_file():
        candidates.extend(
            p for p in output_dir.rglob("*_content_list.json") if p.name == target_filename
        )
    for jf in candidates:
        if not jf.is_file():
            continue
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
            if isinstance(data, dict) and "content_list" in data:
                return data["content_list"]
        except Exception as exc:  # noqa: BLE001
            logger.debug("[parser] failed to read output file {}: {}", jf, exc)
    return []


def _read_middle_json_for_stem(output_dir: Path, name: str) -> dict | None:
    """Read one file's MinerU ``_middle.json`` by its unique synthetic name.

    Direct analogue of ``_read_content_list_for_stem`` for the pipeline
    ``_middle.json`` dump (which carries block layout-detection and span
    OCR-recognition scores). Keys on the exact UNIQUE SYNTHETIC name handed to
    ``do_parse`` so same-bare-stem files from different directories never
    collide. Best-effort and non-raising.

    Args:
        output_dir: Root directory produced by a chunk's ``do_parse`` call.
        name: The unique synthetic name handed to ``do_parse`` for this file.

    Returns:
        The parsed middle_json ``dict`` on a hit (via either the deterministic
        path or the recursive fallback), or ``None`` on a miss / read error /
        corrupt or non-dict JSON (debug-logged). Never raises.
    """
    # NOTE: ``name`` is a filename (``f"{i}_{stem}"``) and MUST NOT be fed into
    # a glob pattern — a stem containing glob metacharacters (``[ ] * ?``, all
    # legal in filenames) would be mis-parsed and silently fail to match.
    # do_parse writes to the deterministic path
    # ``{output_dir}/{name}/auto/{name}_middle.json``; try that first, then fall
    # back to a LITERAL recursive search (constant glob pattern + exact name
    # compare) in case the layout differs by MinerU version.
    target_filename = f"{name}_middle.json"
    candidates: list[Path] = [output_dir / name / "auto" / target_filename]
    if not candidates[0].is_file():
        candidates.extend(p for p in output_dir.rglob("*_middle.json") if p.name == target_filename)
    for jf in candidates:
        if not jf.is_file():
            continue
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except Exception as exc:  # noqa: BLE001
            logger.debug("[parser] failed to read middle_json {}: {}", jf, exc)
    logger.debug("[parser] middle_json not found for name {} in {}", name, output_dir)
    return None


def _run_mineru_inprocess(
    file_path: Path, backend: str = "pipeline"
) -> tuple[list[dict], dict | None]:
    """Run MinerU in-process via ``mineru.cli.common.do_parse``; return (content_list, middle_json).

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
        A ``(content_list, middle_json | None)`` pair: the MinerU content_list
        (list of block dicts; empty list when no content was produced) and the
        captured pipeline ``_middle.json`` dict (layout/OCR confidence scores),
        or ``None`` when none was captured (miss / corrupt).

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
        # NOTE: share the inference gate so a single-file parse (incl. the
        # per-file batch fallback) cannot run do_parse concurrently with a batch.
        with _mineru_inference_gate():
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
                f_dump_middle_json=True,
                f_dump_model_output=False,
                f_dump_orig_pdf=False,
                f_dump_content_list=True,
            )
        # Capture the dumped _middle.json (layout/OCR confidence scores) inside
        # the tempdir before it is cleaned up, and SURFACE it alongside the
        # content_list as a deliberate "list-or-pair" union. ``middle_json`` is
        # None on miss/corrupt, which never breaks the content_list return path.
        middle_json = _read_middle_json_for_stem(out_dir, file_path.stem)
        return _read_output_files(out_dir), middle_json


def _run_mineru_batch_chunk(
    chunk_paths: list[Path],
    name_map: dict[Path, str],
    backend: str = "pipeline",
) -> dict[Path, tuple[list[dict], dict | None]]:
    """Run exactly ONE ``do_parse`` for a chunk; return per-file (content_list, middle_json).

    Mirrors ``_run_mineru_inprocess`` but with a multi-item ``pdf_bytes_list``,
    so a single inference window covers all files in the chunk. Models stay
    resident across chunks via MinerU's pipeline ``ModelSingleton``.

    Each file is handed its UNIQUE SYNTHETIC name (``name_map[path]``), and its
    result is read back by that exact name via ``_read_content_list_for_stem``.

    # WARN: ``do_parse`` mutates ``pdf_file_names`` / ``pdf_bytes_list`` (it
    # deletes office-doc entries by index), so COPIES are always passed.

    Args:
        chunk_paths: The files in this chunk, in order.
        name_map: Maps each path to its batch-global unique synthetic name.
        backend: MinerU backend identifier.

    Returns:
        ``{path: (content_list, middle_json | None)}`` for this chunk's files —
        each value a deliberate "list-or-pair" union carrying the captured
        pipeline ``_middle.json`` dict (or ``None`` on miss/corrupt) alongside
        the content_list, keyed on the same path.

    Raises:
        ImportError: When the MinerU Python API is not importable.
        Exception: Propagated from ``do_parse``; the caller (``parse_batch``)
            owns the per-chunk fallback.
    """
    from mineru.cli.common import do_parse, read_fn  # noqa: PLC0415

    names = [name_map[p] for p in chunk_paths]
    # read_fn returns PDF bytes for PDFs and rasterised PDF bytes for images.
    bytes_list = [read_fn(p) for p in chunk_paths]
    lang_list = ["en"] * len(chunk_paths)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        # NOTE: pass COPIES — do_parse mutates its name/bytes/lang inputs by index.
        # NOTE: the inference gate serialises GPU inference across concurrent
        # parse_batch callers; held only around do_parse, not the cheap read-back.
        with _mineru_inference_gate():
            do_parse(
                output_dir=str(out_dir),
                pdf_file_names=list(names),
                pdf_bytes_list=list(bytes_list),
                p_lang_list=list(lang_list),
                backend=backend,
                parse_method="auto",
                f_draw_layout_bbox=False,
                f_draw_span_bbox=False,
                f_dump_md=False,
                f_dump_middle_json=True,
                f_dump_model_output=False,
                f_dump_orig_pdf=False,
                f_dump_content_list=True,
            )
        # Capture each file's dumped _middle.json inside the tempdir, keyed on
        # the SAME unique synthetic name (name_map[p]) the content_list read-back
        # uses so same-bare-stem files never collide, and SURFACE it alongside
        # each content_list as a deliberate "list-or-pair" union per path. None
        # on miss/corrupt, which never breaks the content_list return path.
        middle_json_by_path = {
            p: _read_middle_json_for_stem(out_dir, name_map[p]) for p in chunk_paths
        }
        return {
            p: (
                _read_content_list_for_stem(out_dir, name_map[p]),
                middle_json_by_path[p],
            )
            for p in chunk_paths
        }


def _run_mineru_batch(
    file_paths: list[Path], backend: str = "pipeline"
) -> dict[Path, tuple[list[dict], dict | None]]:
    """Orchestrate chunked MinerU batch inference over ``file_paths``.

    Thin orchestrator: computes the BATCH-GLOBAL unique name map
    (``f"{i}_{path.stem}"`` keyed on the file's 0-based position in the full
    ordered list), slices ``file_paths`` into chunks of ``MINERU_BATCH_SIZE``,
    and invokes ``_run_mineru_batch_chunk`` once per chunk (one ``do_parse``
    per chunk). Aggregates into ``{path: (content_list, middle_json | None)}``.

    # NOTE: This orchestrator does NOT swallow a chunk failure — a chunk's
    # ``do_parse`` exception propagates out so the per-chunk fallback decision
    # can live in one place (``parse_batch``).

    Args:
        file_paths: Full ordered list of uncached MinerU files to parse.
        backend: MinerU backend identifier.

    Returns:
        ``{path: (content_list, middle_json | None)}`` for every file in
        ``file_paths`` — each value the "list-or-pair" union from the chunk
        runner.
    """
    name_map = _build_batch_name_map(file_paths)
    chunk_size = _read_mineru_batch_size()
    results: dict[Path, tuple[list[dict], dict | None]] = {}
    for chunk_paths in _iter_chunks(file_paths, chunk_size):
        results.update(_run_mineru_batch_chunk(chunk_paths, name_map, backend))
    return results


def _iter_chunks(items: list, size: int):
    """Yield successive ``size``-length slices of ``items``.

    Single source of truth for the batch chunk slicing used by both
    ``_run_mineru_batch`` and ``parse_batch`` so the two cannot drift.

    Args:
        items: The list to slice.
        size: Chunk length (assumed ``>= 1``; see :func:`_read_mineru_batch_size`).

    Yields:
        Consecutive sub-lists of ``items`` of length ``size`` (the last may be shorter).
    """
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _build_batch_name_map(file_paths: list[Path]) -> dict[Path, str]:
    """Build the batch-global unique synthetic name map for ``do_parse``.

    The synthetic name is ``f"{i}_{path.stem}"`` where ``i`` is the file's
    0-based position in the full ordered list. This guarantees global
    uniqueness across chunks even when two files share a bare stem, so no
    duplicate-stem guard or ValueError is ever needed.

    Args:
        file_paths: Full ordered list of files to parse.

    Returns:
        Maps each path to its batch-global unique synthetic name.
    """
    return {path: f"{i}_{path.stem}" for i, path in enumerate(file_paths)}


def _read_mineru_batch_size() -> int:
    """Read MINERU_BATCH_SIZE from the environment, defensively.

    Mirrors the ``MINERU_BACKEND`` env-var read style. A non-parseable string
    or a value ``< 1`` falls back to the bounded default (``8``); never crashes
    and never allows ``0``, negative, or unbounded values.

    Returns:
        A positive, bounded chunk size.
    """
    raw = os.environ.get("MINERU_BATCH_SIZE", str(_DEFAULT_MINERU_BATCH_SIZE))
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return _DEFAULT_MINERU_BATCH_SIZE
    if value < 1:
        return _DEFAULT_MINERU_BATCH_SIZE
    return value


def _read_mineru_max_inflight() -> int:
    """Read MINERU_MAX_INFLIGHT from the environment, defensively.

    Controls how many MinerU ``do_parse`` inference windows may run concurrently
    across ALL in-process callers (default ``1`` — a single shared GPU). A
    non-parseable string or a value ``< 1`` falls back to ``1``.

    Returns:
        A positive concurrency bound.
    """
    raw = os.environ.get("MINERU_MAX_INFLIGHT", str(_DEFAULT_MINERU_MAX_INFLIGHT))
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return _DEFAULT_MINERU_MAX_INFLIGHT
    if value < 1:
        return _DEFAULT_MINERU_MAX_INFLIGHT
    return value


def _mineru_inference_gate() -> threading.BoundedSemaphore:
    """Return the process-wide MinerU inference gate (lazy singleton).

    Serialises in-process ``do_parse`` calls across concurrent ``parse_batch``
    invocations (and per-file fallback parses) so two batches cannot drive the
    same GPU/``ModelSingleton`` at once. Sized from ``MINERU_MAX_INFLIGHT`` on
    first use (default ``1``). Built under a lock so concurrent first-callers
    create exactly one semaphore.

    Returns:
        The shared ``threading.BoundedSemaphore``.
    """
    global _MINERU_INFERENCE_SEMA
    if _MINERU_INFERENCE_SEMA is None:
        with _MINERU_INFERENCE_SEMA_LOCK:
            if _MINERU_INFERENCE_SEMA is None:
                _MINERU_INFERENCE_SEMA = threading.BoundedSemaphore(_read_mineru_max_inflight())
    return _MINERU_INFERENCE_SEMA


def _run_mineru(file_path: Path, backend: str = "pipeline") -> tuple[list[dict], dict | None]:
    """Invoke MinerU and return ``(content_list, middle_json | None)``.

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
        A ``(content_list, middle_json | None)`` pair — a deliberate
        "list-or-pair" union. The in-process path surfaces the captured
        pipeline ``_middle.json`` dict; the CLI fallback has no ``_middle.json``
        and returns ``(content, None)``.

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
        content, middle_json = _run_mineru_inprocess(file_path, backend=backend)
        if content:
            return content, middle_json
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
                # CLI fallback has no captured _middle.json -> middle_json is None.
                return content, None
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
            logger.warning("[docling] base64 decode failed for element {}: {}", element_idx, exc)
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
            from docling.datamodel.base_models import InputFormat  # noqa: PLC0415
            from docling.datamodel.pipeline_options import PdfPipelineOptions  # noqa: PLC0415
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


def _get_paddle_pipeline() -> Any:
    """Return the process-wide cached PaddleOCR-VL pipeline, building it lazily.

    Uses PaddleOCR's ``PaddleOCRVL`` pipeline — the end-to-end PaddleOCR-VL-0.9B
    vision-language model (NaViT visual encoder + ERNIE-4.5-0.3B decoder) — not
    the older modular PP-StructureV3. The pipeline version defaults to ``v1.6``
    (override via ``PADDLE_VL_VERSION``). The VL weights (~1.8 GB) download to the
    PaddleX/HF cache on first use, then load on the GPU; the instance is built
    once and reused across all PaddleOCR parses in the process.

    Raises:
        ImportError: When paddleocr / paddlex[ocr] are not installed.
    """
    global _PADDLE_PIPELINE
    if _PADDLE_PIPELINE is not None:
        return _PADDLE_PIPELINE
    with _PADDLE_PIPELINE_LOCK:
        if _PADDLE_PIPELINE is None:
            try:
                from paddleocr import PaddleOCRVL  # noqa: PLC0415
            except ImportError as exc:
                raise ImportError(
                    "PaddleOCR is not installed. Run: uv pip install paddleocr "
                    "'paddlex[ocr]' and paddlepaddle-gpu"
                ) from exc
            version = os.environ.get("PADDLE_VL_VERSION", "v1.6")
            # Optional fast path: route the VL recognition step to an external
            # OpenAI-compatible serving engine (vLLM/SGLang) instead of the slow
            # in-process ``native`` generator. Set PADDLE_VL_BACKEND (e.g.
            # ``vllm-server``) and PADDLE_VL_SERVER_URL (e.g.
            # ``http://127.0.0.1:8118/v1``). When unset, the native backend runs.
            kwargs: dict[str, Any] = {"pipeline_version": version}
            vl_backend = os.environ.get("PADDLE_VL_BACKEND")
            vl_server_url = os.environ.get("PADDLE_VL_SERVER_URL")
            if vl_backend:
                kwargs["vl_rec_backend"] = vl_backend
            if vl_server_url:
                kwargs["vl_rec_server_url"] = vl_server_url
            _PADDLE_PIPELINE = PaddleOCRVL(**kwargs)
        return _PADDLE_PIPELINE


def _paddle_bbox_to_permille(
    bbox: object, width: float | None, height: float | None
) -> list[float]:
    """Convert a PP-StructureV3 pixel ``[x0, y0, x1, y1]`` box to per-mille.

    PP-StructureV3 reports boxes in pixels relative to the rendered page image
    (``res["width"]`` × ``res["height"]``). We normalise to per-mille (0..1000,
    top-left origin) so the stored ``ElementRecord.bbox`` matches MinerU's
    convention exactly and the viewer needs no page-size context (see
    ``viz/coords.py``). Returns ``[]`` (no geometry) on any unusable input.
    """
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return []
    if not width or not height:
        return []
    try:
        x0, y0, x1, y1 = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return []
    return [
        x0 * 1000.0 / width,
        y0 * 1000.0 / height,
        x1 * 1000.0 / width,
        y1 * 1000.0 / height,
    ]


def _run_paddleocr(file_path: Path, config: EnrichmentConfig) -> list[ElementRecord]:
    """Invoke the PaddleOCR-VL pipeline and return pre-routed ElementRecords.

    PaddleOCR-VL (the 0.9B VLM) produces one result object per page; like every
    PaddleX pipeline it carries a ``parsing_res_list`` of layout blocks
    (``block_label``, ``block_content``, pixel ``block_bbox``) plus the page image
    ``width``/``height``. Each block is mapped to an ElementType via
    ``_PADDLE_LABEL_MAP`` and its box normalised to per-mille top-left (MinerU's
    convention) so the rest of the pipeline and the viewer treat PaddleOCR exactly
    like MinerU.

    Args:
        file_path: Path to the document to parse (PDF or image).
        config: EnrichmentConfig (PaddleOCR has no extra pipeline controls yet).

    Returns:
        ElementRecord list in page / reading order.

    Raises:
        ImportError: When PaddleOCR is not installed.
        RuntimeError: When the PaddleOCR-VL prediction raises.
    """
    pipeline = _get_paddle_pipeline()

    # Optional predict-time tuning. ``PADDLE_VL_MERGE_BBOXES`` ("small"/"large")
    # controls layout box merging: "small" measurably reduces detected-but-
    # unrecognised text boxes on dense/scanned pages without dropping content
    # (verified on a scanned form: blank text boxes 9 -> 3, char count kept),
    # whereas "large" and doc-unwarping over-merge and lose text, so neither is
    # enabled by default. Unset leaves the pipeline default.
    predict_kwargs: dict[str, Any] = {"input": str(file_path)}
    merge_mode = os.environ.get("PADDLE_VL_MERGE_BBOXES")
    if merge_mode:
        predict_kwargs["layout_merge_bboxes_mode"] = merge_mode

    try:
        results = list(pipeline.predict(**predict_kwargs))
    except Exception as exc:
        raise RuntimeError(f"PaddleOCR-VL predict failed: {exc}") from exc

    sha256 = _file_sha256(file_path)
    elements: list[ElementRecord] = []
    element_idx = 0

    for page_pos, res in enumerate(results):
        data = getattr(res, "json", None)
        if not isinstance(data, dict):
            continue
        inner = data.get("res", data)
        if not isinstance(inner, dict):
            continue

        # page_index is the absolute 0-based page; fall back to enumeration order.
        page_idx = inner.get("page_index")
        page_idx = int(page_idx) if isinstance(page_idx, (int, float)) else page_pos

        width = inner.get("width")
        height = inner.get("height")

        # parsing_res_list is the reading-order layout result; block_order is
        # already 1-based reading order, so we keep PP-StructureV3's order.
        blocks = inner.get("parsing_res_list") or []
        for blk in blocks:
            if not isinstance(blk, dict):
                element_idx += 1
                continue
            label = str(blk.get("block_label", "") or "").lower()
            etype = _PADDLE_LABEL_MAP.get(label, ElementType.unknown)
            content = blk.get("block_content", "")
            text = content if isinstance(content, str) else str(content or "")
            bbox_pm = _paddle_bbox_to_permille(blk.get("block_bbox"), width, height)
            try:
                elements.append(
                    ElementRecord(
                        element_id=_build_element_id(sha256, page_idx, element_idx),
                        type=etype,
                        text=text.strip(),
                        bbox=bbox_pm,
                        page_idx=page_idx,
                    )
                )
            except Exception as exc:  # noqa: BLE001 — per-block routing is non-fatal
                logger.debug(
                    "[paddleocr] skipping block label={} idx={}: {}", label, element_idx, exc
                )
            element_idx += 1

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
            element = element.model_copy(update={"description": description, "is_enriched": True})

        enriched.append(element)

    return enriched


def _split_mineru_result(value: object) -> tuple[list[dict], dict | None]:
    """Tolerantly unpack a runner result into ``(content_list, middle_json | None)``.

    The MinerU runners (and the ~15 existing test mocks) return one of two
    shapes — a deliberate "list-or-pair" union:

    - the real pair ``(content_list, middle_json | None)`` produced by the
      wired runners, or
    - a bare ``content_list`` (the historical shape that mocks still return).

    For the batch path, the ``{path: value}`` map is split per value upstream,
    so each per-path ``value`` is itself one of the two shapes above.

    A bare list yields ``middle_json=None``. This helper is defensive and
    NEVER raises: an unexpected shape degrades to ``([], None)`` rather than
    propagating, preserving the package's never-raises contract.

    Args:
        value: A runner result — either the ``(content_list, middle_json)``
            pair or a bare ``content_list``.

    Returns:
        A ``(content_list, middle_json | None)`` tuple. ``middle_json`` is
        ``None`` for a bare list or any unexpected input.
    """
    # Pair shape: a 2-tuple/2-list of (content_list, middle_json|None).
    if isinstance(value, tuple) and len(value) == 2:  # noqa: PLR2004
        content, middle = value
        content_list = content if isinstance(content, list) else []
        middle_json = middle if isinstance(middle, dict) else None
        return content_list, middle_json
    # Bare content_list shape (mock / unwired path): middle_json is absent.
    if isinstance(value, list):
        return value, None
    # Unexpected shape: degrade safely rather than raise.
    return [], None


def _normalise_mineru_content(content_list: list[dict]) -> list[dict]:
    """Filter non-dict blocks and normalise MinerU field aliases.

    VERBATIM extraction of the filter+normalise block previously inlined in
    ``parse()``. Drops any item that is not a dict (debug-logging the dropped
    count) and returns ``_normalise_aliases(b)`` for each surviving block.
    Shared by both ``parse()`` and ``parse_batch()``.

    Args:
        content_list: Raw MinerU content_list, possibly containing ``None`` or
            other non-dict entries.

    Returns:
        List of alias-normalised block dicts.
    """
    # NOTE: Filter out None and non-dict items before normalising aliases so
    # that malformed MinerU output does not crash the whole pipeline.
    valid_blocks = [b for b in content_list if isinstance(b, dict)]
    if len(valid_blocks) < len(content_list):
        logger.debug(
            "[parser] skipped {} non-dict block(s) in content_list",
            len(content_list) - len(valid_blocks),
        )

    # Normalise aliases across all valid blocks.
    return [_normalise_aliases(b) for b in valid_blocks]


def _route_mineru_content_list(content_list: list[dict], sha256: str) -> list[ElementRecord]:
    """Group normalised MinerU blocks by page and route each to an ElementRecord.

    VERBATIM extraction of the page-grouping + routing block previously inlined
    in ``parse()``. Groups blocks by ``page_idx``, computes the page count, and
    routes each block via ``_route_block``. Per-block routing errors are
    debug-logged and skipped (non-fatal), exactly as before.

    Args:
        content_list: Alias-normalised MinerU block dicts (already filtered).
        sha256: SHA-256 digest of the source file, used for element IDs.

    Returns:
        ElementRecord list in page order.
    """
    # Group blocks by page index.
    pages_map: dict[int, list[dict]] = {}
    for block in content_list:
        pidx = int(block.get("page_idx", 0))
        pages_map.setdefault(pidx, []).append(block)

    page_count_from_blocks = (max(pages_map.keys()) + 1) if pages_map else 0

    # Build ElementRecord list in page order.
    all_elements: list[ElementRecord] = []
    element_idx = 0
    for pidx in range(page_count_from_blocks):
        for block in pages_map.get(pidx, []):
            try:
                all_elements.append(_route_block(block, pidx, element_idx, sha256))
            except Exception as exc:  # noqa: BLE001
                # NOTE: Per-block routing errors are non-fatal; log and skip
                # the offending block rather than aborting the entire parse.
                logger.debug(
                    "[parser] skipping block at page={} idx={}: {}", pidx, element_idx, exc
                )
            element_idx += 1

    return all_elements


def _build_parser_output(
    file_path: Path,
    sha256: str,
    elements: list[ElementRecord],
    content_list: list[dict],
    config: EnrichmentConfig,
    middle_json: dict | None = None,
) -> ParserOutput:
    """Assemble a ParserOutput from routed elements; non-raising by contract.

    Shared by both ``parse()`` and ``parse_batch()``. Owns, identical to the
    behaviour previously inlined in ``parse()``: Layer-1 token counts (PDF only,
    under ``_PDFIUM_LOCK``), the per-page quality gate (``evaluate_page`` ->
    ``quality_gate_escalation`` warnings + ``PageRecord``s), serial per-file VLM
    enrichment (``_enrich_elements`` with the ``enrichment_not_supported`` /
    ``enrichment_error`` guards), ``ParserOutput`` assembly, and the per-file
    ``cache_mod.put``.

    # NOTE: This helper owns its OWN outer try/except so it never propagates.
    # On any internal failure it returns a ParserOutput carrying a
    # backend-specific code (``docling_error`` for Docling, else ``mineru_error``).

    # NOTE: ``_PDFIUM_LOCK`` is acquired ONLY inside this helper. In the batch
    # path it is called per-file sequentially after inference, so the lock
    # serialises cleanly with no deadlock.

    Args:
        file_path: Source document path.
        sha256: SHA-256 digest of the source file.
        elements: Routed ElementRecord list (MinerU- or Docling-derived).
        content_list: Raw MinerU blocks for enrichment context (``[]`` for Docling).
        config: Enrichment and backend configuration.
        middle_json: The captured MinerU pipeline ``_middle.json`` dict, or
            ``None``. Aggregated into ``ParserOutput.confidence`` ONLY on the
            MinerU pipeline path; ``None`` (and ``confidence=None``) for Docling
            / mineru-vlm and for unwired / mock paths.

    Returns:
        Validated ParserOutput. Always returned — never raises.
    """
    try:
        all_elements = elements
        warnings: list[WarningRecord] = []

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
            logger.debug("[quality_gate] skipping Layer 1 for non-PDF input: {}", file_path)

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
                all_elements = _enrich_elements(all_elements, content_list, config, file_path)
            except Exception as exc:
                logger.warning("[parser] enrichment failed for {}: {}", file_path, exc)
                warnings.append(
                    WarningRecord(
                        code="enrichment_error",
                        message=f"Enrichment failed: {exc}",
                    )
                )

        # Confidence decision table (MinerU PIPELINE only). Lives INSIDE this
        # helper's outer try/except so a malformed middle_json can never break
        # assembly — it degrades to confidence=None per the table below.
        #   (a) parser != "mineru"            -> None, no warning
        #   (b) mineru, middle_json absent     -> None, no warning (mock/unwired)
        #   (c) mineru, present-but-unusable   -> None + confidence_unavailable
        #   (d) mineru, usable (total_pages>0) -> populated, source_path set
        confidence = None
        if config.parser == "mineru" and middle_json is not None:
            # extract_confidence is pure / never-raising and uses its FROZEN
            # item-21 defaults (merge_discarded=False, threshold=0.70).
            doc_confidence = extract_confidence(middle_json)
            if doc_confidence.total_pages > 0:
                confidence = doc_confidence.model_copy(update={"source_path": str(file_path)})
            else:
                # Captured a dump but it aggregated to nothing usable — a real
                # degradation signal. NOT a batch-failure code.
                warnings.append(
                    WarningRecord(
                        code="confidence_unavailable",
                        message=(
                            "MinerU pipeline confidence requested but the captured "
                            "_middle.json was missing, corrupt, or empty"
                        ),
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
            confidence=confidence,
        )

        cache_mod.put(file_path, output, config)
        return output

    except Exception as exc:  # noqa: BLE001
        # NOTE: Last-resort net for this helper — never propagate. Carries a
        # backend-specific error code so callers stay consistent with parse().
        logger.warning("[parser] _build_parser_output failed for {}: {}", file_path, exc)
        error_code = {"docling": "docling_error", "paddleocr": "paddleocr_error"}.get(
            config.parser, "mineru_error"
        )
        return ParserOutput(
            file_path=str(file_path),
            file_sha256=sha256,
            page_count=0,
            pages=[],
            elements=[],
            warnings=[
                WarningRecord(
                    code=error_code,
                    message=f"Output assembly error: {exc}",
                )
            ],
            enrichment_config=config,
        )


def _classify_batch_paths(
    paths: list[Path], config: EnrichmentConfig
) -> tuple[dict[Path, ParserOutput], dict[Path, ParserOutput], dict[Path, str]]:
    """Classify batch paths into invalid / cache-hit / needs-parse buckets.

    Synchronous and IO-bound (``_file_sha256`` reads each file in full,
    ``cache_mod.get`` reads cache files), so ``parse_batch`` runs it via
    ``asyncio.to_thread`` to keep the event loop responsive on large batches.

    Args:
        paths: The input paths, in order.
        config: Shared enrichment/backend configuration (drives the accepted
            extension set).

    Returns:
        ``(invalid_outputs, cache_hits, needs_parse)`` where ``invalid_outputs``
        maps missing/unsupported paths to a warning ParserOutput, ``cache_hits``
        maps paths to their cached ParserOutput, and ``needs_parse`` maps each
        remaining path to its SHA-256 (``""`` when hashing failed — deferred to
        the per-file ``parse()`` fallback). Insertion order is preserved.
    """
    invalid_outputs: dict[Path, ParserOutput] = {}
    cache_hits: dict[Path, ParserOutput] = {}
    needs_parse: dict[Path, str] = {}  # ordered: path -> sha256

    for path in paths:
        if not path.exists():
            invalid_outputs[path] = ParserOutput(
                file_path=str(path),
                file_sha256="",
                page_count=0,
                pages=[],
                elements=[],
                warnings=[
                    WarningRecord(
                        code="file_not_found",
                        message=f"File not found: {path}",
                    )
                ],
                enrichment_config=config,
            )
            continue
        if path.suffix.lower() not in _accepted_extensions(config):
            invalid_outputs[path] = ParserOutput(
                file_path=str(path),
                file_sha256="",
                page_count=0,
                pages=[],
                elements=[],
                warnings=[
                    WarningRecord(
                        code="unsupported_type",
                        message=f"Unsupported file extension: {path.suffix!r}",
                    )
                ],
                enrichment_config=config,
            )
            continue
        try:
            cached = cache_mod.get(path, config)
        except Exception as exc:  # noqa: BLE001
            logger.debug("[parser] cache lookup failed for {}: {}", path, exc)
            cached = None
        if cached is not None:
            cache_hits[path] = cached
            continue
        try:
            needs_parse[path] = _file_sha256(path)
        except Exception as exc:  # noqa: BLE001
            # NOTE: defer to per-file parse() fallback, which owns the never-raise net.
            logger.debug("[parser] sha256 failed for {}; deferring to parse(): {}", path, exc)
            needs_parse[path] = ""

    return invalid_outputs, cache_hits, needs_parse


def _assemble_chunk_outputs(
    chunk_paths: list[Path],
    name_map: dict[Path, str],
    needs_parse: dict[Path, str],
    backend: str,
    config: EnrichmentConfig,
) -> dict[Path, ParserOutput]:
    """Run one chunk's ``do_parse`` and assemble per-file ParserOutputs.

    Synchronous and CPU/GPU/IO-heavy — ``parse_batch`` runs it via
    ``asyncio.to_thread`` so the event loop is never blocked during inference,
    pypdfium2 token counting, or VLM enrichment.

    # NOTE: This raises ONLY if the chunk's ``do_parse`` / ``read_fn`` raises;
    # the caller (``parse_batch``) catches that and falls the chunk back to
    # per-file ``parse()``. Per-file assembly via ``_build_parser_output`` is
    # itself non-raising, and an empty ``content_list`` for a valid file yields
    # a ``mineru_failed`` ParserOutput (Q4 — no silent data loss).

    Args:
        chunk_paths: Files in this chunk, in order.
        name_map: Batch-global unique synthetic name per path.
        needs_parse: Maps each path to its precomputed SHA-256.
        backend: MinerU backend identifier.
        config: Shared enrichment/backend configuration.

    Returns:
        ``{path: ParserOutput}`` for every file in ``chunk_paths``.

    Raises:
        Exception: Propagated from ``_run_mineru_batch_chunk`` (chunk inference
            failure) so the caller can fall back per chunk.
    """
    chunk_results = _run_mineru_batch_chunk(chunk_paths, name_map, backend)
    outputs: dict[Path, ParserOutput] = {}
    for path in chunk_paths:
        sha256 = needs_parse[path]
        raw_content, middle_json = _split_mineru_result(chunk_results.get(path, []))
        content_list = _normalise_mineru_content(raw_content)
        if not content_list:
            # HARD requirement (Q4): no silent data loss for valid files.
            logger.warning("[parser] do_parse produced no output for {}", path)
            outputs[path] = ParserOutput(
                file_path=str(path),
                file_sha256=sha256,
                page_count=0,
                pages=[],
                elements=[],
                warnings=[
                    WarningRecord(
                        code="mineru_failed",
                        message="do_parse produced no output",
                    )
                ],
                enrichment_config=config,
            )
            continue
        elements = _route_mineru_content_list(content_list, sha256)
        outputs[path] = _build_parser_output(
            path, sha256, elements, content_list, config, middle_json=middle_json
        )
    return outputs


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

    # NOTE: The body is wrapped in try/except so that no unexpected exception
    # can propagate to the caller. The outer handler is the LAST-RESORT net
    # producing a minimal ParserOutput with a backend-specific error code
    # (``"docling_error"`` or ``"mineru_error"``). The dedicated engine
    # try/except blocks below preserve the more specific ``mineru_failed`` /
    # ``docling_failed`` codes and are NOT collapsed into that bucket.

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
        # Config is part of the key so a different backend/settings never hits.
        cached = cache_mod.get(file_path, config)
        if cached is not None:
            logger.debug("[parser] cache hit for {}", file_path)
            return cached

        mineru_backend = os.environ.get("MINERU_BACKEND", "pipeline")

        # Backend dispatch: Docling or MinerU path.
        all_elements: list[ElementRecord]
        content_list: list[dict] = []
        # middle_json is only ever populated on the MinerU pipeline path; stays
        # None for Docling and for the MinerU CLI-fallback / mock paths.
        middle_json: dict | None = None

        if config.parser == "docling":
            # NOTE: dedicated engine try/except — preserves the specific
            # ``docling_failed`` code; not collapsed into the outer net.
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
            content_list = []
        elif config.parser == "paddleocr":
            # NOTE: dedicated engine try/except — preserves the specific
            # ``paddleocr_failed`` code; not collapsed into the outer net.
            try:
                all_elements = _run_paddleocr(file_path, config)
            except Exception as exc:
                logger.warning("[parser] PaddleOCR failed for {}: {}", file_path, exc)
                return ParserOutput(
                    file_path=str(file_path),
                    file_sha256=sha256,
                    page_count=0,
                    pages=[],
                    elements=[],
                    warnings=[
                        WarningRecord(
                            code="paddleocr_failed",
                            message=f"PaddleOCR failed: {exc}",
                        )
                    ],
                    enrichment_config=config,
                )
            content_list = []
        else:
            # NOTE: dedicated engine try/except — preserves the specific
            # ``mineru_failed`` code; not collapsed into the outer net.
            try:
                raw = _run_mineru(file_path, backend=mineru_backend)
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

            raw_content, middle_json = _split_mineru_result(raw)
            content_list = _normalise_mineru_content(raw_content)
            all_elements = _route_mineru_content_list(content_list, sha256)

        return _build_parser_output(
            file_path, sha256, all_elements, content_list, config, middle_json=middle_json
        )

    except Exception as exc:
        # Last-resort never-raise net only.
        logger.warning("[parser] unhandled exception for {}: {}", file_path, exc)
        try:
            sha = _file_sha256(file_path) if file_path.exists() else ""
        except Exception:  # noqa: BLE001
            sha = ""
        error_code = {"docling": "docling_error", "paddleocr": "paddleocr_error"}.get(
            config.parser, "mineru_error"
        )
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
    """Parse multiple documents and return one result per input, in input order.

    For the MinerU backend (default), uncached files are routed through
    chunked, single-call ``do_parse`` inference: the ordered ``needs_parse``
    list is sliced into chunks of ``MINERU_BATCH_SIZE`` (env var, finite default
    ``8``) and each chunk is run through ONE ``do_parse``. This costs
    ``ceil(N / MINERU_BATCH_SIZE)`` inference windows for N uncached files
    instead of N. A chunk whose ``do_parse`` raises falls back to per-file
    ``parse()`` for ONLY that chunk's files. Cache hits and invalid paths bypass
    inference entirely.

    Individual document failures do not abort the batch — each failed path
    produces a ParserOutput with warnings, matching the never-raise contract of
    parse().

    Args:
        paths: List of document paths to parse.
        config: Shared enrichment and backend configuration.
        max_concurrency: Maximum number of concurrent ``parse()`` threads. It
            governs the per-file fallback (when a chunk's ``do_parse`` raises)
            and the Docling / non-MinerU per-file path ONLY — it does NOT govern
            MinerU batch inference concurrency. ``MINERU_BATCH_SIZE`` is the
            separate, explicit knob for inference batching.

    Returns:
        List of ParserOutput instances in the same order as the input paths.
    """
    if config is None:
        config = EnrichmentConfig()

    semaphore = asyncio.Semaphore(max_concurrency)

    async def _parse_one(path: Path) -> ParserOutput:
        async with semaphore:
            return await asyncio.to_thread(parse, path, config)

    # Non-MinerU (e.g. Docling) path is UNCHANGED: per-file fan-out.
    if config.parser != "mineru":
        return list(await asyncio.gather(*[_parse_one(p) for p in paths]))

    mineru_backend = os.environ.get("MINERU_BACKEND", "pipeline")
    batch_size = _read_mineru_batch_size()

    # ---- STEP 1: classify all paths (offloaded — sha256 reads whole files). ----
    # NOTE: run off the event loop; _file_sha256 reads each file in full and
    # cache_mod.get reads cache files, so a large batch would otherwise block.
    invalid_outputs, cache_hits, needs_parse = await asyncio.to_thread(
        _classify_batch_paths, paths, config
    )

    # ---- STEP 2: chunked batch inference with per-chunk fallback. ----
    parsed_outputs: dict[Path, ParserOutput] = {}
    fallback_used = False

    ordered_needs = list(needs_parse.keys())
    name_map = _build_batch_name_map(ordered_needs)

    for chunk_paths in _iter_chunks(ordered_needs, batch_size):
        try:
            # NOTE: the synchronous chunk work (do_parse inference, pypdfium2
            # token counting, VLM enrichment) is offloaded to a worker thread so
            # this async coroutine never blocks the event loop during inference.
            chunk_outputs = await asyncio.to_thread(
                _assemble_chunk_outputs,
                chunk_paths,
                name_map,
                needs_parse,
                mineru_backend,
                config,
            )
        except Exception as exc:  # noqa: BLE001
            # Per-chunk fallback: only THIS chunk's files fall back to parse().
            fallback_used = True
            logger.warning(
                "[parser] batch chunk do_parse failed ({} files); "
                "falling back to per-file parse(): {}",
                len(chunk_paths),
                exc,
            )
            fallback_results = await asyncio.gather(*[_parse_one(p) for p in chunk_paths])
            parsed_outputs.update(dict(zip(chunk_paths, fallback_results, strict=True)))
            continue

        parsed_outputs.update(chunk_outputs)

    # ---- STEP 3: merge in original input order. ----
    results: list[ParserOutput] = []
    for p in paths:
        out = invalid_outputs.get(p) or cache_hits.get(p) or parsed_outputs.get(p)
        if out is None:
            # NOTE: defensive — every path is classified into exactly one bucket
            # and every needs_parse path is assembled, so this should be
            # unreachable. Synthesise an error output rather than returning None
            # and violating the list[ParserOutput] contract.
            logger.warning("[parser] no output produced for {}; synthesising error", p)
            out = ParserOutput(
                file_path=str(p),
                file_sha256="",
                page_count=0,
                pages=[],
                elements=[],
                warnings=[
                    WarningRecord(
                        code="mineru_error",
                        message="No output produced for path in batch",
                    )
                ],
                enrichment_config=config,
            )
        results.append(out)

    # One INFO summary line per parse_batch call (whole-batch rollup).
    # NOTE: empty_or_failed counts documents whose output carries a hard-failure
    # warning code — a true failure count, not "zero elements" (a blank page is
    # a clean parse) and including assembly failures (mineru_error) too.
    empty_or_failed = sum(
        1
        for out in parsed_outputs.values()
        if any(w.code in _BATCH_FAILURE_CODES for w in out.warnings)
    )
    logger.info(
        "[parser] batch summary: requested={}, cache_hits={}, parsed={}, "
        "empty_or_failed={}, fallback={}",
        len(paths),
        len(cache_hits),
        len(parsed_outputs),
        empty_or_failed,
        fallback_used,
    )

    return results
