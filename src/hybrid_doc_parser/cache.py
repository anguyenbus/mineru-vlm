"""File-based parse result cache keyed on sha256 + mtime + parser config.

The cache key folds the file content hash, its mtime, AND the
``EnrichmentConfig`` used for the parse, so the same file parsed with different
settings (e.g. ``parser="mineru"`` vs ``parser="docling"``, or enrichment on vs
off) never collides — each combination is cached independently. Omitting the
config (``config=None``) reproduces the legacy content+mtime-only key.

Cache files are written atomically via a .json.tmp rename to prevent corrupt
reads under concurrent writers. All public functions are non-raising: errors
are logged at WARNING or DEBUG level and the caller receives a safe fallback
(``None`` for reads, silent no-op for writes/deletes).

Typical usage::

    from pathlib import Path
    from hybrid_doc_parser.cache import get, put, invalidate

    path = Path("/data/report.pdf")
    cached = get(path)
    if cached is None:
        result = expensive_parse(path)
        put(path, result)
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from hybrid_doc_parser.models import EnrichmentConfig, ParserOutput


def _cache_dir() -> Path:
    """Return the cache directory resolved from the environment.

    Reads ``HYBRID_DOC_PARSER_CACHE_DIR``; falls back to
    ``~/.cache/hybrid_doc_parser`` when the variable is absent.

    Returns:
        Expanded ``Path`` to the cache directory. The directory may not
        exist yet — callers that need it to exist must create it themselves.
    """
    value = os.environ.get("HYBRID_DOC_PARSER_CACHE_DIR", "~/.cache/hybrid_doc_parser")
    return Path(value).expanduser()


def _cache_key(file_path: Path, config: EnrichmentConfig | None = None) -> str:
    """Compute a stable cache key from file content, mtime, and parser config.

    The key encodes the SHA-256 prefix of the file bytes (with the
    ``EnrichmentConfig`` folded into the same digest) plus the
    millisecond-resolution mtime, so that any change — content, timestamp, OR
    the config used to parse (parser backend, enrichment flags, Docling
    options) — produces a new key. Folding the config into the digest keeps the
    key FORMAT unchanged while making backends/settings cache independently.

    Args:
        file_path: Path to the source file whose content and mtime are used.
        config: The ``EnrichmentConfig`` for this parse. When ``None`` the key
            reproduces the legacy content+mtime-only behavior.

    Returns:
        String of the form ``'{sha256_prefix}_{mtime_ms}'`` where
        ``sha256_prefix`` is the first 32 hex characters of the SHA-256
        digest and ``mtime_ms`` is the integer mtime in milliseconds.
    """
    hasher = hashlib.sha256()
    hasher.update(file_path.read_bytes())
    if config is not None:
        # NUL separator so file-bytes and config bytes cannot run together; the
        # frozen model serializes deterministically (stable field order).
        hasher.update(b"\x00config\x00")
        hasher.update(config.model_dump_json().encode("utf-8"))
    sha_prefix = hasher.hexdigest()[:32]
    mtime_ms = int(file_path.stat().st_mtime * 1000)
    return f"{sha_prefix}_{mtime_ms}"


def _cache_path(key: str) -> Path:
    """Return the absolute path to the cache JSON file for a given key.

    Uses the first 16 characters of the key as the filename to keep cache
    directory listings manageable while preserving adequate uniqueness.

    Args:
        key: Cache key string as returned by :func:`_cache_key`.

    Returns:
        Absolute ``Path`` to the ``{key[:16]}.json`` file inside the cache
        directory. The file may not exist yet.
    """
    return _cache_dir() / f"{key[:16]}.json"


def get(file_path: Path, config: EnrichmentConfig | None = None) -> ParserOutput | None:
    """Return a cached ``ParserOutput`` for *file_path*, or ``None`` on any miss.

    A miss occurs when:
    - The cache file does not exist.
    - The stored ``_cache_key`` does not match the key computed from the
      current file content, mtime, and config (i.e. the file changed since
      caching, or it was cached under a different parser config).
    - Any exception is raised during reading or deserialisation.

    Args:
        file_path: Path to the source file for which a cached result is sought.
        config: The ``EnrichmentConfig`` for this parse; must match the config
            the entry was written with (``put``) to hit. ``None`` uses the
            legacy content+mtime-only key.

    Returns:
        A validated ``ParserOutput`` on cache hit, ``None`` on any miss or
        error. Never raises.
    """
    try:
        key = _cache_key(file_path, config)
        path = _cache_path(key)
        if not path.exists():
            return None
        text = path.read_text(encoding="utf-8")
        data = json.loads(text)
        if data.get("_cache_key") != key:
            return None
        # NOTE Remove internal bookkeeping fields before handing to Pydantic so
        # that model_validate does not reject unknown keys if model is strict.
        data.pop("_cache_key", None)
        data.pop("_cached_at", None)
        return ParserOutput.model_validate(data)
    except Exception as exc:
        logger.debug("[cache] get miss for {}: {}", file_path, exc)
        return None


def put(
    file_path: Path, output: ParserOutput, config: EnrichmentConfig | None = None
) -> None:
    """Persist *output* to the cache for *file_path*. Never raises.

    Writes are atomic: the JSON payload is first flushed to a ``.json.tmp``
    sibling file and then renamed over the target ``.json`` file, so
    concurrent readers never observe a partial write.

    Args:
        file_path: Path to the source file being cached.
        output: The fully-validated ``ParserOutput`` to store.
        config: The ``EnrichmentConfig`` this result was produced with; pass the
            SAME value to ``get`` to retrieve it. ``None`` uses the legacy
            content+mtime-only key. (Not auto-derived from
            ``output.enrichment_config`` so callers control key identity
            explicitly and symmetrically with ``get``.)
    """
    try:
        key = _cache_key(file_path, config)
        cache_file = _cache_path(key)
        directory = _cache_dir()
        directory.mkdir(parents=True, exist_ok=True)

        data = json.loads(output.model_dump_json())
        data["_cache_key"] = key
        data["_cached_at"] = datetime.now(tz=timezone.utc).isoformat()

        json_text = json.dumps(data, ensure_ascii=False)
        # NOTE Write to a .tmp sibling first, then rename for atomicity.
        tmp_path = cache_file.with_suffix(".json.tmp")
        tmp_path.write_text(json_text, encoding="utf-8")
        tmp_path.rename(cache_file)
    except Exception as exc:
        logger.warning("[cache] write error for {}: {}", file_path, exc)


def invalidate(file_path: Path, config: EnrichmentConfig | None = None) -> None:
    """Delete the cache entry for *file_path* if it exists. Never raises.

    Silently succeeds if no cache entry is present or if deletion fails for
    any reason (e.g. race condition where another process already removed it).

    Args:
        file_path: Path to the source file whose cache entry should be removed.
        config: The ``EnrichmentConfig`` whose entry should be removed; ``None``
            targets the legacy content+mtime-only key.
    """
    try:
        key = _cache_key(file_path, config)
        path = _cache_path(key)
        if path.exists():
            path.unlink()
    except Exception as exc:
        logger.debug("[cache] invalidate error for {}: {}", file_path, exc)
