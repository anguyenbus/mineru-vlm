"""File-based verification cache for the standalone advisory ``verify()``.

This cache is SEPARATE from the parse cache in ``cache.py`` and is touched ONLY
inside ``verify()``'s per-page flow. It is keyed on the 4-tuple
``(content_hash, page_idx, model_id, prompt_version)`` where ``content_hash`` is
the ``ParserOutput.file_sha256``, ``model_id`` is ``VerifierConfig.model``, and
``prompt_version`` is the verifier's ``PROMPT_VERSION``. A change to either
``model_id`` or ``prompt_version`` produces a new key and therefore busts the
cached verdict.

The cached unit is a single page's verdict (:class:`PageVerification`), so it
round-trips losslessly via the Pydantic model. It deliberately does NOT piggyback
on the parse cache: a parse-cache hit never triggers a verification-cache read or
write, because the verification cache is reached only from ``verify()``.

Mirroring ``cache.py``'s discipline, all public functions are non-raising: read
and write failures are logged at DEBUG/WARNING level and degrade silently
(``None`` for reads, a no-op for writes). Writes are atomic via a ``.json.tmp``
rename so concurrent readers never observe a partial write.

Typical usage (inside ``verify()``)::

    cached = get(content_hash, page_idx, model_id, prompt_version)
    if cached is not None:
        page_verdict = cached            # skip render + client call
    else:
        page_verdict = expensive_verify(...)
        put(content_hash, page_idx, model_id, prompt_version, page_verdict)
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from hybrid_doc_parser.models import PageVerification


def _cache_dir() -> Path:
    """Return the verification cache directory resolved from the environment.

    Reads ``HYBRID_DOC_PARSER_VERIFIER_CACHE_DIR``; falls back to
    ``~/.cache/hybrid_doc_parser/verifier`` when the variable is absent. The
    fallback nests under the same base as the parse cache
    (``~/.cache/hybrid_doc_parser``) but in a SEPARATE ``verifier`` subdirectory
    so the two caches never collide or piggyback on one another.

    Returns:
        Expanded ``Path`` to the cache directory. The directory may not exist
        yet — callers that need it to exist must create it themselves.
    """
    value = os.environ.get("HYBRID_DOC_PARSER_VERIFIER_CACHE_DIR")
    if value:
        return Path(value).expanduser()
    base = os.environ.get(
        "HYBRID_DOC_PARSER_CACHE_DIR", "~/.cache/hybrid_doc_parser"
    )
    return Path(base).expanduser() / "verifier"


def _cache_key(
    content_hash: str, page_idx: int, model_id: str, prompt_version: str
) -> str:
    """Compute a stable cache key from the verification 4-tuple.

    The key is the SHA-256 digest of the joined tuple, so any change to the
    file content hash, page index, model id, or prompt version yields a new key
    (and thus a cache miss / bust).

    Args:
        content_hash: ``ParserOutput.file_sha256`` of the verified document.
        page_idx: Zero-indexed page the verdict covers.
        model_id: ``VerifierConfig.model`` used to produce the verdict.
        prompt_version: The verifier ``PROMPT_VERSION`` used for the run.

    Returns:
        A 64-character lowercase hex SHA-256 digest of the 4-tuple.
    """
    # NUL-join so component boundaries are unambiguous (a NUL byte cannot appear
    # in any of the textual components), preventing key collisions between, e.g.,
    # ("ab", 1) and ("a", "b1").
    raw = "\x00".join([content_hash, str(page_idx), model_id, prompt_version])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


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


def get(
    content_hash: str, page_idx: int, model_id: str, prompt_version: str
) -> PageVerification | None:
    """Return a cached page verdict for the 4-tuple, or ``None`` on any miss.

    A miss occurs when:
    - The cache file does not exist.
    - The stored ``_cache_key`` does not match the key computed from the
      4-tuple (a defensive guard against the rare ``key[:16]`` filename
      collision).
    - Any exception is raised during reading or deserialisation.

    Args:
        content_hash: ``ParserOutput.file_sha256`` of the verified document.
        page_idx: Zero-indexed page the verdict covers.
        model_id: ``VerifierConfig.model`` used to produce the verdict.
        prompt_version: The verifier ``PROMPT_VERSION`` used for the run.

    Returns:
        A validated :class:`PageVerification` on cache hit, ``None`` on any miss
        or error. Never raises.
    """
    try:
        key = _cache_key(content_hash, page_idx, model_id, prompt_version)
        path = _cache_path(key)
        if not path.exists():
            return None
        text = path.read_text(encoding="utf-8")
        data = json.loads(text)
        if data.get("_cache_key") != key:
            return None
        # NOTE Strip internal bookkeeping fields before validation so the
        # Pydantic model does not see unknown keys.
        data.pop("_cache_key", None)
        data.pop("_cached_at", None)
        return PageVerification.model_validate(data)
    except Exception as exc:
        logger.debug(
            "[verifier_cache] get miss for ({}, {}, {}, {}): {}",
            content_hash,
            page_idx,
            model_id,
            prompt_version,
            exc,
        )
        return None


def put(
    content_hash: str,
    page_idx: int,
    model_id: str,
    prompt_version: str,
    verdict: PageVerification,
) -> None:
    """Persist a page verdict to the verification cache. Never raises.

    Writes are atomic: the JSON payload is first flushed to a ``.json.tmp``
    sibling file and then renamed over the target ``.json`` file, so concurrent
    readers never observe a partial write.

    Args:
        content_hash: ``ParserOutput.file_sha256`` of the verified document.
        page_idx: Zero-indexed page the verdict covers.
        model_id: ``VerifierConfig.model`` used to produce the verdict.
        prompt_version: The verifier ``PROMPT_VERSION`` used for the run.
        verdict: The :class:`PageVerification` to store.
    """
    try:
        key = _cache_key(content_hash, page_idx, model_id, prompt_version)
        cache_file = _cache_path(key)
        directory = _cache_dir()
        directory.mkdir(parents=True, exist_ok=True)

        data = json.loads(verdict.model_dump_json())
        data["_cache_key"] = key
        data["_cached_at"] = datetime.now(tz=timezone.utc).isoformat()

        json_text = json.dumps(data, ensure_ascii=False)
        # NOTE Write to a .tmp sibling first, then rename for atomicity.
        tmp_path = cache_file.with_suffix(".json.tmp")
        tmp_path.write_text(json_text, encoding="utf-8")
        tmp_path.rename(cache_file)
    except Exception as exc:
        logger.warning(
            "[verifier_cache] write error for ({}, {}, {}, {}): {}",
            content_hash,
            page_idx,
            model_id,
            prompt_version,
            exc,
        )
