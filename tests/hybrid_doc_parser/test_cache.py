"""Tests for cache.py file-based parse result cache."""

from __future__ import annotations

import json
import unittest.mock as mock
from pathlib import Path

import pytest

from hybrid_doc_parser.models import EnrichmentConfig, ParserOutput


def make_output(file_path: str = "/tmp/test.pdf") -> ParserOutput:
    """Construct a minimal valid ParserOutput for cache round-trip tests.

    Args:
        file_path: String path to embed in the output.

    Returns:
        A minimal ParserOutput with no pages, elements, or warnings.
    """
    return ParserOutput(
        file_path=file_path,
        file_sha256="a" * 64,
        page_count=1,
        pages=[],
        elements=[],
        warnings=[],
        enrichment_config=EnrichmentConfig(),
    )


def make_pdf(tmp_path: Path, name: str = "test.pdf", content: bytes = b"pdf content") -> Path:
    """Write a fake PDF file to tmp_path and return its Path.

    Args:
        tmp_path: Pytest tmp_path fixture directory.
        name: Filename for the fake PDF.
        content: Byte content to write.

    Returns:
        Path to the created file.
    """
    p = tmp_path / name
    p.write_bytes(content)
    return p


def test_cache_key_format(tmp_path):
    """_cache_key returns a string matching '{32 hex chars}_{integer}'."""
    import re

    from hybrid_doc_parser.cache import _cache_key

    p = make_pdf(tmp_path)
    key = _cache_key(p)
    assert re.match(r"[0-9a-f]{32}_\d+", key)


def test_cache_path_suffix(monkeypatch, tmp_path):
    """_cache_path returns a .json file named by the first 16 chars of the key."""
    monkeypatch.setenv("HYBRID_DOC_PARSER_CACHE_DIR", str(tmp_path))
    from hybrid_doc_parser.cache import _cache_path

    key = "abcdef1234567890abcdef1234567890_1234567890"
    path = _cache_path(key)
    assert path.suffix == ".json"
    assert path.name == key[:16] + ".json"


def test_put_get_roundtrip(monkeypatch, tmp_path):
    """put followed by get returns an equivalent ParserOutput."""
    cache_dir = tmp_path / "cache"
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    monkeypatch.setenv("HYBRID_DOC_PARSER_CACHE_DIR", str(cache_dir))

    from hybrid_doc_parser.cache import get, put

    p = source_dir / "doc.pdf"
    p.write_bytes(b"pdf content here")
    output = make_output(str(p))
    put(p, output)
    result = get(p)
    assert result is not None
    assert result.file_sha256 == output.file_sha256
    assert result.page_count == output.page_count


def test_get_miss(monkeypatch, tmp_path):
    """get returns None when no cache entry exists for the file."""
    cache_dir = tmp_path / "cache"
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    monkeypatch.setenv("HYBRID_DOC_PARSER_CACHE_DIR", str(cache_dir))

    from hybrid_doc_parser.cache import get

    p = source_dir / "doc.pdf"
    p.write_bytes(b"content")
    result = get(p)
    assert result is None


def test_get_key_mismatch(monkeypatch, tmp_path):
    """get returns None when the stored _cache_key does not match the computed key."""
    cache_dir = tmp_path / "cache"
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    monkeypatch.setenv("HYBRID_DOC_PARSER_CACHE_DIR", str(cache_dir))

    from hybrid_doc_parser.cache import _cache_key, _cache_path, get, put

    p = source_dir / "doc.pdf"
    p.write_bytes(b"content")
    output = make_output(str(p))
    put(p, output)

    # Corrupt the _cache_key in the stored file
    key = _cache_key(p)
    cache_file = _cache_path(key)
    data = json.loads(cache_file.read_text())
    data["_cache_key"] = "wrong_key_0000000_9999999"
    cache_file.write_text(json.dumps(data))

    result = get(p)
    assert result is None


def test_put_never_raises(monkeypatch, tmp_path):
    """put does not propagate exceptions even when mkdir fails with PermissionError."""
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    monkeypatch.setenv("HYBRID_DOC_PARSER_CACHE_DIR", str(tmp_path / "cache"))

    from hybrid_doc_parser.cache import put

    p = source_dir / "doc.pdf"
    p.write_bytes(b"content")
    output = make_output(str(p))

    with mock.patch("pathlib.Path.mkdir", side_effect=PermissionError("no write")):
        # Must not raise
        put(p, output)


def test_invalidate(monkeypatch, tmp_path):
    """invalidate deletes the cache entry so subsequent get returns None."""
    cache_dir = tmp_path / "cache"
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    monkeypatch.setenv("HYBRID_DOC_PARSER_CACHE_DIR", str(cache_dir))

    from hybrid_doc_parser.cache import get, invalidate, put

    p = source_dir / "doc.pdf"
    p.write_bytes(b"content")
    output = make_output(str(p))
    put(p, output)
    assert get(p) is not None
    invalidate(p)
    assert get(p) is None
