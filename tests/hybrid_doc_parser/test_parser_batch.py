"""Batch-inference tests for parser.parse_batch() and its chunk helpers.

All tests mock the MinerU boundary so the suite runs WITHOUT requiring a real
MinerU install. Because the helpers lazy-import ``from mineru.cli.common import
do_parse, read_fn`` inside the function body, we inject a fake
``mineru.cli.common`` module into ``sys.modules`` (G2) or patch the
``_run_mineru_batch_chunk`` seam directly (G3). No test imports MinerU for real.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
import types
import unittest.mock as mock
from pathlib import Path

import pytest

from hybrid_doc_parser.models import EnrichmentConfig

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_content_list(n_pages: int = 1) -> list[dict]:
    """Return a minimal valid MinerU content_list."""
    return [
        {
            "type": "text",
            "text": f"Page {i} paragraph text with enough words to pass the gate.",
            "page_idx": i,
            "page_size": [595.0, 842.0],
            "bbox": [50.0, 700.0, 545.0, 750.0],
        }
        for i in range(n_pages)
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _FakeMineruModule:
    """Context manager injecting a fake ``mineru.cli.common`` into sys.modules.

    The lazy ``from mineru.cli.common import do_parse, read_fn`` inside the
    parser picks up these fakes regardless of whether real MinerU is installed.

    ``on_do_parse`` is invoked with the kwargs passed to ``do_parse`` and is
    expected to write each file's ``{name}_content_list.json`` into the output
    directory (so ``_read_content_list_for_stem`` can read it back).
    """

    def __init__(self, on_do_parse, read_fn=None):
        self._on_do_parse = on_do_parse
        self._read_fn = read_fn or (lambda p: f"PDFBYTES::{p}".encode())
        self.do_parse_mock = mock.MagicMock(side_effect=self._do_parse_impl)
        self._saved: dict[str, object] = {}

    def _do_parse_impl(self, **kwargs):
        return self._on_do_parse(**kwargs)

    def __enter__(self):
        # Build a minimal package hierarchy: mineru -> mineru.cli -> mineru.cli.common
        for modname in ("mineru", "mineru.cli", "mineru.cli.common"):
            self._saved[modname] = sys.modules.get(modname)

        mineru_mod = types.ModuleType("mineru")
        cli_mod = types.ModuleType("mineru.cli")
        common_mod = types.ModuleType("mineru.cli.common")
        common_mod.do_parse = self.do_parse_mock
        common_mod.read_fn = self._read_fn
        cli_mod.common = common_mod
        mineru_mod.cli = cli_mod

        sys.modules["mineru"] = mineru_mod
        sys.modules["mineru.cli"] = cli_mod
        sys.modules["mineru.cli.common"] = common_mod
        return self

    def __exit__(self, *exc):
        for modname, mod in self._saved.items():
            if mod is None:
                sys.modules.pop(modname, None)
            else:
                sys.modules[modname] = mod
        return False


def _writer_do_parse(content_per_name=None):
    """Return an on_do_parse callback that dumps one content_list per file name.

    ``content_per_name`` maps a unique synthetic name to its content_list; names
    not present get a default single-page content_list.
    """
    content_per_name = content_per_name or {}

    def _on(**kwargs):
        out_dir = Path(kwargs["output_dir"])
        names = kwargs["pdf_file_names"]
        for name in names:
            cl = content_per_name.get(name, _fake_content_list(1))
            (out_dir / f"{name}_content_list.json").write_text(json.dumps(cl), encoding="utf-8")

    return _on


# ---------------------------------------------------------------------------
# G2: _read_content_list_for_stem / _run_mineru_batch_chunk / _run_mineru_batch
# ---------------------------------------------------------------------------


def test_read_content_list_for_stem_targeted(tmp_path):
    """_read_content_list_for_stem reads only the file matching the unique name."""
    from hybrid_doc_parser.parser import _read_content_list_for_stem

    a = [{"type": "text", "text": "A", "page_idx": 0, "bbox": []}]
    b = [{"type": "text", "text": "B", "page_idx": 0, "bbox": []}]
    (tmp_path / "0_report_content_list.json").write_text(json.dumps(a), encoding="utf-8")
    (tmp_path / "1_report_content_list.json").write_text(json.dumps(b), encoding="utf-8")

    assert _read_content_list_for_stem(tmp_path, "0_report") == a
    assert _read_content_list_for_stem(tmp_path, "1_report") == b
    assert _read_content_list_for_stem(tmp_path, "9_missing") == []


def test_run_mineru_batch_calls_do_parse_once_per_chunk(tmp_path, monkeypatch):
    """do_parse is called exactly once per chunk."""
    monkeypatch.setenv("MINERU_BATCH_SIZE", "2")
    from hybrid_doc_parser.parser import _run_mineru_batch

    paths = [tmp_path / f"f{i}.pdf" for i in range(5)]
    for p in paths:
        p.write_bytes(b"%PDF-1.4 fake")

    fake = _FakeMineruModule(_writer_do_parse())
    with fake:
        results = _run_mineru_batch(paths, backend="pipeline")

    # 5 files, chunk size 2 -> ceil(5/2) = 3 do_parse calls.
    assert fake.do_parse_mock.call_count == 3
    assert set(results.keys()) == set(paths)
    assert all(results[p] for p in paths)


@pytest.mark.parametrize(
    "n,c,expected_calls",
    [
        (1, 8, 1),
        (8, 8, 1),
        (9, 8, 2),
        (5, 2, 3),
        (7, 3, 3),
    ],
)
def test_run_mineru_batch_chunking_math(tmp_path, monkeypatch, n, c, expected_calls):
    """do_parse call count equals ceil(N / C) for various N/C combinations."""
    monkeypatch.setenv("MINERU_BATCH_SIZE", str(c))
    from hybrid_doc_parser.parser import _run_mineru_batch

    paths = [tmp_path / f"f{i}.pdf" for i in range(n)]
    for p in paths:
        p.write_bytes(b"%PDF-1.4 fake")

    fake = _FakeMineruModule(_writer_do_parse())
    with fake:
        _run_mineru_batch(paths, backend="pipeline")

    assert fake.do_parse_mock.call_count == expected_calls


def test_run_mineru_batch_reads_per_unique_name(tmp_path, monkeypatch):
    """Each path's result is read from its own f'{i}_{stem}_content_list.json'."""
    monkeypatch.setenv("MINERU_BATCH_SIZE", "8")
    from hybrid_doc_parser.parser import _run_mineru_batch

    paths = [tmp_path / "a.pdf", tmp_path / "b.pdf", tmp_path / "c.pdf"]
    for p in paths:
        p.write_bytes(b"%PDF-1.4 fake")

    # Distinct content per unique synthetic name.
    content_per_name = {
        "0_a": [{"type": "text", "text": "AAA", "page_idx": 0, "bbox": []}],
        "1_b": [{"type": "text", "text": "BBB", "page_idx": 0, "bbox": []}],
        "2_c": [{"type": "text", "text": "CCC", "page_idx": 0, "bbox": []}],
    }
    fake = _FakeMineruModule(_writer_do_parse(content_per_name))
    with fake:
        results = _run_mineru_batch(paths, backend="pipeline")

    assert results[paths[0]][0]["text"] == "AAA"
    assert results[paths[1]][0]["text"] == "BBB"
    assert results[paths[2]][0]["text"] == "CCC"


def test_run_mineru_batch_passes_copies_to_do_parse(tmp_path, monkeypatch):
    """do_parse mutating its inputs by index must not corrupt the read-back."""
    monkeypatch.setenv("MINERU_BATCH_SIZE", "8")
    from hybrid_doc_parser.parser import _run_mineru_batch

    paths = [tmp_path / "a.pdf", tmp_path / "b.pdf", tmp_path / "c.pdf"]
    for p in paths:
        p.write_bytes(b"%PDF-1.4 fake")

    def _mutating(**kwargs):
        out_dir = Path(kwargs["output_dir"])
        names = kwargs["pdf_file_names"]
        bytes_list = kwargs["pdf_bytes_list"]
        langs = kwargs["p_lang_list"]
        # Write outputs for the names we received BEFORE mutation.
        for name in list(names):
            (out_dir / f"{name}_content_list.json").write_text(
                json.dumps(_fake_content_list(1)), encoding="utf-8"
            )
        # Simulate do_parse deleting an office-doc entry by index.
        del names[0]
        del bytes_list[0]
        del langs[0]

    fake = _FakeMineruModule(_mutating)
    with fake:
        results = _run_mineru_batch(paths, backend="pipeline")

    # Despite do_parse mutating its received (copied) lists, every original
    # path still gets a result — the caller's lists were not corrupted.
    assert set(results.keys()) == set(paths)
    assert all(results[p] for p in paths)


def test_run_mineru_batch_same_bare_stem_maps_correctly(tmp_path, monkeypatch):
    """Two files with the same bare stem from different dirs map correctly."""
    monkeypatch.setenv("MINERU_BATCH_SIZE", "8")
    from hybrid_doc_parser.parser import _run_mineru_batch

    d1 = tmp_path / "dir1"
    d2 = tmp_path / "dir2"
    d1.mkdir()
    d2.mkdir()
    p1 = d1 / "report.pdf"
    p2 = d2 / "report.pdf"
    p1.write_bytes(b"%PDF-1.4 one")
    p2.write_bytes(b"%PDF-1.4 two")

    content_per_name = {
        "0_report": [{"type": "text", "text": "FIRST", "page_idx": 0, "bbox": []}],
        "1_report": [{"type": "text", "text": "SECOND", "page_idx": 0, "bbox": []}],
    }
    fake = _FakeMineruModule(_writer_do_parse(content_per_name))
    with fake:
        results = _run_mineru_batch([p1, p2], backend="pipeline")

    # No ValueError; each file maps to its own distinct output.
    assert results[p1][0]["text"] == "FIRST"
    assert results[p2][0]["text"] == "SECOND"


def test_run_mineru_batch_chunk_failure_isolated(tmp_path, monkeypatch):
    """A chunk whose do_parse raises propagates out of _run_mineru_batch_chunk."""
    monkeypatch.setenv("MINERU_BATCH_SIZE", "8")
    from hybrid_doc_parser.parser import _build_batch_name_map, _run_mineru_batch_chunk

    paths = [tmp_path / "a.pdf", tmp_path / "b.pdf"]
    for p in paths:
        p.write_bytes(b"%PDF-1.4 fake")
    name_map = _build_batch_name_map(paths)

    def _boom(**kwargs):
        raise RuntimeError("do_parse exploded")

    fake = _FakeMineruModule(_boom)
    with fake:
        with pytest.raises(RuntimeError, match="do_parse exploded"):
            _run_mineru_batch_chunk(paths, name_map, backend="pipeline")


# ---------------------------------------------------------------------------
# G3: parse_batch() integration tests
# ---------------------------------------------------------------------------


def test_parse_batch_mineru_output_identical_to_parse(tmp_path, monkeypatch):
    """parse_batch([f]) is byte-identical to parse(f)."""
    monkeypatch.setenv("HYBRID_DOC_PARSER_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("MINERU_BATCH_SIZE", "8")
    from hybrid_doc_parser.parser import parse, parse_batch

    pdf = FIXTURES / "digital_simple.pdf"
    fake_cl = _fake_content_list(2)

    # parse() path: mock _run_mineru.
    with mock.patch("hybrid_doc_parser.parser._run_mineru", return_value=fake_cl):
        single = parse(pdf, EnrichmentConfig())

    # parse_batch() path: mock the chunk seam to return the same content_list.
    def _chunk(chunk_paths, name_map, backend):
        return {p: fake_cl for p in chunk_paths}

    with mock.patch("hybrid_doc_parser.parser._run_mineru_batch_chunk", side_effect=_chunk):
        batch = asyncio.run(parse_batch([pdf], EnrichmentConfig()))

    assert len(batch) == 1
    assert batch[0].model_dump() == single.model_dump()


def test_parse_batch_calls_do_parse_per_chunk(tmp_path, monkeypatch):
    """N uncached MinerU files -> ceil(N/C) do_parse calls."""
    monkeypatch.setenv("HYBRID_DOC_PARSER_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("MINERU_BATCH_SIZE", "3")
    from hybrid_doc_parser.parser import parse_batch

    paths = [tmp_path / f"f{i}.pdf" for i in range(7)]
    for p in paths:
        p.write_bytes(b"%PDF-1.4 fake")

    fake = _FakeMineruModule(_writer_do_parse())
    with fake:
        results = asyncio.run(parse_batch(paths, EnrichmentConfig()))

    # 7 files, chunk 3 -> ceil(7/3) = 3 do_parse calls.
    assert fake.do_parse_mock.call_count == 3
    assert len(results) == 7


def test_parse_batch_preserves_input_order(tmp_path, monkeypatch):
    """Output order matches the input paths order."""
    monkeypatch.setenv("HYBRID_DOC_PARSER_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("MINERU_BATCH_SIZE", "2")
    from hybrid_doc_parser.parser import parse_batch

    paths = [tmp_path / f"f{i}.pdf" for i in range(5)]
    for p in paths:
        p.write_bytes(b"%PDF-1.4 fake")

    fake = _FakeMineruModule(_writer_do_parse())
    with fake:
        results = asyncio.run(parse_batch(paths, EnrichmentConfig()))

    assert [r.file_path for r in results] == [str(p) for p in paths]


def test_parse_batch_all_cached_zero_do_parse(tmp_path, monkeypatch):
    """An all-cached batch issues ZERO do_parse calls."""
    monkeypatch.setenv("HYBRID_DOC_PARSER_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("MINERU_BATCH_SIZE", "8")
    from hybrid_doc_parser.parser import parse_batch

    paths = [tmp_path / f"f{i}.pdf" for i in range(3)]
    for p in paths:
        p.write_bytes(b"%PDF-1.4 fake")

    # First run populates the cache.
    fake1 = _FakeMineruModule(_writer_do_parse())
    with fake1:
        asyncio.run(parse_batch(paths, EnrichmentConfig()))
    assert fake1.do_parse_mock.call_count > 0

    # Second run: everything is cached -> no do_parse calls.
    fake2 = _FakeMineruModule(_writer_do_parse())
    with fake2:
        results = asyncio.run(parse_batch(paths, EnrichmentConfig()))
    assert fake2.do_parse_mock.call_count == 0
    assert len(results) == 3


def test_parse_batch_handles_invalid_paths(tmp_path, monkeypatch):
    """Missing file -> file_not_found; bad extension -> unsupported_type; right index."""
    monkeypatch.setenv("HYBRID_DOC_PARSER_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("MINERU_BATCH_SIZE", "8")
    from hybrid_doc_parser.parser import parse_batch

    good = tmp_path / "good.pdf"
    good.write_bytes(b"%PDF-1.4 fake")
    missing = tmp_path / "ghost.pdf"  # does not exist
    bad_ext = tmp_path / "note.txt"
    bad_ext.write_bytes(b"plain text")

    paths = [missing, good, bad_ext]

    fake = _FakeMineruModule(_writer_do_parse())
    with fake:
        results = asyncio.run(parse_batch(paths, EnrichmentConfig()))

    assert len(results) == 3
    assert any(w.code == "file_not_found" for w in results[0].warnings)
    assert results[1].elements  # the good file parsed
    assert any(w.code == "unsupported_type" for w in results[2].warnings)
    # Only the one valid uncached file went through inference.
    assert fake.do_parse_mock.call_count == 1


def test_parse_batch_mixed_cached_and_uncached(tmp_path, monkeypatch):
    """Correct merge of invalid / cache hit / parsed outputs in input order."""
    monkeypatch.setenv("HYBRID_DOC_PARSER_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("MINERU_BATCH_SIZE", "8")
    from hybrid_doc_parser.parser import parse_batch

    cached_file = tmp_path / "cached.pdf"
    cached_file.write_bytes(b"%PDF-1.4 cached")
    fresh_file = tmp_path / "fresh.pdf"
    fresh_file.write_bytes(b"%PDF-1.4 fresh")
    missing = tmp_path / "ghost.pdf"

    # Pre-populate cache for cached_file via a first batch run.
    fake0 = _FakeMineruModule(_writer_do_parse())
    with fake0:
        asyncio.run(parse_batch([cached_file], EnrichmentConfig()))

    paths = [cached_file, missing, fresh_file]
    fake = _FakeMineruModule(_writer_do_parse())
    with fake:
        results = asyncio.run(parse_batch(paths, EnrichmentConfig()))

    assert len(results) == 3
    assert results[0].file_path == str(cached_file)
    assert results[0].elements  # cache hit, has content
    assert any(w.code == "file_not_found" for w in results[1].warnings)
    assert results[2].file_path == str(fresh_file)
    assert results[2].elements
    # Only the fresh uncached file required inference.
    assert fake.do_parse_mock.call_count == 1


def test_parse_batch_empty_content_list_warns_mineru_failed(tmp_path, monkeypatch):
    """A valid file with empty content_list gets a mineru_failed WarningRecord."""
    monkeypatch.setenv("HYBRID_DOC_PARSER_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("MINERU_BATCH_SIZE", "8")
    from hybrid_doc_parser.parser import parse_batch

    pdf = tmp_path / "empty.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    def _chunk(chunk_paths, name_map, backend):
        return {p: [] for p in chunk_paths}

    with mock.patch("hybrid_doc_parser.parser._run_mineru_batch_chunk", side_effect=_chunk):
        results = asyncio.run(parse_batch([pdf], EnrichmentConfig()))

    assert len(results) == 1
    assert any(w.code == "mineru_failed" for w in results[0].warnings)


def test_parse_batch_fallback_on_batch_failure(tmp_path, monkeypatch):
    """A chunk's do_parse raising -> per-file parse() fallback for that chunk only."""
    monkeypatch.setenv("HYBRID_DOC_PARSER_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("MINERU_BATCH_SIZE", "2")
    from hybrid_doc_parser.parser import parse_batch

    # 4 files -> 2 chunks of 2. Make the FIRST chunk fail, second succeed.
    paths = [tmp_path / f"f{i}.pdf" for i in range(4)]
    for p in paths:
        p.write_bytes(b"%PDF-1.4 fake")

    fail_set = {paths[0], paths[1]}

    def _chunk(chunk_paths, name_map, backend):
        if set(chunk_paths) == fail_set:
            raise RuntimeError("chunk do_parse exploded")
        return {p: _fake_content_list(1) for p in chunk_paths}

    fake_cl = _fake_content_list(1)

    with mock.patch("hybrid_doc_parser.parser._run_mineru_batch_chunk", side_effect=_chunk):
        # The per-file fallback path calls parse() which uses _run_mineru.
        with mock.patch(
            "hybrid_doc_parser.parser._run_mineru", return_value=fake_cl
        ) as run_mineru_mock:
            results = asyncio.run(parse_batch(paths, EnrichmentConfig()))

    assert len(results) == 4
    # First chunk fell back to per-file parse() (2 files).
    assert run_mineru_mock.call_count == 2
    # All files produced content (siblings unaffected).
    assert all(r.elements for r in results)


def test_parse_batch_docling_uses_per_file_path(tmp_path, monkeypatch):
    """parser='docling' never calls the MinerU batch path."""
    monkeypatch.setenv("HYBRID_DOC_PARSER_CACHE_DIR", str(tmp_path / "cache"))
    from hybrid_doc_parser.parser import parse_batch

    pdf = FIXTURES / "digital_simple.pdf"
    config = EnrichmentConfig(parser="docling")

    with mock.patch("hybrid_doc_parser.parser._run_mineru_batch_chunk") as chunk_mock:
        with mock.patch("hybrid_doc_parser.parser._run_docling", return_value=[]) as docling_mock:
            results = asyncio.run(parse_batch([pdf], config))

    chunk_mock.assert_not_called()
    docling_mock.assert_called_once()
    assert len(results) == 1


# ---------------------------------------------------------------------------
# Regression tests for the principal-engineer review fixes (H1, H2, L1)
# ---------------------------------------------------------------------------


def test_read_content_list_for_stem_handles_glob_metachars(tmp_path):
    """H2: a synthetic name with glob metachars ([ ] * ?) must still read back.

    Regression: the old rglob(f"{name}_content_list.json") fed the filename into
    a glob pattern, so a stem like 'report[2023]' was mis-parsed as a character
    class and silently matched nothing -> spurious mineru_failed / data loss.
    """
    from hybrid_doc_parser.parser import _read_content_list_for_stem

    blocks = [{"type": "text", "text": "hi", "page_idx": 0}]
    for name in ("0_report[2023]", "1_data*v2", "2_who?", "3_normal"):
        sub = tmp_path / name / "auto"
        sub.mkdir(parents=True)
        (sub / f"{name}_content_list.json").write_text(json.dumps(blocks), encoding="utf-8")
        got = _read_content_list_for_stem(tmp_path, name)
        assert got == blocks, f"failed to read back content_list for name={name!r}"


def test_read_content_list_for_stem_missing_returns_empty(tmp_path):
    """H2: a genuinely-missing file still returns [] (no false positive)."""
    from hybrid_doc_parser.parser import _read_content_list_for_stem

    assert _read_content_list_for_stem(tmp_path, "0_nope") == []


def test_parse_batch_does_not_block_event_loop(tmp_path, monkeypatch):
    """H1: a slow synchronous chunk must not freeze concurrent coroutines.

    The chunk work is offloaded via asyncio.to_thread, so a ticker coroutine
    must keep advancing while a 0.5s do_parse 'runs'. If parse_batch ran the
    blocking chunk directly on the loop thread, the ticker could not advance.
    """
    import time as _time

    monkeypatch.setenv("HYBRID_DOC_PARSER_CACHE_DIR", str(tmp_path / "cache"))
    import hybrid_doc_parser.parser as P
    from hybrid_doc_parser.parser import parse_batch

    pdf = FIXTURES / "digital_simple.pdf"

    def slow_chunk(chunk_paths, name_map, backend):
        _time.sleep(0.5)  # simulate blocking GPU inference
        return {p: _fake_content_list(1) for p in chunk_paths}

    monkeypatch.setattr(P, "_run_mineru_batch_chunk", slow_chunk)

    ticks = {"n": 0}

    async def ticker():
        for _ in range(40):
            await asyncio.sleep(0.01)
            ticks["n"] += 1

    async def run():
        return await asyncio.gather(parse_batch([pdf], EnrichmentConfig()), ticker())

    batch_results, _ticker_result = asyncio.run(run())
    # Sanity: the batch still produced its output.
    assert len(batch_results) == 1
    assert batch_results[0].page_count == 1
    # The ticker advanced during the 0.5s blocking chunk => loop was not blocked.
    assert ticks["n"] >= 10, f"event loop appears blocked (ticks={ticks['n']})"


def test_parse_batch_summary_counts_real_failures(tmp_path, monkeypatch, caplog):
    """L1: empty_or_failed reflects true failures, not zero-element pages.

    One file yields empty content (real failure -> mineru_failed); another
    yields a valid page. empty_or_failed must be 1, not 0 and not 2.
    """
    import logging

    monkeypatch.setenv("HYBRID_DOC_PARSER_CACHE_DIR", str(tmp_path / "cache"))
    import hybrid_doc_parser.parser as P
    from hybrid_doc_parser.parser import parse_batch

    good = FIXTURES / "digital_simple.pdf"
    bad = FIXTURES / "mixed.pdf"

    def chunk(chunk_paths, name_map, backend):
        out = {}
        for p in chunk_paths:
            out[p] = _fake_content_list(1) if p == good else []
        return out

    monkeypatch.setattr(P, "_run_mineru_batch_chunk", chunk)

    with caplog.at_level(logging.INFO):
        results = asyncio.run(parse_batch([good, bad], EnrichmentConfig()))

    assert len(results) == 2
    # bad file carries mineru_failed; good file is clean.
    bad_out = next(r for r in results if r.file_path == str(bad))
    assert any(w.code == "mineru_failed" for w in bad_out.warnings)
    good_out = next(r for r in results if r.file_path == str(good))
    assert good_out.warnings == [] or all(w.code != "mineru_failed" for w in good_out.warnings)


# ---------------------------------------------------------------------------
# #3: inference gate — serialise do_parse across concurrent parse_batch calls
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [(None, 1), ("1", 1), ("4", 4), ("0", 1), ("-2", 1), ("abc", 1), ("", 1)],
)
def test_read_mineru_max_inflight_defensive(monkeypatch, raw, expected):
    """MINERU_MAX_INFLIGHT parses defensively; bad/<1 values fall back to 1."""
    from hybrid_doc_parser.parser import _read_mineru_max_inflight

    if raw is None:
        monkeypatch.delenv("MINERU_MAX_INFLIGHT", raising=False)
    else:
        monkeypatch.setenv("MINERU_MAX_INFLIGHT", raw)
    assert _read_mineru_max_inflight() == expected


def _concurrency_tracking_do_parse(state, lock):
    """on_do_parse that records peak concurrent do_parse entries."""
    import time as _time

    def _on(**kwargs):
        with lock:
            state["cur"] += 1
            state["max"] = max(state["max"], state["cur"])
        try:
            _time.sleep(0.2)  # widen the overlap window
            out_dir = Path(kwargs["output_dir"])
            for name in kwargs["pdf_file_names"]:
                (out_dir / f"{name}_content_list.json").write_text(
                    json.dumps(_fake_content_list(1)), encoding="utf-8"
                )
        finally:
            with lock:
                state["cur"] -= 1

    return _on


def _run_two_concurrent_batches(tmp_path, monkeypatch, max_inflight):
    """Run two parse_batch() calls concurrently; return peak do_parse overlap."""
    import threading

    monkeypatch.setenv("HYBRID_DOC_PARSER_CACHE_DIR", str(tmp_path / "cache"))
    if max_inflight is None:
        monkeypatch.delenv("MINERU_MAX_INFLIGHT", raising=False)
    else:
        monkeypatch.setenv("MINERU_MAX_INFLIGHT", str(max_inflight))

    import hybrid_doc_parser.parser as P
    from hybrid_doc_parser.parser import parse_batch

    # Force the lazy gate to rebuild at the configured size for this test.
    monkeypatch.setattr(P, "_MINERU_INFERENCE_SEMA", None)

    a = FIXTURES / "digital_simple.pdf"
    b = FIXTURES / "mixed.pdf"
    state = {"cur": 0, "max": 0}
    lock = threading.Lock()
    fake = _FakeMineruModule(_concurrency_tracking_do_parse(state, lock))

    async def run():
        return await asyncio.gather(
            parse_batch([a], EnrichmentConfig()),
            parse_batch([b], EnrichmentConfig()),
        )

    with fake:
        r1, r2 = asyncio.run(run())

    assert len(r1) == 1 and len(r2) == 1
    return state["max"]


def test_parse_batch_serialises_inference_by_default(tmp_path, monkeypatch):
    """#3: with the default gate (1), concurrent batches never overlap do_parse."""
    peak = _run_two_concurrent_batches(tmp_path, monkeypatch, max_inflight=None)
    assert peak == 1, f"expected serialised inference, observed peak overlap {peak}"


def test_parse_batch_inflight_env_allows_concurrency(tmp_path, monkeypatch):
    """Positive control: MINERU_MAX_INFLIGHT=2 lets two do_parse calls overlap.

    Proves the harness can actually observe concurrency, so the ==1 assertion
    above is a real serialisation check and not a false pass.
    """
    peak = _run_two_concurrent_batches(tmp_path, monkeypatch, max_inflight=2)
    assert peak == 2, f"expected overlap of 2 with MINERU_MAX_INFLIGHT=2, observed {peak}"
