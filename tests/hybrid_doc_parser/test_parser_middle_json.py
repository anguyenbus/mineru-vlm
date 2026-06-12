"""Tests for the middle_json capture capability (Deliverable A, items 20).

Two focused groups:

- Group 1: ``_read_middle_json_for_stem`` — the non-raising read-back helper
  mirroring ``_read_content_list_for_stem`` (deterministic path then literal
  recursive fallback; ``None`` on miss/corrupt; never raises).
- Group 2: the ``f_dump_middle_json=True`` capture wiring in
  ``_run_mineru_inprocess`` and ``_run_mineru_batch_chunk``.

All tests mock the MinerU boundary (no real MinerU run). The fake injects a
``mineru.cli.common`` module into ``sys.modules`` so the lazy
``from mineru.cli.common import do_parse, read_fn`` inside the parser picks it
up regardless of whether real MinerU is installed.
"""

from __future__ import annotations

import json
import sys
import types
import unittest.mock as mock
from pathlib import Path

# ---------------------------------------------------------------------------
# Group 1: _read_middle_json_for_stem
# ---------------------------------------------------------------------------


def test_read_middle_json_for_stem_deterministic_path(tmp_path):
    """Reads the middle_json via the deterministic {dir}/{name}/auto/ path."""
    from hybrid_doc_parser.parser import _read_middle_json_for_stem

    name = "0_report"
    middle = {"pdf_info": [{"page_idx": 0, "para_blocks": []}]}
    sub = tmp_path / name / "auto"
    sub.mkdir(parents=True)
    (sub / f"{name}_middle.json").write_text(json.dumps(middle), encoding="utf-8")

    assert _read_middle_json_for_stem(tmp_path, name) == middle


def test_read_middle_json_for_stem_recursive_fallback(tmp_path):
    """Finds the file via the literal recursive rglob + exact p.name compare."""
    from hybrid_doc_parser.parser import _read_middle_json_for_stem

    name = "1_data"
    middle = {"pdf_info": [{"page_idx": 0}], "_backend": "pipeline"}
    # Not at the deterministic path — somewhere deeper in the tree.
    deep = tmp_path / "weird" / "layout"
    deep.mkdir(parents=True)
    (deep / f"{name}_middle.json").write_text(json.dumps(middle), encoding="utf-8")

    assert _read_middle_json_for_stem(tmp_path, name) == middle


def test_read_middle_json_for_stem_handles_glob_metachars(tmp_path):
    """A synthetic name with glob metachars ([ ] * ?) must still read back.

    Regression guard mirroring the content_list helper: ``name`` must never be
    fed into a glob pattern.
    """
    from hybrid_doc_parser.parser import _read_middle_json_for_stem

    middle = {"pdf_info": [{"page_idx": 0}]}
    for name in ("0_report[2023]", "1_data*v2", "2_who?", "3_normal"):
        sub = tmp_path / name / "auto"
        sub.mkdir(parents=True)
        (sub / f"{name}_middle.json").write_text(json.dumps(middle), encoding="utf-8")
        assert _read_middle_json_for_stem(tmp_path, name) == middle


def test_read_middle_json_for_stem_missing_returns_none(tmp_path):
    """A genuinely-missing file returns None (no raise)."""
    from hybrid_doc_parser.parser import _read_middle_json_for_stem

    assert _read_middle_json_for_stem(tmp_path, "0_nope") is None


def test_read_middle_json_for_stem_corrupt_returns_none(tmp_path):
    """Unparseable JSON returns None and does NOT raise."""
    from hybrid_doc_parser.parser import _read_middle_json_for_stem

    name = "0_corrupt"
    sub = tmp_path / name / "auto"
    sub.mkdir(parents=True)
    (sub / f"{name}_middle.json").write_text("{not valid json", encoding="utf-8")

    assert _read_middle_json_for_stem(tmp_path, name) is None


def test_read_middle_json_for_stem_non_dict_returns_none(tmp_path):
    """A JSON value that is not a dict (e.g. a list) returns None."""
    from hybrid_doc_parser.parser import _read_middle_json_for_stem

    name = "0_list"
    sub = tmp_path / name / "auto"
    sub.mkdir(parents=True)
    (sub / f"{name}_middle.json").write_text(json.dumps([1, 2, 3]), encoding="utf-8")

    assert _read_middle_json_for_stem(tmp_path, name) is None


# ---------------------------------------------------------------------------
# Group 2: capture wiring in _run_mineru_inprocess / _run_mineru_batch_chunk
# ---------------------------------------------------------------------------


class _FakeMineruModule:
    """Context manager injecting a fake ``mineru.cli.common`` into sys.modules.

    ``on_do_parse`` is invoked with the kwargs passed to ``do_parse``; it is
    expected to write each file's dump artefacts into the output directory.
    """

    def __init__(self, on_do_parse, read_fn=None):
        self._on_do_parse = on_do_parse
        self._read_fn = read_fn or (lambda p: f"PDFBYTES::{p}".encode())
        self.do_parse_mock = mock.MagicMock(side_effect=self._do_parse_impl)
        self._saved: dict[str, object] = {}

    def _do_parse_impl(self, **kwargs):
        return self._on_do_parse(**kwargs)

    def __enter__(self):
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


def _fake_content_list(n_pages: int = 1) -> list[dict]:
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


def _fast_path_flags_ok(kwargs: dict) -> None:
    """Assert the fast-path flags stay False and content_list/middle stay True."""
    assert kwargs["f_dump_middle_json"] is True
    assert kwargs["f_dump_content_list"] is True
    assert kwargs["f_dump_md"] is False
    assert kwargs["f_dump_model_output"] is False
    assert kwargs["f_dump_orig_pdf"] is False
    assert kwargs["f_draw_layout_bbox"] is False
    assert kwargs["f_draw_span_bbox"] is False


def test_inprocess_flips_middle_json_flag_and_keeps_fast_path(tmp_path):
    """_run_mineru_inprocess sets f_dump_middle_json=True; fast path intact."""
    from hybrid_doc_parser.parser import _run_mineru_inprocess

    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    def _on(**kwargs):
        _fast_path_flags_ok(kwargs)
        out_dir = Path(kwargs["output_dir"])
        name = kwargs["pdf_file_names"][0]
        (out_dir / f"{name}_content_list.json").write_text(
            json.dumps(_fake_content_list(1)), encoding="utf-8"
        )
        (out_dir / f"{name}_middle.json").write_text(
            json.dumps({"pdf_info": [{"page_idx": 0}]}), encoding="utf-8"
        )

    fake = _FakeMineruModule(_on)
    with fake:
        content_list, middle_json = _run_mineru_inprocess(pdf, backend="pipeline")

    # Content_list fast path is unchanged.
    assert content_list and content_list[0]["text"].startswith("Page 0")
    # The captured middle_json is now SURFACED alongside the content_list.
    assert middle_json == {"pdf_info": [{"page_idx": 0}]}
    fake.do_parse_mock.assert_called_once()


def test_batch_chunk_reads_middle_json_per_unique_name(tmp_path, monkeypatch):
    """_run_mineru_batch_chunk dumps + reads middle_json keyed on name_map[p]."""
    from hybrid_doc_parser.parser import _build_batch_name_map, _run_mineru_batch_chunk

    paths = [tmp_path / "a.pdf", tmp_path / "b.pdf"]
    for p in paths:
        p.write_bytes(b"%PDF-1.4 fake")
    name_map = _build_batch_name_map(paths)

    captured: dict[str, dict] = {}

    def _on(**kwargs):
        _fast_path_flags_ok(kwargs)
        out_dir = Path(kwargs["output_dir"])
        for name in kwargs["pdf_file_names"]:
            (out_dir / f"{name}_content_list.json").write_text(
                json.dumps(_fake_content_list(1)), encoding="utf-8"
            )
            middle = {"pdf_info": [{"page_idx": 0}], "_name": name}
            (out_dir / f"{name}_middle.json").write_text(json.dumps(middle), encoding="utf-8")

    # Capture the middle_json the chunk reads back via the helper, keyed by name.
    import hybrid_doc_parser.parser as P

    real_reader = P._read_middle_json_for_stem

    def _spy(out_dir, name):
        got = real_reader(out_dir, name)
        if got is not None:
            captured[name] = got
        return got

    monkeypatch.setattr(P, "_read_middle_json_for_stem", _spy)

    fake = _FakeMineruModule(_on)
    with fake:
        results = _run_mineru_batch_chunk(paths, name_map, backend="pipeline")

    # Content_list read-back fast path is unchanged; each value is now a pair.
    assert set(results.keys()) == set(paths)
    assert all(results[p][0] for p in paths)
    # The middle_json is surfaced as the second element of each pair.
    assert results[paths[0]][1]["_name"] == name_map[paths[0]]
    assert results[paths[1]][1]["_name"] == name_map[paths[1]]
    # The middle_json was read back keyed on the SAME synthetic name.
    assert captured[name_map[paths[0]]]["_name"] == name_map[paths[0]]
    assert captured[name_map[paths[1]]]["_name"] == name_map[paths[1]]


def test_batch_chunk_missing_middle_json_does_not_break_content_list(tmp_path):
    """A missing/corrupt middle_json degrades to None; content_list still returns."""
    from hybrid_doc_parser.parser import _build_batch_name_map, _run_mineru_batch_chunk

    paths = [tmp_path / "a.pdf"]
    paths[0].write_bytes(b"%PDF-1.4 fake")
    name_map = _build_batch_name_map(paths)

    def _on(**kwargs):
        _fast_path_flags_ok(kwargs)
        out_dir = Path(kwargs["output_dir"])
        # Write content_list only; deliberately NO middle.json dumped.
        for name in kwargs["pdf_file_names"]:
            (out_dir / f"{name}_content_list.json").write_text(
                json.dumps(_fake_content_list(1)), encoding="utf-8"
            )

    fake = _FakeMineruModule(_on)
    with fake:
        results = _run_mineru_batch_chunk(paths, name_map, backend="pipeline")

    # The content_list return path is unaffected by the absent middle_json;
    # the value is a (content_list, None) pair when no middle_json was captured.
    content_list, middle_json = results[paths[0]]
    assert content_list
    assert middle_json is None
    assert content_list[0]["text"].startswith("Page 0")


# ---------------------------------------------------------------------------
# Group 3 (item 22): _split_mineru_result tolerant "list-or-pair" unpack
# ---------------------------------------------------------------------------


def test_split_mineru_result_pair():
    """A (content_list, middle_json) pair is returned unchanged."""
    from hybrid_doc_parser.parser import _split_mineru_result

    content = [{"type": "text", "text": "x"}]
    middle = {"pdf_info": [{"page_idx": 0}]}
    assert _split_mineru_result((content, middle)) == (content, middle)


def test_split_mineru_result_bare_list():
    """A bare content_list yields (content_list, None) — the mock-compatible path."""
    from hybrid_doc_parser.parser import _split_mineru_result

    content = [{"type": "text", "text": "x"}]
    assert _split_mineru_result(content) == (content, None)


def test_split_mineru_result_pair_with_none_middle():
    """A (content_list, None) pair yields (content_list, None)."""
    from hybrid_doc_parser.parser import _split_mineru_result

    content = [{"type": "text", "text": "x"}]
    assert _split_mineru_result((content, None)) == (content, None)


def test_split_mineru_result_batch_bare_value():
    """A per-path bare list (batch dict value) splits to (content_list, None)."""
    from hybrid_doc_parser.parser import _split_mineru_result

    # The batch {path: value} map is split per value upstream; a bare list value
    # must still yield middle_json=None so existing batch mocks keep working.
    content = [{"type": "text", "text": "x"}]
    assert _split_mineru_result(content) == (content, None)


def test_split_mineru_result_unexpected_shape_degrades():
    """An unexpected shape degrades to ([], None) rather than raising."""
    from hybrid_doc_parser.parser import _split_mineru_result

    assert _split_mineru_result(None) == ([], None)
    assert _split_mineru_result(42) == ([], None)
    assert _split_mineru_result("nonsense") == ([], None)
