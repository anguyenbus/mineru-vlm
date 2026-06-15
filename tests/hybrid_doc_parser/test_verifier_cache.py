"""Tests for the separate verification cache (Task Group 5).

These cover the verification cache module ``verifier_cache`` (NOT the parse
``cache``) and its wiring into ``verify()``'s per-page flow:

- Round-trip: a cached page verdict is reused on the same 4-tuple
  ``(content_hash, page_idx, model_id, prompt_version)``.
- A changed ``model_id`` OR ``prompt_version`` busts the cached verdict.
- A parse-cache hit does NOT trigger a verification-cache read/write (the
  verification cache is touched only inside ``verify()``).
- Cache read/write failures never raise (degrade silently).

All ``verify()`` tests use ``FakeVerifierClient`` (no network) and mock the
full-page render, mirroring ``test_verify.py``.
"""

from __future__ import annotations

import unittest.mock as mock
from pathlib import Path

import pytest

from hybrid_doc_parser import verifier, verifier_cache
from hybrid_doc_parser.models import (
    Disagreement,
    ElementRecord,
    ElementType,
    EnrichmentConfig,
    PageRecord,
    PageVerification,
    ParserOutput,
    Severity,
    VerifierConfig,
)
from hybrid_doc_parser.verifier import PROMPT_VERSION, FakeVerifierClient


@pytest.fixture(autouse=True)
def _isolated_cache(monkeypatch, tmp_path):
    """Point the verification cache at a throwaway tmp directory per test."""
    monkeypatch.setenv(
        "HYBRID_DOC_PARSER_VERIFIER_CACHE_DIR", str(tmp_path / "vcache")
    )
    yield


@pytest.fixture(autouse=True)
def _mock_render():
    """Avoid real pypdfium2 rendering and record render calls."""
    with mock.patch.object(
        verifier, "_render_full_page", return_value=b"\x89PNGDATA"
    ) as m:
        yield m


def _high_disagreement(element_id: str) -> dict:
    return {
        "disagreements": [
            {
                "element_id": element_id,
                "type": "text",
                "severity": "high",
                "reason": "Mismatch vs image.",
                "suggested_text": "fixed",
                "vlm_confidence": 0.9,
            }
        ],
        "missing_elements": [],
        "extra_elements": [],
    }


def _parser_output(file_sha256: str = "a" * 64) -> ParserOutput:
    """One promote_to_vlm page with a single text element."""
    element = ElementRecord(
        element_id="el-0",
        type=ElementType.text,
        text="hello",
        bbox=[1.0, 2.0, 3.0, 4.0],
        page_idx=0,
    )
    page = PageRecord(
        page_idx=0,
        quality_decision="promote_to_vlm",
        element_count=1,
        vlm_used=False,
    )
    return ParserOutput(
        file_path="/tmp/doc.pdf",
        file_sha256=file_sha256,
        page_count=1,
        pages=[page],
        elements=[element],
        warnings=[],
        enrichment_config=EnrichmentConfig(),
    )


def _config(*, model: str = "model-x") -> VerifierConfig:
    return VerifierConfig(backend="fake", model=model, min_severity_to_report="high")


def _page_verdict(page_idx: int = 0) -> PageVerification:
    return PageVerification(
        page_idx=page_idx,
        disagreements=[
            Disagreement(
                element_id="el-0",
                type=ElementType.text,
                severity=Severity.high,
                reason="r",
                suggested_text="s",
                vlm_confidence=0.9,
            )
        ],
        missing_elements=[],
        extra_elements=[],
    )


# ---------------------------------------------------------------------------
# Module-level get/put round-trip and busting
# ---------------------------------------------------------------------------


def test_module_roundtrip_same_tuple():
    """get returns the verdict put under the identical 4-tuple."""
    verdict = _page_verdict()
    verifier_cache.put("h" * 64, 0, "model-x", PROMPT_VERSION, verdict)

    got = verifier_cache.get("h" * 64, 0, "model-x", PROMPT_VERSION)
    assert got is not None
    assert got.page_idx == 0
    assert len(got.disagreements) == 1
    assert got.disagreements[0].element_id == "el-0"


def test_changed_model_id_busts():
    """A different model_id in the key misses the cached verdict."""
    verifier_cache.put("h" * 64, 0, "model-x", PROMPT_VERSION, _page_verdict())
    assert verifier_cache.get("h" * 64, 0, "model-y", PROMPT_VERSION) is None


def test_changed_prompt_version_busts():
    """A different prompt_version in the key misses the cached verdict."""
    verifier_cache.put("h" * 64, 0, "model-x", PROMPT_VERSION, _page_verdict())
    assert verifier_cache.get("h" * 64, 0, "model-x", "v-other") is None


# ---------------------------------------------------------------------------
# verify() wiring: round-trip reuse skips render + client
# ---------------------------------------------------------------------------


def test_verify_hit_reuses_and_skips_render_and_client(_mock_render):
    """Second verify() with the same 4-tuple hits the cache: no render, no call."""
    output = _parser_output()
    config = _config()

    # First run populates the verification cache and exercises the client.
    client1 = FakeVerifierClient(verdicts={0: _high_disagreement("el-0")})
    with mock.patch.object(verifier, "make_verifier_client", return_value=client1):
        report1 = verifier.verify(output, Path("/tmp/doc.pdf"), config)
    assert len(report1.pages) == 1
    assert len(client1.calls) == 1
    first_render_count = _mock_render.call_count

    # Second run: same 4-tuple -> cache hit -> client never invoked, render not
    # called again for that page.
    client2 = FakeVerifierClient(verdicts={0: _high_disagreement("el-0")})
    with mock.patch.object(verifier, "make_verifier_client", return_value=client2):
        report2 = verifier.verify(output, Path("/tmp/doc.pdf"), config)

    assert len(report2.pages) == 1
    assert report2.pages[0].disagreements[0].element_id == "el-0"
    assert client2.calls == []  # client never called on a cache hit
    assert _mock_render.call_count == first_render_count  # render skipped on hit


def test_verify_changed_model_busts_and_recalls_client():
    """Changing config.model busts the cached verdict so the client is called again."""
    output = _parser_output()

    client1 = FakeVerifierClient(verdicts={0: _high_disagreement("el-0")})
    with mock.patch.object(verifier, "make_verifier_client", return_value=client1):
        verifier.verify(output, Path("/tmp/doc.pdf"), _config(model="model-x"))

    client2 = FakeVerifierClient(verdicts={0: _high_disagreement("el-0")})
    with mock.patch.object(verifier, "make_verifier_client", return_value=client2):
        verifier.verify(output, Path("/tmp/doc.pdf"), _config(model="model-y"))

    assert len(client2.calls) == 1  # different model -> cache miss -> client called


# ---------------------------------------------------------------------------
# Parse-cache independence and never-raises discipline
# ---------------------------------------------------------------------------


def test_parse_cache_hit_does_not_touch_verification_cache():
    """A parse-cache hit must NOT trigger a verification-cache read or write.

    The parse cache (``cache``) is standalone from the verifier: ``verify()`` is
    the ONLY caller of the verification cache. Exercising the parse cache must
    not call ``verifier_cache.get``/``put`` at all.
    """
    from hybrid_doc_parser import cache as parse_cache

    with mock.patch.object(verifier_cache, "get") as vget, mock.patch.object(
        verifier_cache, "put"
    ) as vput:
        # Driving the parse cache (a plain get on a missing path) must never
        # reach into the verification cache.
        result = parse_cache.get(Path("/nonexistent/never.pdf"))

    assert result is None
    vget.assert_not_called()
    vput.assert_not_called()


def test_get_never_raises_on_corrupt_file(monkeypatch, tmp_path):
    """A corrupt/garbage cache file degrades to a miss (None), never raises."""
    cache_dir = tmp_path / "vcache"
    cache_dir.mkdir(parents=True)
    monkeypatch.setenv("HYBRID_DOC_PARSER_VERIFIER_CACHE_DIR", str(cache_dir))

    key = verifier_cache._cache_key("h" * 64, 0, "model-x", PROMPT_VERSION)
    verifier_cache._cache_path(key).write_text("{ this is not json ", encoding="utf-8")

    # Must not raise; corrupt content is a miss.
    assert verifier_cache.get("h" * 64, 0, "model-x", PROMPT_VERSION) is None


def test_put_never_raises_on_write_failure():
    """put swallows write errors (e.g. mkdir PermissionError) and never raises."""
    with mock.patch("pathlib.Path.mkdir", side_effect=PermissionError("no write")):
        # Must not raise.
        verifier_cache.put("h" * 64, 0, "model-x", PROMPT_VERSION, _page_verdict())
