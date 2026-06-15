"""Tests for verifier batch concurrency + throttling resilience (Task Group 6).

These cover the concurrency-and-resilience layer added to ``verify()``:

- A SEPARATE ``asyncio.Semaphore(config.max_concurrency)`` bounds the number of
  in-flight per-page verifier calls — independent of ``parse_batch``'s own
  concurrency budget.
- A bounded retry with jittered backoff retries on a THROTTLING error and then
  succeeds (the sleep is mocked so the test runs instantly).
- Retry exhaustion degrades the page to a ``verification_failed`` warning rather
  than raising.

All tests use ``FakeVerifierClient`` (no network) and mock the full-page render,
mirroring ``test_verify.py``.
"""

from __future__ import annotations

import threading
import unittest.mock as mock
from pathlib import Path

import pytest

from hybrid_doc_parser import verifier
from hybrid_doc_parser.models import (
    EnrichmentConfig,
    PageRecord,
    ParserOutput,
    VerifierConfig,
)
from hybrid_doc_parser.verifier import FakeVerifierClient, verify


def _parser_output(*, n_pages: int, file_path: str = "/tmp/doc.pdf") -> ParserOutput:
    """Build a ParserOutput with ``n_pages`` quality-gate-flagged pages."""
    pages = [
        PageRecord(
            page_idx=idx,
            quality_decision="promote_to_vlm",
            element_count=0,
            vlm_used=False,
        )
        for idx in range(n_pages)
    ]
    return ParserOutput(
        file_path=file_path,
        file_sha256="a" * 64,
        page_count=len(pages),
        pages=pages,
        elements=[],
        warnings=[],
        enrichment_config=EnrichmentConfig(),
    )


@pytest.fixture(autouse=True)
def _isolated_verifier_cache(monkeypatch, tmp_path):
    """Isolate the verification cache per test so a verdict never short-circuits."""
    monkeypatch.setenv(
        "HYBRID_DOC_PARSER_VERIFIER_CACHE_DIR", str(tmp_path / "vcache")
    )
    yield


@pytest.fixture(autouse=True)
def _mock_render():
    """Avoid real pypdfium2 rendering in every test."""
    with mock.patch.object(
        verifier, "_render_full_page", return_value=b"\x89PNGDATA"
    ):
        yield


def _no_sleep(_delay: float) -> None:
    """A sleeper stand-in that records nothing and never actually waits."""


def test_separate_semaphore_bounds_in_flight_page_calls(monkeypatch):
    """The verifier's own semaphore caps concurrent in-flight page calls."""
    max_concurrency = 2
    n_pages = 8

    lock = threading.Lock()
    state = {"in_flight": 0, "peak": 0}
    barrier = threading.Barrier(max_concurrency)

    class ConcurrencyProbe(FakeVerifierClient):
        """A fake that measures the peak number of simultaneous verify_page calls."""

        def verify_page(self, image_bytes, prompt, config):  # type: ignore[override]
            with lock:
                state["in_flight"] += 1
                state["peak"] = max(state["peak"], state["in_flight"])
            # Block until `max_concurrency` callers are concurrently inside so a
            # too-large semaphore would be observed as a higher peak. The timeout
            # keeps the test from hanging if fewer than `max_concurrency` ever run.
            try:
                barrier.wait(timeout=2.0)
            except threading.BrokenBarrierError:
                pass
            try:
                return super().verify_page(image_bytes, prompt, config)
            finally:
                with lock:
                    state["in_flight"] -= 1

    probe = ConcurrencyProbe()
    monkeypatch.setattr(verifier, "make_verifier_client", lambda cfg: probe)

    config = VerifierConfig(
        backend="fake", max_concurrency=max_concurrency, max_pages_per_doc=n_pages
    )
    report = verify(_parser_output(n_pages=n_pages), Path("/tmp/doc.pdf"), config)

    # The verifier's separate semaphore must never let more than its own
    # max_concurrency run at once, regardless of how many pages are queued.
    assert state["peak"] == max_concurrency
    assert len(probe.calls) == n_pages
    assert [p.page_idx for p in report.pages] == list(range(n_pages))


def test_verifier_semaphore_independent_of_parse_batch(monkeypatch):
    """The verifier semaphore is sized by VerifierConfig, not parse_batch."""
    # A single in-flight call is allowed; with two pages the peak must be 1 even
    # though parse_batch defaults to a wider concurrency. This proves the
    # verifier owns a SEPARATE budget.
    lock = threading.Lock()
    state = {"in_flight": 0, "peak": 0}
    release = threading.Event()

    class SerialProbe(FakeVerifierClient):
        def verify_page(self, image_bytes, prompt, config):  # type: ignore[override]
            with lock:
                state["in_flight"] += 1
                state["peak"] = max(state["peak"], state["in_flight"])
            # Briefly hold the slot so any concurrent caller would bump the peak.
            release.wait(timeout=0.2)
            try:
                return super().verify_page(image_bytes, prompt, config)
            finally:
                with lock:
                    state["in_flight"] -= 1

    probe = SerialProbe()
    monkeypatch.setattr(verifier, "make_verifier_client", lambda cfg: probe)

    config = VerifierConfig(backend="fake", max_concurrency=1, max_pages_per_doc=4)
    verify(_parser_output(n_pages=4), Path("/tmp/doc.pdf"), config)

    assert state["peak"] == 1


def test_retry_with_jitter_retries_on_throttling_then_succeeds(monkeypatch):
    """A throttling error is retried (mocked sleep) and the page then succeeds."""
    attempts = {"n": 0}
    clean = {"disagreements": [], "missing_elements": [], "extra_elements": []}

    class FlakyThrottleClient(FakeVerifierClient):
        def verify_page(self, image_bytes, prompt, config):  # type: ignore[override]
            self.calls.append((image_bytes is not None, prompt, config))
            attempts["n"] += 1
            if attempts["n"] == 1:
                return {"error": "ThrottlingException: rate exceeded"}
            return clean

    client = FlakyThrottleClient()
    monkeypatch.setattr(verifier, "make_verifier_client", lambda cfg: client)

    slept: list[float] = []

    config = VerifierConfig(backend="fake", max_concurrency=1)
    report = verify(
        _parser_output(n_pages=1),
        Path("/tmp/doc.pdf"),
        config,
        sleeper=slept.append,
    )

    # Retried exactly once (two attempts total), slept once between them, and the
    # second attempt produced a clean verdict with NO verification_failed warning.
    assert attempts["n"] == 2
    assert len(slept) == 1
    assert slept[0] >= 0.0
    assert [p.page_idx for p in report.pages] == [0]
    assert [w for w in report.warnings if w.code == "verification_failed"] == []


def test_retry_exhaustion_degrades_to_verification_failed(monkeypatch):
    """Persistent throttling exhausts retries and degrades to a warning (no raise)."""
    attempts = {"n": 0}

    class AlwaysThrottleClient(FakeVerifierClient):
        def verify_page(self, image_bytes, prompt, config):  # type: ignore[override]
            self.calls.append((image_bytes is not None, prompt, config))
            attempts["n"] += 1
            return {"error": "ThrottlingException: slow down"}

    client = AlwaysThrottleClient()
    monkeypatch.setattr(verifier, "make_verifier_client", lambda cfg: client)

    slept: list[float] = []

    config = VerifierConfig(backend="fake", max_concurrency=1)
    report = verify(
        _parser_output(n_pages=1),
        Path("/tmp/doc.pdf"),
        config,
        sleeper=slept.append,
    )

    # All bounded attempts were made, sleeping between each (attempts - 1 times),
    # the page yielded NO verdict, and it degraded to a verification_failed
    # warning rather than raising.
    assert attempts["n"] == verifier._MAX_VERIFY_ATTEMPTS
    assert len(slept) == verifier._MAX_VERIFY_ATTEMPTS - 1
    assert report.pages == []
    failed = [w for w in report.warnings if w.code == "verification_failed"]
    assert len(failed) == 1
    assert failed[0].page_idx == 0


def test_non_throttling_error_is_not_retried(monkeypatch):
    """A non-throttling failure degrades immediately without any retry."""
    attempts = {"n": 0}

    class HardFailClient(FakeVerifierClient):
        def verify_page(self, image_bytes, prompt, config):  # type: ignore[override]
            self.calls.append((image_bytes is not None, prompt, config))
            attempts["n"] += 1
            return {"error": "ValidationException: bad request"}

    client = HardFailClient()
    monkeypatch.setattr(verifier, "make_verifier_client", lambda cfg: client)

    slept: list[float] = []

    config = VerifierConfig(backend="fake", max_concurrency=1)
    report = verify(
        _parser_output(n_pages=1),
        Path("/tmp/doc.pdf"),
        config,
        sleeper=slept.append,
    )

    # Exactly one attempt, no sleeps, and a single verification_failed warning.
    assert attempts["n"] == 1
    assert slept == []
    assert report.pages == []
    assert len([w for w in report.warnings if w.code == "verification_failed"]) == 1
