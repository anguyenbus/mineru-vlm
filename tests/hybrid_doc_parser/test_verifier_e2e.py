"""End-to-end integration tests for the standalone ``verify()`` workflow (TG7).

Task Group 7 reviewed the per-layer coverage from Task Groups 1-6 and fills the
remaining CRITICAL integration gaps in the full ``verify()`` pipeline
(flagged trigger -> render/serialize/prompt -> client -> filter -> report ->
cache). The earlier groups cover each layer in isolation:

- TG1 asserts the canonical envelope on a HAND-BUILT ``VerificationReport``.
- TG4 asserts ``.pages`` / ``.warnings`` attributes of a ``verify()`` result but
  never the FULL serialized canonical envelope the pipeline actually produces.
- TG5 asserts a cache hit skips render + client, and TG6 asserts concurrency /
  retry behavior.

These E2E tests exercise the seams BETWEEN those layers with
``FakeVerifierClient`` (no network) and a mocked full-page render:

1. A full happy-path ``verify()`` whose serialized output matches the canonical
   ``{"verification": {...}}`` envelope VERBATIM (top-level key + all three
   channels surviving the filter).
2. A second ``verify()`` short-circuiting on the verification cache so it
   produces a byte-identical envelope with NO second client/render call.
3. A mixed document: one flagged PDF page yielding a verdict carrying BOTH a
   disagreement and a ``missing_elements`` / ``extra_elements`` finding.
4. Disagreement ``type`` backfilled from MinerU's authoritative element type,
   overriding the model's claimed type.
5. A whole-run failure degrading to the canonical EMPTY envelope with a
   document-level ``verification_failed`` warning (never raises).

All tests use ``FakeVerifierClient``; rendering is mocked; nothing touches the
network or the parse cache.
"""

from __future__ import annotations

import unittest.mock as mock
from pathlib import Path

import pytest

from hybrid_doc_parser import verifier
from hybrid_doc_parser.models import (
    ElementRecord,
    ElementType,
    EnrichmentConfig,
    PageRecord,
    ParserOutput,
    VerificationReport,
    VerifierConfig,
)
from hybrid_doc_parser.verifier import PROMPT_VERSION, FakeVerifierClient, verify


# ---------------------------------------------------------------------------
# Shared fixtures / builders (mirroring test_verify.py conventions)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated_verifier_cache(monkeypatch, tmp_path):
    """Point the verification cache at a throwaway dir per test.

    Tests share a fixed ``file_sha256`` and default model, so without isolation
    one test's cached page verdict would short-circuit another's client call.
    """
    monkeypatch.setenv(
        "HYBRID_DOC_PARSER_VERIFIER_CACHE_DIR", str(tmp_path / "vcache")
    )
    yield


@pytest.fixture(autouse=True)
def _mock_render():
    """Avoid real pypdfium2 rendering; record render calls for short-circuit checks."""
    with mock.patch.object(
        verifier, "_render_full_page", return_value=b"\x89PNGDATA"
    ) as m:
        yield m


def _element(element_id: str, page_idx: int, etype: ElementType, text: str):
    return ElementRecord(
        element_id=element_id,
        type=etype,
        text=text,
        bbox=[1.0, 2.0, 3.0, 4.0],
        page_idx=page_idx,
    )


def _parser_output(
    *,
    decisions: list[str],
    elements: list[ElementRecord] | None = None,
    file_path: str = "/tmp/doc.pdf",
    file_sha256: str = "a" * 64,
) -> ParserOutput:
    pages = [
        PageRecord(
            page_idx=idx,
            quality_decision=decision,  # type: ignore[arg-type]
            element_count=0,
            vlm_used=False,
        )
        for idx, decision in enumerate(decisions)
    ]
    return ParserOutput(
        file_path=file_path,
        file_sha256=file_sha256,
        page_count=len(pages),
        pages=pages,
        elements=elements or [],
        warnings=[],
        enrichment_config=EnrichmentConfig(),
    )


def _full_channel_verdict() -> dict:
    """A verdict exercising all three channels with high-severity findings."""
    return {
        "disagreements": [
            {
                "element_id": "p3-e7",
                "type": "table",
                "severity": "high",
                "reason": "Row 4 merged two columns; values misaligned vs image.",
                "suggested_text": "...",
                "vlm_confidence": 0.86,
            }
        ],
        "missing_elements": [
            {
                "severity": "high",
                "reason": "Footnote at page bottom not extracted.",
                "approx_location": "below last paragraph",
            }
        ],
        "extra_elements": [
            {
                "severity": "high",
                "reason": "Spurious header repeated.",
                "approx_location": "top of page",
            }
        ],
    }


# ---------------------------------------------------------------------------
# 1. Full happy-path E2E -> canonical envelope verbatim
# ---------------------------------------------------------------------------


def test_e2e_happy_path_produces_canonical_verification_envelope(monkeypatch):
    """verify() output serializes to the canonical {"verification": {...}} envelope.

    A single flagged page with a disagreement keyed to a real MinerU element plus
    a missing and an extra finding, all high-severity, must survive the filter and
    serialize verbatim to the spec's canonical shape with the top-level
    ``verification`` key.
    """
    elements = [_element("p3-e7", 3, ElementType.table, "Q1 Q2 Q3")]
    # Page 3 is the only promoted page; pages 0-2 are kept (not verified).
    output = _parser_output(
        decisions=["keep", "keep", "keep", "promote_to_vlm"], elements=elements
    )
    fake = FakeVerifierClient(verdicts={3: _full_channel_verdict()})
    monkeypatch.setattr(verifier, "make_verifier_client", lambda cfg: fake)

    config = VerifierConfig(
        backend="fake", model="anthropic.claude-...", min_severity_to_report="high"
    )
    report = verify(output, Path("/tmp/doc.pdf"), config)

    dumped = report.model_dump()
    # Top-level envelope.
    assert set(dumped.keys()) == {"verification"}
    inner = dumped["verification"]
    assert set(inner.keys()) == {"model_id", "prompt_version", "pages", "warnings"}
    assert inner["model_id"] == "anthropic.claude-..."
    assert inner["prompt_version"] == PROMPT_VERSION
    assert inner["warnings"] == []

    # Exactly the one verified page, with all three channels populated.
    assert len(inner["pages"]) == 1
    page = inner["pages"][0]
    assert page["page_idx"] == 3

    dis = page["disagreements"][0]
    assert set(dis.keys()) == {
        "element_id",
        "type",
        "severity",
        "reason",
        "suggested_text",
        "vlm_confidence",
    }
    assert dis["element_id"] == "p3-e7"
    assert dis["type"] == "table"
    assert dis["severity"] == "high"
    assert dis["vlm_confidence"] == 0.86

    miss = page["missing_elements"][0]
    assert "element_id" not in miss
    assert set(miss.keys()) == {"severity", "reason", "approx_location"}
    assert miss["approx_location"] == "below last paragraph"

    extra = page["extra_elements"][0]
    assert "element_id" not in extra
    assert set(extra.keys()) == {"severity", "reason", "approx_location"}
    assert extra["reason"] == "Spurious header repeated."


def test_e2e_json_dump_wraps_pipeline_output_in_verification(monkeypatch):
    """model_dump_json() of a verify() result also carries the top-level envelope."""
    elements = [_element("p0-e0", 0, ElementType.text, "hello")]
    output = _parser_output(decisions=["promote_to_vlm"], elements=elements)
    verdict = {
        "disagreements": [
            {
                "element_id": "p0-e0",
                "type": "text",
                "severity": "high",
                "reason": "mismatch",
                "suggested_text": "fixed",
                "vlm_confidence": 0.91,
            }
        ],
        "missing_elements": [],
        "extra_elements": [],
    }
    fake = FakeVerifierClient(verdicts={0: verdict})
    monkeypatch.setattr(verifier, "make_verifier_client", lambda cfg: fake)

    report = verify(output, Path("/tmp/doc.pdf"), VerifierConfig(backend="fake"))

    json_str = report.model_dump_json()
    assert json_str.startswith('{"verification":')
    # Round-trips back through the envelope-unwrapping validator.
    restored = VerificationReport.model_validate_json(json_str)
    assert restored == report


# ---------------------------------------------------------------------------
# 2. Cache hit short-circuits a SECOND verify() (same envelope, no re-call)
# ---------------------------------------------------------------------------


def test_e2e_cache_hit_short_circuits_second_verify(monkeypatch, _mock_render):
    """A second verify() on the same 4-tuple reuses the cache: no client/render.

    Distinct from TG5's wiring test in that it asserts the SECOND run reproduces
    the full canonical envelope identically while the second client is never
    touched and rendering is not repeated.
    """
    elements = [_element("p0-e0", 0, ElementType.text, "hello")]
    output = _parser_output(decisions=["promote_to_vlm"], elements=elements)
    config = VerifierConfig(backend="fake", model="model-x", min_severity_to_report="high")
    verdict = {
        "disagreements": [
            {
                "element_id": "p0-e0",
                "type": "text",
                "severity": "high",
                "reason": "mismatch",
                "suggested_text": "fixed",
                "vlm_confidence": 0.9,
            }
        ],
        "missing_elements": [],
        "extra_elements": [],
    }

    client1 = FakeVerifierClient(verdicts={0: verdict})
    monkeypatch.setattr(verifier, "make_verifier_client", lambda cfg: client1)
    report1 = verify(output, Path("/tmp/doc.pdf"), config)
    assert len(client1.calls) == 1
    renders_after_first = _mock_render.call_count

    # Second run with a DIFFERENT client instance that would raise if used (its
    # canned verdict differs) — a cache hit means it is never consulted.
    client2 = FakeVerifierClient(default_verdict={"disagreements": [], "missing_elements": [], "extra_elements": []})
    monkeypatch.setattr(verifier, "make_verifier_client", lambda cfg: client2)
    report2 = verify(output, Path("/tmp/doc.pdf"), config)

    assert client2.calls == []  # cache hit -> client never invoked
    assert _mock_render.call_count == renders_after_first  # render not repeated
    # Identical canonical envelope produced from cache.
    assert report2.model_dump() == report1.model_dump()
    assert report2.pages[0].disagreements[0].element_id == "p0-e0"


# ---------------------------------------------------------------------------
# 3. Mixed document: a verdict AND an unsupported warning are both possible
# ---------------------------------------------------------------------------


def test_e2e_pdf_page_yields_all_three_channels_in_one_verdict(monkeypatch):
    """A single flagged PDF page can yield disagreement + missing + extra together."""
    elements = [_element("p0-e0", 0, ElementType.text, "body")]
    output = _parser_output(decisions=["promote_to_vlm"], elements=elements)
    verdict = {
        "disagreements": [
            {
                "element_id": "p0-e0",
                "type": "text",
                "severity": "high",
                "reason": "wrong",
                "suggested_text": "right",
                "vlm_confidence": 0.8,
            }
        ],
        "missing_elements": [
            {"severity": "high", "reason": "dropped table", "approx_location": "center"}
        ],
        "extra_elements": [
            {"severity": "high", "reason": "phantom line", "approx_location": "footer"}
        ],
    }
    fake = FakeVerifierClient(verdicts={0: verdict})
    monkeypatch.setattr(verifier, "make_verifier_client", lambda cfg: fake)

    report = verify(
        output, Path("/tmp/doc.pdf"), VerifierConfig(backend="fake", min_severity_to_report="high")
    )

    page = report.pages[0]
    assert [d.element_id for d in page.disagreements] == ["p0-e0"]
    assert [m.reason for m in page.missing_elements] == ["dropped table"]
    assert [e.reason for e in page.extra_elements] == ["phantom line"]
    assert report.warnings == []


def test_e2e_docx_input_yields_only_unsupported_warning(monkeypatch):
    """A DOCX input (no source image) no-ops to a single unsupported warning.

    This is the "unsupported half" of the mixed PDF+DOCX scenario from the spec:
    even a flagged DOCX page yields a ``verification_unsupported`` warning and no
    verdicts, while the verifier client is never invoked.
    """
    output = _parser_output(decisions=["promote_to_vlm"], file_path="/tmp/doc.docx")
    fake = FakeVerifierClient(verdicts={0: _full_channel_verdict()})
    monkeypatch.setattr(verifier, "make_verifier_client", lambda cfg: fake)

    report = verify(output, Path("/tmp/doc.docx"), VerifierConfig(backend="fake"))

    assert report.pages == []
    assert fake.calls == []
    assert [w.code for w in report.warnings] == ["verification_unsupported"]
    # The canonical envelope is still well-formed for an unsupported input.
    inner = report.model_dump()["verification"]
    assert inner["pages"] == []
    assert inner["warnings"][0]["code"] == "verification_unsupported"


# ---------------------------------------------------------------------------
# 4. Disagreement type backfilled from MinerU's authoritative type
# ---------------------------------------------------------------------------


def test_e2e_disagreement_type_backfilled_from_mineru(monkeypatch):
    """The disagreement's type comes from MinerU, overriding the model's claim.

    The model claims the element is ``text`` but MinerU's authoritative type for
    ``p0-e0`` is ``table``; the produced disagreement must carry MinerU's type.
    """
    elements = [_element("p0-e0", 0, ElementType.table, "Q1 Q2")]
    output = _parser_output(decisions=["promote_to_vlm"], elements=elements)
    verdict = {
        "disagreements": [
            {
                "element_id": "p0-e0",
                "type": "text",  # WRONG claim; MinerU says table
                "severity": "high",
                "reason": "mismatch",
                "suggested_text": "fixed",
                "vlm_confidence": 0.7,
            }
        ],
        "missing_elements": [],
        "extra_elements": [],
    }
    fake = FakeVerifierClient(verdicts={0: verdict})
    monkeypatch.setattr(verifier, "make_verifier_client", lambda cfg: fake)

    report = verify(
        output, Path("/tmp/doc.pdf"), VerifierConfig(backend="fake", min_severity_to_report="high")
    )

    assert report.pages[0].disagreements[0].type == ElementType.table


# ---------------------------------------------------------------------------
# 5. Whole-run failure -> canonical empty envelope, never raises
# ---------------------------------------------------------------------------


def test_e2e_whole_run_failure_returns_empty_canonical_envelope(monkeypatch):
    """An unexpected internal failure degrades to an empty report (never raises).

    Forcing ``_select_pages`` to raise exercises the last-resort try/except in
    ``verify()``: it must return a well-formed empty envelope with the model id /
    prompt version populated and a document-level ``verification_failed`` warning,
    rather than propagating.
    """
    output = _parser_output(decisions=["promote_to_vlm"])
    monkeypatch.setattr(
        verifier,
        "_select_pages",
        mock.Mock(side_effect=RuntimeError("boom")),
    )

    config = VerifierConfig(backend="fake", model="model-z")
    report = verify(output, Path("/tmp/doc.pdf"), config)

    assert isinstance(report, VerificationReport)
    assert report.pages == []
    failed = [w for w in report.warnings if w.code == "verification_failed"]
    assert len(failed) == 1
    assert failed[0].page_idx is None  # document-level, not page-level
    inner = report.model_dump()["verification"]
    assert inner["model_id"] == "model-z"
    assert inner["prompt_version"] == PROMPT_VERSION
    assert inner["pages"] == []
