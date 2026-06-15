"""Tests for the standalone verify() orchestration (Task Group 4).

All tests use ``FakeVerifierClient`` (no network) and mock the full-page render
so no real rendering is required. They cover: the flagged-only default trigger,
``force_verify_all``, severity filtering, the never-raises discipline, the
advisory (no-mutation) guarantee, and the DOCX/HTML no-op.
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
from hybrid_doc_parser.verifier import FakeVerifierClient, verify


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
) -> ParserOutput:
    """Build a ParserOutput with one PageRecord per decision in ``decisions``."""
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
        file_sha256="a" * 64,
        page_count=len(pages),
        pages=pages,
        elements=elements or [],
        warnings=[],
        enrichment_config=EnrichmentConfig(),
    )


@pytest.fixture(autouse=True)
def _isolated_verifier_cache(monkeypatch, tmp_path):
    """Isolate the verification cache per test.

    ``verify()`` now reads/writes a separate verification cache keyed on
    ``(file_sha256, page_idx, model, prompt_version)``. These tests share a
    fixed ``file_sha256`` and the default empty ``model``, so without isolation
    one test's cached page-0 verdict would short-circuit another's client call
    and break ``client.calls`` assertions. Point the cache at a throwaway dir.
    """
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


def test_default_verifies_only_promoted_pages(monkeypatch):
    output = _parser_output(decisions=["keep", "promote_to_vlm", "keep"])
    fake = FakeVerifierClient(
        verdicts={1: _high_disagreement("p1-e0")},
    )
    monkeypatch.setattr(verifier, "make_verifier_client", lambda cfg: fake)

    report = verify(output, Path("/tmp/doc.pdf"), VerifierConfig(backend="fake"))

    # Only the single promoted page (idx 1) was rendered/verified.
    assert [page_idx for (_, prompt, _) in fake.calls
            for page_idx in [int(prompt.split("page_idx:")[1].split()[0])]] == [1]
    assert [p.page_idx for p in report.pages] == [1]


def test_force_verify_all_verifies_every_page(monkeypatch):
    output = _parser_output(decisions=["keep", "keep", "keep"])
    fake = FakeVerifierClient()
    monkeypatch.setattr(verifier, "make_verifier_client", lambda cfg: fake)

    config = VerifierConfig(backend="fake", force_verify_all=True)
    report = verify(output, Path("/tmp/doc.pdf"), config)

    assert len(fake.calls) == 3
    assert [p.page_idx for p in report.pages] == [0, 1, 2]


def test_filters_to_disagreements_and_drops_below_min_severity(monkeypatch):
    elements = [_element("p0-e0", 0, ElementType.text, "a")]
    output = _parser_output(decisions=["promote_to_vlm"], elements=elements)
    verdict = {
        "disagreements": [
            {
                "element_id": "p0-e0",
                "type": "text",
                "severity": "low",  # below the "high" floor -> dropped
                "reason": "minor",
                "suggested_text": "",
                "vlm_confidence": 0.4,
            }
        ],
        "missing_elements": [
            {"severity": "high", "reason": "footnote", "approx_location": "bottom"}
        ],
        "extra_elements": [
            {"severity": "medium", "reason": "spurious", "approx_location": "top"}
        ],
    }
    fake = FakeVerifierClient(verdicts={0: verdict})
    monkeypatch.setattr(verifier, "make_verifier_client", lambda cfg: fake)

    config = VerifierConfig(backend="fake", min_severity_to_report="high")
    report = verify(output, Path("/tmp/doc.pdf"), config)

    page = report.pages[0]
    # Low-severity disagreement dropped; medium extra dropped; high missing kept.
    assert page.disagreements == []
    assert page.extra_elements == []
    assert len(page.missing_elements) == 1
    assert page.missing_elements[0].reason == "footnote"


def test_never_raises_records_verification_failed_warning(monkeypatch):
    output = _parser_output(decisions=["promote_to_vlm", "promote_to_vlm"])
    # Page 0 returns an error; page 1 succeeds.
    fake = FakeVerifierClient(
        verdicts={
            0: {"error": "boom"},
            1: {"disagreements": [], "missing_elements": [], "extra_elements": []},
        }
    )
    monkeypatch.setattr(verifier, "make_verifier_client", lambda cfg: fake)

    report = verify(output, Path("/tmp/doc.pdf"), VerifierConfig(backend="fake"))

    # Errored page yields NO verdict but a verification_failed warning.
    assert [p.page_idx for p in report.pages] == [1]
    failed = [w for w in report.warnings if w.code == "verification_failed"]
    assert len(failed) == 1
    assert failed[0].page_idx == 0


def test_advisory_does_not_mutate_parser_output(monkeypatch):
    elements = [_element("p0-e0", 0, ElementType.text, "original")]
    output = _parser_output(decisions=["promote_to_vlm"], elements=elements)
    fake = FakeVerifierClient(verdicts={0: _high_disagreement("p0-e0")})
    monkeypatch.setattr(verifier, "make_verifier_client", lambda cfg: fake)

    report = verify(output, Path("/tmp/doc.pdf"), VerifierConfig(backend="fake"))

    # parser_output, element text, and is_enriched are all unchanged.
    assert output.elements[0].text == "original"
    assert output.elements[0].is_enriched is False
    assert isinstance(report, VerificationReport)
    assert report is not output  # separate return value


def test_docx_input_no_op_with_unsupported_warning(monkeypatch):
    output = _parser_output(
        decisions=["promote_to_vlm"], file_path="/tmp/doc.docx"
    )
    fake = FakeVerifierClient()
    monkeypatch.setattr(verifier, "make_verifier_client", lambda cfg: fake)

    report = verify(output, Path("/tmp/doc.docx"), VerifierConfig(backend="fake"))

    assert report.pages == []
    assert fake.calls == []  # no rendering / no client calls
    assert [w.code for w in report.warnings] == ["verification_unsupported"]


def test_max_pages_per_doc_truncates_with_warning(monkeypatch):
    output = _parser_output(decisions=["promote_to_vlm"] * 4)
    fake = FakeVerifierClient()
    monkeypatch.setattr(verifier, "make_verifier_client", lambda cfg: fake)

    config = VerifierConfig(
        backend="fake", force_verify_all=True, max_pages_per_doc=2
    )
    report = verify(output, Path("/tmp/doc.pdf"), config)

    assert [p.page_idx for p in report.pages] == [0, 1]
    assert len(fake.calls) == 2
    truncated = [w for w in report.warnings if w.code == "verification_truncated"]
    assert len(truncated) == 1
    assert "skipped pages [2, 3]" in truncated[0].message
