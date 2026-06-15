"""Small LABELED disagreement set for the standalone ``verify()`` eval harness.

This is the doc-bench labeled disagreement set required by the HARD eval gate
(Task Group 8). It is deliberately SELF-CONTAINED: each "document" is an
in-memory :class:`ParserOutput` plus a ground-truth label set describing the
KNOWN MinerU disagreements / missing / extra elements for that document, and a
canned :class:`FakeVerifierClient` verdict that a perfect (or imperfect) verifier
would emit. No binary PDFs/images are committed; the harness mocks the full-page
render, so the labeled set runs with no PDF toolchain and no network.

Why synthetic stand-ins instead of real PDF fixtures: the deterministic FAKE
path is what PROVES the harness is correct — the fake returns KNOWN verdicts, so
precision/recall against the labels are deterministic and reproducible in CI. The
real-model precision/recall (a different, noisy number) is measured separately by
the harness's ``@live`` path against a real backend; this labeled set supplies
the ground truth both paths score against.

The set intentionally includes the full mix of cases the gate must score:
- A clean page (no findings) the verifier must NOT flag (precision pressure).
- True-positive disagreements / missing / extra findings (recall).
- A low-severity finding that the precision-favoring severity floor should DROP.
- A page the quality gate KEEPS (not ``promote_to_vlm``) carrying a real defect,
  so the flagged-only-vs-``force_verify_all`` recall gap is measurable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from hybrid_doc_parser.models import (
    ElementRecord,
    ElementType,
    EnrichmentConfig,
    PageRecord,
    ParserOutput,
)

Channel = Literal["disagreement", "missing", "extra"]


@dataclass(frozen=True)
class Label:
    """A single ground-truth finding the verifier is expected to surface.

    Attributes:
        page_idx: Zero-indexed page the finding belongs to.
        channel: Which report channel the finding belongs in
            (``"disagreement"`` / ``"missing"`` / ``"extra"``).
        key: A stable identifier for matching a predicted finding to this label.
            For disagreements this is the MinerU ``element_id``; for
            missing/extra findings it is a curator-chosen ``approx_location``.
        severity: True severity of the finding; used to model what the
            precision-favoring ``min_severity_to_report`` floor should keep.
    """

    page_idx: int
    channel: Channel
    key: str
    severity: Literal["low", "medium", "high"]


@dataclass(frozen=True)
class LabeledDocument:
    """One labeled document in the eval set.

    Attributes:
        name: Human-readable identifier used in the eval report.
        parser_output: The in-memory MinerU parse output to verify.
        file_path: A ``.pdf``/image path string so ``verify()`` treats the input
            as renderable (the render is mocked; only the suffix matters).
        labels: Ground-truth findings for this document.
        fake_verdicts: Canned per-page verdicts for :class:`FakeVerifierClient`,
            keyed by ``page_idx``. Models what the (fake) verifier "sees" — it may
            differ from the labels to exercise false positives / negatives.
    """

    name: str
    parser_output: ParserOutput
    file_path: str
    labels: list[Label]
    fake_verdicts: dict[int, dict[str, Any]] = field(default_factory=dict)


def _element(element_id: str, page_idx: int, etype: ElementType, text: str) -> ElementRecord:
    """Build a minimal :class:`ElementRecord` for the labeled set."""
    return ElementRecord(
        element_id=element_id,
        type=etype,
        text=text,
        bbox=[0.0, 0.0, 1.0, 1.0],
        page_idx=page_idx,
    )


def _parser_output(
    *,
    name: str,
    decisions: list[str],
    elements: list[ElementRecord],
) -> ParserOutput:
    """Assemble a :class:`ParserOutput` with one :class:`PageRecord` per decision."""
    pages = [
        PageRecord(
            page_idx=idx,
            quality_decision=decision,  # type: ignore[arg-type]
            element_count=sum(1 for el in elements if el.page_idx == idx),
            vlm_used=False,
        )
        for idx, decision in enumerate(decisions)
    ]
    return ParserOutput(
        file_path=f"/eval/{name}.pdf",
        file_sha256=_stable_sha(name),
        page_count=len(pages),
        pages=pages,
        elements=elements,
        warnings=[],
        enrichment_config=EnrichmentConfig(),
    )


def _stable_sha(name: str) -> str:
    """Return a deterministic 64-hex digest unique per document name.

    Distinct per document so the per-document verification cache keys never
    collide across the labeled set within a single harness run.
    """
    import hashlib  # noqa: PLC0415

    return hashlib.sha256(name.encode("utf-8")).hexdigest()


def _doc_clean() -> LabeledDocument:
    """A flagged page that is actually CORRECT: verifier must surface NOTHING.

    Precision pressure: a confidently-wrong verifier that invents a disagreement
    here is penalised. The fake emits an empty verdict (a faithful verifier).
    """
    name = "clean_page"
    elements = [
        _element("clean-e0", 0, ElementType.heading, "Quarterly Report"),
        _element("clean-e1", 0, ElementType.text, "Revenue rose 4% year over year."),
    ]
    output = _parser_output(name=name, decisions=["promote_to_vlm"], elements=elements)
    return LabeledDocument(
        name=name,
        parser_output=output,
        file_path=output.file_path,
        labels=[],
        fake_verdicts={
            0: {"disagreements": [], "missing_elements": [], "extra_elements": []}
        },
    )


def _doc_table_disagreement() -> LabeledDocument:
    """A flagged page whose table row is mis-extracted: one HIGH disagreement.

    The fake emits exactly the labeled disagreement (a true positive).
    """
    name = "table_misalign"
    elements = [
        _element("tbl-e0", 0, ElementType.heading, "Results"),
        _element("tbl-e1", 0, ElementType.table, "Q1 100 | Q2 200 | Q3 300"),
    ]
    output = _parser_output(name=name, decisions=["promote_to_vlm"], elements=elements)
    return LabeledDocument(
        name=name,
        parser_output=output,
        file_path=output.file_path,
        labels=[Label(page_idx=0, channel="disagreement", key="tbl-e1", severity="high")],
        fake_verdicts={
            0: {
                "disagreements": [
                    {
                        "element_id": "tbl-e1",
                        "type": "table",
                        "severity": "high",
                        "reason": "Row 2 merged two columns; values misaligned vs image.",
                        "suggested_text": "Q1 100 | Q2 200 | Q3 300 | Q4 400",
                        "vlm_confidence": 0.88,
                    }
                ],
                "missing_elements": [],
                "extra_elements": [],
            }
        },
    )


def _doc_missing_and_extra() -> LabeledDocument:
    """A flagged page with one MISSING (footnote) and one EXTRA (phantom) finding.

    The fake emits both, plus a spurious LOW-severity disagreement that the
    precision-favoring floor at ``high`` should DROP (so it must not count as a
    false positive once filtered).
    """
    name = "missing_extra"
    elements = [
        _element("me-e0", 0, ElementType.text, "Body paragraph one."),
        _element("me-e1", 0, ElementType.text, "Body paragraph two."),
    ]
    output = _parser_output(name=name, decisions=["promote_to_vlm"], elements=elements)
    return LabeledDocument(
        name=name,
        parser_output=output,
        file_path=output.file_path,
        labels=[
            Label(page_idx=0, channel="missing", key="below last paragraph", severity="high"),
            Label(page_idx=0, channel="extra", key="top of page", severity="high"),
        ],
        fake_verdicts={
            0: {
                "disagreements": [
                    {
                        "element_id": "me-e0",
                        "type": "text",
                        "severity": "low",
                        "reason": "Minor whitespace difference (noise).",
                        "suggested_text": "Body paragraph one. ",
                        "vlm_confidence": 0.3,
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
                        "reason": "Repeated running header extracted as content.",
                        "approx_location": "top of page",
                    }
                ],
            }
        },
    )


def _doc_gate_missed_defect() -> LabeledDocument:
    """A page the quality gate KEEPS that nonetheless has a real HIGH defect.

    This is the recall-gap document: flagged-only verification never sees page 0
    (decision ``"keep"``), so its labeled disagreement is unrecoverable without
    ``force_verify_all``. Page 1 is flagged and clean. The fake would surface the
    defect on page 0 IF page 0 were verified.
    """
    name = "gate_missed"
    elements = [
        _element("gm-e0", 0, ElementType.equation, "E = mc"),
        _element("gm-e1", 1, ElementType.text, "Conclusion."),
    ]
    output = _parser_output(
        name=name, decisions=["keep", "promote_to_vlm"], elements=elements
    )
    return LabeledDocument(
        name=name,
        parser_output=output,
        file_path=output.file_path,
        labels=[Label(page_idx=0, channel="disagreement", key="gm-e0", severity="high")],
        fake_verdicts={
            0: {
                "disagreements": [
                    {
                        "element_id": "gm-e0",
                        "type": "equation",
                        "severity": "high",
                        "reason": "Truncated equation: missing exponent (E = mc^2).",
                        "suggested_text": "E = mc^2",
                        "vlm_confidence": 0.92,
                    }
                ],
                "missing_elements": [],
                "extra_elements": [],
            },
            1: {"disagreements": [], "missing_elements": [], "extra_elements": []},
        },
    )


def labeled_documents() -> list[LabeledDocument]:
    """Return the full labeled disagreement set for the eval harness.

    Returns:
        The curated list of :class:`LabeledDocument` covering clean,
        disagreement, missing/extra, severity-floor, and quality-gate-recall-gap
        cases.
    """
    return [
        _doc_clean(),
        _doc_table_disagreement(),
        _doc_missing_and_extra(),
        _doc_gate_missed_defect(),
    ]
