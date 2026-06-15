"""Eval harness for the standalone advisory ``verify()`` (HARD eval gate, TG8).

Modeled on ``scripts/bench_docbench.py``: it iterates a small LABELED
disagreement set (``scripts/eval_verifier_fixtures.py``), runs ``verify()`` over
it, scores the produced :class:`VerificationReport` against the ground-truth
labels, and reports PRECISION and RECALL — separately for the disagreement and
the missing/extra channels — plus the recall gap left by the quality gate
(flagged-only selection vs ``force_verify_all``).

Backends
--------
- FAKE (default): uses :class:`FakeVerifierClient` with the per-document canned
  verdicts baked into the labeled set. DETERMINISTIC and NETWORK-FREE, so it runs
  in CI and is what PROVES the harness correct (known verdicts -> known
  precision/recall). This path produces the gate artifact as HARNESS VALIDATION.
- LIVE (``--live``): selects a real backend (Bedrock or an OpenAI-compatible
  local vllm/Ollama) so real-model precision/recall can be measured WITHOUT any
  required AWS spend in CI. Off by default; never runs unless explicitly asked.

Run (deterministic fake path, no network)::

    PYTHONPATH=src python scripts/eval_verifier.py
    PYTHONPATH=src python scripts/eval_verifier.py --write-artifact

Run (live, real backend — requires creds / a local endpoint)::

    PYTHONPATH=src python scripts/eval_verifier.py --live --backend bedrock \
        --model anthropic.claude-3-5-sonnet-20240620-v1:0 --region us-east-1

Tuning ``min_severity_to_report``
---------------------------------
The harness sweeps the severity floor (``low`` / ``medium`` / ``high``) and
records the floor that maximises PRECISION (ties broken toward higher recall),
favoring precision over recall as the spec requires. The chosen floor and the
resulting precision/recall are written to the gate artifact.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import unittest.mock as mock
from dataclasses import dataclass
from pathlib import Path

# Allow ``python scripts/eval_verifier.py`` to import the sibling fixtures module
# regardless of the caller's cwd, mirroring bench_docbench.py's import-by-name.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hybrid_doc_parser import verifier as _verifier  # noqa: E402
from hybrid_doc_parser.models import (  # noqa: E402
    VerificationReport,
    VerifierConfig,
)
from hybrid_doc_parser.verifier import FakeVerifierClient, verify  # noqa: E402

from eval_verifier_fixtures import (  # noqa: E402
    Channel,
    Label,
    LabeledDocument,
    labeled_documents,
)

# The severity floors swept when tuning ``min_severity_to_report``, weakest to
# strongest. A stronger floor keeps fewer findings (more precision-favoring).
_SEVERITY_FLOORS: tuple[str, ...] = ("low", "medium", "high")

# Channels scored as two groups: disagreements alone, and missing+extra together
# (they share an element-id-less, location-keyed shape).
_DISAGREEMENT = "disagreement"
_LOCATION_CHANNELS: frozenset[str] = frozenset({"missing", "extra"})


@dataclass(frozen=True)
class PrecisionRecall:
    """Precision/recall counts and ratios for one scored channel group."""

    true_positives: int
    false_positives: int
    false_negatives: int

    @property
    def precision(self) -> float:
        """Precision = TP / (TP + FP); ``1.0`` when nothing was predicted."""
        denom = self.true_positives + self.false_positives
        return 1.0 if denom == 0 else self.true_positives / denom

    @property
    def recall(self) -> float:
        """Recall = TP / (TP + FN); ``1.0`` when there is nothing to find."""
        denom = self.true_positives + self.false_negatives
        return 1.0 if denom == 0 else self.true_positives / denom


def _predicted_keys(
    report: VerificationReport, channel: Channel
) -> set[tuple[int, str]]:
    """Extract ``(page_idx, key)`` pairs the report predicts for ``channel``.

    Disagreements are keyed by ``element_id``; missing/extra findings are keyed
    by their ``approx_location`` (matching the labels' curator-chosen keys).

    Args:
        report: The :class:`VerificationReport` produced by ``verify()``.
        channel: ``"disagreement"`` / ``"missing"`` / ``"extra"``.

    Returns:
        The set of predicted ``(page_idx, key)`` pairs for the channel.
    """
    keys: set[tuple[int, str]] = set()
    for page in report.pages:
        if channel == _DISAGREEMENT:
            for dis in page.disagreements:
                keys.add((page.page_idx, dis.element_id))
        elif channel == "missing":
            for miss in page.missing_elements:
                keys.add((page.page_idx, miss.approx_location))
        elif channel == "extra":
            for extra in page.extra_elements:
                keys.add((page.page_idx, extra.approx_location))
    return keys


def _label_keys(labels: list[Label], channels: frozenset[str] | str) -> set[tuple[int, str]]:
    """Return ``(page_idx, key)`` pairs for labels in the given channel(s)."""
    wanted = {channels} if isinstance(channels, str) else channels
    return {(lbl.page_idx, lbl.key) for lbl in labels if lbl.channel in wanted}


def _score_channel(
    reports: dict[str, VerificationReport],
    docs: list[LabeledDocument],
    channels: frozenset[str] | str,
) -> PrecisionRecall:
    """Score one channel group across all documents into a :class:`PrecisionRecall`.

    Args:
        reports: Mapping of document name -> its produced report.
        docs: The labeled documents (carry the ground-truth labels).
        channels: The channel or set of channels to score together.

    Returns:
        Aggregate precision/recall counts for the channel group.
    """
    tp = fp = fn = 0
    channel_list = [channels] if isinstance(channels, str) else sorted(channels)
    for doc in docs:
        report = reports[doc.name]
        predicted: set[tuple[int, str]] = set()
        for ch in channel_list:
            predicted |= _predicted_keys(report, ch)  # type: ignore[arg-type]
        labeled = _label_keys(doc.labels, channels)
        tp += len(predicted & labeled)
        fp += len(predicted - labeled)
        fn += len(labeled - predicted)
    return PrecisionRecall(true_positives=tp, false_positives=fp, false_negatives=fn)


def _run_reports(
    docs: list[LabeledDocument],
    config: VerifierConfig,
    *,
    use_fake: bool,
) -> dict[str, VerificationReport]:
    """Run ``verify()`` over every labeled document and collect the reports.

    On the FAKE path the full-page render is mocked (the fake ignores image
    bytes) and a per-document :class:`FakeVerifierClient` returns that document's
    canned verdicts. On the LIVE path the real backend selected by ``config``
    renders and calls the model; documents in the labeled set use synthetic file
    paths, so a live run requires real fixtures — see ``--live`` notes in the
    module docstring.

    Args:
        docs: The labeled documents to verify.
        config: The active verifier config (its ``force_verify_all`` /
            ``min_severity_to_report`` drive selection and filtering).
        use_fake: When ``True`` (CI default) mock render + inject the fake client.

    The verification cache is pointed at a FRESH directory for this call. The
    severity floor (``min_severity_to_report``) is NOT part of the cache key and
    ``verify()`` stores the already-FILTERED verdict, so a shared cache would let
    a verdict filtered at one floor leak into another floor's score. A per-call
    cache dir keeps every floor's precision/recall scored from the raw verdict.

    Returns:
        Mapping of document name -> its :class:`VerificationReport`.
    """
    import os  # noqa: PLC0415

    os.environ["HYBRID_DOC_PARSER_VERIFIER_CACHE_DIR"] = tempfile.mkdtemp(
        prefix="eval_vcache_floor_"
    )
    reports: dict[str, VerificationReport] = {}
    for doc in docs:
        if use_fake:
            fake = FakeVerifierClient(verdicts=doc.fake_verdicts)
            with mock.patch.object(
                _verifier, "_render_full_page", return_value=b"\x89PNGDATA"
            ), mock.patch.object(
                _verifier, "make_verifier_client", lambda _cfg, _f=fake: _f
            ):
                reports[doc.name] = verify(
                    doc.parser_output, Path(doc.file_path), config
                )
        else:
            reports[doc.name] = verify(
                doc.parser_output, Path(doc.file_path), config
            )
    return reports


@dataclass(frozen=True)
class EvalResult:
    """The full eval outcome the harness computes and the artifact records."""

    chosen_floor: str
    disagreement_pr: PrecisionRecall
    location_pr: PrecisionRecall
    flagged_only_recall: float
    force_all_recall: float
    per_floor: dict[str, tuple[PrecisionRecall, PrecisionRecall]]


def _overall_recall(disagreement: PrecisionRecall, location: PrecisionRecall) -> float:
    """Combined recall across both channel groups (pooled TP / (TP + FN))."""
    tp = disagreement.true_positives + location.true_positives
    fn = disagreement.false_negatives + location.false_negatives
    denom = tp + fn
    return 1.0 if denom == 0 else tp / denom


def _overall_precision(disagreement: PrecisionRecall, location: PrecisionRecall) -> float:
    """Combined precision across both channel groups (pooled TP / (TP + FP))."""
    tp = disagreement.true_positives + location.true_positives
    fp = disagreement.false_positives + location.false_positives
    denom = tp + fp
    return 1.0 if denom == 0 else tp / denom


def run_eval(
    docs: list[LabeledDocument],
    *,
    use_fake: bool,
    base_config: VerifierConfig,
) -> EvalResult:
    """Run the full eval: tune the severity floor and measure the recall gap.

    Sweeps ``min_severity_to_report`` over :data:`_SEVERITY_FLOORS` (all with
    ``force_verify_all=True``), scores precision/recall per channel group at each
    floor, and selects the floor maximising precision (ties -> higher recall).
    Separately measures combined recall under flagged-only selection vs
    ``force_verify_all`` at the chosen floor to quantify the quality-gate gap.

    Args:
        docs: The labeled documents.
        use_fake: Whether to use the deterministic fake path (CI) or a live
            backend.
        base_config: Config template; ``force_verify_all`` /
            ``min_severity_to_report`` are overridden per sweep.

    Returns:
        The :class:`EvalResult` capturing the chosen floor, per-channel
        precision/recall, the per-floor sweep, and the recall gap.
    """
    per_floor: dict[str, tuple[PrecisionRecall, PrecisionRecall]] = {}
    for floor in _SEVERITY_FLOORS:
        config = base_config.model_copy(
            update={"force_verify_all": True, "min_severity_to_report": floor}
        )
        reports = _run_reports(docs, config, use_fake=use_fake)
        dis_pr = _score_channel(reports, docs, _DISAGREEMENT)
        loc_pr = _score_channel(reports, docs, _LOCATION_CHANNELS)
        per_floor[floor] = (dis_pr, loc_pr)

    # Tune for PRECISION over recall: highest combined precision wins; ties go to
    # the floor with higher combined recall.
    def _key(floor: str) -> tuple[float, float]:
        dis_pr, loc_pr = per_floor[floor]
        return (_overall_precision(dis_pr, loc_pr), _overall_recall(dis_pr, loc_pr))

    chosen_floor = max(_SEVERITY_FLOORS, key=_key)
    chosen_dis, chosen_loc = per_floor[chosen_floor]

    # Recall gap: combined recall when only quality-gate-flagged pages are
    # verified, vs every page (force_verify_all), at the chosen floor.
    flagged_cfg = base_config.model_copy(
        update={"force_verify_all": False, "min_severity_to_report": chosen_floor}
    )
    flagged_reports = _run_reports(docs, flagged_cfg, use_fake=use_fake)
    flagged_dis = _score_channel(flagged_reports, docs, _DISAGREEMENT)
    flagged_loc = _score_channel(flagged_reports, docs, _LOCATION_CHANNELS)
    flagged_recall = _overall_recall(flagged_dis, flagged_loc)
    force_all_recall = _overall_recall(chosen_dis, chosen_loc)

    return EvalResult(
        chosen_floor=chosen_floor,
        disagreement_pr=chosen_dis,
        location_pr=chosen_loc,
        flagged_only_recall=flagged_recall,
        force_all_recall=force_all_recall,
        per_floor=per_floor,
    )


def _format_pr(pr: PrecisionRecall) -> str:
    """Render one :class:`PrecisionRecall` as a compact human-readable line."""
    return (
        f"precision={pr.precision:.3f} recall={pr.recall:.3f} "
        f"(TP={pr.true_positives} FP={pr.false_positives} FN={pr.false_negatives})"
    )


def render_artifact(result: EvalResult, *, mode: str) -> str:
    """Render the gate artifact (Markdown) for the spec's verifications folder.

    Args:
        result: The computed :class:`EvalResult`.
        mode: ``"fake"`` (harness validation) or ``"live"`` (real-model figure).

    Returns:
        The Markdown artifact body.
    """
    sweep_lines = []
    for floor in _SEVERITY_FLOORS:
        dis_pr, loc_pr = result.per_floor[floor]
        chosen = " (CHOSEN)" if floor == result.chosen_floor else ""
        sweep_lines.append(
            f"| {floor}{chosen} | {dis_pr.precision:.3f} | {dis_pr.recall:.3f} "
            f"| {loc_pr.precision:.3f} | {loc_pr.recall:.3f} |"
        )
    sweep_table = "\n".join(sweep_lines)

    live_note = (
        "These numbers are the DETERMINISTIC FAKE-PATH result: they validate that "
        "the harness computes precision/recall correctly against the labeled set "
        "(the `FakeVerifierClient` returns known verdicts). They are NOT a "
        "real-model figure. A real-model precision/recall measurement via the "
        "`--live` path (Bedrock or a local vllm/Ollama endpoint, against real PDF "
        "fixtures) is STILL REQUIRED before the feature is recommended for "
        "production use."
        if mode == "fake"
        else "These numbers are a LIVE real-model measurement against the labeled set."
    )

    return f"""# Verifier eval gate — precision/recall artifact

Mode: **{mode}**
Generated by: `scripts/eval_verifier.py`
Labeled set: `scripts/eval_verifier_fixtures.py`

> {live_note}

## Chosen severity floor (precision-favoring tuning)

`min_severity_to_report = "{result.chosen_floor}"` — selected by maximising
combined precision over the labeled set (ties broken toward higher recall), per
the spec's precision-over-recall requirement.

## Measured precision / recall at the chosen floor (force_verify_all)

- Disagreements: {_format_pr(result.disagreement_pr)}
- Missing/Extra: {_format_pr(result.location_pr)}

## Quality-gate recall gap (flagged-only vs force_verify_all)

- Combined recall, flagged-only selection: {result.flagged_only_recall:.3f}
- Combined recall, force_verify_all:        {result.force_all_recall:.3f}
- Recall gap (defects on KEEP pages, unseen by default): \
{result.force_all_recall - result.flagged_only_recall:.3f}

The gap quantifies defects that live on pages the quality gate KEEPS: the
flagged-only default never verifies them, so verifier recall is bounded by the
quality gate's recall. `force_verify_all` exists solely to measure this.

## Severity-floor sweep

| floor | disagreement precision | disagreement recall | missing/extra precision | missing/extra recall |
|-------|------------------------|---------------------|-------------------------|----------------------|
{sweep_table}

## Production recommendation

NOT YET RECOMMENDED on the strength of the fake path alone. The fake path proves
the harness; a `--live` run with real fixtures must report an acceptable
(precision-favoring) figure before the Task Group 8 gate is satisfied for a
production recommendation.
"""


_ARTIFACT_DIR = (
    Path(__file__).resolve().parents[1]
    / "agent-os"
    / "specs"
    / "2026-06-15-llm-disagreement-verifier"
    / "verifications"
)
_ARTIFACT_PATH = _ARTIFACT_DIR / "eval-results.md"


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: run the eval and optionally write the gate artifact."""
    parser = argparse.ArgumentParser(description="Verifier precision/recall eval gate")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use a real backend instead of the deterministic fake path.",
    )
    parser.add_argument(
        "--backend",
        default="bedrock",
        choices=["bedrock", "openai_compatible"],
        help="Live backend to use (only with --live).",
    )
    parser.add_argument("--model", default="", help="Model id for the live backend.")
    parser.add_argument("--region", default="", help="Region for the live backend.")
    parser.add_argument(
        "--write-artifact",
        action="store_true",
        help="Write the precision/recall gate artifact to the spec folder.",
    )
    args = parser.parse_args(argv)

    use_fake = not args.live
    mode = "fake" if use_fake else "live"
    if use_fake:
        base_config = VerifierConfig(backend="fake", model="fake-verifier")
    else:
        base_config = VerifierConfig(
            backend=args.backend, model=args.model, region=args.region
        )

    docs = labeled_documents()
    print(f"[eval] mode={mode}  labeled documents={len(docs)}")
    for doc in docs:
        print(f"[eval]   {doc.name:18s} pages={doc.parser_output.page_count} labels={len(doc.labels)}")

    result = run_eval(docs, use_fake=use_fake, base_config=base_config)

    print("\n[eval] === SEVERITY-FLOOR SWEEP (force_verify_all) ===")
    for floor in _SEVERITY_FLOORS:
        dis_pr, loc_pr = result.per_floor[floor]
        marker = " <- CHOSEN" if floor == result.chosen_floor else ""
        print(f"[eval]   floor={floor:6s} disagreements: {_format_pr(dis_pr)}{marker}")
        print(f"[eval]   floor={floor:6s} missing/extra: {_format_pr(loc_pr)}")

    print(f"\n[eval] chosen min_severity_to_report = {result.chosen_floor!r}")
    print(f"[eval] disagreements: {_format_pr(result.disagreement_pr)}")
    print(f"[eval] missing/extra: {_format_pr(result.location_pr)}")
    print(
        f"[eval] recall gap: flagged_only={result.flagged_only_recall:.3f} "
        f"force_all={result.force_all_recall:.3f} "
        f"gap={result.force_all_recall - result.flagged_only_recall:.3f}"
    )

    if args.write_artifact:
        _ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        _ARTIFACT_PATH.write_text(render_artifact(result, mode=mode), encoding="utf-8")
        print(f"\n[eval] wrote gate artifact -> {_ARTIFACT_PATH}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
