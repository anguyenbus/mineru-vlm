"""Confidence aggregation core for MinerU pipeline ``_middle.json`` dumps.

MinerU's pipeline ``_middle.json`` carries two confidence signals per page:

- a per-block ``score`` from layout detection, and
- a per-span ``score`` from OCR recognition.

This module turns that raw structure into per-page (:class:`PageConfidence`)
and document-level (:class:`DocumentConfidence`) aggregates via the pure
:func:`extract_confidence` function (plus the convenience
:func:`extract_confidence_from_path` wrapper). It is the standalone,
separately-tested aggregation core; wiring it into the parse flow and
``ParserOutput`` is downstream (roadmap item 22), which is why
:class:`PageConfidence` / :class:`DocumentConfidence` are defined locally here
rather than in ``models.py``.

Every public entry point follows the never-raises contract used across the
package: malformed, empty, or non-pipeline input degrades to a well-formed
empty model (``total_pages == 0``, ``None`` aggregates) rather than throwing.

Typical usage:

    from hybrid_doc_parser.confidence import extract_confidence
    confidence = extract_confidence(middle_json)
    if confidence.pages_flagged:
        ...  # surface low-confidence pages
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Final

from loguru import logger
from pydantic import BaseModel, ConfigDict

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

DEFAULT_LOW_CONFIDENCE_THRESHOLD: Final[float] = 0.70
"""Shared threshold below which a block or span score is "low confidence"."""


# ---------------------------------------------------------------------------
# Models (defined locally this slice; item 22 re-exports/moves them)
# ---------------------------------------------------------------------------


class PageConfidence(BaseModel):
    """Per-page confidence aggregates for one MinerU pipeline page.

    The headline fields aggregate over the page's *headline* blocks (exactly
    one of ``para_blocks`` or ``preproc_blocks`` — never both). The parallel
    ``discarded_*`` fields aggregate the same way over the separate
    ``discarded_blocks`` diagnostic bucket and never affect flagging.

    A ``mean_*`` / ``min_*`` field is ``None`` when its dimension's count is
    ``0`` (no data), which is deliberately distinguishable from a genuine
    ``0.0`` score. Count fields are always non-null integers.

    Attributes:
        page_idx: Zero-indexed page number.
        block_count: Number of headline blocks with a numeric ``score``.
        mean_block_score: Mean headline block score, or ``None`` when none.
        min_block_score: Minimum headline block score, or ``None`` when none.
        low_confidence_blocks: Headline blocks scoring strictly below the
            threshold.
        span_count: Number of headline spans with a numeric ``score``.
        mean_span_score: Mean headline span score, or ``None`` when none.
        min_span_score: Minimum headline span score, or ``None`` when none.
        low_confidence_spans: Headline spans scoring strictly below the
            threshold.
        discarded_block_count: As ``block_count`` over ``discarded_blocks``.
        discarded_mean_block_score: As ``mean_block_score`` over the bucket.
        discarded_min_block_score: As ``min_block_score`` over the bucket.
        discarded_low_confidence_blocks: As ``low_confidence_blocks`` over it.
        discarded_span_count: As ``span_count`` over the bucket.
        discarded_mean_span_score: As ``mean_span_score`` over the bucket.
        discarded_min_span_score: As ``min_span_score`` over the bucket.
        discarded_low_confidence_spans: As ``low_confidence_spans`` over it.
        flagged: ``True`` when either headline ``mean_block_score`` or
            ``mean_span_score`` is non-``None`` and strictly below the
            threshold; the discarded bucket never flags.
    """

    model_config = ConfigDict(frozen=True)

    page_idx: int

    block_count: int
    mean_block_score: float | None
    min_block_score: float | None
    low_confidence_blocks: int
    span_count: int
    mean_span_score: float | None
    min_span_score: float | None
    low_confidence_spans: int

    discarded_block_count: int
    discarded_mean_block_score: float | None
    discarded_min_block_score: float | None
    discarded_low_confidence_blocks: int
    discarded_span_count: int
    discarded_mean_span_score: float | None
    discarded_min_span_score: float | None
    discarded_low_confidence_spans: int

    flagged: bool


class DocumentConfidence(BaseModel):
    """Document-level confidence aggregates over all pages.

    Overall means/mins are computed only over pages/dimensions that have data
    (``None`` page dimensions are skipped); an overall dimension with no
    contributing data is ``None``. Document-wide ``low_confidence_*`` are
    summed counts over the headline dimensions.

    Attributes:
        total_pages: Number of pages aggregated.
        pages: Per-page :class:`PageConfidence` records in page order.
        overall_mean_block_score: Mean of headline block scores across pages,
            or ``None`` when no page had block data.
        overall_mean_span_score: Mean of headline span scores across pages,
            or ``None`` when no page had span data.
        overall_min_block_score: Minimum headline block score across pages,
            or ``None``.
        overall_min_span_score: Minimum headline span score across pages, or
            ``None``.
        low_confidence_blocks: Summed headline low-confidence block count.
        low_confidence_spans: Summed headline low-confidence span count.
        pages_flagged: Ordered list of ``page_idx`` values whose page is
            ``flagged``.
        version_name: MinerU ``_version_name`` from the dump, or ``None``.
        backend: MinerU ``_backend`` from the dump, or ``None``.
        source_path: Originating file path; ``None`` this slice (item 22).
    """

    model_config = ConfigDict(frozen=True)

    total_pages: int
    pages: list[PageConfidence]
    overall_mean_block_score: float | None
    overall_mean_span_score: float | None
    overall_min_block_score: float | None
    overall_min_span_score: float | None
    low_confidence_blocks: int
    low_confidence_spans: int
    pages_flagged: list[int]
    version_name: str | None
    backend: str | None
    source_path: str | None


# ---------------------------------------------------------------------------
# Internal score helpers
# ---------------------------------------------------------------------------


def _is_number(value: object) -> bool:
    """``True`` for a real numeric score (``bool`` is rejected)."""
    return isinstance(value, int | float) and not isinstance(value, bool)


class _Stats:
    """Mutable accumulator for one score dimension (blocks or spans)."""

    def __init__(self) -> None:
        self._sum: float = 0.0
        self.count: int = 0
        self.minimum: float | None = None
        self.low_confidence: int = 0

    def add(self, score: float, threshold: float) -> None:
        self._sum += score
        self.count += 1
        if self.minimum is None or score < self.minimum:
            self.minimum = score
        if score < threshold:
            self.low_confidence += 1

    @property
    def mean(self) -> float | None:
        if self.count == 0:
            return None
        return self._sum / self.count


# ---------------------------------------------------------------------------
# Internal iterators (defensive; never raise)
# ---------------------------------------------------------------------------


def _iter_span_scores(block: object) -> Iterator[float]:
    """Yield every numeric span score reachable from ``block``.

    Handles BOTH block shapes with one traversal:

    - SIMPLE blocks: ``block["lines"]`` -> ``line["spans"]``.
    - HIERARCHICAL blocks: ``block["blocks"]`` -> recurse into each sub-block
      -> ``lines`` -> ``spans``.

    Spans without a numeric ``"score"`` key are skipped (a ``0.0`` score is
    kept — it is the bad-OCR signal). Any block/line/span of the wrong type is
    skipped rather than raised on.
    """
    if not isinstance(block, dict):
        return

    sub_blocks = block.get("blocks")
    if isinstance(sub_blocks, list):
        for sub in sub_blocks:
            yield from _iter_span_scores(sub)

    lines = block.get("lines")
    if isinstance(lines, list):
        for line in lines:
            if not isinstance(line, dict):
                continue
            spans = line.get("spans")
            if not isinstance(spans, list):
                continue
            for span in spans:
                if not isinstance(span, dict) or "score" not in span:
                    continue
                score = span["score"]
                if _is_number(score):
                    yield float(score)


def _accumulate_blocks(
    blocks: object,
    threshold: float,
) -> tuple[_Stats, _Stats]:
    """Accumulate block-dimension and span-dimension stats over ``blocks``.

    A block contributes its layout ``score`` to the block dimension only when
    that score is numeric (``None`` or non-numeric is skipped — never counted
    as ``0``). Regardless of its block score, every block contributes any OCR
    span scores it carries to the span dimension. Pure-image blocks (no spans)
    add nothing to the span dimension and so are not low-confidence there.
    """
    block_stats = _Stats()
    span_stats = _Stats()
    if not isinstance(blocks, list):
        return block_stats, span_stats

    for block in blocks:
        if not isinstance(block, dict):
            continue
        score = block.get("score")
        if _is_number(score):
            block_stats.add(float(score), threshold)
        for span_score in _iter_span_scores(block):
            span_stats.add(span_score, threshold)

    return block_stats, span_stats


def _select_headline_blocks(page: dict) -> object:
    """Return the single headline block collection for a page.

    ``para_blocks`` when present and non-empty, else ``preproc_blocks``. Never
    both — in real dumps the two carry identical counts, so counting both would
    double-count every block.
    """
    para = page.get("para_blocks")
    if isinstance(para, list) and para:
        return para
    return page.get("preproc_blocks")


# ---------------------------------------------------------------------------
# Public aggregation entry points
# ---------------------------------------------------------------------------


def _build_page_confidence(page: dict, threshold: float) -> PageConfidence:
    """Aggregate one ``pdf_info`` page into a :class:`PageConfidence`."""
    page_idx_raw = page.get("page_idx")
    page_idx = page_idx_raw if isinstance(page_idx_raw, int) else 0

    head_blocks, head_spans = _accumulate_blocks(_select_headline_blocks(page), threshold)
    disc_blocks, disc_spans = _accumulate_blocks(page.get("discarded_blocks"), threshold)

    flagged = (head_blocks.mean is not None and head_blocks.mean < threshold) or (
        head_spans.mean is not None and head_spans.mean < threshold
    )

    return PageConfidence(
        page_idx=page_idx,
        block_count=head_blocks.count,
        mean_block_score=head_blocks.mean,
        min_block_score=head_blocks.minimum,
        low_confidence_blocks=head_blocks.low_confidence,
        span_count=head_spans.count,
        mean_span_score=head_spans.mean,
        min_span_score=head_spans.minimum,
        low_confidence_spans=head_spans.low_confidence,
        discarded_block_count=disc_blocks.count,
        discarded_mean_block_score=disc_blocks.mean,
        discarded_min_block_score=disc_blocks.minimum,
        discarded_low_confidence_blocks=disc_blocks.low_confidence,
        discarded_span_count=disc_spans.count,
        discarded_mean_span_score=disc_spans.mean,
        discarded_min_span_score=disc_spans.minimum,
        discarded_low_confidence_spans=disc_spans.low_confidence,
        flagged=flagged,
    )


def _mean_or_none(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _min_or_none(values: list[float]) -> float | None:
    return min(values) if values else None


def extract_confidence(
    middle_json: dict,
    low_confidence_threshold: float = DEFAULT_LOW_CONFIDENCE_THRESHOLD,
) -> DocumentConfidence:
    """Aggregate a MinerU pipeline ``_middle.json`` into a DocumentConfidence.

    Pure and never-raising: a non-pipeline, empty, or malformed ``middle_json``
    (missing ``pdf_info``, non-list blocks, non-numeric scores) degrades to a
    well-formed :class:`DocumentConfidence` with ``total_pages == 0``, empty
    ``pages``, ``None`` overall aggregates, zero document-wide counts, and
    ``pages_flagged == []`` — with ``version_name`` / ``backend`` taken from
    whatever the dict provides.

    Args:
        middle_json: The in-memory parsed ``_middle.json`` dict.
        low_confidence_threshold: Shared threshold below which a block or span
            score counts as low confidence; defaults to
            :data:`DEFAULT_LOW_CONFIDENCE_THRESHOLD`.

    Returns:
        A validated :class:`DocumentConfidence`.
    """
    version_name: str | None = None
    backend: str | None = None
    pages: list[PageConfidence] = []

    try:
        if isinstance(middle_json, dict):
            raw_version = middle_json.get("_version_name")
            if isinstance(raw_version, str):
                version_name = raw_version
            raw_backend = middle_json.get("_backend")
            if isinstance(raw_backend, str):
                backend = raw_backend

            pdf_info = middle_json.get("pdf_info")
            if isinstance(pdf_info, list):
                for page in pdf_info:
                    if isinstance(page, dict):
                        pages.append(_build_page_confidence(page, low_confidence_threshold))
    except Exception as exc:  # noqa: BLE001
        logger.debug("[confidence] failed to aggregate middle_json: {}", exc)
        pages = []

    block_means = [p.mean_block_score for p in pages if p.mean_block_score is not None]
    span_means = [p.mean_span_score for p in pages if p.mean_span_score is not None]
    block_mins = [p.min_block_score for p in pages if p.min_block_score is not None]
    span_mins = [p.min_span_score for p in pages if p.min_span_score is not None]

    return DocumentConfidence(
        total_pages=len(pages),
        pages=pages,
        overall_mean_block_score=_mean_or_none(block_means),
        overall_mean_span_score=_mean_or_none(span_means),
        overall_min_block_score=_min_or_none(block_mins),
        overall_min_span_score=_min_or_none(span_mins),
        low_confidence_blocks=sum(p.low_confidence_blocks for p in pages),
        low_confidence_spans=sum(p.low_confidence_spans for p in pages),
        pages_flagged=[p.page_idx for p in pages if p.flagged],
        version_name=version_name,
        backend=backend,
        source_path=None,
    )


def extract_confidence_from_path(
    path: str | Path,
    low_confidence_threshold: float = DEFAULT_LOW_CONFIDENCE_THRESHOLD,
) -> DocumentConfidence:
    """Read a ``_middle.json`` file and delegate to :func:`extract_confidence`.

    Best-effort and never-raising: a missing, unreadable, or unparseable file
    (or one whose top level is not a dict) degrades to an empty
    :class:`DocumentConfidence`, debug-logged with the ``[confidence]`` prefix.

    Args:
        path: Filesystem path to a ``_middle.json`` dump.
        low_confidence_threshold: Passed through to :func:`extract_confidence`.

    Returns:
        A validated :class:`DocumentConfidence`.
    """
    middle_json: dict = {}
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(data, dict):
            middle_json = data
    except Exception as exc:  # noqa: BLE001
        logger.debug("[confidence] failed to read middle_json {}: {}", path, exc)
    return extract_confidence(middle_json, low_confidence_threshold)
