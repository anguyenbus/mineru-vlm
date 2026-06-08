"""Two-layer text quality gate for parsed PDF pages.

Layer 1: Block coverage ratio — compares extracted token count vs
         the PDF embedded text layer token count.
Layer 2: Five heuristic text quality signals — garbled token ratio,
         mean word length, dictionary hit rate, repeated char runs,
         ASCII printable ratio.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Final, Literal

from loguru import logger

from hybrid_doc_parser.models import ElementRecord, ElementType

# NOTE: These thresholds mirror the doc-parser reference implementation.
GARBLED_TOKEN_RATIO_MAX: Final[float] = 0.20
MEAN_WORD_LENGTH_MIN: Final[float] = 2.0
DICT_HIT_RATE_MIN: Final[float] = 0.50
MAX_REPEATED_CHAR_RUN: Final[int] = 6
ASCII_PRINTABLE_MIN: Final[float] = 0.90
COVERAGE_MIN_TEXT_LAYER_TOKENS: Final[int] = 50
COVERAGE_RATIO_MIN: Final[float] = 0.30

_GARBLED_RE: Final[re.Pattern] = re.compile(r"(?<=[A-Za-z])\d|(?<=\d)[A-Za-z]")
_REPEATED_RE: Final[re.Pattern] = re.compile(r"(.)\1{6,}")


@dataclass
class QualitySignals:
    """Heuristic quality signals for a block of text.

    Attributes:
        garbled_token_ratio: Fraction of tokens with mixed alpha/digit characters.
        mean_word_length: Average whitespace-split token length.
        dict_hit_rate: Fraction of tokens that are clean words or numbers.
        has_repeated_char_run: True if any character repeats 7+ times in a row.
        ascii_printable_ratio: Fraction of printable unicode characters in the text.
        failing_signals: Names of signals that exceeded their threshold.
    """

    garbled_token_ratio: float
    mean_word_length: float
    dict_hit_rate: float
    has_repeated_char_run: bool
    ascii_printable_ratio: float
    failing_signals: list[str] = field(default_factory=list)

    @property
    def passes(self) -> bool:
        """Return True if no signals are failing."""
        return len(self.failing_signals) == 0


@dataclass
class Decision:
    """Quality gate decision for a single page.

    Attributes:
        action: Either 'keep' (use MinerU output as-is) or 'promote_to_vlm'.
        reason: Human-readable reason string, or None for clean keeps.
        layer: Which gate layer triggered (1 or 2), or None for clean keeps.
    """

    action: Literal["keep", "promote_to_vlm"]
    reason: str | None
    layer: int | None


def _is_content_token(token: str) -> bool:
    """Return True if token is a clean word or number.

    Alphabetic words and cleanly formatted numbers (thousands separators,
    decimal points, currency symbols, dates, ranges) are considered content.
    Number-dense pages such as financial tables are valid content, not OCR
    garble; genuine garble is caught by the other four heuristic signals.

    Args:
        token: Whitespace-split token to classify.

    Returns:
        True if the token is purely alphabetic or a clean number
        (after stripping common punctuation characters and numeric separators).
    """
    if token.isalpha():
        return True
    stripped = token.strip("()[].,%$:/+-")
    # NOTE: Remove thousands separators, decimal points, and date/range slashes
    # before the digit check so that tokens like "$1,234.00" are treated as
    # content rather than flagged as noise — matching the reference implementation.
    digits = stripped.replace(",", "").replace(".", "").replace("/", "")
    return len(digits) > 0 and digits.isdigit()


def _measure_text_quality(text: str) -> QualitySignals:
    """Compute text quality heuristics for a block of text.

    Args:
        text: Raw concatenated text from page elements.

    Returns:
        QualitySignals with individual signal values and a list of failing signal names.
    """
    tokens = text.split()
    if not tokens:
        return QualitySignals(
            garbled_token_ratio=0.0,
            mean_word_length=0.0,
            dict_hit_rate=0.0,
            has_repeated_char_run=False,
            ascii_printable_ratio=1.0,
            failing_signals=["empty_text"],
        )

    n = len(tokens)
    garbled_token_ratio = sum(1 for t in tokens if _GARBLED_RE.search(t)) / n
    mean_word_length = sum(len(t) for t in tokens) / n
    dict_hit_rate = sum(1 for t in tokens if _is_content_token(t)) / n
    has_repeated_char_run = bool(_REPEATED_RE.search(text))

    char_count = max(len(text), 1)
    printable_count = sum(
        1 for c in text if unicodedata.category(c) != "Cc" and c.isprintable()
    )
    ascii_printable_ratio = printable_count / char_count

    failing: list[str] = []
    if garbled_token_ratio > GARBLED_TOKEN_RATIO_MAX:
        failing.append("garbled_token_ratio")
    if mean_word_length < MEAN_WORD_LENGTH_MIN:
        failing.append("mean_word_length")
    if dict_hit_rate < DICT_HIT_RATE_MIN:
        failing.append("dict_hit_rate")
    if has_repeated_char_run:
        failing.append("repeated_char_run")
    if ascii_printable_ratio < ASCII_PRINTABLE_MIN:
        failing.append("ascii_printable_ratio")

    return QualitySignals(
        garbled_token_ratio=garbled_token_ratio,
        mean_word_length=mean_word_length,
        dict_hit_rate=dict_hit_rate,
        has_repeated_char_run=has_repeated_char_run,
        ascii_printable_ratio=ascii_printable_ratio,
        failing_signals=failing,
    )


_TEXT_TYPES: Final[frozenset[ElementType]] = frozenset(
    {
        ElementType.text,
        ElementType.heading,
        ElementType.list_item,
    }
)
_FURNITURE_TYPES: Final[frozenset[ElementType]] = frozenset(
    {
        ElementType.header,
        ElementType.footer,
        ElementType.page_number,
    }
)


def evaluate_page(
    page_idx: int,
    elements: list[ElementRecord],
    pdf_text_layer_tokens: int | None,
) -> Decision:
    """Evaluate the quality of a parsed page and decide whether to keep or escalate.

    Layer 1 (coverage): If pdf_text_layer_tokens is known and >= COVERAGE_MIN_TEXT_LAYER_TOKENS,
    checks that extracted text tokens cover at least COVERAGE_RATIO_MIN of the PDF text layer.

    Layer 2 (heuristics): Applies five text quality signals to the combined page text.
    Pages with failing signals are escalated to VLM re-extraction.

    Args:
        page_idx: Zero-based page index (used for logging).
        elements: Parsed elements from this page.
        pdf_text_layer_tokens: Token count from the PDF embedded text layer, or None
            to skip Layer 1.

    Returns:
        Decision with action='keep' or action='promote_to_vlm' and the triggering layer.
    """
    try:
        # Layer 1: coverage check
        if (
            pdf_text_layer_tokens is not None
            and pdf_text_layer_tokens >= COVERAGE_MIN_TEXT_LAYER_TOKENS
        ):
            text_elements = [e for e in elements if e.type in _TEXT_TYPES]
            extracted_tokens = sum(len(e.text.split()) for e in text_elements)
            # NOTE: coverage = fraction of PDF text layer tokens represented in extracted elements
            coverage = extracted_tokens / pdf_text_layer_tokens
            if coverage < COVERAGE_RATIO_MIN:
                logger.debug(
                    "[quality_gate] page {} layer1 fail: coverage={:.2f} < {}",
                    page_idx,
                    coverage,
                    COVERAGE_RATIO_MIN,
                )
                return Decision(
                    action="promote_to_vlm",
                    reason=f"coverage={coverage:.2f}<{COVERAGE_RATIO_MIN}",
                    layer=1,
                )

        # Layer 2: text quality heuristics
        content_elements = [
            e for e in elements if e.type not in _FURNITURE_TYPES and e.text
        ]
        combined = "\n".join(e.text for e in content_elements)

        if not combined.strip():
            return Decision(action="keep", reason=None, layer=None)

        signals = _measure_text_quality(combined)

        if signals.failing_signals == ["empty_text"]:
            return Decision(action="keep", reason=None, layer=None)

        if not signals.passes:
            reason = "heuristic_failed: " + ", ".join(signals.failing_signals)
            logger.debug("[quality_gate] page {} layer2 fail: {}", page_idx, reason)
            return Decision(action="promote_to_vlm", reason=reason, layer=2)

        return Decision(action="keep", reason=None, layer=None)

    except Exception as exc:
        logger.warning("[quality_gate] gate_error on page {}: {}", page_idx, exc)
        return Decision(action="keep", reason=f"gate_error: {exc}", layer=None)
