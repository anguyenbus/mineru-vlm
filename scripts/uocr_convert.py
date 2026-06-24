"""Convert Unlimited-OCR intermediate JSON into a library ``ParserOutput``.

Unlimited-OCR (prompt ``document parsing.``) emits grounded segments per page::

    <|det|>LABEL [x0, y0, x1, y1]<|/det|>CONTENT

where LABEL is ``header/title/text/table/footer/page_number/image`` and the
coordinates are normalised per-axis to 0..999 (per-mille, top-left). This module
parses those into ``ElementRecord``s (bbox kept as per-mille so the viewer's
``unlimited`` coordinate transform maps it correctly) and assembles a
``ParserOutput``. Runs in the PROJECT venv (it imports the library models).
"""

from __future__ import annotations

import re

from hybrid_doc_parser.models import (
    ElementRecord,
    ElementType,
    EnrichmentConfig,
    PageRecord,
    ParserOutput,
)

_SEG = re.compile(
    r"<\|det\|>\s*([A-Za-z_]+)\s*\[([0-9.,\s]*)\]\s*<\|/det\|>(.*?)(?=<\|det\|>|\Z)",
    re.DOTALL,
)

# Unlimited-OCR label -> library ElementType.
_LABEL_MAP = {
    "title": ElementType.heading,
    "header": ElementType.header,
    "text": ElementType.text,
    "table": ElementType.table,
    "footer": ElementType.footer,
    "page_number": ElementType.page_number,
    "image": ElementType.image,
    "figure": ElementType.image,
    "caption": ElementType.caption,
    "list": ElementType.list_item,
}


def _clean(s: str) -> str:
    s = re.sub(r"<\|/?det\|>", "", s)
    s = re.sub(r"\[Non-[Tt]ext\]", "", s)
    s = re.sub(r"<[^>]+>", " ", s)  # flatten any html (e.g. inline tables)
    return re.sub(r"\s+", " ", s).strip()


def _clean_keep_html(s: str) -> str:
    """Like _clean but preserve HTML markup — used for table blocks so the
    ``<table>...</table>`` structure Unlimited-OCR emits survives for TEDS."""
    s = re.sub(r"<\|/?det\|>", "", s)
    s = re.sub(r"\[Non-[Tt]ext\]", "", s)
    return re.sub(r"[ \t]+", " ", s).strip()


def _bbox(coords: str) -> list[float]:
    nums = [float(x) for x in re.split(r"[,\s]+", coords.strip()) if x]
    if len(nums) != 4:
        return []
    # clamp into the per-mille range so the viewer's transform stays valid
    return [max(0.0, min(999.0, n)) for n in nums]


def _page_elements(raw_text: str, page_index: int, start_seq: int) -> list[ElementRecord]:
    out: list[ElementRecord] = []
    segs = list(_SEG.finditer(raw_text))
    seq = start_seq

    if not segs:
        # No grounding markers — emit each non-empty line as a text element.
        for line in (ln.strip() for ln in raw_text.splitlines()):
            if not line:
                continue
            out.append(
                ElementRecord(
                    element_id=f"uocr-{seq:05d}",
                    type=ElementType.text,
                    text=_clean(line),
                    bbox=[],
                    page_idx=page_index,
                )
            )
            seq += 1
        return out

    for m in segs:
        label = m.group(1).lower()
        etype = _LABEL_MAP.get(label, ElementType.text)
        raw = m.group(3)
        # Preserve <table> HTML so table structure survives for TEDS; flatten
        # everything else to plain text.
        text = _clean_keep_html(raw) if (etype is ElementType.table and "<table" in raw.lower()) else _clean(raw)
        bbox = _bbox(m.group(2))
        if not text and etype is not ElementType.image:
            continue
        out.append(
            ElementRecord(
                element_id=f"uocr-{seq:05d}",
                type=etype,
                text=text,
                bbox=bbox,
                page_idx=page_index,
            )
        )
        seq += 1
    return out


def build_parser_output(intermediate: dict) -> ParserOutput:
    """Assemble a ``ParserOutput`` from run_unlimited_ocr.py's intermediate JSON."""
    elements: list[ElementRecord] = []
    seq = 0
    page_counts: dict[int, int] = {}
    for page in intermediate.get("pages", []):
        pi = int(page.get("page_index", 0))
        page_els = _page_elements(page.get("raw_text", ""), pi, seq)
        seq += len(page_els)
        elements.extend(page_els)
        page_counts[pi] = len(page_els)

    page_count = int(intermediate.get("page_count", len(page_counts) or 1))
    pages = [
        PageRecord(
            page_idx=i,
            quality_decision="keep",
            element_count=page_counts.get(i, 0),
            vlm_used=True,
        )
        for i in range(page_count)
    ]

    return ParserOutput(
        file_path=intermediate.get("file_path", ""),
        file_sha256=intermediate.get("file_sha256", ""),
        page_count=page_count,
        pages=pages,
        elements=elements,
        warnings=[],
        # Placeholder: Unlimited-OCR is not one of EnrichmentConfig's parser
        # literals (it runs out-of-process in its own venv). The viewer keys the
        # coordinate transform on the explicit backend name, not this field.
        enrichment_config=EnrichmentConfig(),
        confidence=None,
    )
