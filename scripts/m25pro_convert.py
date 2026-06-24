"""Convert MinerU2.5-Pro intermediate JSON into a library ``ParserOutput``.

Mirrors ``hybrid_doc_parser.parser._run_mineru25pro``: bbox is normalised
[0,1] top-left from the VLM and scaled to per-mille (0..1000) so the viewer's
``mineru25pro`` coordinate transform maps it; block type is mapped via the same
label map. Runs in the PROJECT venv (imports the library models).
"""

from __future__ import annotations

from hybrid_doc_parser.models import (
    ElementRecord,
    ElementType,
    EnrichmentConfig,
    PageRecord,
    ParserOutput,
)

# Identical to parser._MINERU25PRO_LABEL_MAP.
_LABEL_MAP = {
    "text": ElementType.text,
    "aside_text": ElementType.text,
    "ref_text": ElementType.text,
    "index": ElementType.text,
    "phonetic": ElementType.text,
    "algorithm": ElementType.text,
    "code": ElementType.text,
    "title": ElementType.heading,
    "table": ElementType.table,
    "equation": ElementType.equation,
    "equation_block": ElementType.equation,
    "formula_number": ElementType.equation,
    "list": ElementType.list_item,
    "list_item": ElementType.list_item,
    "table_caption": ElementType.caption,
    "image_caption": ElementType.caption,
    "code_caption": ElementType.caption,
    "table_footnote": ElementType.caption,
    "image_footnote": ElementType.caption,
    "header": ElementType.header,
    "footer": ElementType.footer,
    "page_footnote": ElementType.footer,
    "page_number": ElementType.page_number,
    "image": ElementType.image,
    "image_block": ElementType.image,
    "chart": ElementType.image,
}


def build_parser_output(intermediate: dict) -> ParserOutput:
    """Assemble a ``ParserOutput`` from run_mineru25pro.py's intermediate JSON."""
    sha = intermediate.get("file_sha256", "")
    elements: list[ElementRecord] = []
    seq = 0
    page_counts: dict[int, int] = {}
    for page in intermediate.get("pages", []):
        pi = int(page.get("page_index", 0))
        n = 0
        for blk in page.get("blocks", []):
            label = str(blk.get("type", "") or "").lower()
            etype = _LABEL_MAP.get(label, ElementType.unknown)
            content = blk.get("content", "") or ""
            text = content if isinstance(content, str) else str(content)
            bbox = blk.get("bbox")
            bbox_pm: list[float] = []
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                try:
                    bbox_pm = [float(v) * 1000.0 for v in bbox]
                except (TypeError, ValueError):
                    bbox_pm = []
            elements.append(
                ElementRecord(
                    element_id=f"m25pro-{seq:05d}",
                    type=etype,
                    text=text.strip(),
                    bbox=bbox_pm,
                    page_idx=pi,
                )
            )
            seq += 1
            n += 1
        page_counts[pi] = n

    page_count = int(intermediate.get("page_count", len(page_counts) or 1))
    pages = [
        PageRecord(page_idx=i, quality_decision="keep", element_count=page_counts.get(i, 0), vlm_used=True)
        for i in range(page_count)
    ]
    return ParserOutput(
        file_path=intermediate.get("file_path", ""),
        file_sha256=sha,
        page_count=page_count,
        pages=pages,
        elements=elements,
        warnings=[],
        enrichment_config=EnrichmentConfig(parser="mineru25pro"),
        confidence=None,
    )
