"""Synthesize golden ``ParserOutput`` JSON with KNOWN raw per-backend bboxes.

Run with the project venv:

    .venv/bin/python tests/hybrid_doc_parser/viz/fixtures/make_golden_json.py

Why synthesized, not real-parsed?
---------------------------------
``import mineru`` / ``import docling`` are not available in this environment
(heavy/optional deps), so per view.md §9 we construct deterministic
``ParserOutput`` JSON with raw bboxes in each backend's NATIVE space:

* MinerU  — per-mille (0..1000), top-left origin.
* Docling — PDF points, bottom-left origin, stored ``[l, b, r, t]``.

This makes the golden ``to_canonical`` assertions exact and fully offline. The
one human/GPU step that remains is the live visual-eyeball confirmation that
real boxes align in all three tabs on a real document.

Each emitted file is a real ``ParserOutput.model_dump_json()`` (schema 1.1), so
Group 2's Pydantic-backed ``doc_from_parser_output`` can consume them as-is.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from hybrid_doc_parser.models import (
    ElementRecord,
    ElementType,
    EnrichmentConfig,
    ParserOutput,
)

HERE = Path(__file__).resolve().parent

# Deterministic synthetic sha256 (32 bytes of 0xAB) — never a real document.
_FAKE_SHA = "ab" * 32


def _eid(page_idx: int, block: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{_FAKE_SHA}:{page_idx}:{block}"))


def _record(
    page_idx: int, block: int, etype: ElementType, text: str, bbox: list[float]
) -> ElementRecord:
    return ElementRecord(
        element_id=_eid(page_idx, block),
        type=etype,
        text=text,
        description="",
        bbox=bbox,
        page_idx=page_idx,
        is_enriched=False,
        image_bytes=None,
    )


def _write(path: Path, file_path: str, records: list[ElementRecord]) -> None:
    out = ParserOutput(
        file_path=file_path,
        file_sha256=_FAKE_SHA,
        page_count=1,
        pages=[],
        elements=records,
        warnings=[],
        enrichment_config=EnrichmentConfig(),
    )
    path.write_text(out.model_dump_json(indent=2))


def main() -> None:
    """Write the synthesized golden ParserOutput JSON fixtures."""
    # --- MinerU (per-mille 0..1000, top-left). Page size irrelevant. ---------
    # heading 0.25..0.75 x, 0.10..0.20 y ; left/right body columns.
    mineru = [
        _record(0, 0, ElementType.heading, "# Heading", [250, 100, 750, 200]),
        _record(0, 1, ElementType.text, "left column body", [100, 250, 480, 800]),
        _record(0, 2, ElementType.text, "right column body", [520, 250, 900, 800]),
    ]
    _write(HERE / "us_letter.mineru.json", "us_letter.pdf", mineru)
    # A4 MinerU is identical raw — per-mille is page-relative; this is the point.
    _write(HERE / "a4.mineru.json", "a4.pdf", mineru)
    # Rotated MinerU: per-mille against the rotated page, no extra handling.
    _write(HERE / "rotate90.mineru.json", "rotate90.pdf", mineru)

    # --- Docling (PDF points, bottom-left, stored [l, b, r, t]). -------------
    # Target canonical (top-left): heading 0.10..0.90 x, 0.08..0.15 y on a
    # US-Letter page (612 x 792 pt). bottom-left y: top y=0.92*792=728.64,
    # bottom y=0.85*792=673.20 -> stored [l, b, r, t].
    docling_letter = [
        _record(0, 0, ElementType.heading, "# Heading",
                [61.2, 673.2, 550.8, 728.64]),
        # left column: x 0.10..0.48, y 0.20..0.80 (canonical top-left).
        # bottom-left: top y=0.80*792=633.6, bottom y=0.20*792=158.4.
        _record(0, 1, ElementType.text, "left column body",
                [61.2, 158.4, 293.76, 633.6]),
    ]
    _write(HERE / "us_letter.docling.json", "us_letter.pdf", docling_letter)

    # A4 Docling (595.28 x 841.89 pt). Target canonical heading 0.10..0.90 x,
    # 0.08..0.15 y. top y=0.92*841.89=774.5388, bottom y=0.85*841.89=715.6065.
    docling_a4 = [
        _record(0, 0, ElementType.heading, "# Heading",
                [59.528, 715.6065, 535.752, 774.5388]),
    ]
    _write(HERE / "a4.docling.json", "a4.pdf", docling_a4)

    for p in sorted(HERE.glob("*.json")):
        print(f"  {p.name}: {p.stat().st_size} bytes")


if __name__ == "__main__":
    main()
