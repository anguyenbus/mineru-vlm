"""Generate the four Task Group 1 calibration fixtures (regenerable, tiny).

Run with the project venv:

    .venv/bin/python tests/hybrid_doc_parser/viz/fixtures/make_fixtures.py

Produces, alongside this file:

* ``us_letter.pdf``   — vanilla US-Letter (612 x 792 pt) single page.
* ``a4.pdf``          — A4 / non-Letter (595.28 x 841.89 pt) single page.
* ``rotate90.pdf``    — a single page carrying ``/Rotate 90``.
* ``image_input.png`` — a small raster "page" (no point system; pixels = units).

These are the source artifacts the calibration spike overlays against the
pdfplumber reference tab. The golden ``to_canonical`` assertions live in the
sibling test module and use *synthesized* raw per-backend bboxes (MinerU
per-mille 0..1000 top-left; Docling PDF points bottom-left) so the goldens are
exact and fully offline — no MinerU/Docling/GPU dependency. See
``planning/calibration-notes.md``.
"""

from __future__ import annotations

import re
from pathlib import Path

HERE = Path(__file__).resolve().parent

# US-Letter and A4 sizes in PDF points (1 pt = 1/72 in).
US_LETTER_PT = (612.0, 792.0)
A4_PT = (595.28, 841.89)


def _draw_boxes(pdf, w_pt: float, h_pt: float) -> None:
    """Draw a few labelled rectangles at known fractions of the page."""
    pdf.set_font("Helvetica", size=14)
    # heading: x 10%..90%, y 8%..15%
    pdf.rect(w_pt * 0.10, h_pt * 0.08, w_pt * 0.80, h_pt * 0.07)
    pdf.text(w_pt * 0.12, h_pt * 0.13, "CALIBRATION FIXTURE - heading")
    # left column body: x 10%..48%, y 20%..80%
    pdf.rect(w_pt * 0.10, h_pt * 0.20, w_pt * 0.38, h_pt * 0.60)
    pdf.text(w_pt * 0.12, h_pt * 0.25, "left column body text")
    # right column body: x 52%..90%, y 20%..80%
    pdf.rect(w_pt * 0.52, h_pt * 0.20, w_pt * 0.38, h_pt * 0.60)
    pdf.text(w_pt * 0.54, h_pt * 0.25, "right column body text")


def _make_pdf(path: Path, size_pt: tuple[float, float]) -> None:
    from fpdf import FPDF  # noqa: PLC0415

    w_pt, h_pt = size_pt
    pdf = FPDF(unit="pt", format=(w_pt, h_pt))
    pdf.add_page()
    _draw_boxes(pdf, w_pt, h_pt)
    pdf.output(str(path))


def _make_rotated_pdf(path: Path) -> None:
    """US-Letter page with a /Rotate 90 entry injected into the page dict."""
    import pypdfium2 as pdfium  # noqa: PLC0415

    base = HERE / "_rotate90_base.pdf"
    _make_pdf(base, US_LETTER_PT)
    data = base.read_bytes()
    # Match the leaf page object only: "/Type /Page" NOT followed by "s" (the
    # "/Type /Pages" page-tree root must be left intact). fpdf2 writes plain
    # (uncompressed) page objects, so a regex insert is reliable.
    if b"/Rotate" not in data:
        data = re.sub(
            rb"/Type\s*/Page(?![s])",
            b"/Type /Page /Rotate 90",
            data,
            count=1,
        )
    path.write_bytes(data)
    base.unlink(missing_ok=True)
    # sanity: confirm pypdfium2 reports the rotation
    doc = pdfium.PdfDocument(str(path))
    rot = doc[0].get_rotation()
    doc.close()
    if rot != 90:
        raise RuntimeError(f"rotate90.pdf rotation injection failed: rot={rot}")


def _make_image(path: Path) -> None:
    from PIL import Image, ImageDraw  # noqa: PLC0415

    # tiny: 400 x 300 px so pixel = unit math is obvious in goldens.
    img = Image.new("RGB", (400, 300), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([40, 24, 360, 45], outline="#000")  # heading 10%..90%, 8%..15%
    d.rectangle([40, 60, 192, 240], outline="#000")  # left col
    d.text((48, 70), "image-input fixture", fill="#000")
    img.save(str(path))


def main() -> None:
    """Generate all four calibration fixtures alongside this file."""
    _make_pdf(HERE / "us_letter.pdf", US_LETTER_PT)
    _make_pdf(HERE / "a4.pdf", A4_PT)
    _make_rotated_pdf(HERE / "rotate90.pdf")
    _make_image(HERE / "image_input.png")
    for name in ("us_letter.pdf", "a4.pdf", "rotate90.pdf", "image_input.png"):
        p = HERE / name
        print(f"  {name}: {p.stat().st_size} bytes")


if __name__ == "__main__":
    main()
