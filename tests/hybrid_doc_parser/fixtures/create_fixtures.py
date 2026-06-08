"""Fixture PDF creation utility.

One-time script to generate minimal PDF fixtures used in tests. Run with:

    python tests/hybrid_doc_parser/fixtures/create_fixtures.py

Requires fpdf2 (pip install fpdf2) or falls back to manually crafted minimal
PDF bytes when neither reportlab nor fpdf2 is available.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent


def _build_minimal_pdf_with_text(text: str) -> bytes:
    """Build a minimal single-page PDF containing machine-readable text.

    Uses raw PDF syntax so no third-party library is required. The text is
    embedded as a Type 1 Helvetica font string so that pypdfium2's text
    extraction layer can read it.

    Args:
        text: The text string to embed on page 1.

    Returns:
        Raw PDF bytes.
    """
    # Escape parentheses for PDF string syntax
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    content_stream = (
        f"BT\n"
        f"/F1 12 Tf\n"
        f"50 750 Td\n"
        f"({escaped}) Tj\n"
        f"ET\n"
    ).encode("latin-1")

    objects: list[bytes] = []

    def add_obj(body: str) -> int:
        idx = len(objects) + 1
        objects.append(f"{idx} 0 obj\n{body}\nendobj\n".encode())
        return idx

    # obj 1: catalog
    catalog_idx = add_obj("<< /Type /Catalog /Pages 2 0 R >>")
    # obj 2: pages node (forward reference to obj 3)
    pages_idx = add_obj("<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    # obj 3: page
    page_idx = add_obj(
        "<< /Type /Page /Parent 2 0 R "
        "/MediaBox [0 0 612 792] "
        "/Contents 4 0 R "
        "/Resources << /Font << /F1 5 0 R >> >> >>"
    )
    # obj 4: content stream
    stream_len = len(content_stream)
    stream_obj = (
        f"4 0 obj\n"
        f"<< /Length {stream_len} >>\n"
        f"stream\n"
    ).encode() + content_stream + b"\nendstream\nendobj\n"
    objects.append(stream_obj)
    # obj 5: font
    font_idx = add_obj(
        "<< /Type /Font /Subtype /Type1 "
        "/BaseFont /Helvetica "
        "/Encoding /WinAnsiEncoding >>"
    )

    header = b"%PDF-1.4\n"
    body = b"".join(objects)
    xref_offset = len(header) + len(body)

    offsets: list[int] = []
    pos = len(header)
    for obj_bytes in objects:
        offsets.append(pos)
        pos += len(obj_bytes)

    xref_count = len(objects) + 1
    xref = f"xref\n0 {xref_count}\n0000000000 65535 f \n".encode()
    for off in offsets:
        xref += f"{off:010d} 00000 n \n".encode()

    trailer = (
        f"trailer\n"
        f"<< /Size {xref_count} /Root 1 0 R >>\n"
        f"startxref\n"
        f"{xref_offset}\n"
        f"%%EOF\n"
    ).encode()

    return header + body + xref + trailer


def _build_minimal_pdf_with_image() -> bytes:
    """Build a minimal single-page PDF with a white PNG image and no text layer.

    The page contains only an XObject image resource, so pypdfium2's text
    extraction returns an empty string (simulating a scanned page).

    Returns:
        Raw PDF bytes.
    """
    # Minimal 4x4 white PNG (smallest valid RGB PNG)
    w, h = 4, 4
    raw_rows = b"".join(b"\x00" + bytes([255, 255, 255] * w) for _ in range(h))
    def _png_chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    png_bytes = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + _png_chunk(b"IDAT", zlib.compress(raw_rows))
        + _png_chunk(b"IEND", b"")
    )
    img_len = len(png_bytes)

    objects: list[bytes] = []

    def _raw(body_bytes: bytes) -> None:
        objects.append(body_bytes)

    # obj 1: catalog
    objects.append(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
    # obj 2: pages
    objects.append(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n")
    # obj 3: page with image XObject
    objects.append(
        b"3 0 obj\n"
        b"<< /Type /Page /Parent 2 0 R "
        b"/MediaBox [0 0 612 792] "
        b"/Contents 4 0 R "
        b"/Resources << /XObject << /Im1 5 0 R >> >> >>\n"
        b"endobj\n"
    )
    # obj 4: content stream (draws the image)
    cs = b"q 612 0 0 792 0 0 cm /Im1 Do Q\n"
    objects.append(
        f"4 0 obj\n<< /Length {len(cs)} >>\nstream\n".encode()
        + cs
        + b"\nendstream\nendobj\n"
    )
    # obj 5: image XObject (PNG embedded as raw stream)
    objects.append(
        f"5 0 obj\n"
        f"<< /Type /XObject /Subtype /Image "
        f"/Width {w} /Height {h} "
        f"/ColorSpace /DeviceRGB /BitsPerComponent 8 "
        f"/Filter /FlateDecode "
        f"/Length {img_len} >>\nstream\n".encode()
        + png_bytes
        + b"\nendstream\nendobj\n"
    )

    header = b"%PDF-1.4\n"
    body = b"".join(objects)
    xref_offset = len(header) + len(body)

    offsets: list[int] = []
    pos = len(header)
    for obj_bytes in objects:
        offsets.append(pos)
        pos += len(obj_bytes)

    xref_count = len(objects) + 1
    xref = f"xref\n0 {xref_count}\n0000000000 65535 f \n".encode()
    for off in offsets:
        xref += f"{off:010d} 00000 n \n".encode()

    trailer = (
        f"trailer\n<< /Size {xref_count} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n"
    ).encode()

    return header + body + xref + trailer


def _build_two_page_pdf(page0_text: str) -> bytes:
    """Build a two-page PDF: page 0 has text, page 1 is image-only.

    Args:
        page0_text: Text string to embed on page 0.

    Returns:
        Raw PDF bytes.
    """
    escaped = page0_text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    cs0 = (
        f"BT\n/F1 12 Tf\n50 750 Td\n({escaped}) Tj\nET\n"
    ).encode("latin-1")

    # Minimal white PNG for page 1
    w, h = 4, 4
    raw_rows = b"".join(b"\x00" + bytes([255, 255, 255] * w) for _ in range(h))
    def _png_chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    png_bytes = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + _png_chunk(b"IDAT", zlib.compress(raw_rows))
        + _png_chunk(b"IEND", b"")
    )
    cs1 = b"q 612 0 0 792 0 0 cm /Im1 Do Q\n"

    objs: list[bytes] = []
    # 1: catalog
    objs.append(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
    # 2: pages (2 kids)
    objs.append(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R 6 0 R] /Count 2 >>\nendobj\n")
    # 3: page 0 (text)
    objs.append(
        b"3 0 obj\n"
        b"<< /Type /Page /Parent 2 0 R "
        b"/MediaBox [0 0 612 792] "
        b"/Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>\n"
        b"endobj\n"
    )
    # 4: content stream page 0
    objs.append(
        f"4 0 obj\n<< /Length {len(cs0)} >>\nstream\n".encode()
        + cs0
        + b"\nendstream\nendobj\n"
    )
    # 5: font
    objs.append(
        b"5 0 obj\n"
        b"<< /Type /Font /Subtype /Type1 "
        b"/BaseFont /Helvetica "
        b"/Encoding /WinAnsiEncoding >>\n"
        b"endobj\n"
    )
    # 6: page 1 (image only)
    objs.append(
        b"6 0 obj\n"
        b"<< /Type /Page /Parent 2 0 R "
        b"/MediaBox [0 0 612 792] "
        b"/Contents 7 0 R "
        b"/Resources << /XObject << /Im1 8 0 R >> >> >>\n"
        b"endobj\n"
    )
    # 7: content stream page 1
    objs.append(
        f"7 0 obj\n<< /Length {len(cs1)} >>\nstream\n".encode()
        + cs1
        + b"\nendstream\nendobj\n"
    )
    # 8: image XObject
    img_len = len(png_bytes)
    objs.append(
        f"8 0 obj\n"
        f"<< /Type /XObject /Subtype /Image "
        f"/Width {w} /Height {h} "
        f"/ColorSpace /DeviceRGB /BitsPerComponent 8 "
        f"/Filter /FlateDecode "
        f"/Length {img_len} >>\nstream\n".encode()
        + png_bytes
        + b"\nendstream\nendobj\n"
    )

    header = b"%PDF-1.4\n"
    body_bytes = b"".join(objs)
    xref_offset = len(header) + len(body_bytes)

    offsets: list[int] = []
    pos = len(header)
    for ob in objs:
        offsets.append(pos)
        pos += len(ob)

    xref_count = len(objs) + 1
    xref = f"xref\n0 {xref_count}\n0000000000 65535 f \n".encode()
    for off in offsets:
        xref += f"{off:010d} 00000 n \n".encode()

    trailer = (
        f"trailer\n<< /Size {xref_count} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n"
    ).encode()

    return header + body_bytes + xref + trailer


def create_digital_simple() -> None:
    """Create digital_simple.pdf — single page with machine-readable text."""
    text = (
        "This is a digital PDF document with extractable text. "
        "It contains multiple sentences for testing purposes."
    )
    pdf_bytes = _build_minimal_pdf_with_text(text)
    out = FIXTURES_DIR / "digital_simple.pdf"
    out.write_bytes(pdf_bytes)
    print(f"Created {out} ({len(pdf_bytes)} bytes)")


def create_scanned() -> None:
    """Create scanned.pdf — single page with embedded image only, no text layer."""
    pdf_bytes = _build_minimal_pdf_with_image()
    out = FIXTURES_DIR / "scanned.pdf"
    out.write_bytes(pdf_bytes)
    print(f"Created {out} ({len(pdf_bytes)} bytes)")


def create_mixed() -> None:
    """Create mixed.pdf — page 0 has text, page 1 is image-only."""
    text = "This is the digital page of the mixed PDF document."
    pdf_bytes = _build_two_page_pdf(text)
    out = FIXTURES_DIR / "mixed.pdf"
    out.write_bytes(pdf_bytes)
    print(f"Created {out} ({len(pdf_bytes)} bytes)")


def create_equation_heavy() -> None:
    """Create equation_heavy.pdf — single page with LaTeX-like notation in text."""
    text = (
        "The integral formula is defined as: "
        "int_0^1 f(x) dx = F(1) - F(0). "
        "This is used in calculus."
    )
    pdf_bytes = _build_minimal_pdf_with_text(text)
    out = FIXTURES_DIR / "equation_heavy.pdf"
    out.write_bytes(pdf_bytes)
    print(f"Created {out} ({len(pdf_bytes)} bytes)")


if __name__ == "__main__":
    create_digital_simple()
    create_scanned()
    create_mixed()
    create_equation_heavy()
    print("All fixtures created successfully.")
