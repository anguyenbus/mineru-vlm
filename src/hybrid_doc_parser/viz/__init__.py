"""Parse Report viewer package (``viz``).

A static-HTML linked review tool for ``hybrid-doc-parser``. This package MAY
import the library; the library MUST NEVER import ``viz`` (the runtime package
must import with zero ``viz`` deps installed).

Pure modules (``coords``, ``model``, ``normalize``, ``html``) import without the
``viz`` optional deps; ``render`` defers its ``pypdfium2``/Pillow imports so it,
too, is pure-importable.
"""

from __future__ import annotations

from hybrid_doc_parser.viz.coords import COORD, to_canonical
from hybrid_doc_parser.viz.html import build_html
from hybrid_doc_parser.viz.model import Doc, Span
from hybrid_doc_parser.viz.normalize import doc_from_parser_output, doc_from_pdfplumber

__all__ = [
    "COORD",
    "to_canonical",
    "build_html",
    "Doc",
    "Span",
    "doc_from_parser_output",
    "doc_from_pdfplumber",
]
