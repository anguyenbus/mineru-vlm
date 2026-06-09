"""Pure data model for the viewer — the one normalized shape every source maps into.

Heterogeneous sources (MinerU ``ParserOutput``, Docling ``ParserOutput``, the
live pdfplumber baseline) all collapse into a list of :class:`Span` grouped into
a :class:`Doc`. These are plain dataclasses with no PDF/parser dependencies, so
this module is pure-importable and snapshot-testable.

A :class:`Span` is the *clickable unit*: linking happens at element granularity
via ``element_id`` (NOT via the lossy ``render_markdown()`` string), and the
canonical bbox is already normalized to ``0..1`` top-left coordinates. A span
with ``bbox is None`` has no geometry (e.g. a DOCX/HTML element) — it is listed
in the text pane labelled "no geometry" but never drawn on the page.
"""

from __future__ import annotations

from dataclasses import dataclass, field

CanonicalBox = tuple[float, float, float, float]


@dataclass
class Span:
    """One extracted unit, linked to its source region.

    Attributes:
        element_id: Stable per-element id used for two-way hover/click linking.
            For library backends this is the source ``ElementRecord.element_id``;
            for the pdfplumber baseline it is a synthetic ``pdfplumber-<pg>-<j>``.
        page: Zero-indexed page this span belongs to.
        type: Semantic type string (``text``/``heading``/``table``/…), used for
            color coding and the text-pane label.
        text: Inert, document-derived display text. Rendered escaped by the HTML
            layer — never raw HTML/markdown.
        bbox: Canonical ``(x0, y0, x1, y1)`` in ``0..1`` top-left coords, or
            ``None`` when the element has no usable geometry.
    """

    element_id: str
    page: int
    type: str
    text: str
    bbox: CanonicalBox | None


@dataclass
class Doc:
    """All spans for a single backend tab.

    Attributes:
        backend: Backend key (``mineru``/``docling``/``pdfplumber``).
        spans: Spans in source/reading order.
    """

    backend: str
    spans: list[Span] = field(default_factory=list)
