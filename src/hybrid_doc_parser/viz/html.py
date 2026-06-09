"""Pure HTML emitter — ``(page images, docs) -> html string``. No PDF deps here.

The overlay boxes are positioned with CSS percentages over the page image using
canonical ``0..1`` top-left coordinates, so the overlay is resolution-independent
— there is NO DPI/pixel math at overlay time. The only thing that must be correct
is the per-backend transform already applied in :mod:`hybrid_doc_parser.viz.coords`.

Layout: a tab per backend (MinerU / Docling / pdfplumber); each tab is a two-pane
linked view — the rendered page (left) with overlay boxes, the extracted units
(right) as clickable blocks. Spans with no geometry are listed in the text pane
labelled "no geometry" and NOT drawn. A span whose ``page`` exceeds the rendered
page count is listed with an "out of range" note and likewise never drawn. Hover
or click an element on either side to highlight the matching region on the other;
click also scrolls it into view.

Every document-derived string is HTML-escaped and rendered inert — document text
and VLM descriptions are untrusted and must never become an injection vector.
"""

from __future__ import annotations

import html

from hybrid_doc_parser.viz.model import Doc

# Colors keyed by element type. An unknown type falls back to a visible grey
# default rather than vanishing.
TYPE_COLORS = {
    "text": "#3b82f6",
    "heading": "#8b5cf6",
    "title": "#8b5cf6",
    "table": "#ef4444",
    "image": "#10b981",
    "figure": "#10b981",
    "equation": "#f59e0b",
    "caption": "#ec4899",
    "list": "#06b6d4",
    "list_item": "#06b6d4",
    "header": "#a3a3a3",
    "footer": "#a3a3a3",
    "page_number": "#a3a3a3",
    "warning": "#dc2626",
}
DEFAULT_COLOR = "#9ca3af"

# Tab order; only backends present in ``docs`` are emitted.
_TAB_ORDER = ("mineru", "docling", "pdfplumber")


def _color(t: str) -> str:
    return TYPE_COLORS.get(str(t).lower(), DEFAULT_COLOR)


def build_html(
    pages_b64: list[str],
    docs: dict[str, Doc],
    title: str = "Parse report",
    notes: list[str] | None = None,
    page_indices: list[int] | None = None,
) -> str:
    """Render the linked two-pane report to a single self-contained HTML string.

    Args:
        pages_b64: One base64 PNG string per RENDERED page (shared across tabs).
        docs: Mapping of backend key -> :class:`Doc`. Tabs are emitted in
            ``mineru``, ``docling``, ``pdfplumber`` order for the keys present.
        title: Report title (escaped).
        notes: Optional report-level notices (e.g. a page-cap truncation note or
            a render-failure message) shown as an escaped banner at the top.
        page_indices: Absolute 0-based page index for each entry in
            ``pages_b64`` (from the renderer's selection). When ``None`` the
            images are assumed to be the document's leading pages
            (``0..len(pages_b64)-1``). This is what lets a non-prefix selection
            such as ``--pages 3-5`` draw each backend's boxes on the CORRECT
            page image instead of the positional one.

    Returns:
        A complete standalone HTML document string.
    """
    order = [b for b in _TAB_ORDER if b in docs]
    npages = len(pages_b64)
    # Map render position -> absolute page index. Default: leading pages.
    abs_pages = page_indices if page_indices is not None else list(range(npages))
    abs_pages = abs_pages[:npages]

    tabs = "".join(
        f'<button class="tab{" active" if k == 0 else ""}" data-tab="{html.escape(b)}">'
        f"{html.escape(b)}</button>"
        for k, b in enumerate(order)
    )

    banner = ""
    if notes:
        items = "".join(f"<li>{html.escape(str(n))}</li>" for n in notes if n)
        if items:
            banner = f'<div class="notes"><ul>{items}</ul></div>'

    panels = [
        _build_panel(b, docs[b], pages_b64, abs_pages, active=(k == 0))
        for k, b in enumerate(order)
    ]

    return _TEMPLATE.format(
        title=html.escape(title), tabs=tabs, banner=banner, panels="".join(panels)
    )


def _build_panel(
    backend: str, doc: Doc, pages_b64: list[str], abs_pages: list[int], active: bool
) -> str:
    """Build one backend tab: image column (boxes) + text column (spans).

    ``abs_pages[pos]`` is the absolute page index of the image at ``pages_b64[pos]``,
    so a span is drawn over an image only when ``span.page`` equals that image's
    ABSOLUTE page index — never its position in a (possibly non-prefix) subset.
    """
    rendered = set(abs_pages)
    img_col: list[str] = []
    for pos, abs_pg in enumerate(abs_pages):
        boxes = []
        for s in doc.spans:
            # Only spans whose absolute page matches THIS image; boxless spans
            # are never drawn. Both still appear in the text pane below.
            if s.page != abs_pg or not s.bbox:
                continue
            x0, y0, x1, y1 = s.bbox
            boxes.append(
                f'<div class="box" data-eid="{html.escape(s.element_id)}" '
                f"style=\"left:{x0 * 100:.3f}%;top:{y0 * 100:.3f}%;"
                f"width:{(x1 - x0) * 100:.3f}%;height:{(y1 - y0) * 100:.3f}%;"
                f"border-color:{_color(s.type)};background:{_color(s.type)}22\"></div>"
            )
        img_col.append(
            f'<div class="page"><img src="data:image/png;base64,{pages_b64[pos]}" '
            f'alt="page {abs_pg}">{"".join(boxes)}</div>'
        )

    text_col: list[str] = []
    # Group spans by page in source order; include spans whose page image was not
    # rendered (out of the selected range) so nothing is silently dropped.
    pages_with_spans = sorted({s.page for s in doc.spans})
    for pg in pages_with_spans:
        page_spans = [s for s in doc.spans if s.page == pg]
        if not page_spans:
            continue
        out_of_range = pg not in rendered
        label = f"page {pg}" + (" — not rendered (not drawn)" if out_of_range else "")
        text_col.append(f'<div class="pagelabel">{html.escape(label)}</div>')
        for s in page_spans:
            # No box drawn when the span has no geometry OR its page was not
            # rendered; both fall back to a labelled, non-drawn text-pane entry.
            drawable = bool(s.bbox) and not out_of_range
            nogeo = "" if drawable else " nogeo"
            if not s.bbox:
                badge = '<span class="badge">no geometry</span>'
            elif out_of_range:
                badge = '<span class="badge">page out of range</span>'
            else:
                badge = ""
            display = html.escape(s.text) or "&nbsp;"
            text_col.append(
                f'<div class="span{nogeo}" data-eid="{html.escape(s.element_id)}" '
                f'style="border-left-color:{_color(s.type)}">'
                f'<span class="etype">{html.escape(s.type)}</span>{badge}'
                f'<div class="etext">{display}</div></div>'
            )

    return (
        f'<div class="panel{" active" if active else ""}" id="panel-{html.escape(backend)}">'
        f'<div class="twopane"><div class="imgcol">{"".join(img_col)}</div>'
        f'<div class="textcol">{"".join(text_col)}</div></div></div>'
    )


_TEMPLATE = """<!doctype html><html><head><meta charset="utf-8">
<title>{title}</title><style>
*{{box-sizing:border-box}}
body{{margin:0;font:14px/1.5 system-ui,sans-serif;color:#111;background:#f6f7f9}}
header{{padding:10px 16px;background:#111;color:#fff;font-weight:600}}
.notes{{background:#fef3c7;color:#92400e;padding:6px 16px;font-size:13px;
  border-bottom:1px solid #fde68a}}
.notes ul{{margin:4px 0;padding-left:20px}}
.tabs{{display:flex;gap:4px;padding:8px 16px;background:#1f2937}}
.tab{{padding:6px 14px;border:0;border-radius:6px 6px 0 0;background:#374151;
  color:#d1d5db;cursor:pointer;font:inherit}}
.tab.active{{background:#f6f7f9;color:#111;font-weight:600}}
.panel{{display:none}} .panel.active{{display:block}}
.twopane{{display:grid;grid-template-columns:1fr 1fr;gap:0;height:calc(100vh - 92px)}}
.imgcol,.textcol{{overflow:auto;padding:16px}}
.imgcol{{background:#e5e7eb}} .textcol{{background:#fff;border-left:1px solid #ddd}}
.page{{position:relative;display:inline-block;margin:0 auto 16px;box-shadow:0 1px 6px #0003}}
.page img{{display:block;width:100%;height:auto}}
.box{{position:absolute;border:2px solid;border-radius:2px;cursor:pointer;
  transition:background .08s,box-shadow .08s}}
.box.hl{{box-shadow:0 0 0 3px #facc15;background:#facc1555 !important;z-index:5}}
.pagelabel{{font-weight:700;color:#6b7280;margin:14px 0 6px;font-size:12px;
  text-transform:uppercase;letter-spacing:.05em}}
.span{{border-left:4px solid;padding:6px 10px;margin:4px 0;border-radius:0 6px 6px 0;
  background:#fafafa;cursor:pointer}}
.span.hl{{background:#fef9c3;box-shadow:0 0 0 2px #facc15}}
.span.nogeo{{opacity:.7;cursor:default}}
.etype{{font-size:11px;color:#6b7280;text-transform:uppercase;letter-spacing:.04em}}
.etext{{white-space:pre-wrap}}
.badge{{font-size:10px;background:#fee2e2;color:#b91c1c;border-radius:4px;
  padding:1px 5px;margin-left:6px}}
</style></head><body>
<header>{title} — hover or click an extracted unit to locate it on the page</header>
{banner}
<div class="tabs">{tabs}</div>
{panels}
<script>
function clear(){{document.querySelectorAll('.hl').forEach(e=>e.classList.remove('hl'));}}
function mark(eid){{document.querySelectorAll('[data-eid="'+eid+'"]')
  .forEach(e=>e.classList.add('hl'));}}
document.querySelectorAll('[data-eid]').forEach(el=>{{
  el.addEventListener('mouseenter',()=>{{mark(el.dataset.eid);}});
  el.addEventListener('mouseleave',clear);
  el.addEventListener('click',()=>{{
    clear();mark(el.dataset.eid);
    const box=document.querySelector('.box[data-eid="'+el.dataset.eid+'"]');
    if(box)box.scrollIntoView({{behavior:'smooth',block:'center'}});
  }});
}});
document.querySelectorAll('.tab').forEach(t=>t.addEventListener('click',()=>{{
  document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));
  document.querySelectorAll('.panel').forEach(x=>x.classList.remove('active'));
  t.classList.add('active');
  document.getElementById('panel-'+t.dataset.tab).classList.add('active');
}}));
</script></body></html>"""
