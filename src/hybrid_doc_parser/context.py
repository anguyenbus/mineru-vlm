"""Context extraction utilities for modal processor prompts.

Extracts surrounding text from MinerU content_list to provide context
when prompting a VLM to describe images, tables, or equations.

Typical usage::

    extractor = ContextExtractor(content_list, window=3, max_tokens=512)
    block_context = extractor.extract_for_block(block_index)
    page_context = extractor.extract_for_page(page_idx)
"""

from __future__ import annotations

# NOTE: This module has no external dependencies beyond the Python stdlib.
#       It is intentionally kept dependency-free so it can be imported early
#       in the pipeline without triggering heavy imports (e.g. MinerU).

_TEXT_TYPES: frozenset[str] = frozenset({"text", "title"})


class ContextExtractor:
    """Extracts surrounding text context from a MinerU content_list.

    Both public methods are designed to never raise: any internal error is
    caught and an empty string is returned instead, so callers can safely
    embed the result in a VLM prompt without defensive wrapping.

    Attributes:
        _content_list: Raw MinerU output list of block dicts.
        _window: Number of surrounding blocks/pages to collect.
        _max_tokens: Maximum whitespace-split token count before truncation.
    """

    __slots__ = ("_content_list", "_window", "_max_tokens")

    def __init__(
        self,
        content_list: list[dict],
        window: int = 3,
        max_tokens: int = 512,
    ) -> None:
        """Initialise the extractor with a content list and windowing parameters.

        Args:
            content_list: Raw MinerU output list of block dicts.
            window: Number of surrounding blocks (for ``extract_for_block``) or
                adjacent pages (for ``extract_for_page``) to collect context from.
                Defaults to 3.
            max_tokens: Maximum whitespace-split word count before the combined
                context string is truncated and ``"..."`` is appended.
                Defaults to 512.
        """
        self._content_list = content_list
        self._window = window
        self._max_tokens = max_tokens

    def _truncate(self, text: str) -> str:
        """Truncate *text* to ``_max_tokens`` words, appending ``"..."`` if truncated.

        Args:
            text: The combined context string to potentially truncate.

        Returns:
            The original text if within the word limit, otherwise the first
            ``_max_tokens`` words joined by spaces followed by ``"..."``.
        """
        words = text.split()
        if len(words) <= self._max_tokens:
            return text
        return " ".join(words[: self._max_tokens]) + "..."

    def extract_for_block(self, block_index: int) -> str:
        """Extract surrounding text for a specific block by index.

        Collects text from items within ``_window`` positions of
        ``block_index``, excluding the item at ``block_index`` itself. Only
        items whose ``type`` field is in ``{"text", "title"}`` are included.
        The collected strings are joined with ``"\\n"`` and truncated at
        ``_max_tokens`` words.

        Args:
            block_index: Zero-based index into the content_list.

        Returns:
            Concatenated surrounding text, or ``""`` on any error.
        """
        try:
            start = max(0, block_index - self._window)
            end = min(len(self._content_list), block_index + self._window + 1)
            texts = []
            for i in range(start, end):
                if i == block_index:
                    continue
                item = self._content_list[i]
                if item.get("type") not in _TEXT_TYPES:
                    continue
                t = item.get("text", "") or item.get("content", "")
                if t:
                    texts.append(t)
            return self._truncate("\n".join(texts))
        except Exception:
            return ""

    def extract_for_page(self, page_idx: int) -> str:
        """Extract surrounding text for an entire page by page index.

        Collects text from items on pages ``page_idx - 1``, ``page_idx``, and
        ``page_idx + 1``. Only items whose ``type`` field is in
        ``{"text", "title"}`` are included. The collected strings are joined
        with ``"\\n"`` and truncated at ``_max_tokens`` words.

        Args:
            page_idx: Zero-based page index.

        Returns:
            Concatenated surrounding text, or ``""`` on any error.
        """
        try:
            target_pages = {page_idx - 1, page_idx, page_idx + 1}
            texts = []
            for item in self._content_list:
                if item.get("page_idx") not in target_pages:
                    continue
                if item.get("type") not in _TEXT_TYPES:
                    continue
                t = item.get("text", "") or item.get("content", "")
                if t:
                    texts.append(t)
            return self._truncate("\n".join(texts))
        except Exception:
            return ""
