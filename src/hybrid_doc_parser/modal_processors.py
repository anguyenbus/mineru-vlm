"""Modal processors for image, table, and equation enrichment.

Each processor:
1. Extracts surrounding text context via ContextExtractor
2. Constructs a type-specific VLM prompt
3. Calls VLMClient
4. Returns a plain-language description string

Enrichment is only invoked by parser.py when EnrichmentConfig enables
the corresponding modality.
"""

from __future__ import annotations

from typing import Any, Final

from loguru import logger

from hybrid_doc_parser.context import ContextExtractor
from hybrid_doc_parser.vlm_client import VLMClient

# NOTE: Prompt templates mirror RAG-Anything's modal processor prompts,
# adapted for plain-language output compatible with keyword-based retrieval.

IMAGE_PROMPT_TEMPLATE: Final[str] = """\
You are analyzing a figure or image extracted from a document.

Surrounding document context:
{context}

Describe the image in plain language. Include: what type of visualization it is
(chart, diagram, photo, etc.), what data or concept it represents, key labels
or annotations visible, and any notable trends or findings. Output JSON:
{{"description": "<plain language description>"}}"""

TABLE_PROMPT_TEMPLATE: Final[str] = """\
You are analyzing a table extracted from a document.

Surrounding document context:
{context}

Table content:
{table_content}

Describe the table in plain language. Include: the table's topic, column headers,
row structure, and key data points or patterns. Output JSON:
{{"description": "<plain language description>"}}"""

EQUATION_PROMPT_TEMPLATE: Final[str] = """\
You are analyzing a mathematical equation or formula extracted from a document.

Surrounding document context:
{context}

Equation (LaTeX):
{latex}

Explain this equation in plain language: what it represents, what each symbol
means, and its significance in the document context. Output JSON:
{{"description": "<plain language description>"}}"""

GENERIC_PROMPT_TEMPLATE: Final[str] = """\
You are analyzing a {content_type} element extracted from a document.

Surrounding document context:
{context}

Content:
{content}

Describe this content in plain language. Output JSON:
{{"description": "<plain language description>"}}"""


class ImageModalProcessor:
    """Enriches image blocks by generating a plain-language VLM description.

    Args:
        vlm: VLMClient instance for inference calls.
        content_list: Raw MinerU content_list for context extraction.
        context_window: Number of surrounding blocks to include in the prompt.
    """

    __slots__ = ("_vlm", "_extractor")

    def __init__(
        self,
        vlm: VLMClient,
        content_list: list[dict],
        context_window: int = 3,
    ) -> None:
        """Initialise the processor with a VLM client and content list.

        Args:
            vlm: VLMClient instance for inference calls.
            content_list: Raw MinerU content_list for context extraction.
            context_window: Number of surrounding blocks to include in the prompt.
        """
        self._vlm = vlm
        self._extractor = ContextExtractor(content_list, window=context_window)

    def process(
        self,
        block: dict[str, Any],
        block_index: int,
        image_bytes: bytes | None,
    ) -> str:
        """Generate a plain-language description for an image block.

        Args:
            block: MinerU content_list block dict of type 'image'.
            block_index: Zero-based index of the block in the content_list.
            image_bytes: Optional rasterized PNG bytes of the image region.

        Returns:
            Plain-language description string, or '' on any error or VLM failure.
        """
        try:
            context = self._extractor.extract_for_block(block_index)
            prompt = IMAGE_PROMPT_TEMPLATE.format(context=context or "(none)")
            response = self._vlm.call(image_bytes, prompt, "image")
            if "error" in response:
                logger.debug("[image_processor] VLM error: {}", response["error"])
                return ""
            return str(response.get("description", "")).strip()
        except Exception as exc:
            logger.warning("[image_processor] process failed: {}", exc)
            return ""

    def enrich(
        self,
        block: dict[str, Any],
        block_index: int,
        content_list: list[dict],
        vlm_client: VLMClient,
        config: Any,
    ) -> str:
        """Enrich an image block using the provided VLM client and config.

        Compatibility method matching the tasks.md enrich() interface.
        Delegates to process() using image_bytes from block.get("image_bytes").

        Args:
            block: MinerU content_list block dict of type 'image'.
            block_index: Zero-based index of the block in the content_list.
            content_list: Full content_list for context extraction.
            vlm_client: VLMClient instance for inference calls.
            config: EnrichmentConfig with context_window and max_context_tokens.

        Returns:
            Plain-language description string, or '' on any error or VLM failure.
        """
        try:
            extractor = ContextExtractor(
                content_list,
                window=config.context_window,
                max_tokens=config.max_context_tokens,
            )
            context = extractor.extract_for_block(block_index)
            prompt = IMAGE_PROMPT_TEMPLATE.format(context=context or "(none)")
            img_bytes: bytes | None = block.get("image_bytes")
            response = vlm_client.call(img_bytes, prompt, "image")
            if "error" in response:
                logger.debug("[image_processor] VLM error: {}", response["error"])
                return ""
            return str(response.get("description", "")).strip()
        except Exception as exc:
            logger.warning("[image_processor] enrich failed: {}", exc)
            return ""


class TableModalProcessor:
    """Enriches table blocks by generating a plain-language VLM description.

    Args:
        vlm: VLMClient instance for inference calls.
        content_list: Raw MinerU content_list for context extraction.
        context_window: Number of surrounding blocks to include in the prompt.
    """

    __slots__ = ("_vlm", "_extractor")

    def __init__(
        self,
        vlm: VLMClient,
        content_list: list[dict],
        context_window: int = 3,
    ) -> None:
        """Initialise the processor with a VLM client and content list.

        Args:
            vlm: VLMClient instance for inference calls.
            content_list: Raw MinerU content_list for context extraction.
            context_window: Number of surrounding blocks to include in the prompt.
        """
        self._vlm = vlm
        self._extractor = ContextExtractor(content_list, window=context_window)

    def process(
        self,
        block: dict[str, Any],
        block_index: int,
        image_bytes: bytes | None,
    ) -> str:
        """Generate a plain-language description for a table block.

        Args:
            block: MinerU content_list block dict of type 'table'.
            block_index: Zero-based index of the block in the content_list.
            image_bytes: Optional rasterized PNG bytes of the table region.

        Returns:
            Plain-language description string, or '' on any error or VLM failure.
        """
        try:
            context = self._extractor.extract_for_block(block_index)
            # NOTE: prefer HTML from block["html"] if present, then fall back
            # to block["text"] which may also contain HTML from MinerU output.
            table_content = (
                block.get("html", "")
                or block.get("text", "")
                or block.get("table_body", "")
                or "(table image provided, no HTML available)"
            )
            prompt = TABLE_PROMPT_TEMPLATE.format(
                context=context or "(none)",
                table_content=table_content,
            )
            response = self._vlm.call(image_bytes, prompt, "table")
            if "error" in response:
                logger.debug("[table_processor] VLM error: {}", response["error"])
                return ""
            return str(response.get("description", "")).strip()
        except Exception as exc:
            logger.warning("[table_processor] process failed: {}", exc)
            return ""

    def enrich(
        self,
        block: dict[str, Any],
        block_index: int,
        content_list: list[dict],
        vlm_client: VLMClient,
        config: Any,
    ) -> str:
        """Enrich a table block using the provided VLM client and config.

        Compatibility method matching the tasks.md enrich() interface.
        Uses HTML from block.get("html") if available, otherwise falls back
        to block.get("text") or block.get("table_body").

        Args:
            block: MinerU content_list block dict of type 'table'.
            block_index: Zero-based index of the block in the content_list.
            content_list: Full content_list for context extraction.
            vlm_client: VLMClient instance for inference calls.
            config: EnrichmentConfig with context_window and max_context_tokens.

        Returns:
            Plain-language description string, or '' on any error or VLM failure.
        """
        try:
            extractor = ContextExtractor(
                content_list,
                window=config.context_window,
                max_tokens=config.max_context_tokens,
            )
            context = extractor.extract_for_block(block_index)
            html = block.get("html", "")
            table_content = (
                html
                or block.get("text", "")
                or block.get("table_body", "")
                or "(table image provided, no HTML available)"
            )
            # NOTE: use image_bytes only when no HTML is available
            img_bytes: bytes | None = None if html else block.get("image_bytes")
            prompt = TABLE_PROMPT_TEMPLATE.format(
                context=context or "(none)",
                table_content=table_content,
            )
            response = vlm_client.call(img_bytes, prompt, "table")
            if "error" in response:
                logger.debug("[table_processor] VLM error: {}", response["error"])
                return ""
            return str(response.get("description", "")).strip()
        except Exception as exc:
            logger.warning("[table_processor] enrich failed: {}", exc)
            return ""


class EquationModalProcessor:
    """Enriches equation blocks by generating a plain-language VLM description.

    Args:
        vlm: VLMClient instance for inference calls.
        content_list: Raw MinerU content_list for context extraction.
        context_window: Number of surrounding blocks to include in the prompt.
    """

    __slots__ = ("_vlm", "_extractor")

    def __init__(
        self,
        vlm: VLMClient,
        content_list: list[dict],
        context_window: int = 3,
    ) -> None:
        """Initialise the processor with a VLM client and content list.

        Args:
            vlm: VLMClient instance for inference calls.
            content_list: Raw MinerU content_list for context extraction.
            context_window: Number of surrounding blocks to include in the prompt.
        """
        self._vlm = vlm
        self._extractor = ContextExtractor(content_list, window=context_window)

    def process(
        self,
        block: dict[str, Any],
        block_index: int,
        image_bytes: bytes | None,
    ) -> str:
        """Generate a plain-language description for an equation block.

        Args:
            block: MinerU content_list block dict of type 'interline_equation'.
            block_index: Zero-based index of the block in the content_list.
            image_bytes: Optional rasterized PNG bytes of the equation region.

        Returns:
            Plain-language description string, or '' on any error or VLM failure.
        """
        try:
            context = self._extractor.extract_for_block(block_index)
            latex = block.get("text", "") or block.get("latex", "") or block.get("content", "") or ""
            prompt = EQUATION_PROMPT_TEMPLATE.format(
                context=context or "(none)",
                latex=latex,
            )
            response = self._vlm.call(image_bytes, prompt, "equation")
            if "error" in response:
                logger.debug("[equation_processor] VLM error: {}", response["error"])
                return ""
            return str(response.get("description", "")).strip()
        except Exception as exc:
            logger.warning("[equation_processor] process failed: {}", exc)
            return ""

    def enrich(
        self,
        block: dict[str, Any],
        block_index: int,
        content_list: list[dict],
        vlm_client: VLMClient,
        config: Any,
    ) -> str:
        """Enrich an equation block using the provided VLM client and config.

        Compatibility method matching the tasks.md enrich() interface.
        Reads LaTeX from block.get("content") or block.get("text").

        Args:
            block: MinerU content_list block dict of type 'interline_equation'.
            block_index: Zero-based index of the block in the content_list.
            content_list: Full content_list for context extraction.
            vlm_client: VLMClient instance for inference calls.
            config: EnrichmentConfig with context_window and max_context_tokens.

        Returns:
            Plain-language description string, or '' on any error or VLM failure.
        """
        try:
            extractor = ContextExtractor(
                content_list,
                window=config.context_window,
                max_tokens=config.max_context_tokens,
            )
            context = extractor.extract_for_block(block_index)
            latex = block.get("content", "") or block.get("text", "") or block.get("latex", "") or ""
            prompt = EQUATION_PROMPT_TEMPLATE.format(
                context=context or "(none)",
                latex=latex,
            )
            response = vlm_client.call(None, prompt, "equation")
            if "error" in response:
                logger.debug("[equation_processor] VLM error: {}", response["error"])
                return ""
            return str(response.get("description", "")).strip()
        except Exception as exc:
            logger.warning("[equation_processor] enrich failed: {}", exc)
            return ""


class GenericModalProcessor:
    """Enriches arbitrary content blocks by generating a plain-language VLM description.

    Used as a fallback for block types without a dedicated processor.

    Args:
        vlm: VLMClient instance for inference calls.
        content_list: Raw MinerU content_list for context extraction.
        context_window: Number of surrounding blocks to include in the prompt.
    """

    __slots__ = ("_vlm", "_extractor")

    def __init__(
        self,
        vlm: VLMClient,
        content_list: list[dict],
        context_window: int = 3,
    ) -> None:
        """Initialise the processor with a VLM client and content list.

        Args:
            vlm: VLMClient instance for inference calls.
            content_list: Raw MinerU content_list for context extraction.
            context_window: Number of surrounding blocks to include in the prompt.
        """
        self._vlm = vlm
        self._extractor = ContextExtractor(content_list, window=context_window)

    def process(
        self,
        block: dict[str, Any],
        block_index: int,
        image_bytes: bytes | None,
    ) -> str:
        """Generate a plain-language description for a generic content block.

        Args:
            block: MinerU content_list block dict of any type.
            block_index: Zero-based index of the block in the content_list.
            image_bytes: Optional rasterized PNG bytes of the region.

        Returns:
            Plain-language description string, or '' on any error or VLM failure.
        """
        try:
            context = self._extractor.extract_for_block(block_index)
            content = block.get("text", "") or block.get("content", "") or ""
            content_type = block.get("type", "unknown")
            prompt = GENERIC_PROMPT_TEMPLATE.format(
                context=context or "(none)",
                content=content,
                content_type=content_type,
            )
            response = self._vlm.call(image_bytes, prompt, "text")
            if "error" in response:
                logger.debug("[generic_processor] VLM error: {}", response["error"])
                return ""
            return str(response.get("description", "")).strip()
        except Exception as exc:
            logger.warning("[generic_processor] process failed: {}", exc)
            return ""

    def enrich(
        self,
        block: dict[str, Any],
        block_index: int,
        content_list: list[dict],
        vlm_client: VLMClient,
        config: Any,
    ) -> str:
        """Enrich a generic block using the provided VLM client and config.

        Compatibility method matching the tasks.md enrich() interface.

        Args:
            block: MinerU content_list block dict of any type.
            block_index: Zero-based index of the block in the content_list.
            content_list: Full content_list for context extraction.
            vlm_client: VLMClient instance for inference calls.
            config: EnrichmentConfig with context_window and max_context_tokens.

        Returns:
            Plain-language description string, or '' on any error or VLM failure.
        """
        try:
            extractor = ContextExtractor(
                content_list,
                window=config.context_window,
                max_tokens=config.max_context_tokens,
            )
            context = extractor.extract_for_block(block_index)
            content = block.get("text", "") or block.get("content", "") or ""
            content_type = block.get("type", "unknown")
            prompt = GENERIC_PROMPT_TEMPLATE.format(
                context=context or "(none)",
                content=content,
                content_type=content_type,
            )
            response = vlm_client.call(None, prompt, "text")
            if "error" in response:
                logger.debug("[generic_processor] VLM error: {}", response["error"])
                return ""
            return str(response.get("description", "")).strip()
        except Exception as exc:
            logger.warning("[generic_processor] enrich failed: {}", exc)
            return ""
