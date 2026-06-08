"""Pluggable VLM/LLM client backends for modal enrichment.

Concrete implementations:
- OpenAICompatibleClient: any OpenAI-compatible API endpoint
- BedrockClient: AWS Bedrock via boto3

Both clients never raise; they return {"error": ...} on failure.
"""

from __future__ import annotations

import base64
import json
import os
import re
from typing import Any, Final, Protocol, runtime_checkable

from loguru import logger

from hybrid_doc_parser.models import EnrichmentConfig


@runtime_checkable
class VLMClient(Protocol):
    """Protocol for VLM/LLM backend clients.

    All implementations must be callable with image_bytes, prompt, and mode,
    and must never raise — returning {"error": ...} on failure instead.
    """

    def call(self, image_bytes: bytes | None, prompt: str, mode: str) -> dict[str, Any]:
        """Call the VLM with optional image and text prompt.

        Args:
            image_bytes: Raw image bytes to include, or None for text-only.
            prompt: Text prompt for the VLM.
            mode: Hint for the processor type ('image', 'table', 'equation', 'text').

        Returns:
            Parsed JSON response dict. Never raises; returns {"error": ...} on failure.
        """
        ...


def _detect_media_type(image_bytes: bytes) -> str:
    """Detect image media type from magic bytes.

    Args:
        image_bytes: Raw image bytes.

    Returns:
        MIME type string. Falls back to 'image/png' if unknown.
    """
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if image_bytes.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"


def _strip_thinking_tags(text: str) -> str:
    """Strip <think>...</think> and <thinking>...</thinking> tags from VLM output.

    Supports reasoning models that emit chain-of-thought in think tags
    (DeepSeek-R1, Qwen).

    Args:
        text: Raw VLM response text.

    Returns:
        Text with all thinking tag content removed, whitespace-stripped.
    """
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<thinking>.*?</thinking>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()


def _robust_json_parse(response: str) -> dict[str, Any]:
    """Parse JSON from a VLM response using 4 fallback strategies. Never raises.

    Strategy 1: Direct json.loads after stripping thinking tags.
    Strategy 2: Strip markdown code fences (```json ... ```) then json.loads.
    Strategy 3: Extract first balanced {...} block then json.loads.
    Strategy 4 (last resort): Return {"description": cleaned_text}.

    Args:
        response: Raw VLM response string.

    Returns:
        Parsed dict. Always returns a dict, never raises.
    """
    cleaned = _strip_thinking_tags(response)

    # Strategy 1: direct parse
    # NOTE: attempt the cheapest path first — no string manipulation needed
    try:
        return json.loads(cleaned.strip())
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 2: strip markdown fences
    try:
        lines = cleaned.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        return json.loads("\n".join(lines).strip())
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 3: extract first balanced {...} block
    # NOTE: walk the string tracking brace depth to find the outermost {...}
    try:
        depth = 0
        start = -1
        for i, ch in enumerate(cleaned):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start != -1:
                    candidate = cleaned[start: i + 1]
                    return json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 4: last resort — wrap entire cleaned text as description
    return {"description": cleaned}


def _build_bedrock_request(image_bytes: bytes | None, prompt: str) -> dict[str, Any]:
    """Build the Bedrock invoke_model request body. Pure function.

    Args:
        image_bytes: Optional image bytes to include in the request.
        prompt: Text prompt for the model.

    Returns:
        Request body dict compatible with Anthropic Claude on Bedrock.
    """
    content: list[dict[str, Any]] = []
    if image_bytes is not None:
        media_type = _detect_media_type(image_bytes)
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": b64},
        })
    content.append({"type": "text", "text": prompt})
    return {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 8192,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": content}],
    }


class OpenAICompatibleClient:
    """VLM client for any OpenAI-compatible API endpoint.

    Reads OPENAI_BASE_URL, OPENAI_API_KEY, VLM_MODEL_NAME from environment
    on each call. Never raises; returns {"error": ...} on any failure.
    """

    __slots__ = ()

    def call(self, image_bytes: bytes | None, prompt: str, mode: str) -> dict[str, Any]:
        """Call the OpenAI-compatible API with optional image and text prompt.

        Args:
            image_bytes: Raw image bytes to include, or None for text-only.
            prompt: Text prompt for the VLM.
            mode: Processor hint ('image', 'table', 'equation', 'text').

        Returns:
            Parsed JSON dict from VLM response. Returns {"error": ...} on failure.
        """
        try:
            import openai  # noqa: PLC0415

            base_url = os.environ.get("OPENAI_BASE_URL", "")
            api_key = os.environ.get("OPENAI_API_KEY", "none")
            model = os.environ.get("VLM_MODEL_NAME", "")

            content: list[dict[str, Any]] | str
            if image_bytes is not None:
                media_type = _detect_media_type(image_bytes)
                b64 = base64.b64encode(image_bytes).decode("utf-8")
                content = [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{media_type};base64,{b64}"},
                    },
                    {"type": "text", "text": prompt},
                ]
            else:
                content = prompt

            messages: list[dict[str, Any]] = [{"role": "user", "content": content}]
            client = openai.OpenAI(base_url=base_url, api_key=api_key)
            response = client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                max_tokens=4096,
                temperature=0.0,
            )
            raw = response.choices[0].message.content or ""
            return _robust_json_parse(raw)
        except Exception as exc:
            logger.warning("[VLM] OpenAI call failed: {}", exc)
            return {"error": str(exc)}


class BedrockClient:
    """VLM client for AWS Bedrock.

    Reads AWS_REGION and BEDROCK_VLM_MODEL from environment on each call.
    Creates a boto3 client per call to respect IAM role rotation.
    Never raises; returns {"error": ...} on any failure.
    """

    __slots__ = ()

    def call(self, image_bytes: bytes | None, prompt: str, mode: str) -> dict[str, Any]:
        """Call AWS Bedrock with optional image and text prompt.

        Args:
            image_bytes: Raw image bytes to include, or None for text-only.
            prompt: Text prompt for the VLM.
            mode: Processor hint ('image', 'table', 'equation', 'text').

        Returns:
            Parsed JSON dict from VLM response. Returns {"error": ...} on failure.
        """
        try:
            import boto3  # noqa: PLC0415

            region = os.environ.get("AWS_REGION", "us-east-1")
            model_id = os.environ.get("BEDROCK_VLM_MODEL", "")
            body = _build_bedrock_request(image_bytes, prompt)
            runtime = boto3.client("bedrock-runtime", region_name=region)
            resp = runtime.invoke_model(
                modelId=model_id,
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json",
            )
            payload = json.loads(resp["body"].read())
            raw = "".join(
                block.get("text", "")
                for block in payload.get("content", [])
                if block.get("type") == "text"
            )
            return _robust_json_parse(raw)
        except Exception as exc:
            logger.warning("[VLM] Bedrock call failed: {}", exc)
            return {"error": str(exc)}


def make_vlm_client(config: EnrichmentConfig) -> VLMClient:
    """Factory returning the appropriate VLMClient for the given config.

    Args:
        config: EnrichmentConfig with vlm_backend set to 'bedrock' or
            'openai_compatible'.

    Returns:
        Concrete VLMClient implementation.
    """
    if config.vlm_backend == "bedrock":
        return BedrockClient()
    return OpenAICompatibleClient()


# Re-export Final constant for type-checking convenience
_FINAL_MARKER: Final[str] = "vlm_client"
