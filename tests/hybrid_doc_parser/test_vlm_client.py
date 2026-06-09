"""Tests for vlm_client.py VLM backend implementations."""
from __future__ import annotations

import json
import unittest.mock as mock

import pytest


def test_detect_media_type_jpeg():
    from hybrid_doc_parser.vlm_client import _detect_media_type
    assert _detect_media_type(b"\xff\xd8\xff" + b"\x00" * 10) == "image/jpeg"


def test_detect_media_type_png():
    from hybrid_doc_parser.vlm_client import _detect_media_type
    assert _detect_media_type(b"\x89PNG\r\n\x1a\n" + b"\x00" * 10) == "image/png"


def test_detect_media_type_gif():
    from hybrid_doc_parser.vlm_client import _detect_media_type
    assert _detect_media_type(b"GIF87a" + b"\x00" * 10) == "image/gif"


def test_detect_media_type_webp():
    from hybrid_doc_parser.vlm_client import _detect_media_type
    assert _detect_media_type(b"RIFF" + b"\x00\x00\x00\x00" + b"WEBP" + b"\x00" * 10) == "image/webp"


def test_detect_media_type_unknown():
    from hybrid_doc_parser.vlm_client import _detect_media_type
    assert _detect_media_type(b"\x00\x01\x02\x03") == "image/png"


def test_strip_thinking_tags_basic():
    from hybrid_doc_parser.vlm_client import _strip_thinking_tags
    assert _strip_thinking_tags("<think>hidden</think>visible") == "visible"


def test_strip_thinking_tags_case_insensitive():
    from hybrid_doc_parser.vlm_client import _strip_thinking_tags
    result = _strip_thinking_tags("<THINK>multiline\nhidden\n</THINK>visible")
    assert result == "visible"


def test_strip_thinking_tags_multiple():
    from hybrid_doc_parser.vlm_client import _strip_thinking_tags
    result = _strip_thinking_tags("<thinking>a</thinking>b<think>c</think>d")
    assert result == "bd"


def test_strip_thinking_tags_no_tags():
    from hybrid_doc_parser.vlm_client import _strip_thinking_tags
    assert _strip_thinking_tags("plain text") == "plain text"


def test_robust_json_parse_direct():
    from hybrid_doc_parser.vlm_client import _robust_json_parse
    result = _robust_json_parse('{"description": "hello"}')
    assert result == {"description": "hello"}


def test_robust_json_parse_markdown_fence():
    from hybrid_doc_parser.vlm_client import _robust_json_parse
    result = _robust_json_parse('```json\n{"description": "world"}\n```')
    assert result == {"description": "world"}
    result2 = _robust_json_parse('```\n{"key": 1}\n```')
    assert result2 == {"key": 1}


def test_robust_json_parse_extract_braces():
    from hybrid_doc_parser.vlm_client import _robust_json_parse
    result = _robust_json_parse('Some text before {"description": "extracted"} and after')
    assert result == {"description": "extracted"}


def test_robust_json_parse_fallback():
    from hybrid_doc_parser.vlm_client import _robust_json_parse
    result = _robust_json_parse("completely unparseable text no json here")
    assert isinstance(result, dict)
    assert "description" in result
    assert "unparseable" in result["description"]


def test_robust_json_parse_strips_thinking_then_fallback():
    from hybrid_doc_parser.vlm_client import _robust_json_parse
    result = _robust_json_parse("<think>strip this</think>fallback text")
    assert isinstance(result, dict)
    assert "description" in result
    assert "strip" not in result["description"]


def test_openai_client_call_text(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "none")
    monkeypatch.setenv("VLM_MODEL_NAME", "test-model")

    from hybrid_doc_parser.vlm_client import OpenAICompatibleClient

    mock_response = mock.MagicMock()
    mock_response.choices[0].message.content = '{"description": "mocked"}'

    with mock.patch("openai.OpenAI") as mock_openai_class:
        mock_client_instance = mock.MagicMock()
        mock_openai_class.return_value = mock_client_instance
        mock_client_instance.chat.completions.create.return_value = mock_response

        client = OpenAICompatibleClient()
        result = client.call(None, "test prompt", "text")
        assert isinstance(result, dict)
        assert result.get("description") == "mocked"


def test_openai_client_call_with_image(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "none")
    monkeypatch.setenv("VLM_MODEL_NAME", "test-model")

    from hybrid_doc_parser.vlm_client import OpenAICompatibleClient

    mock_response = mock.MagicMock()
    mock_response.choices[0].message.content = '{"description": "image result"}'

    with mock.patch("openai.OpenAI") as mock_openai_class:
        mock_client_instance = mock.MagicMock()
        mock_openai_class.return_value = mock_client_instance
        mock_client_instance.chat.completions.create.return_value = mock_response

        jpeg_bytes = b"\xff\xd8\xff" + b"\x00" * 10
        client = OpenAICompatibleClient()
        result = client.call(jpeg_bytes, "describe this", "image")
        assert isinstance(result, dict)

        # Verify image_url block was included in the messages
        call_kwargs = mock_client_instance.chat.completions.create.call_args
        messages = call_kwargs.kwargs.get("messages", call_kwargs.args[0] if call_kwargs.args else [])
        content = messages[0]["content"]
        image_block = next((b for b in content if b.get("type") == "image_url"), None)
        assert image_block is not None
        assert "image/jpeg" in image_block["image_url"]["url"]


def test_bedrock_client_never_raises(monkeypatch):
    monkeypatch.setenv("AWS_REGION", "ap-southeast-2")
    monkeypatch.setenv("BEDROCK_VLM_MODEL", "test-model")

    from hybrid_doc_parser.vlm_client import BedrockClient

    with mock.patch("boto3.client") as mock_boto3:
        mock_boto3.side_effect = RuntimeError("no AWS credentials")
        client = BedrockClient()
        result = client.call(None, "prompt", "text")
        assert isinstance(result, dict)
        assert "error" in result


def test_bedrock_request_structure(monkeypatch):
    monkeypatch.setenv("AWS_REGION", "ap-southeast-2")
    monkeypatch.setenv("BEDROCK_VLM_MODEL", "anthropic.claude-3-sonnet")

    from hybrid_doc_parser.vlm_client import BedrockClient

    mock_response_body = mock.MagicMock()
    mock_response_body.read.return_value = json.dumps({
        "content": [{"type": "text", "text": '{"description": "bedrock result"}'}]
    }).encode()

    with mock.patch("boto3.client") as mock_boto3_client:
        mock_runtime = mock.MagicMock()
        mock_boto3_client.return_value = mock_runtime
        mock_runtime.invoke_model.return_value = {"body": mock_response_body}

        client = BedrockClient()
        result = client.call(None, "test prompt", "text")

        call_kwargs = mock_runtime.invoke_model.call_args.kwargs
        body = json.loads(call_kwargs["body"])
        assert body["anthropic_version"] == "bedrock-2023-05-31"
        assert body["max_tokens"] == 8192
        assert isinstance(result, dict)
        assert result.get("description") == "bedrock result"
