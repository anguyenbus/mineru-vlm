"""Tests for verifier.py VerifierClient protocol and backends."""
from __future__ import annotations

import json
import unittest.mock as mock

from hybrid_doc_parser.models import VerifierConfig
from hybrid_doc_parser.verifier import (
    BedrockVerifierClient,
    FakeVerifierClient,
    OpenAICompatibleVerifierClient,
    VerifierClient,
    make_verifier_client,
)


def test_fake_verifier_returns_canned_verdict_no_network():
    config = VerifierConfig(backend="fake")
    verdict = {
        "disagreements": [{"element_id": "p3-e7", "severity": "high"}],
        "missing_elements": [],
        "extra_elements": [],
    }
    client = FakeVerifierClient(verdicts={3: verdict})

    result = client.verify_page(b"\x89PNG\r\n", "page_idx: 3\nelements...", config)

    assert result == verdict
    assert client.calls == [(True, "page_idx: 3\nelements...", config)]


def test_fake_verifier_default_verdict_for_unknown_page():
    config = VerifierConfig(backend="fake")
    client = FakeVerifierClient(verdicts={3: {"disagreements": ["x"]}})

    result = client.verify_page(None, "page_idx: 9", config)

    assert result == {
        "disagreements": [],
        "missing_elements": [],
        "extra_elements": [],
    }


def test_all_backends_satisfy_protocol():
    for client in (
        FakeVerifierClient(),
        BedrockVerifierClient(),
        OpenAICompatibleVerifierClient(),
    ):
        assert isinstance(client, VerifierClient)


def test_make_verifier_client_dispatch():
    assert isinstance(make_verifier_client(VerifierConfig(backend="fake")), FakeVerifierClient)
    assert isinstance(
        make_verifier_client(VerifierConfig(backend="openai_compatible")),
        OpenAICompatibleVerifierClient,
    )
    assert isinstance(
        make_verifier_client(VerifierConfig(backend="bedrock")), BedrockVerifierClient
    )


def test_bedrock_client_never_raises_on_failure():
    config = VerifierConfig(backend="bedrock", model="anthropic.claude-x", region="us-east-1")

    with mock.patch("boto3.client", side_effect=RuntimeError("no AWS credentials")):
        result = BedrockVerifierClient().verify_page(None, "prompt", config)

    assert isinstance(result, dict)
    assert "error" in result


def test_bedrock_request_uses_bare_model_id_and_image_block():
    config = VerifierConfig(
        backend="bedrock", model="anthropic.claude-3-sonnet", region="us-east-1"
    )

    response_body = mock.MagicMock()
    response_body.read.return_value = json.dumps(
        {"content": [{"type": "text", "text": '{"disagreements": []}'}]}
    ).encode()

    with mock.patch("boto3.client") as mock_boto3_client:
        runtime = mock.MagicMock()
        mock_boto3_client.return_value = runtime
        runtime.invoke_model.return_value = {"body": response_body}

        jpeg_bytes = b"\xff\xd8\xff" + b"\x00" * 10
        result = BedrockVerifierClient().verify_page(jpeg_bytes, "verify", config)

    # Fresh client created with the configured region.
    assert mock_boto3_client.call_args.kwargs["region_name"] == "us-east-1"

    invoke_kwargs = runtime.invoke_model.call_args.kwargs
    # BARE on-demand model id — no inference-profile prefix (e.g. "us.").
    assert invoke_kwargs["modelId"] == "anthropic.claude-3-sonnet"
    assert "." in invoke_kwargs["modelId"]
    assert not invoke_kwargs["modelId"].startswith(("us.", "eu.", "apac."))

    body = json.loads(invoke_kwargs["body"])
    image_block = next(
        (b for b in body["messages"][0]["content"] if b.get("type") == "image"), None
    )
    assert image_block is not None
    assert image_block["source"]["media_type"] == "image/jpeg"
    assert result == {"disagreements": []}


def test_openai_client_never_raises_on_failure():
    config = VerifierConfig(backend="openai_compatible", model="local-model")

    with mock.patch("openai.OpenAI", side_effect=RuntimeError("connection refused")):
        result = OpenAICompatibleVerifierClient().verify_page(None, "prompt", config)

    assert isinstance(result, dict)
    assert "error" in result
