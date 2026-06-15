"""Tests for hybrid_doc_parser.models — Pydantic v2 schema definitions.

Tests follow strict TDD: each test validates a specific contract from the spec.
Run in isolation:
    uv run pytest tests/hybrid_doc_parser/test_models.py -v
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from hybrid_doc_parser.models import (
    SCHEMA_VERSION,
    ElementRecord,
    ElementType,
    EnrichmentConfig,
    PageRecord,
    ParserOutput,
    WarningRecord,
)


class TestElementType:
    """Test 1: ElementType has all 11 expected string values."""

    def test_element_type_has_all_11_members(self) -> None:
        """ElementType enum must contain exactly the 11 specified string values."""
        expected = {
            "text",
            "heading",
            "list_item",
            "image",
            "table",
            "equation",
            "caption",
            "header",
            "footer",
            "page_number",
            "unknown",
        }
        actual = {member.value for member in ElementType}
        assert actual == expected, f"Missing or extra members: {actual.symmetric_difference(expected)}"

    def test_element_type_is_str_subclass(self) -> None:
        """ElementType members must be usable as plain strings."""
        assert isinstance(ElementType.text, str)
        assert ElementType.text == "text"


class TestEnrichmentConfigDefaults:
    """Test 2: EnrichmentConfig default values."""

    def test_default_values(self) -> None:
        """EnrichmentConfig constructed with no args must have spec-defined defaults."""
        cfg = EnrichmentConfig()
        assert cfg.enabled is False
        assert cfg.image is True
        assert cfg.table is True
        assert cfg.equation is True
        assert cfg.context_window == 3
        assert cfg.max_context_tokens == 512
        assert cfg.vlm_backend == "openai_compatible"


class TestEnrichmentConfigValidation:
    """Test 3: EnrichmentConfig field validators enforce ge/le constraints."""

    def test_context_window_negative_raises(self) -> None:
        """context_window=-1 violates ge=0 and must raise ValidationError."""
        with pytest.raises(ValidationError):
            EnrichmentConfig(context_window=-1)

    def test_max_context_tokens_below_minimum_raises(self) -> None:
        """max_context_tokens=63 violates ge=64 and must raise ValidationError."""
        with pytest.raises(ValidationError):
            EnrichmentConfig(max_context_tokens=63)

    def test_context_window_above_maximum_raises(self) -> None:
        """context_window=21 violates le=20 and must raise ValidationError."""
        with pytest.raises(ValidationError):
            EnrichmentConfig(context_window=21)

    def test_max_context_tokens_above_maximum_raises(self) -> None:
        """max_context_tokens=4097 violates le=4096 and must raise ValidationError."""
        with pytest.raises(ValidationError):
            EnrichmentConfig(max_context_tokens=4097)


class TestElementRecord:
    """Test 4: ElementRecord construction and defaults."""

    def test_constructs_with_required_fields(self) -> None:
        """ElementRecord must construct successfully with all required fields provided."""
        record = ElementRecord(
            element_id="some-uuid",
            type=ElementType.text,
            text="hello world",
            bbox=[0.0, 0.0, 100.0, 20.0],
            page_idx=0,
        )
        assert record.element_id == "some-uuid"
        assert record.type == ElementType.text
        assert record.text == "hello world"
        assert record.bbox == [0.0, 0.0, 100.0, 20.0]
        assert record.page_idx == 0

    def test_is_enriched_defaults_to_false(self) -> None:
        """is_enriched must default to False when not explicitly set."""
        record = ElementRecord(
            element_id="abc",
            type=ElementType.image,
            text="",
            bbox=[],
            page_idx=1,
        )
        assert record.is_enriched is False

    def test_description_defaults_to_empty_string(self) -> None:
        """description must default to empty string when not explicitly set."""
        record = ElementRecord(
            element_id="abc",
            type=ElementType.table,
            text="some table",
            bbox=[10.0, 20.0, 200.0, 150.0],
            page_idx=0,
        )
        assert record.description == ""

    def test_image_bytes_defaults_to_none(self) -> None:
        """image_bytes must default to None when not explicitly set."""
        record = ElementRecord(
            element_id="x",
            type=ElementType.equation,
            text=r"\int_0^1",
            bbox=[],
            page_idx=2,
        )
        assert record.image_bytes is None


class TestParserOutputFrozen:
    """Test 5: ParserOutput is frozen — mutation raises TypeError or ValidationError."""

    def _make_output(self) -> ParserOutput:
        """Build a minimal valid ParserOutput for mutation tests."""
        return ParserOutput(
            file_path="/tmp/test.pdf",
            file_sha256="a" * 64,
            page_count=1,
            pages=[],
            elements=[],
            warnings=[],
            enrichment_config=EnrichmentConfig(),
        )

    def test_mutation_raises(self) -> None:
        """Attempting to mutate a frozen ParserOutput must raise TypeError or ValidationError."""
        output = self._make_output()
        with pytest.raises((TypeError, ValidationError)):
            output.page_count = 99  # type: ignore[misc]


class TestParserOutputSchemaVersion:
    """Test 6: ParserOutput.schema_version defaults to '1.0'; SCHEMA_VERSION == '1.0'."""

    def test_schema_version_constant(self) -> None:
        """Module-level SCHEMA_VERSION must equal '1.0'."""
        assert SCHEMA_VERSION == "1.0"

    def test_schema_version_default(self) -> None:
        """ParserOutput without explicit schema_version must default to SCHEMA_VERSION."""
        output = ParserOutput(
            file_path="/tmp/doc.pdf",
            file_sha256="b" * 64,
            page_count=0,
            pages=[],
            elements=[],
            warnings=[],
            enrichment_config=EnrichmentConfig(),
        )
        assert output.schema_version == "1.0"


class TestParserOutputRoundTrip:
    """Test 7: ParserOutput round-trips through JSON serialisation."""

    def test_json_round_trip(self) -> None:
        """model_dump_json() -> model_validate_json() must reproduce an equal object."""
        page = PageRecord(
            page_idx=0,
            quality_decision="keep",
            element_count=1,
            vlm_used=False,
        )
        element = ElementRecord(
            element_id="test-id-1",
            type=ElementType.heading,
            text="# Introduction",
            description="",
            bbox=[0.0, 700.0, 500.0, 720.0],
            page_idx=0,
            is_enriched=False,
        )
        warning = WarningRecord(
            page_idx=None,
            code="test_code",
            message="test warning",
        )
        original = ParserOutput(
            schema_version="1.0",
            file_path="/abs/path/doc.pdf",
            file_sha256="c" * 64,
            page_count=1,
            pages=[page],
            elements=[element],
            warnings=[warning],
            enrichment_config=EnrichmentConfig(enabled=True),
        )

        json_str = original.model_dump_json()
        restored = ParserOutput.model_validate_json(json_str)

        assert restored == original


class TestWarningRecordNullPageIdx:
    """Test 8: WarningRecord with page_idx=None constructs and serialises without error."""

    def test_document_level_warning(self) -> None:
        """WarningRecord with page_idx=None must construct and serialise cleanly."""
        warning = WarningRecord(
            page_idx=None,
            code="unsupported_type",
            message="File extension .xyz is not supported.",
        )
        assert warning.page_idx is None
        assert warning.code == "unsupported_type"

        # Must also serialise without error
        json_str = warning.model_dump_json()
        assert "unsupported_type" in json_str

    def test_page_level_warning(self) -> None:
        """WarningRecord with an explicit page_idx must store the value correctly."""
        warning = WarningRecord(
            page_idx=3,
            code="render_failed",
            message="render_region raised ValueError",
        )
        assert warning.page_idx == 3


class TestEnrichmentConfigAutoParser:
    """Test Group 1: EnrichmentConfig.parser accepts 'auto' as a valid Literal value."""

    def test_auto_parser_constructs_without_error(self) -> None:
        """EnrichmentConfig(parser='auto') must construct without a ValidationError.

        The 'auto' value selects the backend per file at classify-time via
        MinerU's classify() heuristic rather than statically.
        """
        cfg = EnrichmentConfig(parser="auto")
        assert cfg.parser == "auto"

    def test_invalid_parser_value_raises_validation_error(self) -> None:
        """EnrichmentConfig(parser='invalid_value') must raise ValidationError.

        This is a regression guard confirming the Literal is still constrained
        to only the valid backend identifiers.
        """
        with pytest.raises(ValidationError):
            EnrichmentConfig(parser="invalid_value")


# ---------------------------------------------------------------------------
# Task Group 1: Verifier schema layer (VerifierConfig / VerificationReport)
# ---------------------------------------------------------------------------

from hybrid_doc_parser.models import (  # noqa: E402
    Disagreement,
    ExtraElement,
    MissingElement,
    PageVerification,
    Severity,
    VerificationReport,
    VerifierConfig,
)


class TestVerifierConfigDefaults:
    """VerifierConfig constructs with zero args and is safe (verifier off)."""

    def test_zero_arg_defaults_are_safe(self) -> None:
        """VerifierConfig() must default to a disabled, precision-favoring config."""
        cfg = VerifierConfig()
        assert cfg.enabled is False
        assert cfg.force_verify_all is False
        assert cfg.min_severity_to_report == "high"
        assert cfg.max_concurrency >= 1
        assert cfg.max_pages_per_doc >= 1
        assert cfg.timeout > 0


class TestVerifierConfigBounds:
    """Field(...) bounds reject out-of-range values."""

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"max_concurrency": 0},  # ge=1
            {"max_pages_per_doc": 0},  # ge=1
            {"timeout": 0},  # gt=0
            {"render_dpi": 71},  # below lower bound
            {"render_dpi": 601},  # above upper bound
        ],
    )
    def test_out_of_range_raises(self, kwargs: dict) -> None:
        """Each out-of-range field value must raise ValidationError."""
        with pytest.raises(ValidationError):
            VerifierConfig(**kwargs)


class TestVerifierModelsFrozen:
    """All verifier models are frozen — mutation raises."""

    def test_config_is_frozen(self) -> None:
        """Mutating VerifierConfig must raise."""
        cfg = VerifierConfig()
        with pytest.raises((TypeError, ValidationError)):
            cfg.enabled = True  # type: ignore[misc]

    def test_report_is_frozen(self) -> None:
        """Mutating VerificationReport must raise."""
        report = VerificationReport(model_id="m", prompt_version="v1")
        with pytest.raises((TypeError, ValidationError)):
            report.model_id = "other"  # type: ignore[misc]


class TestVerificationReportEnvelope:
    """A populated VerificationReport serializes to the canonical envelope shape."""

    def _make_report(self) -> VerificationReport:
        return VerificationReport(
            model_id="anthropic.claude-...",
            prompt_version="v1",
            pages=[
                PageVerification(
                    page_idx=3,
                    disagreements=[
                        Disagreement(
                            element_id="p3-e7",
                            type=ElementType.table,
                            severity=Severity.high,
                            reason="Row 4 merged two columns; values misaligned vs image.",
                            suggested_text="...",
                            vlm_confidence=0.86,
                        )
                    ],
                    missing_elements=[
                        MissingElement(
                            severity=Severity.medium,
                            reason="Footnote at page bottom not extracted.",
                            approx_location="below last paragraph",
                        )
                    ],
                    extra_elements=[
                        ExtraElement(
                            severity=Severity.low,
                            reason="Spurious header repeated.",
                            approx_location="top of page",
                        )
                    ],
                )
            ],
            warnings=[
                WarningRecord(
                    code="verification_failed",
                    page_idx=9,
                    message="Bedrock throttled after 3 retries",
                )
            ],
        )

    def test_serializes_under_top_level_verification_key(self) -> None:
        """Top-level output must nest everything under the 'verification' key."""
        dumped = self._make_report().model_dump()
        assert set(dumped.keys()) == {"verification"}
        inner = dumped["verification"]
        assert set(inner.keys()) == {"model_id", "prompt_version", "pages", "warnings"}
        assert inner["model_id"] == "anthropic.claude-..."
        assert inner["prompt_version"] == "v1"

    def test_envelope_matches_canonical_shape(self) -> None:
        """The serialized payload must match the canonical disagreement/missing/extra shape."""
        inner = self._make_report().model_dump()["verification"]

        page = inner["pages"][0]
        assert page["page_idx"] == 3

        dis = page["disagreements"][0]
        assert dis["element_id"] == "p3-e7"
        assert dis["type"] == "table"
        assert dis["severity"] == "high"
        assert dis["vlm_confidence"] == 0.86
        assert set(dis.keys()) == {
            "element_id",
            "type",
            "severity",
            "reason",
            "suggested_text",
            "vlm_confidence",
        }

        miss = page["missing_elements"][0]
        assert "element_id" not in miss
        assert set(miss.keys()) == {"severity", "reason", "approx_location"}

        extra = page["extra_elements"][0]
        assert "element_id" not in extra
        assert set(extra.keys()) == {"severity", "reason", "approx_location"}

        warn = inner["warnings"][0]
        assert warn["code"] == "verification_failed"
        assert warn["page_idx"] == 9

    def test_json_dump_wraps_in_verification(self) -> None:
        """model_dump_json() must also produce the top-level 'verification' envelope."""
        json_str = self._make_report().model_dump_json()
        assert json_str.lstrip().startswith('{"verification":')
