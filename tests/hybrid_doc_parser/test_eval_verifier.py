"""CI test for the verifier eval harness deterministic FAKE path (TG8.5).

This is the network-free, deterministic guard that the HARD eval gate's harness
(``scripts/eval_verifier.py``) actually computes precision/recall against the
labeled disagreement set (``scripts/eval_verifier_fixtures.py``). Because the
``FakeVerifierClient`` returns KNOWN verdicts, the scored numbers are
deterministic and assertable, which is what proves the harness correct.

No network, no AWS, no PDF toolchain: the harness mocks the full-page render and
injects the fake client. The ``@live`` real-backend path is intentionally NOT
exercised here (it needs creds / a local endpoint and is measured separately).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load(module_name: str, filename: str):
    """Load a ``scripts/`` module by path (scripts/ is not an installed package)."""
    sys.path.insert(0, str(_SCRIPTS))
    spec = importlib.util.spec_from_file_location(module_name, _SCRIPTS / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def eval_mod():
    """Import the eval fixtures + harness modules from ``scripts/``."""
    _load("eval_verifier_fixtures", "eval_verifier_fixtures.py")
    return _load("eval_verifier", "eval_verifier.py")


def test_harness_computes_deterministic_precision_recall(eval_mod, monkeypatch, tmp_path):
    """The fake path runs end-to-end and computes the expected precision/recall.

    Asserts the harness:
    - selects ``medium`` as the precision-favoring severity floor (the spurious
      LOW disagreement is dropped at/above ``medium``, lifting precision to 1.0);
    - reports perfect precision/recall for both channel groups at that floor; and
    - quantifies a non-zero quality-gate recall gap (a defect on a KEEP page is
      invisible to flagged-only selection but recovered by ``force_verify_all``).
    """
    monkeypatch.setenv(
        "HYBRID_DOC_PARSER_VERIFIER_CACHE_DIR", str(tmp_path / "vcache")
    )
    from hybrid_doc_parser.models import VerifierConfig

    docs = eval_mod.labeled_documents()
    result = eval_mod.run_eval(
        docs,
        use_fake=True,
        base_config=VerifierConfig(backend="fake", model="fake-verifier"),
    )

    # Precision-favoring tuning lands on the floor that drops the LOW false
    # positive (ties broken toward higher recall -> "medium", not "high").
    assert result.chosen_floor == "medium"

    # At the chosen floor every labeled finding is recovered with no false
    # positives, in BOTH the disagreement and missing/extra channel groups.
    assert result.disagreement_pr.precision == pytest.approx(1.0)
    assert result.disagreement_pr.recall == pytest.approx(1.0)
    assert result.location_pr.precision == pytest.approx(1.0)
    assert result.location_pr.recall == pytest.approx(1.0)

    # At the weakest floor the spurious LOW disagreement leaks in, lowering
    # disagreement precision below 1.0 -> the sweep is genuinely discriminating.
    low_dis, _low_loc = result.per_floor["low"]
    assert low_dis.precision < 1.0
    assert low_dis.false_positives == 1

    # The quality gate hides a real defect on a KEEP page: flagged-only recall is
    # strictly below force_verify_all recall.
    assert result.flagged_only_recall < result.force_all_recall
    assert result.force_all_recall == pytest.approx(1.0)


def test_harness_renders_a_gate_artifact_with_numbers(eval_mod, monkeypatch, tmp_path):
    """render_artifact() emits Markdown carrying the measured numbers and caveats."""
    monkeypatch.setenv(
        "HYBRID_DOC_PARSER_VERIFIER_CACHE_DIR", str(tmp_path / "vcache")
    )
    from hybrid_doc_parser.models import VerifierConfig

    docs = eval_mod.labeled_documents()
    result = eval_mod.run_eval(
        docs,
        use_fake=True,
        base_config=VerifierConfig(backend="fake", model="fake-verifier"),
    )
    artifact = eval_mod.render_artifact(result, mode="fake")

    assert "precision" in artifact and "recall" in artifact
    assert 'min_severity_to_report = "medium"' in artifact
    # The fake path must NOT claim production readiness on its own.
    assert "NOT YET RECOMMENDED" in artifact
    assert "STILL REQUIRED" in artifact
