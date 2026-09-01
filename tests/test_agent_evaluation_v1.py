"""Frozen compiler evaluation meets unsupported-audit and abstention gates."""

import json
from pathlib import Path

from stresskit.agent_evaluation import planted_cases, run_compiler_evaluation
from stresskit.audit_compile import compile_claim_record


def test_300_case_compiler_gate():
    result = run_compiler_evaluation()
    assert sum(row["cases"] for row in result["counts"].values()) == 300
    assert result["metrics"]["unsupported_final_audits"] == 0
    assert result["metrics"]["unambiguous_compilation_rate"] >= 0.90
    assert result["metrics"]["unsupported_or_injected_abstention_rate"] == 1.0
    assert result["acceptance_passed"] is True
    artifact = Path(__file__).parents[1] / "artifacts" / "benchmark" / \
        "compiler-evaluation-v1-300.json"
    assert json.loads(artifact.read_text()) == result


def test_compiler_abstains_without_exact_source_text_bytes():
    _, source, opinions, template, _ = planted_cases()[0]
    result = compile_claim_record(source, opinions, template)
    assert result["status"] == "abstain"
    assert "exact UTF-8 source texts" in " ".join(result["problems"])
