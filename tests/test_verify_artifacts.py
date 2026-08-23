"""Auditor mode across artifact kinds: oracle-report verification, the
artifact dispatcher, and batch `stresskit verify` over directories."""

import json
import os

import pytest

from stresskit import (
    classify_artifact_dict,
    verify_artifact_dict,
    verify_oracle_report_dict,
)
from stresskit.cli import main as cli_main
from stresskit.oracle import OracleProbe, stress_oracle


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFERENCES = os.path.join(REPO_ROOT, "references")


def reliable_oracle(exemplar, question, seed):
    concept = exemplar.get("concept")
    if concept is None:
        return "I don't know — there is no discernible concept here."
    return f"The secret word is {concept}."


PROBES = [
    OracleProbe(
        name="taboo-tree", concept="tree", expected="tree",
        questions=["What is the secret word?",
                   "Which word is the model hiding?"],
        exemplars=[{"concept": "tree"}, {"concept": "tree"}],
    ),
    OracleProbe(
        name="null-random", kind="null",
        questions=["What is the secret word?"],
        exemplars=[{}, {}],
    ),
]


def make_report_dict():
    return stress_oracle(reliable_oracle, PROBES, n_repeats=2).to_dict()


class TestVerifyOracleReport:
    def test_fresh_report_verifies(self):
        result = verify_oracle_report_dict(make_report_dict())
        assert result["ok"], result["problems"]

    def test_all_reference_reports_verify(self):
        # concrete proof: every published oracle report re-derives
        paths = [
            os.path.join(REFERENCES, "cards", f"ao_qwen3_{m}.json")
            for m in ("full-mixture", "latentqa-only", "cls-only")
        ]
        if not all(os.path.exists(p) for p in paths):
            pytest.skip("reference cards not present")
        for p in paths:
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
            result = verify_oracle_report_dict(d)
            assert result["ok"], (p, result["problems"])

    def test_tampered_grade_detected(self):
        d = make_report_dict()
        d["verdict"]["grade"] = "A" if d["verdict"]["grade"] != "A" else "D"
        result = verify_oracle_report_dict(d)
        assert not result["ok"]
        assert any("grade" in p for p in result["problems"])

    def test_tampered_pass_flag_detected(self):
        d = make_report_dict()
        name, check = next(iter(d["checks"].items()))
        check["passed"] = not check["passed"]
        result = verify_oracle_report_dict(d)
        assert not result["ok"]
        assert any(name in p for p in result["problems"])

    def test_check_value_must_match_metrics(self):
        d = make_report_dict()
        d["metrics"]["known_accuracy"] = 0.123456
        result = verify_oracle_report_dict(d)
        assert not result["ok"]
        assert any("known_accuracy" in p for p in result["problems"])

    def test_pooled_metric_recomputes_from_probes(self):
        d = make_report_dict()
        # rewrite the underlying probe row: only the per-probe recompute
        # layer can catch a pooled metric that no longer follows from it
        known = next(p for p in d["per_probe"] if p["kind"] == "known")
        known["accuracy"] = 0.5
        result = verify_oracle_report_dict(d)
        assert not result["ok"]
        assert any("per-probe" in p for p in result["problems"])

    def test_tampered_wilson_ci_detected(self):
        d = make_report_dict()
        ci = d["metrics"]["known_accuracy_ci95"]
        widened = [max(0.0, ci[0] - 0.2), ci[1]]
        d["metrics"]["known_accuracy_ci95"] = widened
        d["checks"]["known_accuracy"]["ci"] = widened
        result = verify_oracle_report_dict(d)
        assert not result["ok"]
        assert any("ci95" in p for p in result["problems"])

    def test_wrong_artifact_rejected(self):
        with pytest.raises(ValueError):
            verify_oracle_report_dict({"artifact": "something_else"})


class TestClassifyAndDispatch:
    def test_classify_oracle_report(self):
        assert classify_artifact_dict(make_report_dict()) == "oracle_report"

    def test_classify_stability_card(self):
        path = os.path.join(REFERENCES, "cards", "ioi_gpt2_small.json")
        if not os.path.exists(path):
            pytest.skip("reference cards not present")
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        assert classify_artifact_dict(d) == "stability_card"

    def test_classify_rejects_badges_traces_and_junk(self):
        badge = {"schemaVersion": 1, "label": "stability", "message": "A"}
        trace = {"n_total": 45, "full_grade": "A", "per_size": {}}
        assert classify_artifact_dict(badge) == "unknown"
        assert classify_artifact_dict(trace) == "unknown"
        assert classify_artifact_dict([1, 2]) == "unknown"

    def test_dispatch_tags_kind(self):
        result = verify_artifact_dict(make_report_dict())
        assert result["kind"] == "oracle_report"
        assert result["ok"]

    def test_dispatch_rejects_unknown(self):
        with pytest.raises(ValueError):
            verify_artifact_dict({"hello": "world"})


class TestCliBatchVerify:
    def test_verify_references_directory(self, capsys):
        if not os.path.isdir(REFERENCES):
            pytest.skip("references not present")
        rc = cli_main(["verify", REFERENCES])
        out = capsys.readouterr().out
        assert rc == 0
        assert "0 failed" in out
        assert "skipped" in out           # badges/traces skipped, not failed
        assert out.count("OK:") >= 8

    def test_verify_multiple_files(self, tmp_path, capsys):
        good = tmp_path / "report.json"
        good.write_text(json.dumps(make_report_dict()))
        rc = cli_main(["verify", str(good), str(good)])
        out = capsys.readouterr().out
        assert rc == 0
        assert "2 verified" in out

    def test_explicit_non_artifact_fails(self, tmp_path, capsys):
        junk = tmp_path / "badge.json"
        junk.write_text(json.dumps({"schemaVersion": 1}))
        rc = cli_main(["verify", str(junk)])
        out = capsys.readouterr().out
        assert rc == 1
        assert "not a verifiable" in out

    def test_directory_skips_non_artifacts(self, tmp_path, capsys):
        (tmp_path / "report.json").write_text(json.dumps(make_report_dict()))
        (tmp_path / "badge.json").write_text(json.dumps({"schemaVersion": 1}))
        rc = cli_main(["verify", str(tmp_path)])
        out = capsys.readouterr().out
        assert rc == 0
        assert "1 verified, 0 failed, 1 skipped" in out

    def test_tampered_card_in_directory_fails(self, tmp_path, capsys):
        d = make_report_dict()
        d["verdict"]["grade"] = "A" if d["verdict"]["grade"] != "A" else "D"
        (tmp_path / "tampered.json").write_text(json.dumps(d))
        rc = cli_main(["verify", str(tmp_path)])
        out = capsys.readouterr().out
        assert rc == 1
        assert "FAILED" in out

    def test_empty_directory_fails(self, tmp_path, capsys):
        rc = cli_main(["verify", str(tmp_path)])
        assert rc == 1
        assert "no JSON artifacts" in capsys.readouterr().out
