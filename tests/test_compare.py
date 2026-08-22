"""Stability regression testing: compare_cards and the compare CLI."""

import json
import os

import pytest

import stresskit as sk
from stresskit.cli import main as cli_main
from stresskit.compare import compare_cards, compare_markdown
from stresskit.oracle import OracleProbe, stress_oracle


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_card(jitter=0, thresholds=None, model="gpt2"):
    """A real, verifiable card. jitter=0 is stable; higher degrades it."""
    findings = []
    for i in range(8):
        drift = set(range(100 + i * jitter, 100 + i * jitter + jitter))
        comps = (set(range(20)) - set(range(i * jitter // 2))) | drift
        findings.append(sk.circuit(
            comps or {0}, claim="late" if jitter < 3 or i % 2 == 0 else f"mid{i}",
            score=1.0 + 0.01 * i * (1 + jitter), universe_size=144))
    return sk.from_findings(
        findings, thresholds=thresholds, model=model, task="toy",
        method="toy").card.to_dict()


def make_oracle_report(good=True):
    def ask(exemplar, question, seed):
        c = exemplar.get("concept")
        if c is None:
            return ("I don't know." if good else "The word is banana.")
        return f"The secret word is {c}." if good else "It is about sports."

    probes = [
        OracleProbe(name="p", concept="tree", expected="tree",
                    questions=["What is the secret word?",
                               "Which word is hidden?"],
                    exemplars=[{"concept": "tree"}, {"concept": "tree"}]),
        OracleProbe(name="null", kind="null",
                    questions=["What is the secret word?"],
                    exemplars=[{}, {}]),
    ]
    return stress_oracle(ask, probes, n_repeats=2).to_dict()


class TestCompareCards:
    def test_self_comparison_no_regression(self):
        card = make_card()
        cmp = compare_cards(card, card)
        assert not cmp["regressed"]
        assert cmp["regressions"] == []
        for r in cmp["checks"].values():
            assert r["delta"] == 0

    def test_degradation_detected(self):
        cmp = compare_cards(make_card(jitter=0), make_card(jitter=8))
        assert cmp["regressed"]
        assert cmp["grade_regressed"] or cmp["regressions"]

    def test_improvement_detected(self):
        cmp = compare_cards(make_card(jitter=8), make_card(jitter=0))
        assert not cmp["regressed"]
        assert cmp["grade_improved"] or cmp["improvements"]

    def test_threshold_mismatch_excluded(self):
        loose = sk.Thresholds(jaccard=0.1)
        cmp = compare_cards(make_card(jitter=8),
                            make_card(jitter=8, thresholds=loose))
        row = cmp["checks"]["structural_stability"]
        assert not row["comparable"]
        assert any("thresholds" in c for c in cmp["caveats"])
        # a pass under moved goalposts is not an improvement
        assert "structural_stability" not in cmp["improvements"]

    def test_cross_finding_caveat(self):
        cmp = compare_cards(make_card(model="gpt2"),
                            make_card(model="pythia-160m"))
        assert any("different findings" in c for c in cmp["caveats"])

    def test_mismatched_kinds_rejected(self):
        with pytest.raises(ValueError, match="cannot compare"):
            compare_cards(make_card(), make_oracle_report())

    def test_tampered_baseline_rejected(self):
        bad = make_card()
        bad["verdict"]["grade"] = "A" if bad["verdict"]["grade"] != "A" else "D"
        with pytest.raises(ValueError, match="does not verify"):
            compare_cards(bad, make_card())

    def test_oracle_reports_compare(self):
        cmp = compare_cards(make_oracle_report(good=True),
                            make_oracle_report(good=False))
        assert cmp["kind"] == "oracle_report"
        assert cmp["regressed"]

    def test_markdown_renders(self):
        md = compare_markdown(compare_cards(make_card(), make_card(jitter=8)))
        assert "REGRESSED" in md
        assert "| check |" in md


class TestCompareCli:
    def _write(self, tmp_path, name, d):
        p = tmp_path / name
        p.write_text(json.dumps(d))
        return str(p)

    def test_exit_zero_without_flag_even_on_regression(self, tmp_path, capsys):
        a = self._write(tmp_path, "a.json", make_card())
        b = self._write(tmp_path, "b.json", make_card(jitter=8))
        assert cli_main(["compare", a, b]) == 0
        assert "REGRESSED" in capsys.readouterr().out

    def test_fail_on_regression(self, tmp_path, capsys):
        a = self._write(tmp_path, "a.json", make_card())
        b = self._write(tmp_path, "b.json", make_card(jitter=8))
        assert cli_main(["compare", a, b, "--fail-on-regression"]) == 1

    def test_no_regression_passes_gate(self, tmp_path, capsys):
        a = self._write(tmp_path, "a.json", make_card())
        assert cli_main(["compare", a, a, "--fail-on-regression"]) == 0

    def test_real_reference_cards(self, capsys):
        a = os.path.join(REPO_ROOT, "references/scale/ioi_gpt2_medium.json")
        b = os.path.join(REPO_ROOT, "references/cards/ioi_gpt2_small.json")
        if not (os.path.exists(a) and os.path.exists(b)):
            pytest.skip("reference cards not present")
        assert cli_main(["compare", a, b]) == 0
        out = capsys.readouterr().out
        assert "cross-finding comparison" in out
