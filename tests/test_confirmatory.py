import copy
import json
import random

import pytest

import stresskit as sk
from stresskit.card import classify_artifact_dict, verify_artifact_dict
from stresskit.cli import main as cli_main


def manifest(n, seed=3):
    return sk.SpecificationSpace(
        axes={"seed_policy": ["a", "b"], "threshold": [0.1, 0.2]}
    ).sample_manifest(n, seed)


def stable_findings(n):
    return [
        sk.feature_set(
            range(20), claim="single", universe_size=100, universe="toy"
        )
        for _ in range(n)
    ]


def uniform_findings(n, seed=9):
    rng = random.Random(seed)
    return [
        sk.feature_set(
            rng.sample(range(100), 20),
            claim="single",
            universe_size=100,
            universe="toy",
        )
        for _ in range(n)
    ]


def make_result(n=200, *, null="uniform"):
    nulls = uniform_findings(n) if null == "uniform" else stable_findings(n)
    return sk.confirmatory_from_findings(
        stable_findings(n),
        manifest(n),
        claim_statement="registered toy structure is stable and specific",
        thresholds={
            "structural_stability": 0.8,
            "beats_random": 0.1,
            "claim_stability": 0.8,
            "specificity": 0.2,
        },
        threshold_justifications={
            "structural_stability": "toy recovery criterion",
            "beats_random": "minimum toy null separation",
            "claim_stability": "registered alpha=0.2 policy",
            "specificity": "minimum toy real-null effect",
        },
        claim_classes=["single", "multiple"],
        null_findings=nulls,
        null_manifest=manifest(n, seed=5),
        seed=17,
        model="toy",
        task="recovery",
        method="planted",
        claim_id="toy-001",
    )


def test_positive_control_passes_all_registered_gates():
    result = make_result()
    assert result.state == "pass"
    assert all(check["state"] == "pass" for check in result.checks.values())
    payload = result.card.to_dict()
    verified = sk.verify_confirmatory_card_dict(payload)
    assert verified["ok"], verified["problems"]
    assert "familywise confidence" in result.to_markdown()


def test_non_specific_stable_output_cannot_pass():
    result = make_result(null="stable")
    assert result.checks["specificity"]["state"] != "pass"
    assert result.state != "pass"


@pytest.mark.parametrize("n", [20, 100])
def test_below_calibrated_run_count_is_inconclusive(n):
    result = make_result(n=n)
    assert result.state == "inconclusive"
    assert all(not check["minimum_n_met"] for check in result.checks.values())


def test_diagnostic_oat_manifest_is_rejected():
    space = sk.SpecificationSpace(axes={"a": [0, 1]})
    with pytest.raises(ValueError, match="iid_specification_sample"):
        sk.confirmatory_from_findings(
            stable_findings(2),
            space.diagnostic_oat_manifest({"a": 0}),
            claim_statement="x",
            thresholds={"structural_stability": 0.8, "beats_random": 0.1},
            threshold_justifications={
                "structural_stability": "registered",
                "beats_random": "registered",
            },
        )


def test_thresholds_require_justification_and_validity_gates():
    with pytest.raises(ValueError, match="threshold keys"):
        sk.confirmatory_from_findings(
            stable_findings(2),
            manifest(2),
            claim_statement="x",
            thresholds={"structural_stability": 0.8},
            threshold_justifications={"structural_stability": "registered"},
        )
    with pytest.raises(ValueError, match="justification"):
        sk.confirmatory_from_findings(
            stable_findings(2),
            manifest(2),
            claim_statement="x",
            thresholds={"structural_stability": 0.8, "beats_random": 0.1},
            threshold_justifications={
                "structural_stability": "registered", "beats_random": ""
            },
        )


@pytest.mark.parametrize(
    "mutation,problem",
    [
        (lambda d: d["metrics"].__setitem__("paired_mean_jaccard", 0.0), "metrics"),
        (lambda d: d["checks"]["structural_stability"].__setitem__("state", "fail"), "checks"),
        (lambda d: d["verdict"].__setitem__("state", "fail"), "verdict"),
        (lambda d: d["runs"][0]["manifest"].__setitem__("configuration", {}), "manifest"),
        (lambda d: d["runs"][0]["components"].append(999), "sha256"),
    ],
)
def test_verifier_rejects_tampering(mutation, problem):
    payload = make_result().card.to_dict()
    mutation(payload)
    verified = sk.verify_confirmatory_card_dict(payload)
    assert not verified["ok"]
    assert any(problem in message for message in verified["problems"])


def test_dispatch_and_cli_support_confirmatory_card(tmp_path, capsys):
    payload = make_result().card.to_dict()
    assert classify_artifact_dict(payload) == "confirmatory_card"
    assert verify_artifact_dict(payload)["ok"]
    path = tmp_path / "confirmatory.json"
    path.write_text(json.dumps(payload))
    assert cli_main(["verify", str(path)]) == 0
    assert "verdict pass" in capsys.readouterr().out
