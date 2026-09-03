"""Rebuilding findings from a committed card, regrading them at the recorded
seed, and relabelling a v0.3 card under grade rule v0.4."""

import copy
import json
import os
import random

import pytest

import stresskit as sk
from stresskit.battery import grade_checks
from stresskit.card import verify_card_dict
from stresskit.card_findings import (
    card_thresholds,
    findings_from_card_dict,
    findings_from_manifest_real_rows,
    grade_reasons,
    null_findings_from_manifest,
    recomputed_checks,
    regrade_card,
    regrade_findings,
    relabel_grade,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IOI_CARD = os.path.join(REPO_ROOT, "references", "cards", "ioi_gpt2_small.json")
CONFIRMATORY_DIR = os.path.join(REPO_ROOT, "references", "cards", "confirmatory")
ORACLE_REPORT = os.path.join(
    REPO_ROOT, "references", "cards", "ao_qwen3_full-mixture.json"
)
NOTE_DATE = "2026-09-03"


def finder(data, seed, config):
    """Stable core on real data (positive ints); noise on null data."""
    rng = random.Random(seed * 7919 + len(data))
    if min(data) > 0:
        core = set(range(20)) - set(rng.sample(range(20), 2))
        claim = "late" if rng.random() < 0.8 else "early"
        return sk.feature_set(
            core, claim=claim, score=0.9 + 0.02 * rng.random(), universe_size=144
        )
    return sk.feature_set(
        rng.sample(range(144), 18),
        claim=rng.choice(["late", "early", "mid"]),
        score=0.5 + 0.3 * rng.random(),
        universe_size=144,
    )


@pytest.fixture(scope="module")
def result():
    return sk.stress(
        finder,
        list(range(1, 21)),
        battery=["seeds", "bootstrap"],
        n_runs=5,
        seed=3,
        null_data=[-x for x in range(1, 21)],
        task="toy-task",
        model="toy-model",
    )


@pytest.fixture(scope="module")
def card_dict(result):
    return result.card.to_dict()


def null_manifest(result):
    return {
        "runs": [
            {
                "group": "null",
                "axis": r.axis,
                "variant": r.variant,
                "seed": r.seed,
                "claim": r.finding.claim,
                "score": r.finding.score,
                "size": r.finding.size,
                "components": sorted(str(c) for c in r.finding.components),
            }
            for r in result.null_runs
        ]
    }


def assert_same_checks(recorded, fresh):
    assert set(recorded) == set(fresh)
    for name, c in recorded.items():
        assert fresh[name]["value"] == c["value"], name
        assert fresh[name]["ci"] == c["ci"], name
        assert fresh[name]["state"] == c["state"], name


def legacy_copy(card_dict):
    """The same card as it would have been written before grade rule v0.4."""
    legacy = copy.deepcopy(card_dict)
    legacy["schema_version"] = "0.4"
    del legacy["verdict"]["grade_rule"]
    del legacy["verdict"]["thresholds"]["random_floor"]
    del legacy["verdict"]["thresholds"]["specificity_ratio"]
    legacy["verdict"]["grade"] = grade_checks(recomputed_checks(legacy), rule="v0.3")
    return legacy


class TestRegradeFromCard:
    def test_regrade_reproduces_checks_and_grade(self, result, card_dict):
        findings, axes = findings_from_card_dict(card_dict)
        assert len(findings) == len(card_dict["runs"])
        assert axes == [r["axis"] for r in card_dict["runs"][1:]]
        nulls = null_findings_from_manifest(null_manifest(result), universe_size=144)
        assert len(nulls) == len(result.null_runs)
        fresh = regrade_card(
            card_dict, seed=card_dict["battery"]["seed"], null_findings=nulls
        )
        assert_same_checks(card_dict["verdict"]["checks"], fresh.checks)
        assert fresh.grade == card_dict["verdict"]["grade"]
        assert (
            fresh.pooled["confidence"] == card_dict["metrics"]["pooled"]["confidence"]
        )

    def test_manifest_real_rows_reproduce_a_hash_only_card(self, result, card_dict):
        hash_only = copy.deepcopy(card_dict)
        hash_only["battery"]["components_embedded"] = False
        manifest = {"runs": []}
        for row in hash_only["runs"]:
            manifest["runs"].append({**row, "group": "real"})
            del row["components"]
        findings, axes = findings_from_manifest_real_rows(hash_only, manifest)
        direct, direct_axes = findings_from_card_dict(card_dict)
        assert axes == direct_axes
        assert [f.components for f in findings] == [f.components for f in direct]
        with pytest.raises(ValueError, match="hash-only"):
            regrade_card(hash_only, seed=hash_only["battery"]["seed"])
        nulls = null_findings_from_manifest(null_manifest(result), universe_size=144)
        fresh = regrade_findings(
            hash_only, findings, axes, seed=hash_only["battery"]["seed"], null_findings=nulls
        )
        assert_same_checks(card_dict["verdict"]["checks"], fresh.checks)

        tampered = copy.deepcopy(manifest)
        tampered["runs"][2]["components"].append("L99H99")
        with pytest.raises(ValueError, match="components_sha256"):
            findings_from_manifest_real_rows(hash_only, tampered)

    def test_hash_only_row_names_the_card(self, card_dict):
        bad = copy.deepcopy(card_dict)
        del bad["runs"][3]["components"]
        with pytest.raises(ValueError, match="toy-task / toy-model"):
            findings_from_card_dict(bad)
        with pytest.raises(ValueError, match="my-card"):
            findings_from_card_dict(bad, name="my-card")

    def test_base_must_come_first(self, card_dict):
        bad = copy.deepcopy(card_dict)
        bad["runs"] = bad["runs"][1:] + bad["runs"][:1]
        with pytest.raises(ValueError, match="base"):
            findings_from_card_dict(bad)

    def test_null_manifest_without_components_is_refused(self):
        manifest = {"null": [{"axis": "base", "size": 5, "claim": "x"}] * 3}
        with pytest.raises(ValueError, match="no components"):
            null_findings_from_manifest(manifest, universe_size=144)


class TestThresholds:
    def test_defaults_fill_the_missing_bars(self, card_dict):
        legacy = legacy_copy(card_dict)
        thresholds = card_thresholds(legacy)
        assert thresholds == sk.Thresholds()

    def test_non_default_bar_is_rejected(self, card_dict):
        bad = copy.deepcopy(card_dict)
        bad["verdict"]["thresholds"]["jaccard"] = 0.7
        with pytest.raises(ValueError, match="jaccard=0.7"):
            card_thresholds(bad)


class TestRelabel:
    def test_fixture_separates_the_rules(self, card_dict):
        checks = recomputed_checks(card_dict)
        assert grade_checks(checks, rule="v0.3") != grade_checks(checks, rule="v0.4")

    def test_current_card_is_a_no_op(self, card_dict):
        out = relabel_grade(card_dict, note_date=NOTE_DATE)
        assert out["verdict"]["grade"] == card_dict["verdict"]["grade"]
        assert out["verdict"]["grade_rule"] == "v0.4"
        assert out["notes"] == card_dict["notes"]
        assert out["verdict"]["checks"] == card_dict["verdict"]["checks"]
        assert verify_card_dict(out)["ok"]

    def test_legacy_card_is_relabelled(self, card_dict):
        legacy = legacy_copy(card_dict)
        assert verify_card_dict(legacy)["ok"]
        out = relabel_grade(legacy, note_date=NOTE_DATE)
        report = verify_card_dict(out)
        assert report["ok"], report["problems"]
        assert out["schema_version"] == "0.5"
        assert out["verdict"]["grade_rule"] == "v0.4"
        assert out["verdict"]["grade"] == grade_checks(
            recomputed_checks(out), rule="v0.4"
        )
        assert out["verdict"]["grade"] != legacy["verdict"]["grade"]
        assert out["verdict"]["checks"] == legacy["verdict"]["checks"]
        assert out["verdict"]["thresholds"]["random_floor"] == 1.5
        assert out["verdict"]["thresholds"]["specificity_ratio"] == 1.5
        assert out["notes"][-1] == (
            f"v0.3 grade: {legacy['verdict']['grade']}; regraded {NOTE_DATE} "
            "under grade rule v0.4 from the recorded checks (schema 0.5)"
        )
        assert legacy["schema_version"] == "0.4"
        reasons = grade_reasons(recomputed_checks(out), random_floor=1.5)
        assert reasons["undecided"]
        assert reasons["cap"] == "A"

    def test_schema_02_card_gains_the_required_fields(self, card_dict):
        old = legacy_copy(card_dict)
        old["schema_version"] = "0.2"
        for key in ("profile", "confirmatory_state", "required_checks"):
            del old["verdict"][key]
        for check in old["verdict"]["checks"].values():
            del check["state"]
        assert verify_card_dict(old)["ok"]
        out = relabel_grade(old, note_date=NOTE_DATE)
        report = verify_card_dict(out)
        assert report["ok"], report["problems"]
        assert out["verdict"]["profile"] == "diagnostic"
        for name, check in out["verdict"]["checks"].items():
            assert check["state"] == card_dict["verdict"]["checks"][name]["state"]

    def test_inconsistent_recorded_grade_is_refused(self, card_dict):
        legacy = legacy_copy(card_dict)
        legacy["verdict"]["grade"] = "D"
        with pytest.raises(ValueError, match="does not re-derive"):
            relabel_grade(legacy, note_date=NOTE_DATE)

    def test_refuses_non_diagnostic_artifacts(self, card_dict):
        with pytest.raises(ValueError, match="diagnostic stability card"):
            relabel_grade(
                {"artifact": "stresskit_confirmatory_card"}, note_date=NOTE_DATE
            )
        with pytest.raises(ValueError, match="diagnostic stability card"):
            relabel_grade({"artifact": "stresskit_oracle_report"}, note_date=NOTE_DATE)
        confirmatory_profile = copy.deepcopy(card_dict)
        confirmatory_profile["verdict"]["profile"] = "confirmatory"
        with pytest.raises(ValueError, match="confirmatory"):
            relabel_grade(confirmatory_profile, note_date=NOTE_DATE)

    def test_refuses_committed_certificates_and_reports(self):
        paths = []
        if os.path.isdir(CONFIRMATORY_DIR):
            paths += [
                os.path.join(CONFIRMATORY_DIR, f)
                for f in sorted(os.listdir(CONFIRMATORY_DIR))
                if f.endswith(".confirmatory.json")
            ]
        if os.path.exists(ORACLE_REPORT):
            paths.append(ORACLE_REPORT)
        if not paths:
            pytest.skip("reference certificates and reports not present")
        for path in paths:
            with open(path, encoding="utf-8") as f:
                artifact = json.load(f)
            with pytest.raises(ValueError):
                relabel_grade(artifact, note_date=NOTE_DATE)


@pytest.mark.skipif(not os.path.exists(IOI_CARD), reason="reference card not present")
def test_ioi_card_structural_check_reproduces_exactly():
    with open(IOI_CARD, encoding="utf-8") as f:
        card = json.load(f)
    fresh = regrade_card(card, seed=card["battery"]["seed"])
    recorded = card["verdict"]["checks"]["structural_stability"]
    assert fresh.checks["structural_stability"]["value"] == recorded["value"]
    assert fresh.checks["structural_stability"]["ci"] == recorded["ci"]
