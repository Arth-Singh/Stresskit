"""Tests for verify (auditor mode) and universe-aware structural comparison."""

import json
import random

import pytest

import stresskit as sk
from stresskit.card import verify_card_dict
from stresskit.cli import main as cli_main


def stable_finder(data, seed, config):
    rng = random.Random(seed)
    picked = sorted(rng.sample(sorted(data), 8))
    return sk.feature_set(picked, claim="middle", score=0.9,
                          universe_size=100)


DATA = list(range(30))


def run_battery(**kw):
    return sk.stress(stable_finder, DATA, battery=["seeds", "bootstrap"],
                     n_runs=5, **kw)


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------

def test_verify_accepts_untouched_card():
    card = run_battery().card.to_dict()
    out = verify_card_dict(card)
    assert out["ok"], out["problems"]
    assert out["recomputed_grade"] == card["verdict"]["grade"]


def test_verify_catches_flipped_pass():
    card = run_battery().card.to_dict()
    name = next(iter(card["verdict"]["checks"]))
    card["verdict"]["checks"][name]["passed"] = \
        not card["verdict"]["checks"][name]["passed"]
    out = verify_card_dict(card)
    assert not out["ok"]
    assert any(name in p for p in out["problems"])


def test_verify_catches_inflated_grade():
    card = run_battery().card.to_dict()
    # doctor a value so the stored pass/grade no longer follow
    for c in card["verdict"]["checks"].values():
        c["value"] = c["threshold"] * (0.5 if "stability" in str(c) else 0.5)
    out = verify_card_dict(card)
    assert not out["ok"]


def test_verify_cli(tmp_path):
    result = run_battery()
    path = tmp_path / "card.json"
    result.card.save(str(path))
    assert cli_main(["verify", str(path)]) == 0

    d = json.loads(path.read_text())
    d["verdict"]["grade"] = "A" if d["verdict"]["grade"] != "A" else "D"
    doctored = tmp_path / "doctored.json"
    doctored.write_text(json.dumps(d))
    assert cli_main(["verify", str(doctored)]) == 1


# ---------------------------------------------------------------------------
# universe-aware structural comparison
# ---------------------------------------------------------------------------

def universe_finder(data, seed, config):
    rng = random.Random(seed)
    items = sorted(data)
    picked = sorted(rng.sample(items, 8))
    return sk.feature_set(
        picked, claim="middle", score=0.9, universe_size=100,
        universe="A" if items[0] < 1000 else "B",
    )


def test_cross_universe_runs_excluded_from_jaccard():
    other = list(range(1000, 1030))     # disjoint item namespace
    result = sk.stress(
        universe_finder, DATA,
        battery=["seeds", "templates"], n_runs=5,
        templates={"other-dataset": other},
    )
    pooled = result.pooled
    assert pooled["n_cross_universe_excluded"] == 1
    assert any("different\ncomponent universe".replace("\n", " ") in n
               or "component universe" in n for n in result.card.notes)
    # templates-axis Jaccard must be undefined, not 0.0
    assert result.axis_metrics["templates"]["mean_pairwise_jaccard"] is None
    # pooled Jaccard must equal the same-universe-only computation
    from stresskit.metrics import mean_pairwise_jaccard
    same_universe = [
        r.finding.components for r in result.runs
        if r.finding.meta.get("universe") == "A"
    ]
    assert pooled["mean_pairwise_jaccard"] == pytest.approx(
        mean_pairwise_jaccard(same_universe))
    # the cross-universe run still contributes its claim
    assert pooled["modal_share"] == 1.0


def test_no_universe_label_keeps_old_behavior():
    result = sk.stress(
        stable_finder, DATA,
        battery=["seeds", "templates"], n_runs=4,
        templates={"same-universe": list(range(30))},
    )
    assert "n_cross_universe_excluded" not in result.pooled
    assert result.axis_metrics["templates"]["mean_pairwise_jaccard"] is not None
