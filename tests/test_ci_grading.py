"""Tests for CI-aware grading, confidence, and the Monte-Carlo null."""

import random

import pytest

import stresskit as sk
from stresskit.baselines import empirical_random_jaccard
from stresskit.metrics import expected_random_jaccard


# ---------------------------------------------------------------------------
# empirical (Monte-Carlo) null
# ---------------------------------------------------------------------------

def test_empirical_null_matches_analytic_for_equal_sizes():
    # equal sizes: MC mean should be close to the analytic k/(2N-k)
    mc = empirical_random_jaccard([10] * 12, 200, seed=1, repeats=400)
    an = expected_random_jaccard(10, 200)
    assert mc == pytest.approx(an, abs=0.02)


def test_empirical_null_handles_heterogeneous_sizes():
    mc = empirical_random_jaccard([5, 10, 20, 40], 500, seed=2)
    assert mc is not None and 0.0 < mc < 0.2


def test_empirical_null_none_when_underdetermined():
    assert empirical_random_jaccard([10], 200) is None
    assert empirical_random_jaccard([10, 10], 0) is None


# ---------------------------------------------------------------------------
# CI-aware checks + confidence
# ---------------------------------------------------------------------------

def rock_solid_finder(data, seed, config):
    # genuinely constant finding: same 8 components, claim, score every run
    return sk.feature_set(list(range(8)), claim="late", score=1.0,
                          universe_size=200)


def noisy_finder(data, seed, config):
    # each run keeps ~half the base set and swaps the rest → moderate,
    # high-variance Jaccard so the CI is wide at small n
    rng = random.Random(seed)
    base = list(range(8))
    keep = rng.sample(base, 4)
    extra = rng.sample(range(8, 40), 4)
    claim = rng.choice(["late", "late", "middle"])
    return sk.feature_set(keep + extra, claim=claim, score=0.5 + 0.4 * rng.random(),
                          universe_size=200)


DATA = list(range(60))


def test_checks_carry_ci_and_robust_flags():
    result = sk.stress(rock_solid_finder, DATA,
                       battery=["seeds", "bootstrap"], n_runs=8)
    c = result.checks["structural_stability"]
    assert c["ci"] is not None
    assert set(c) >= {"value", "threshold", "passed", "op", "ci", "robust"}
    # a perfectly stable finder is robustly stable
    assert c["passed"] and c["robust"] is True
    assert result.pooled["confidence"] == "high"


def test_underpowered_pass_is_flagged_low_confidence():
    result = sk.stress(noisy_finder, DATA,
                       battery=["seeds", "bootstrap"], n_runs=6,
                       thresholds=sk.Thresholds(jaccard=0.3))
    ss = result.checks["structural_stability"]
    if ss["passed"] and ss["robust"] is False:
        assert result.pooled["confidence"] == "low"
        assert "structural_stability" in result.pooled["borderline_checks"]
        assert any("underpowered" in n.lower() for n in result.card.notes)
        assert "⚠️" in result.card.to_markdown()
    else:
        pytest.skip("finder happened not to land borderline this seed")


def test_confidence_high_when_ci_clears_bar():
    result = sk.stress(rock_solid_finder, DATA,
                       battery=["seeds", "bootstrap"], n_runs=8,
                       thresholds=sk.Thresholds(jaccard=0.5))
    assert result.pooled["borderline_checks"] == []
    assert result.pooled["confidence"] == "high"


def test_engine_uses_empirical_null_and_keeps_analytic():
    result = sk.stress(rock_solid_finder, DATA,
                       battery=["seeds"], n_runs=5)
    assert result.pooled["expected_random_jaccard"] is not None       # MC
    assert result.pooled["expected_random_jaccard_analytic"] is not None
    assert result.pooled["jaccard_vs_random"] is not None


def test_verify_still_passes_with_ci_fields():
    from stresskit.card import verify_card_dict
    result = sk.stress(rock_solid_finder, DATA, battery=["seeds", "bootstrap"],
                       n_runs=6)
    out = verify_card_dict(result.card.to_dict())
    assert out["ok"], out["problems"]
