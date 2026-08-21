"""Tests for the two-sample ratio bootstrap (specificity / beats-random CIs)
and the verdict-stability trace."""

import random

import pytest

import stresskit as sk
from stresskit.metrics import bootstrap_ci_ratio_pairwise, jaccard


def stable_sets(n, jitter=2, size=30, seed=0):
    rng = random.Random(seed)
    return [frozenset(set(range(size)) - set(rng.sample(range(size), jitter)))
            for _ in range(n)]


def random_sets(n, size=27, universe=144, seed=1):
    rng = random.Random(seed)
    return [frozenset(rng.sample(range(universe), size)) for _ in range(n)]


def make_findings(sets, claim="late", score=0.9, universe_size=144):
    return [sk.feature_set(s, claim=claim, score=score + 0.001 * i,
                           universe_size=universe_size)
            for i, s in enumerate(sets)]


# ---------------------------------------------------------------------------
# bootstrap_ci_ratio_pairwise
# ---------------------------------------------------------------------------

def test_ratio_ci_separated_groups_excludes_one():
    real, null = stable_sets(10), random_sets(10)
    ci = bootstrap_ci_ratio_pairwise(real, null, jaccard, seed=0)
    assert ci is not None and ci[0] > 1.5


def test_ratio_ci_identical_groups_contains_one():
    sets = stable_sets(10)
    ci = bootstrap_ci_ratio_pairwise(sets, list(sets), jaccard, seed=0)
    assert ci is not None and ci[0] <= 1.0 <= ci[1]


def test_ratio_ci_needs_four_per_group():
    assert bootstrap_ci_ratio_pairwise(stable_sets(3), stable_sets(10), jaccard) is None
    assert bootstrap_ci_ratio_pairwise(stable_sets(10), stable_sets(3), jaccard) is None


def test_ratio_ci_point_estimate_inside():
    real, null = stable_sets(12), random_sets(12)
    from stresskit.metrics import mean_pairwise_jaccard
    point = mean_pairwise_jaccard(real) / mean_pairwise_jaccard(null)
    ci = bootstrap_ci_ratio_pairwise(real, null, jaccard, seed=3)
    assert ci[0] < point < ci[1]


def test_ratio_ci_zero_denominator_replicates_dropped():
    # disjoint null sets: pairwise Jaccard is exactly 0 in every replicate
    real = stable_sets(8)
    null = [frozenset({10 * i, 10 * i + 1}) for i in range(8)]
    assert bootstrap_ci_ratio_pairwise(real, null, jaccard, seed=0) is None


# ---------------------------------------------------------------------------
# CI-carrying specificity and beats_random checks
# ---------------------------------------------------------------------------

def test_specificity_check_carries_ci():
    result = sk.from_findings(
        make_findings(stable_sets(10)),
        null_findings=make_findings(random_sets(10), claim="none", score=0.1),
    )
    spec = result.checks["specificity"]
    assert spec["ci"] is not None
    assert spec["robust"] is True  # cleanly separated groups
    assert result.pooled["specificity_ci95"] == spec["ci"]


def test_beats_random_check_carries_ci():
    result = sk.from_findings(make_findings(stable_sets(10)))
    br = result.checks["beats_random"]
    assert br["ci"] is not None and br["robust"] is True
    lo, hi = result.pooled["mean_pairwise_jaccard_ci95"]
    null = result.pooled["expected_random_jaccard"]
    assert br["ci"] == pytest.approx([lo / null, hi / null])


def test_specificity_straddle_lowers_confidence():
    # null nearly as stable as real: ratio sits close to 1.5 with a wide CI
    rng = random.Random(7)
    real = [frozenset(set(range(30)) - set(rng.sample(range(30), 4))) for _ in range(8)]
    null = [frozenset(set(range(100, 130)) - set(rng.sample(range(100, 130), 7)))
            for _ in range(8)]
    result = sk.from_findings(
        make_findings(real),
        null_findings=make_findings(null, claim="none", score=0.1),
        thresholds=sk.Thresholds(specificity_ratio=1.25),
    )
    spec = result.checks["specificity"]
    assert spec["ci"][0] < 1.25 < spec["ci"][1]
    assert spec["robust"] is False
    assert "specificity" in result.pooled["borderline_checks"]
    assert result.pooled["confidence"] == "low"


def test_small_null_group_degrades_to_no_ci():
    result = sk.from_findings(
        make_findings(stable_sets(10)),
        null_findings=make_findings(random_sets(3), claim="none", score=0.1),
    )
    spec = result.checks["specificity"]
    assert spec["ci"] is None and spec["robust"] is None


def test_card_roundtrip_verifies_with_new_cis(tmp_path):
    result = sk.from_findings(
        make_findings(stable_sets(10)),
        null_findings=make_findings(random_sets(10), claim="none", score=0.1),
        model="toy", task="unit",
    )
    path = tmp_path / "card.json"
    result.card.save(str(path))
    card = sk.load_card(str(path))
    report = sk.verify_card_dict(card.to_dict())
    assert report["ok"] is True


# ---------------------------------------------------------------------------
# verdict_trace
# ---------------------------------------------------------------------------

def test_trace_stable_finding_settles_early():
    trace = sk.verdict_trace(make_findings(stable_sets(14)),
                             sizes=[4, 6, 10, 14], n_subsamples=15, seed=0)
    assert trace["full_grade"] == "A"
    assert trace["settled_n"] == 4
    assert trace["sizes"] == [4, 6, 10, 14]
    row = trace["per_size"][6]
    assert row["modal_grade"] == "A" and row["modal_grade_share"] == 1.0
    assert set(row["check_pass_frac"]) == set(
        sk.from_findings(make_findings(stable_sets(14))).checks
    )


def test_trace_borderline_finding_shows_disagreement():
    # jaccard hovers around the 0.8 bar: subset grades should disagree at
    # small n, and the full-sample verdict should be low confidence
    rng = random.Random(3)
    sets = [frozenset(set(range(20)) - set(rng.sample(range(20), rng.choice([2, 3, 4]))))
            for _ in range(16)]
    trace = sk.verdict_trace(make_findings(sets), sizes=[4, 6, 16],
                             n_subsamples=25, seed=1)
    small = trace["per_size"][4]
    assert len(small["grade_dist"]) >= 2 or small["low_confidence_share"] > 0


def test_trace_full_size_uses_single_draw():
    trace = sk.verdict_trace(make_findings(stable_sets(8)), sizes=[4, 8],
                             n_subsamples=10, seed=0)
    assert trace["per_size"][8]["n_subsamples"] == 1


def test_trace_input_validation():
    with pytest.raises(ValueError, match=">= 5"):
        sk.verdict_trace(make_findings(stable_sets(4)))
    with pytest.raises(ValueError, match="must lie in"):
        sk.verdict_trace(make_findings(stable_sets(8)), sizes=[2, 8])
    with pytest.raises(ValueError, match="must lie in"):
        sk.verdict_trace(make_findings(stable_sets(8)), sizes=[4, 30])


def test_trace_deterministic():
    findings = make_findings(stable_sets(10))
    a = sk.verdict_trace(findings, sizes=[4, 6], n_subsamples=8, seed=5)
    b = sk.verdict_trace(findings, sizes=[4, 6], n_subsamples=8, seed=5)
    assert a == b


def test_trace_markdown_renders():
    trace = sk.verdict_trace(make_findings(stable_sets(10)),
                             null_findings=make_findings(random_sets(10),
                                                         claim="none", score=0.1),
                             sizes=[4, 6, 10], n_subsamples=8, seed=0)
    md = sk.verdict_trace_markdown(trace)
    assert "Verdict-stability trace" in md
    assert "| 4 |" in md and "| 10 |" in md
    assert ("settles at" in md) or ("not settle" in md)
