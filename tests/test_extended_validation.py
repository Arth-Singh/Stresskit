from stresskit.extended_validation import (
    run_extended_validation,
    s6_interaction_counterexample,
    s8_score_applicability,
)


def test_s6_oat_hides_crossed_interaction_exactly():
    result = s6_interaction_counterexample()
    assert result["oat_mean_pairwise_jaccard"] == 1.0
    assert result["crossed_mean_pairwise_jaccard"] == 0.5


def test_s8_applicability_controls_match_expectation():
    rows = s8_score_applicability()
    assert all(row["matched_expectation"] for row in rows)
    assert next(row for row in rows if row["scenario"] == "S8_signed")["value"] is None


def test_extended_study_is_reproducible_and_structured():
    kwargs = dict(
        n_trials=10,
        run_counts=[20],
        cluster_counts=[10],
        repeats_per_cluster=3,
        master_seed=7,
    )
    first = run_extended_validation(**kwargs)
    second = run_extended_validation(**kwargs)
    first.pop("provenance")
    second.pop("provenance")
    assert first == second
    assert len(first["S7_specificity"]) == 2
    assert len(first["S9_dependence"]) == 1
    assert first["S7_specificity"][0]["coverage"] >= 0.9
