import json

import pytest

from stresskit.calibration import (
    StructuralScenario,
    default_structural_scenarios,
    main,
    run_calibration_cell,
)


def test_default_scenarios_cover_structural_s1_through_s5():
    scenarios = default_structural_scenarios()
    assert [scenario.name.split("_")[0] for scenario in scenarios] == [
        "S1",
        "S2",
        "S3",
        "S4",
        "S5",
    ]
    assert all(0.0 <= scenario.true_mean_jaccard() <= 1.0 for scenario in scenarios)


def test_two_mode_truth_matches_direct_distribution_sum():
    scenario = StructuralScenario(
        "modes",
        "two_mode",
        universe_size=12,
        finding_size=5,
        mixture_probability=0.7,
        mode_overlap=2,
    )
    between_mode_jaccard = 2 / 8
    expected = 0.7**2 + 0.3**2 + 2 * 0.7 * 0.3 * between_mode_jaccard
    assert scenario.true_mean_jaccard() == pytest.approx(expected)


def test_invalid_scenario_rejected_before_simulation():
    with pytest.raises(ValueError, match="do not fit"):
        StructuralScenario("bad", "two_mode", 10, 8, mode_overlap=0)
    with pytest.raises(ValueError, match="alternate_size"):
        StructuralScenario("bad", "heterogeneous_uniform", 10, 3)


def test_stable_cell_has_exact_estimate_interval_and_decision():
    scenario = StructuralScenario("stable", "stable", 20, 4)
    result = run_calibration_cell(
        scenario,
        "jackknife_normal",
        n_runs=5,
        n_trials=12,
        threshold=0.8,
    ).to_dict()
    assert result["mean_estimate"] == 1.0
    assert result["rmse"] == 0.0
    assert result["empirical_coverage"] == 1.0
    assert result["mean_interval_width"] == 0.0
    assert result["pass_rate"] == 1.0
    assert result["false_fail_rate"] == 0.0


def test_calibration_cell_is_reproducible_and_shardable():
    scenario = StructuralScenario("uniform", "uniform", 30, 6)
    kwargs = dict(
        interval_method="jackknife_normal",
        n_runs=8,
        confidence_level=0.95,
        threshold=0.2,
        master_seed=123,
    )
    full = run_calibration_cell(scenario, n_trials=20, **kwargs)
    repeated = run_calibration_cell(scenario, n_trials=20, **kwargs)
    assert repeated == full

    left = run_calibration_cell(scenario, n_trials=7, **kwargs)
    right = run_calibration_cell(
        scenario, n_trials=13, trial_start=7, **kwargs
    )
    assert left.estimate_count + right.estimate_count == full.estimate_count
    assert left.interval_valid_count + right.interval_valid_count == full.interval_valid_count
    assert left.coverage_count + right.coverage_count == full.coverage_count
    assert left.pass_count + right.pass_count == full.pass_count
    assert left.fail_count + right.fail_count == full.fail_count
    assert left.inconclusive_count + right.inconclusive_count == full.inconclusive_count
    assert left.estimate_sum + right.estimate_sum == pytest.approx(full.estimate_sum)
    assert left.squared_error_sum + right.squared_error_sum == pytest.approx(
        full.squared_error_sum
    )


def test_result_reports_monte_carlo_error_and_three_state_rates():
    scenario = StructuralScenario("uniform", "uniform", 20, 4)
    result = run_calibration_cell(
        scenario,
        "jackknife_normal",
        n_runs=6,
        n_trials=25,
        threshold=0.8,
    ).to_dict()
    assert result["coverage_mcse"] is not None
    assert result["interval_valid_rate"] == 1.0
    assert result["pass_rate"] + result["fail_rate"] + result["inconclusive_rate"] == 1.0
    assert result["false_pass_rate"] == result["pass_rate"]
    assert result["false_fail_rate"] is None


def test_paired_hoeffding_uses_matching_reference_estimator():
    scenario = StructuralScenario("stable", "stable", 20, 4)
    result = run_calibration_cell(
        scenario,
        "paired_hoeffding",
        n_runs=100,
        n_trials=5,
        threshold=0.8,
    ).to_dict()
    assert result["mean_estimate"] == 1.0
    assert result["empirical_coverage"] == 1.0
    assert result["pass_rate"] == 1.0


def test_cli_emits_machine_readable_payload(capsys):
    assert main(
        [
            "--runs",
            "4",
            "--trials",
            "2",
            "--methods",
            "jackknife_normal",
            "--bootstrap-replicates",
            "20",
        ]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == "0.1"
    assert len(payload["provenance"]["source_sha256"]) == 64
    assert payload["configuration"]["trial_start"] == 0
    assert len(payload["results"]) == 5
    assert all(row["estimate_count"] == 2 for row in payload["results"])
