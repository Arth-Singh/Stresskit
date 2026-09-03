import pytest

from stresskit.battery_calibration import (
    battery_calibration_source_digest,
    default_battery_scenarios,
    run_battery_cell,
)
from stresskit.calibration import (
    StructuralScenario,
    calibration_source_digest,
    run_calibration_cell,
)
from stresskit.calibration_merge import merge_calibration_payloads


def payload(result, source_digest=None):
    return {
        "schema_version": "0.1",
        "study": "structural_known_truth_pilot",
        "provenance": {
            "source_sha256": source_digest or calibration_source_digest()
        },
        "results": [result.to_dict()],
    }


def battery_payload(result, source_digest=None):
    return {
        "schema_version": "0.1",
        "study": "battery_known_truth",
        "provenance": {
            "source_sha256": source_digest or battery_calibration_source_digest()
        },
        "results": [result.to_dict()],
    }


def assert_close(actual, expected):
    if isinstance(expected, dict):
        assert set(actual) == set(expected)
        for key, value in expected.items():
            assert_close(actual[key], value)
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected)
    else:
        assert actual == expected


def test_merge_matches_same_trial_range_run_as_one_cell():
    scenario = StructuralScenario("uniform", "uniform", 20, 4)
    kwargs = dict(
        scenario=scenario,
        interval_method="paired_hoeffding",
        n_runs=20,
        master_seed=9,
        threshold=0.8,
    )
    left = run_calibration_cell(n_trials=7, trial_start=0, **kwargs)
    right = run_calibration_cell(n_trials=13, trial_start=7, **kwargs)
    full = run_calibration_cell(n_trials=20, trial_start=0, **kwargs).to_dict()
    merged = merge_calibration_payloads([payload(left), payload(right)])
    row = merged["results"][0]
    assert row["trials_requested"] == 20
    assert row["coverage_count"] == full["coverage_count"]
    assert row["estimate_sum"] == pytest.approx(full["estimate_sum"])
    assert row["rmse"] == pytest.approx(full["rmse"])


def test_merge_rejects_overlap_and_source_mismatch():
    scenario = StructuralScenario("stable", "stable", 20, 4)
    first = run_calibration_cell(
        scenario, "paired_hoeffding", 10, 5, trial_start=0
    )
    overlap = run_calibration_cell(
        scenario, "paired_hoeffding", 10, 5, trial_start=4
    )
    with pytest.raises(ValueError, match="overlapping"):
        merge_calibration_payloads([payload(first), payload(overlap)])
    with pytest.raises(ValueError, match="source_sha256"):
        merge_calibration_payloads([payload(first), payload(overlap, "0" * 64)])


def test_merge_battery_shards_matches_single_run():
    scenario = default_battery_scenarios()[0]
    kwargs = dict(n_runs=6, master_seed=4)
    left = run_battery_cell(scenario, n_trials=3, trial_start=0, **kwargs)
    right = run_battery_cell(scenario, n_trials=5, trial_start=3, **kwargs)
    full = run_battery_cell(scenario, n_trials=8, trial_start=0, **kwargs).to_dict()
    merged = merge_calibration_payloads(
        [battery_payload(right), battery_payload(left)]
    )
    assert merged["study"] == "battery_known_truth"
    assert merged["provenance"]["input_payload_count"] == 2
    assert list(merged["cell_trial_ranges"].values()) == [[[0, 3], [3, 8]]]
    row = merged["results"][0]
    assert row["trial_start"] == 0
    assert row["trials_requested"] == 8
    assert row["counts"] == full["counts"]
    assert row["sums"] == pytest.approx(full["sums"])
    assert_close(row["rates"], full["rates"])
    assert row["truths"] == full["truths"]
    assert row["truth_grade_v04"] == full["truth_grade_v04"]


def test_merge_rejects_battery_overlap_and_source_mismatch():
    scenario = default_battery_scenarios()[0]
    first = run_battery_cell(scenario, 6, 3, trial_start=0)
    overlap = run_battery_cell(scenario, 6, 3, trial_start=2)
    with pytest.raises(ValueError, match="overlapping"):
        merge_calibration_payloads([battery_payload(first), battery_payload(overlap)])
    with pytest.raises(ValueError, match="source_sha256"):
        merge_calibration_payloads(
            [battery_payload(first), battery_payload(overlap, "0" * 64)]
        )


def test_merge_rejects_mixed_and_unknown_studies():
    structural = run_calibration_cell(
        StructuralScenario("stable", "stable", 20, 4), "paired_hoeffding", 10, 2
    )
    battery = run_battery_cell(default_battery_scenarios()[0], 6, 2)
    with pytest.raises(ValueError, match="one study"):
        merge_calibration_payloads([payload(structural), battery_payload(battery)])
    unknown = payload(structural)
    unknown["study"] = "test"
    with pytest.raises(ValueError, match="no merge rule"):
        merge_calibration_payloads([unknown])
