import pytest

from stresskit.calibration import (
    StructuralScenario,
    calibration_source_digest,
    run_calibration_cell,
)
from stresskit.calibration_merge import merge_calibration_payloads


def payload(result, source_digest=None):
    return {
        "schema_version": "0.1",
        "study": "test",
        "provenance": {
            "source_sha256": source_digest or calibration_source_digest()
        },
        "results": [result.to_dict()],
    }


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
