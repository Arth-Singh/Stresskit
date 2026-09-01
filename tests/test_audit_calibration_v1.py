"""Calibration acceptance and fresh-seed reporting for v1 profiles."""

import json
from pathlib import Path

from stresskit.audit_calibration import run_profile_calibration


def test_small_fresh_seed_calibration_has_required_fields():
    artifact = run_profile_calibration(trials=100, units=200, seed=7)
    assert artifact["primary"]["seed"] == 7
    assert artifact["fresh_seed_replication"]["seed"] == 8
    assert artifact["acceptance"]["observed_minimum_coverage"] >= 0.93
    assert artifact["acceptance"][
        "observed_maximum_known_invalid_false_pass_rate"
    ] <= 0.05
    assert artifact["acceptance"]["passed"] is True
    assert all("reported_minimum_detectable_margin" in row
               for row in artifact["profiles"].values())


def test_frozen_2000_trial_artifact_recomputes_exactly():
    path = Path(__file__).parents[1] / "artifacts" / "calibration" / \
        "v1-audit-profiles-2000.json"
    stored = json.loads(path.read_text())
    assert stored == run_profile_calibration(
        trials=2000, units=200, seed=20260901
    )
