import json
import random

import pytest

import stresskit as sk
from stresskit.battery import Thresholds, grade_checks
from stresskit.battery_calibration import (
    CHECKS,
    BatteryScenario,
    default_battery_scenarios,
    main,
    run_battery_cell,
)
from stresskit.calibration import StructuralScenario
from stresskit.metrics import exact_expected_random_jaccard


THRESHOLDS = Thresholds()
CELLS = {scenario.name: scenario for scenario in default_battery_scenarios()}

TRUTH_GRADES = {
    "A_control": ("A", "A"),
    "A_noisy": ("A", "A"),
    "J_pass_edge": ("A", "A"),
    "J_fail_edge": ("B", "B"),
    "J_two_mode": ("B", "B"),
    "claim_pass_edge": ("A", "A"),
    "claim_at_bar": ("A", "A"),
    "claim_fail_edge": ("B", "B"),
    "cv_pass_edge": ("A", "A"),
    "cv_fail_edge": ("B", "B"),
    "spec_same_structure": ("B", "C"),
    "spec_at_bar": ("B", "C"),
    "spec_ranker": ("B", "C"),
    "random_D": ("D", "D"),
    "mixed_C": ("C", "C"),
    "sizes_D": ("D", "D"),
}


def test_truths_match_direct_formulas():
    two_mode = CELLS["J_two_mode"]
    assert two_mode.truths()["structural_stability"] == (
        two_mode.structure.true_mean_jaccard()
    )
    assert two_mode.truths()["structural_stability"] == pytest.approx(
        0.7**2 + 0.3**2 + 2 * 0.7 * 0.3 * (5 / 35)
    )
    near_stable = CELLS["A_noisy"]
    assert near_stable.truths()["structural_stability"] == (
        near_stable.structure.true_mean_jaccard()
    )
    assert near_stable.truths()["structural_stability"] == pytest.approx(
        0.5 + 0.5 * (18 / 22)
    )
    assert near_stable.truths()["beats_random"] == pytest.approx(
        (10 / 11) / exact_expected_random_jaccard(20, 500)
    )
    assert CELLS["J_pass_edge"].truths()["structural_stability"] == pytest.approx(
        0.5 + 0.5 * (16 / 24)
    )
    assert CELLS["J_fail_edge"].truths()["structural_stability"] == pytest.approx(
        0.5 + 0.5 * (14 / 26)
    )
    assert CELLS["mixed_C"].truths()["structural_stability"] == pytest.approx(0.625)
    assert CELLS["claim_fail_edge"].truths()["claim_stability"] == pytest.approx(0.7)
    assert CELLS["mixed_C"].truths()["claim_stability"] == pytest.approx(0.5)
    assert CELLS["cv_fail_edge"].truths()["score_stability"] == 0.35
    assert CELLS["random_D"].truths()["beats_random"] == 1.0
    assert CELLS["sizes_D"].truths()["beats_random"] == 1.0
    spec_at_bar = CELLS["spec_at_bar"].truths()["specificity"]
    assert 1.5 - 1e-3 < spec_at_bar < 1.5
    assert CELLS["spec_ranker"].truths()["specificity"] == pytest.approx(12 / 11)
    assert CELLS["spec_same_structure"].truths()["specificity"] == pytest.approx(1.0)
    assert CELLS["mixed_C"].truths()["specificity"] == pytest.approx(1.0)


def test_truth_grade_table_for_all_sixteen_cells():
    assert list(CELLS) == list(TRUTH_GRADES)
    for name, scenario in CELLS.items():
        assert (
            scenario.truth_grade("v0.3", THRESHOLDS),
            scenario.truth_grade("v0.4", THRESHOLDS),
        ) == TRUTH_GRADES[name], name
    truth_checks = CELLS["spec_at_bar"].truth_checks(THRESHOLDS)
    assert set(truth_checks) == set(CHECKS)
    assert truth_checks["specificity"]["state"] == "fail"
    assert truth_checks["structural_stability"]["state"] == "pass"


def test_no_null_scenario_has_no_specificity_truth_and_caps_at_b():
    scenario = BatteryScenario("no_null", StructuralScenario("stable", "stable", 50, 5))
    assert scenario.truths()["specificity"] is None
    assert "specificity" not in scenario.truth_checks(THRESHOLDS)
    assert scenario.truth_grade("v0.3", THRESHOLDS) == "A"
    assert scenario.truth_grade("v0.4", THRESHOLDS) == "B"


def test_invalid_scenarios_are_rejected():
    structure = StructuralScenario("stable", "stable", 50, 5)
    with pytest.raises(ValueError, match="n_claim_classes"):
        BatteryScenario("bad", structure, n_claim_classes=1)
    with pytest.raises(ValueError, match="claim_modal_probability"):
        BatteryScenario("bad", structure, claim_modal_probability=1.5)
    with pytest.raises(ValueError, match="score_mean"):
        BatteryScenario("bad", structure, score_mean=0.0)
    with pytest.raises(ValueError, match="universe_size"):
        BatteryScenario(
            "bad", structure, null=StructuralScenario("uniform", "uniform", 60, 5)
        )
    with pytest.raises(ValueError, match="no null structure"):
        BatteryScenario("no_null", structure).draw(random.Random(0), 4, null=True)


def test_draw_produces_findings_with_planted_fields():
    scenario = CELLS["mixed_C"]
    findings = scenario.draw(random.Random(3), 40)
    assert len(findings) == 40
    assert all(f.size == 20 and f.universe_size == 500 for f in findings)
    assert {f.claim for f in findings} <= {"c0", "c1", "c2"}
    assert len({f.claim for f in findings}) == 3
    assert all(frozenset(range(8)) <= f.components for f in findings)
    assert {f.components for f in findings} == {
        frozenset(range(20)),
        frozenset(range(8)) | frozenset(range(20, 32)),
    }
    null = scenario.draw(random.Random(3), 40, null=True)
    assert {f.claim for f in null} == {"c0"}
    assert {f.score for f in null} == {1.0}


def test_a_control_small_n_is_graded_a_with_exact_structural_interval():
    result = run_battery_cell(CELLS["A_control"], 6, 5, master_seed=1)
    assert result.counts["trials_done"] == 5
    assert result.counts["errors"] == 0
    assert result.counts["grade_v04:A"] == 5
    assert result.counts["correct_v04"] == 5
    assert result.counts["grade_v03:A"] == 5
    structural = result.to_dict()["rates"]["checks"]["structural_stability"]
    assert structural["coverage"] == 1.0
    assert structural["mean_width"] == 0.0
    assert structural["bias"] == 0.0
    assert structural["false_fail_state_rate"] == 0.0
    assert structural["false_pass_state_rate"] is None


def test_result_grade_is_the_v04_rule_over_its_checks():
    scenario = CELLS["spec_at_bar"]
    rng = random.Random(11)
    real = scenario.draw(rng, 8)
    null = scenario.draw(rng, 8, null=True)
    result = sk.from_findings(real, null_findings=null, seed=5)
    assert result.grade == grade_checks(
        result.checks, rule="v0.4", random_floor=THRESHOLDS.random_floor
    )
    assert set(result.checks) == set(CHECKS)


def test_cell_is_reproducible_and_shardable():
    scenario = CELLS["claim_at_bar"]
    kwargs = dict(n_runs=6, master_seed=123)
    full = run_battery_cell(scenario, n_trials=20, **kwargs)
    repeated = run_battery_cell(scenario, n_trials=20, **kwargs)
    assert repeated == full

    left = run_battery_cell(scenario, n_trials=7, **kwargs)
    right = run_battery_cell(scenario, n_trials=13, trial_start=7, **kwargs)
    assert set(left.counts) == set(full.counts)
    for key, value in full.counts.items():
        assert left.counts[key] + right.counts[key] == value, key
    for key, value in full.sums.items():
        assert left.sums[key] + right.sums[key] == pytest.approx(value), key
    assert full.counts["trials_done"] == 20
    assert full.counts["claim_stability:applicable"] == 20


def test_rates_are_derived_from_counts():
    row = run_battery_cell(CELLS["random_D"], 6, 4, master_seed=2).to_dict()
    assert row["truth_grade_v03"] == "D"
    assert row["rates"]["grade"]["v04"]["truth"] == "D"
    grade_rates = row["rates"]["grade"]["v04"]
    assert sum(grade_rates["distribution"].values()) == pytest.approx(1.0)
    assert grade_rates["accuracy"] == row["counts"]["correct_v04"] / 4
    structural = row["rates"]["checks"]["structural_stability"]
    assert structural["false_pass_state_rate"] is not None
    assert structural["false_fail_state_rate"] is None
    assert sum(structural["state_rates"].values()) == pytest.approx(1.0)
    assert set(row["rates"]["checks"]) == set(CHECKS)


def test_run_battery_cell_rejects_bad_arguments():
    with pytest.raises(ValueError, match="n_runs"):
        run_battery_cell(CELLS["A_control"], 1, 1)
    with pytest.raises(ValueError, match="n_trials"):
        run_battery_cell(CELLS["A_control"], 6, 0)
    with pytest.raises(ValueError, match="trial_start"):
        run_battery_cell(CELLS["A_control"], 6, 1, trial_start=-1)


def test_cli_emits_machine_readable_payload(capsys):
    assert (
        main(
            [
                "--runs",
                "6,10",
                "--trials",
                "2",
                "--cells",
                "A_control,random_D",
                "--seed",
                "1",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == "0.1"
    assert payload["study"] == "battery_known_truth"
    assert len(payload["provenance"]["source_sha256"]) == 64
    assert payload["configuration"]["cells"] == ["A_control", "random_D"]
    assert payload["configuration"]["master_seed"] == 1
    assert payload["configuration"]["thresholds"]["jaccard"] == 0.8
    assert len(payload["results"]) == 4
    assert all(row["counts"]["trials_done"] == 2 for row in payload["results"])
    assert [row["n_runs"] for row in payload["results"]] == [6, 10, 6, 10]


def test_cli_time_one_prints_seconds_per_run_count(capsys):
    assert main(["--runs", "6,10", "--cells", "A_control", "--time-one"]) == 0
    timings = json.loads(capsys.readouterr().out)
    assert set(timings) == {"6", "10"}
    assert all(isinstance(value, float) and value >= 0.0 for value in timings.values())


def test_cli_lists_cells_and_rejects_unknown_ones(capsys):
    assert main(["--list-cells"]) == 0
    assert capsys.readouterr().out.split() == list(TRUTH_GRADES)
    with pytest.raises(SystemExit):
        main(["--cells", "not_a_cell", "--runs", "6", "--trials", "1"])
