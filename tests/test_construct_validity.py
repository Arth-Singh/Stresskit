"""Construct-validity regression table: the grade each degenerate finder
receives under both grade rules, both battery arms, with and without a null
control. Assertions pin relations between rows, not exact intervals."""

from __future__ import annotations

import pytest

from stresskit.construct_validity import (
    ARMS,
    FINDERS,
    POSITIVE_CONTROL,
    degenerate_matrix_markdown,
    run_degenerate_matrix,
    uncaught_rows,
)

ARM_NAMES = ("default", "seeds_only")
DATA_INDEPENDENT = ("constant", "index_ranker", "planted_leak", "fixed_direction")
SEED_IGNORING = DATA_INDEPENDENT + ("size_inflating",)
DATA_IGNORING = tuple(spec.name for spec in FINDERS if spec.ignores_data)
DATA_READING = tuple(spec.name for spec in FINDERS if not spec.ignores_data)


@pytest.fixture(scope="module")
def rows():
    return run_degenerate_matrix(seed=0)


def _row(rows, finder, arm, with_null):
    matches = [
        r
        for r in rows
        if r["finder"] == finder and r["arm"] == arm and r["with_null"] is with_null
    ]
    assert len(matches) == 1, (finder, arm, with_null)
    return matches[0]


def test_matrix_has_one_row_per_configuration(rows):
    assert len(rows) == len(FINDERS) * len(ARMS) * 2
    keys = {(r["finder"], r["arm"], r["with_null"]) for r in rows}
    assert len(keys) == len(rows)
    for r in rows:
        assert ("specificity" in r["checks"]) is r["with_null"]


@pytest.mark.parametrize("arm", ARM_NAMES)
@pytest.mark.parametrize("finder", DATA_INDEPENDENT)
def test_without_null_data_independent_finders_are_A_under_v03_and_B_under_v04(
    rows, finder, arm
):
    row = _row(rows, finder, arm, False)
    assert row["grade_v03"] == "A"
    assert row["grade_v04"] == "B"


@pytest.mark.parametrize("arm", ARM_NAMES)
@pytest.mark.parametrize("finder", SEED_IGNORING)
def test_with_null_data_independent_finders_fail_specificity(rows, finder, arm):
    row = _row(rows, finder, arm, True)
    assert row["checks"]["specificity"]["state"] == "fail"
    assert row["checks"]["specificity"]["value"] <= 1.0
    assert row["grade_v04"] in ("C", "D")
    assert row["grade_v03"] == "B"


@pytest.mark.parametrize("arm", ARM_NAMES)
@pytest.mark.parametrize("with_null", (False, True))
def test_size_inflating_fails_beats_random_by_a_hair(rows, arm, with_null):
    row = _row(rows, "size_inflating", arm, with_null)
    check = row["checks"]["beats_random"]
    assert check["state"] == "fail"
    assert 1.5 < check["value"] < 3.0


@pytest.mark.parametrize("with_null", (False, True))
@pytest.mark.parametrize("finder", ("random_subset", "random_direction"))
def test_random_finders_grade_D_under_seeds_only(rows, finder, with_null):
    row = _row(rows, finder, "seeds_only", with_null)
    assert row["checks"]["beats_random"]["value"] <= 1.5
    assert row["grade_v03"] == "D"
    assert row["grade_v04"] == "D"


@pytest.mark.parametrize("with_null", (False, True))
def test_random_subset_escapes_D_under_default_arm(rows, with_null):
    row = _row(rows, "random_subset", "default", with_null)
    assert row["notes_flags"]["vacuous_bootstrap"]
    assert row["checks"]["beats_random"]["state"] == "pass"
    assert row["checks"]["structural_stability"]["state"] == "fail"
    assert row["grade_v03"] == "C"
    assert row["grade_v04"] == "C"


def test_random_direction_ties_positive_control_under_default_arm_without_null(rows):
    row = _row(rows, "random_direction", "default", False)
    control = _row(rows, POSITIVE_CONTROL, "default", False)
    assert row["notes_flags"]["vacuous_bootstrap"]
    assert row["checks"]["structural_stability"]["state"] == "fail"
    assert row["grade_v04"] == control["grade_v04"] == "B"
    assert row["grade_v03"] == "B"


def test_random_direction_with_null_under_default_arm_grades_C(rows):
    row = _row(rows, "random_direction", "default", True)
    assert row["grade_v04"] == "C"
    assert row["grade_v03"] == "B"


@pytest.mark.parametrize("with_null", (False, True))
@pytest.mark.parametrize("finder", DATA_IGNORING)
def test_data_ignoring_finders_flag_vacuous_bootstrap_under_default_arm(
    rows, finder, with_null
):
    assert _row(rows, finder, "default", with_null)["notes_flags"]["vacuous_bootstrap"]


@pytest.mark.parametrize("with_null", (False, True))
@pytest.mark.parametrize("finder", DATA_READING)
def test_data_reading_finders_do_not_flag_vacuous_bootstrap(rows, finder, with_null):
    flags = _row(rows, finder, "default", with_null)["notes_flags"]
    assert not flags["vacuous_bootstrap"]


@pytest.mark.parametrize("arm", ARM_NAMES)
@pytest.mark.parametrize("finder", SEED_IGNORING)
def test_seed_ignoring_finders_flag_vacuous_seeds(rows, finder, arm):
    assert _row(rows, finder, arm, False)["notes_flags"]["vacuous_seeds"]


@pytest.mark.parametrize("arm", ARM_NAMES)
@pytest.mark.parametrize("finder", ("random_subset", "demo_on_noise"))
def test_seed_varying_set_finders_do_not_flag_vacuous_seeds(rows, finder, arm):
    assert not _row(rows, finder, arm, False)["notes_flags"]["vacuous_seeds"]


@pytest.mark.parametrize("arm", ARM_NAMES)
def test_random_direction_does_not_flag_vacuous_seeds(rows, arm):
    row = _row(rows, "random_direction", arm, False)
    assert row["checks"]["structural_stability"]["value"] < 0.8
    assert not row["notes_flags"]["vacuous_seeds"]


@pytest.mark.parametrize("arm", ARM_NAMES)
def test_positive_control_with_null_grades_A_with_clear_specificity(rows, arm):
    row = _row(rows, POSITIVE_CONTROL, arm, True)
    assert row["checks"]["specificity"]["state"] == "pass"
    assert row["checks"]["specificity"]["value"] > 3
    assert row["grade_v03"] == "A"
    assert row["grade_v04"] == "A"


@pytest.mark.parametrize("arm", ARM_NAMES)
def test_positive_control_without_null_is_capped_at_B_under_v04(rows, arm):
    row = _row(rows, POSITIVE_CONTROL, arm, False)
    assert row["grade_v03"] == "A"
    assert row["grade_v04"] == "B"


@pytest.mark.parametrize("with_null", (False, True))
@pytest.mark.parametrize("arm", ARM_NAMES)
def test_dead_salmon_never_grades_A(rows, arm, with_null):
    row = _row(rows, "demo_on_noise", arm, with_null)
    assert row["grade_v03"] != "A"
    assert row["grade_v04"] != "A"


def test_uncaught_under_v04_are_exactly_the_no_null_ties(rows):
    uncaught = uncaught_rows(rows, rule="v0.4")
    assert all(not u["with_null"] and u["grade"] == "B" for u in uncaught)
    assert {u["finder"] for u in uncaught} == {
        "constant",
        "index_ranker",
        "planted_leak",
        "size_inflating",
        "fixed_direction",
        "random_direction",
    }
    assert len(uncaught) == 11


def test_uncaught_under_v03_are_the_no_null_A_grades(rows):
    uncaught = uncaught_rows(rows, rule="v0.3")
    assert all(not u["with_null"] and u["grade"] == "A" for u in uncaught)
    assert {u["finder"] for u in uncaught} == set(DATA_INDEPENDENT)
    assert len(uncaught) == 8


def test_uncaught_rows_rejects_unknown_rule(rows):
    with pytest.raises(ValueError, match="v9.9"):
        uncaught_rows(rows, rule="v9.9")


def test_markdown_renders_one_row_per_configuration(rows):
    text = degenerate_matrix_markdown(rows)
    table_rows = [
        line
        for line in text.splitlines()
        if line.startswith("| ") and not line.startswith("| finder ")
    ]
    assert len(table_rows) == len(rows)
    for r in rows:
        assert f"| {r['finder']} | {r['arm']} |" in text
