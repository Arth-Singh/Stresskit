import json
from fractions import Fraction
from pathlib import Path

from stresskit import metrics as M
from stresskit.battery import decision_state


VECTORS_PATH = Path(__file__).parents[1] / "formal" / "golden" / "vectors.json"


def _fraction(row):
    return Fraction(row["numerator"], row["denominator"])


def test_python_jaccard_matches_exact_golden_vectors():
    vectors = json.loads(VECTORS_PATH.read_text())["jaccard"]
    for row in vectors:
        got = M.jaccard_fraction(
            frozenset(row["left"]), frozenset(row["right"])
        )
        assert got == _fraction(row)


def test_python_random_null_matches_exact_golden_vectors():
    vectors = json.loads(VECTORS_PATH.read_text())["random_jaccard"]
    for row in vectors:
        got = M.exact_expected_random_jaccard_fraction(
            row["left_size"], row["universe_size"], row["right_size"]
        )
        assert got == _fraction(row)


def test_python_decisions_match_golden_vectors():
    vectors = json.loads(VECTORS_PATH.read_text())["decisions"]
    for row in vectors:
        assert decision_state(
            row["value"],
            row["threshold"],
            row["op"],
            row["ci"],
            minimum_n_met=row["minimum_n_met"],
        ) == row["state"]
