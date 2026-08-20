import random

import pytest

from stresskit import judges
from stresskit.oracle import (
    OracleProbe,
    OracleThresholds,
    blind_spot_matrix,
    stress_oracle,
)


# ---------------------------------------------------------------------------
# Fake oracles. Exemplars are dicts like {"concept": "tree"} standing in for
# activation handles; a real ask_fn would inject activations into an AO.
# ---------------------------------------------------------------------------

def reliable_oracle(exemplar, question, seed):
    concept = exemplar.get("concept")
    if concept is None:
        return "I don't know — there is no discernible concept here."
    return f"The secret word is {concept}."


def hallucinating_oracle(exemplar, question, seed):
    concept = exemplar.get("concept")
    if concept is None:
        return "The secret word is banana."   # confident nonsense on nulls
    return f"The secret word is {concept}."


def prompt_sensitive_oracle(exemplar, question, seed):
    concept = exemplar.get("concept", "nothing")
    if "hiding" in question:                   # only one phrasing works
        return f"It is hiding the word {concept}."
    return "Some vague description of text."


def inconsistent_oracle(exemplar, question, seed):
    rng = random.Random(seed)
    return rng.choice(["The word is tree.", "The word is moon.",
                       "This is about sports.", "A formal tone."])


PROBES = [
    OracleProbe(
        name="taboo-tree", concept="tree", expected="tree",
        questions=["What is the secret word?",
                   "Which word is the model hiding?"],
        exemplars=[{"concept": "tree"}, {"concept": "tree"}],
    ),
    OracleProbe(
        name="taboo-moon", concept="moon", expected="moon",
        questions=["What is the secret word?",
                   "Which word is the model hiding?"],
        exemplars=[{"concept": "moon"}, {"concept": "moon"}],
    ),
    OracleProbe(
        name="null-random", kind="null",
        questions=["What is the secret word?"],
        exemplars=[{}, {}],
    ),
]


class TestStressOracle:
    def test_reliable_oracle_grades_a(self):
        report = stress_oracle(reliable_oracle, PROBES, oracle_name="good")
        assert report.grade == "A"
        assert report.metrics["known_accuracy"] == 1.0
        assert report.metrics["null_hallucination_rate"] == 0.0
        assert report.checks["answer_consistency"]["passed"]

    def test_hallucinator_fails_null_check(self):
        report = stress_oracle(hallucinating_oracle, PROBES)
        assert not report.checks["null_hallucination"]["passed"]
        assert report.metrics["null_hallucination_rate"] == 1.0
        assert report.grade != "A"

    def test_prompt_sensitivity_detected(self):
        report = stress_oracle(prompt_sensitive_oracle, PROBES,
                               judge=judges.token_f1(0.5))
        assert report.metrics["prompt_spread"] == 1.0   # one phrasing works, one doesn't
        assert not report.checks["prompt_sensitivity"]["passed"]

    def test_inconsistent_oracle_low_consistency(self):
        report = stress_oracle(inconsistent_oracle, PROBES, n_repeats=3)
        assert report.metrics["answer_consistency"] < 0.6
        assert not report.checks["answer_consistency"]["passed"]

    def test_no_null_probes_noted(self):
        report = stress_oracle(reliable_oracle, PROBES[:2])
        assert any("null-control" in n for n in report.notes)

    def test_report_serializes(self, tmp_path):
        report = stress_oracle(reliable_oracle, PROBES)
        path = tmp_path / "oracle_report.json"
        report.save(str(path))
        import json
        d = json.loads(path.read_text())
        assert d["artifact"] == "stresskit_oracle_report"
        assert d["verdict"]["grade"] == "A"

    def test_markdown_and_badge(self):
        report = stress_oracle(reliable_oracle, PROBES)
        md = report.to_markdown()
        assert "Oracle Reliability Report" in md
        assert "taboo-tree" in md
        badge = report.badge_dict()
        assert badge["label"] == "oracle reliability"
        assert badge["color"] == "brightgreen"

    def test_probe_validation(self):
        with pytest.raises(ValueError, match="requires expected"):
            OracleProbe(name="x", questions=["q"], exemplars=[1], kind="known")
        with pytest.raises(ValueError, match="known|consistency|null"):
            OracleProbe(name="x", questions=["q"], exemplars=[1],
                        expected="e", kind="bogus")

    def test_nothing_to_grade(self):
        probe = OracleProbe(name="solo", kind="consistency",
                            questions=["q"], exemplars=[{"concept": "tree"}])
        with pytest.raises(ValueError, match="Nothing to grade"):
            stress_oracle(reliable_oracle, [probe])  # single answer, no pairs

    def test_custom_thresholds(self):
        lax = OracleThresholds(hallucination=1.0)
        report = stress_oracle(hallucinating_oracle, PROBES, thresholds=lax)
        assert report.checks["null_hallucination"]["passed"]


class TestBlindSpotMatrix:
    @staticmethod
    def make_blind_oracle(blind_concept):
        def ask(exemplar, question, seed):
            concept = exemplar.get("concept")
            if concept == blind_concept:
                return "I cannot tell what this represents."   # the blind spot
            return f"The secret word is {concept}."
        return ask

    def test_diagonal_blind_spot_flagged(self):
        oracles = {
            "ft-tree": self.make_blind_oracle("tree"),   # blind on its own concept
            "base": reliable_oracle,
            "ft-moon": self.make_blind_oracle("moon"),
        }
        out = blind_spot_matrix(oracles, PROBES)
        flagged = {(f["oracle"], f["concept"]) for f in out["blind_spots"]}
        assert ("ft-tree", "tree") in flagged
        assert ("ft-moon", "moon") in flagged
        assert not any(o == "base" for o, _ in flagged)
        assert out["accuracy"]["base"]["tree"] == 1.0
        assert out["accuracy"]["ft-tree"]["tree"] == 0.0

    def test_requires_concept_probes(self):
        with pytest.raises(ValueError, match="concept"):
            blind_spot_matrix({"o": reliable_oracle}, [PROBES[2]])
