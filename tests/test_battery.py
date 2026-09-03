import itertools
import random

import pytest

import stresskit as sk


N_UNIVERSE = 500
TRUE_COMPONENTS = frozenset(range(10))


def stable_finder(data, seed, config):
    """Always finds the true components; score has tiny jitter."""
    rng = random.Random(seed)
    return sk.circuit(
        TRUE_COMPONENTS,
        claim="early",
        score=0.9 + rng.uniform(-0.005, 0.005),
        universe_size=N_UNIVERSE,
    )


_call_counter = itertools.count()


def unstable_finder(data, seed, config):
    """Returns a fresh (but deterministic) random subset every call — noise."""
    rng = random.Random(next(_call_counter))
    comps = frozenset(rng.sample(range(N_UNIVERSE), 10))
    return sk.circuit(
        comps,
        claim=rng.choice(["early", "middle", "late"]),
        score=rng.uniform(0.1, 0.9),
        universe_size=N_UNIVERSE,
    )


DATA = list(range(50))


def counting_finder(data, seed, config):
    """Reads its data: the 10 most frequent values in a seeded 75% subsample,
    ties broken at random. On REAL that is always the true set; on NULL it
    is a fresh random subset of the sampled values every run."""
    rng = random.Random(seed)
    sample = rng.sample(list(data), max(4, int(0.75 * len(data))))
    counts = {}
    for value in sample:
        counts[value] = counts.get(value, 0) + 1
    top = sorted(counts, key=lambda v: (-counts[v], rng.random()))[:10]
    jitter = config.get("jitter", 0.005)
    return sk.circuit(
        top,
        claim="early",
        score=1.0 + rng.uniform(-jitter, jitter),
        universe_size=N_UNIVERSE,
    )


REAL = [c for c in sorted(TRUE_COMPONENTS) for _ in range(5)]
NULL = random.Random(7).choices(range(N_UNIVERSE), k=50)


class TestStressStable:
    def test_grade_capped_at_b_without_null_control(self):
        result = sk.stress(stable_finder, DATA, n_runs=6, seed=0)
        assert result.grade == "B"
        assert "specificity" not in result.checks
        assert all(c["state"] == "pass" for c in result.checks.values())
        assert result.pooled["mean_pairwise_jaccard"] == 1.0
        assert result.pooled["flip_rate"] == 0.0
        assert result.pooled["modal_share"] == 1.0

    def test_grade_a_needs_a_decided_specificity_pass(self):
        result = sk.stress(counting_finder, REAL, n_runs=6, seed=0,
                           null_data=NULL)
        assert result.grade == "A"
        assert result.checks["specificity"]["state"] == "pass"
        assert result.checks["specificity"]["value"] > 3.0
        assert result.pooled["confidence"] == "high"

    def test_data_ignoring_finder_cannot_grade_above_c_with_a_null(self):
        result = sk.stress(stable_finder, DATA, n_runs=6, seed=0,
                           null_data=list(range(50, 100)))
        assert result.checks["specificity"]["state"] == "fail"
        assert result.grade == "C"
        assert any("bootstrap axis" in n for n in result.card.notes)

    def test_beats_random_check_present(self):
        result = sk.stress(stable_finder, DATA, n_runs=4, seed=0)
        assert result.checks["beats_random"]["passed"]
        assert result.pooled["jaccard_vs_random"] > 10


class TestStressUnstable:
    def test_grade_d(self):
        result = sk.stress(unstable_finder, DATA, n_runs=10, seed=0)
        assert result.grade == "D"
        assert result.pooled["mean_pairwise_jaccard"] < 0.2
        assert result.pooled["flip_rate"] > 0.3

    def test_at_random_null(self):
        result = sk.stress(unstable_finder, DATA, n_runs=10, seed=0)
        assert result.pooled["jaccard_vs_random"] < 1.5


class TestAxes:
    def test_axis_run_counts(self):
        result = sk.stress(
            stable_finder,
            DATA,
            battery=["seeds", "bootstrap", "templates", "hyperparams"],
            n_runs=3,
            templates={"t1": DATA, "t2": DATA},
            hyperparams={"threshold": [0.1, 0.2]},
            config={"threshold": 0.05},
        )
        by_axis = {}
        for r in result.runs:
            by_axis[r.axis] = by_axis.get(r.axis, 0) + 1
        assert by_axis == {
            "base": 1,
            "seeds": 3,
            "bootstrap": 3,
            "templates": 2,
            "hyperparams": 2,
        }

    def test_hyperparam_equal_to_base_skipped(self):
        result = sk.stress(
            stable_finder,
            DATA,
            battery=["hyperparams"],
            hyperparams={"threshold": [0.05, 0.1]},
            config={"threshold": 0.05},
        )
        hp_runs = [r for r in result.runs if r.axis == "hyperparams"]
        assert len(hp_runs) == 1
        assert hp_runs[0].config["threshold"] == 0.1

    def test_missing_templates_noted_not_fatal(self):
        result = sk.stress(stable_finder, DATA, battery=["seeds", "templates"], n_runs=2)
        assert any("templates" in n for n in result.card.notes)

    def test_bootstrap_needs_sized_data(self):
        result = sk.stress(stable_finder, None, battery=["seeds", "bootstrap"], n_runs=2)
        assert any("bootstrap" in n for n in result.card.notes)

    def test_unknown_axis_raises(self):
        with pytest.raises(ValueError, match="Unknown battery axis"):
            sk.stress(stable_finder, DATA, battery=["seeds", "cosmic_rays"])


class TestVacuousSeedDetection:
    def test_deterministic_finder_flagged(self):
        # ignores its seed entirely -> identical finding every seed run
        det = lambda d, s, c: sk.circuit(TRUE_COMPONENTS, claim="early",
                                         score=0.9, universe_size=N_UNIVERSE)
        result = sk.stress(det, DATA, battery=["seeds"], n_runs=4)
        assert any("may ignore its seed" in n for n in result.card.notes)

    def test_seed_using_finder_not_flagged(self):
        result = sk.stress(stable_finder, DATA, battery=["seeds"], n_runs=4)
        assert not any("may ignore its seed" in n for n in result.card.notes)


class TestValidation:
    def test_wrong_return_type(self):
        with pytest.raises(TypeError, match="must return a stresskit.Finding"):
            sk.stress(lambda d, s, c: {"edges": [1]}, DATA, n_runs=1)

    def test_nothing_to_grade(self):
        empty = lambda d, s, c: sk.Finding()
        with pytest.raises(ValueError, match="Nothing to grade"):
            sk.stress(empty, DATA, n_runs=2)

    def test_explicit_empty_circuit_is_structural_and_reported(self):
        empty = lambda d, s, c: sk.circuit([], universe_size=100)
        result = sk.stress(empty, DATA, battery=["seeds"], n_runs=4)
        assert result.pooled["mean_pairwise_jaccard"] == 1.0
        assert result.pooled["empty_finding_rate"] == 1.0
        assert result.pooled["expected_random_jaccard"] == 1.0
        assert result.checks["beats_random"]["passed"] is False
        assert result.grade == "D"
        assert sk.verify_card_dict(result.card.to_dict())["ok"]


class TestThresholds:
    def test_custom_thresholds_change_verdict(self):
        # jitter finder: same components, mildly noisy score
        def jitter(data, seed, config):
            rng = random.Random(seed)
            return sk.circuit(
                TRUE_COMPONENTS, claim="early",
                score=1.0 + rng.uniform(-0.4, 0.4), universe_size=N_UNIVERSE,
            )

        strict = sk.stress(jitter, DATA, n_runs=8,
                           thresholds=sk.Thresholds(score_cv=0.01))
        lax = sk.stress(jitter, DATA, n_runs=8,
                        thresholds=sk.Thresholds(score_cv=5.0))
        assert not strict.checks["score_stability"]["passed"]
        assert lax.checks["score_stability"]["passed"]
        # no null control: both batteries are capped at B, so the letter
        # moves only where the null is present
        assert strict.grade == "B" and lax.grade == "B"
        strict = sk.stress(counting_finder, REAL, n_runs=8, null_data=NULL,
                           config={"jitter": 0.4},
                           thresholds=sk.Thresholds(score_cv=0.01))
        lax = sk.stress(counting_finder, REAL, n_runs=8, null_data=NULL,
                        config={"jitter": 0.4},
                        thresholds=sk.Thresholds(score_cv=5.0))
        assert strict.checks["score_stability"]["state"] == "fail"
        assert strict.grade == "B"
        assert lax.grade == "A"

    def test_random_floor_is_a_threshold(self):
        result = sk.stress(counting_finder, REAL, n_runs=6, null_data=NULL,
                           thresholds=sk.Thresholds(random_floor=1e6))
        assert result.grade == "D"
        assert result.card.verdict["thresholds"]["random_floor"] == 1e6
