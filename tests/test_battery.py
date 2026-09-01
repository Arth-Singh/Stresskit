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


class TestStressStable:
    def test_grade_a(self):
        result = sk.stress(stable_finder, DATA, n_runs=6, seed=0)
        assert result.grade == "A"
        assert result.pooled["mean_pairwise_jaccard"] == 1.0
        assert result.pooled["flip_rate"] == 0.0
        assert result.pooled["modal_share"] == 1.0

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
        assert lax.grade == "A"
