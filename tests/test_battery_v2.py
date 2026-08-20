"""Tests for the v0.2 engine features: semantic claims, specificity
null-control, bootstrap CIs, size-guarded structural pooling."""

import random
from collections import Counter

import pytest

import stresskit as sk
from stresskit import judges, metrics as M

N_UNIVERSE = 400


class TestSemanticClaims:
    def test_cluster_labels(self):
        labels = ["the word is tree", "The word is TREE!", "the word is moon"]
        ids = M.cluster_labels(labels, judges.normalized)
        assert ids[0] == ids[1] != ids[2]

    def test_paraphrased_claims_do_not_flip(self):
        phrasings = ["The secret word is tree",
                     "the secret word is 'tree'",
                     "The Secret Word is tree."]

        def finder(data, seed, config):
            rng = random.Random(seed)
            return sk.circuit(frozenset(range(6)),
                              claim=rng.choice(phrasings),
                              score=0.9, universe_size=N_UNIVERSE)

        strict = sk.stress(finder, list(range(20)), n_runs=8)
        semantic = sk.stress(finder, list(range(20)), n_runs=8,
                             claim_equiv=judges.normalized)
        assert strict.pooled["flip_rate"] > 0.3          # string equality panics
        assert semantic.pooled["flip_rate"] == 0.0       # judge sees one claim
        assert semantic.pooled["n_claim_classes"] == 1
        assert semantic.checks["claim_stability"]["passed"]


class TestSpecificity:
    @staticmethod
    def _data_driven_finder(data, seed, config):
        # finds the 8 most common values in the data (deterministic given data)
        counts = Counter(data)
        top = [v for v, _ in counts.most_common(8)]
        return sk.circuit(top, score=float(counts[top[0]]), universe_size=N_UNIVERSE)

    def test_specific_finder_passes(self):
        rng = random.Random(0)
        # real data: strong repeated structure + noise
        real = [v for v in range(8) for _ in range(30)] + \
               [rng.randrange(N_UNIVERSE) for _ in range(100)]
        # null data: uniform noise, no structure
        null = [rng.randrange(N_UNIVERSE) for _ in range(340)]
        result = sk.stress(self._data_driven_finder, real,
                           battery=["bootstrap"], n_runs=8, null_data=null)
        assert "specificity" in result.checks
        assert result.checks["specificity"]["passed"]
        assert result.null_summary["mean_pairwise_jaccard"] < \
            result.pooled["mean_pairwise_jaccard"]

    def test_artifact_finder_fails_specificity(self):
        # ignores the data entirely -> equally "stable" on null data
        def artifact(data, seed, config):
            return sk.circuit(frozenset(range(8)), score=1.0,
                              universe_size=N_UNIVERSE)

        rng = random.Random(1)
        real = [rng.randrange(N_UNIVERSE) for _ in range(100)]
        null = [rng.randrange(N_UNIVERSE) for _ in range(100)]
        result = sk.stress(artifact, real, battery=["bootstrap"],
                           n_runs=6, null_data=null)
        assert not result.checks["specificity"]["passed"]
        assert result.grade != "A"

    def test_null_control_on_card(self):
        def artifact(data, seed, config):
            return sk.circuit(frozenset(range(8)), score=1.0,
                              universe_size=N_UNIVERSE)

        result = sk.stress(artifact, list(range(30)), battery=["bootstrap"],
                           n_runs=4, null_data=list(range(30, 60)))
        assert "null_control" in result.card.metrics
        assert "null-control" in result.card.to_markdown()


class TestBootstrapCIs:
    def test_ci_present_and_ordered(self):
        def jitter(data, seed, config):
            rng = random.Random(seed)
            comps = set(range(10))
            if rng.random() < 0.5:
                comps.discard(rng.randrange(10))
            return sk.circuit(comps, claim=rng.choice(["a", "b"]),
                              score=0.5, universe_size=N_UNIVERSE)

        result = sk.stress(jitter, list(range(30)), n_runs=10)
        j_ci = result.pooled["mean_pairwise_jaccard_ci95"]
        f_ci = result.pooled["flip_rate_ci95"]
        assert j_ci is not None and j_ci[0] <= j_ci[1]
        assert f_ci is not None and f_ci[0] <= f_ci[1]
        j = result.pooled["mean_pairwise_jaccard"]
        assert j_ci[0] - 0.15 <= j <= j_ci[1] + 0.15

    def test_ci_none_for_tiny_runs(self):
        stable = lambda d, s, c: sk.circuit(frozenset({1, 2}), score=1.0,
                                            universe_size=N_UNIVERSE)
        result = sk.stress(stable, list(range(10)), battery=["seeds"], n_runs=2)
        assert result.pooled["mean_pairwise_jaccard_ci95"] is None


class TestSizeGuard:
    def test_size_mismatched_runs_excluded_from_grading(self):
        def finder(data, seed, config):
            k = config.get("k", 10)
            return sk.circuit(frozenset(range(k)), score=1.0,
                              universe_size=N_UNIVERSE)

        result = sk.stress(finder, list(range(20)),
                           battery=["seeds", "hyperparams"], n_runs=4,
                           config={"k": 10},
                           hyperparams={"k": [50]})  # 5x the base size
        # graded Jaccard ignores the k=50 run; unrestricted value is reported
        assert result.pooled["mean_pairwise_jaccard"] == 1.0
        assert result.pooled["mean_pairwise_jaccard_all_sizes"] < 1.0
        assert result.pooled["n_size_mismatched_excluded"] == 1
        assert any("size-comparable" in n for n in result.card.notes)
