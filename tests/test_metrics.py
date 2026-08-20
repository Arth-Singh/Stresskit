import itertools
import math

import pytest

from stresskit import metrics as M


class TestJaccard:
    def test_identical(self):
        s = frozenset({1, 2, 3})
        assert M.jaccard(s, s) == 1.0

    def test_disjoint(self):
        assert M.jaccard(frozenset({1}), frozenset({2})) == 0.0

    def test_partial(self):
        a, b = frozenset({1, 2}), frozenset({2, 3})
        assert M.jaccard(a, b) == pytest.approx(1 / 3)

    def test_empty_pair_defined_as_one(self):
        assert M.jaccard(frozenset(), frozenset()) == 1.0

    def test_mean_pairwise_none_below_two(self):
        assert M.mean_pairwise_jaccard([frozenset({1})]) is None

    def test_pairwise_count(self):
        sets = [frozenset({i}) for i in range(5)]
        assert len(M.pairwise_jaccard(sets)) == math.comb(5, 2)


class TestFlipRate:
    def _brute_force(self, labels):
        pairs = list(itertools.combinations(labels, 2))
        return sum(1 for a, b in pairs if a != b) / len(pairs)

    def test_matches_brute_force(self):
        labels = ["a", "a", "b", "c", "a", "b"]
        assert M.flip_rate(labels) == pytest.approx(self._brute_force(labels))

    def test_all_same(self):
        assert M.flip_rate(["x"] * 10) == 0.0

    def test_all_distinct(self):
        assert M.flip_rate(["a", "b", "c"]) == 1.0

    def test_undefined_below_two(self):
        assert M.flip_rate(["a"]) is None
        assert M.flip_rate([]) is None

    def test_balanced_two_class_value(self):
        # unbiased estimator for n balanced labels over 2 classes:
        # F = 1 - (n/2)(n/2-1)*2 / (n(n-1)); slightly above the asymptotic 0.5
        labels = ["a", "b"] * 10
        assert M.flip_rate(labels) == pytest.approx(self._brute_force(labels))
        assert 0.5 < M.flip_rate(labels) < 0.55


class TestModalShareFilability:
    def test_modal_share(self):
        assert M.modal_share(["a", "a", "b", "a"]) == 0.75

    def test_filable(self):
        assert M.filable(["a"] * 9 + ["b"], alpha=0.2) is True
        assert M.filable(["a", "b", "c", "d"], alpha=0.2) is False


class TestScoreStats:
    def test_cv(self):
        xs = [10.0, 10.0, 10.0]
        assert M.coefficient_of_variation(xs) == 0.0

    def test_cv_undefined_near_zero_mean(self):
        assert M.coefficient_of_variation([-1.0, 1.0]) is None

    def test_cv_value(self):
        xs = [8.0, 12.0]  # mean 10, pop std 2
        assert M.coefficient_of_variation(xs) == pytest.approx(0.2)


class TestRandomNull:
    def test_analytic_formula(self):
        # k/(2N - k)
        assert M.expected_random_jaccard(200, 32491) == pytest.approx(
            200 / (2 * 32491 - 200)
        )

    def test_matches_empirical(self):
        from stresskit.baselines import random_jaccard_stats

        stats = random_jaccard_stats(k=50, universe_size=500, n=40, seed=1)
        assert stats["empirical_mean"] == pytest.approx(
            stats["analytic_expected"], rel=0.25
        )

    def test_degenerate(self):
        assert M.expected_random_jaccard(0, 100) is None
        assert M.expected_random_jaccard(10, 0) is None


class TestVarianceShares:
    def test_shares_sum_to_one(self):
        shares = M.variance_shares({"seeds": [1.0, 2.0, 3.0], "bootstrap": [1.0, 1.1]})
        assert sum(shares.values()) == pytest.approx(1.0)

    def test_dominant_axis(self):
        shares = M.variance_shares(
            {"noisy": [0.0, 10.0, 0.0, 10.0], "quiet": [5.0, 5.001]}
        )
        assert shares["noisy"] > 0.99

    def test_all_zero(self):
        shares = M.variance_shares({"a": [1.0, 1.0], "b": [2.0, 2.0]})
        assert shares == {"a": 0.0, "b": 0.0}
