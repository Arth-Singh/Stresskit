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

    def test_confirmatory_cv_accepts_declared_ratio_scale(self):
        result = M.score_variation_assessment(
            [8.0, 10.0, 12.0], scale_type="ratio", minimum_abs_mean=1.0
        )
        assert result["applicable"] is True
        assert result["value"] == pytest.approx(M.coefficient_of_variation([8, 10, 12]))

    @pytest.mark.parametrize(
        "values,scale_type,reason",
        [
            ([-1.0, 1.0], "signed", "ratio scale"),
            ([0.001, 0.002], "ratio", "minimum_abs_mean"),
            ([-1.0, 2.0], "ratio", "nonnegative"),
            ([float("nan"), 1.0], "ratio", "non-finite"),
        ],
    )
    def test_confirmatory_cv_rejects_unsupported_regimes(
        self, values, scale_type, reason
    ):
        result = M.score_variation_assessment(
            values, scale_type=scale_type, minimum_abs_mean=0.1
        )
        assert result["applicable"] is False
        assert result["value"] is None
        assert reason in result["reason"]


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

    def test_exact_formula_matches_exhaustive_small_universes(self):
        for universe_size in range(1, 7):
            universe = range(universe_size)
            for k in range(universe_size + 1):
                left = [frozenset(x) for x in itertools.combinations(universe, k)]
                for other_size in range(universe_size + 1):
                    right = [
                        frozenset(x)
                        for x in itertools.combinations(universe, other_size)
                    ]
                    brute = sum(M.jaccard(a, b) for a in left for b in right)
                    brute /= len(left) * len(right)
                    exact = M.exact_expected_random_jaccard(
                        k, universe_size, other_size
                    )
                    assert exact == pytest.approx(brute, abs=1e-14)

    def test_ratio_of_expectations_is_only_approximate(self):
        exact = M.exact_expected_random_jaccard(15, 144)
        approximate = M.expected_random_jaccard(15, 144)
        assert exact == pytest.approx(0.05663774525582543)
        assert approximate == pytest.approx(0.054945054945054944)
        assert exact != pytest.approx(approximate, rel=0.01)

    def test_exact_fraction_preserves_formal_value(self):
        exact = M.exact_expected_random_jaccard_fraction(15, 144)
        assert exact.numerator == 2584850149088656364382653
        assert exact.denominator == 45638295405532475996009088
        assert float(exact) == M.exact_expected_random_jaccard(15, 144)

    def test_exact_heterogeneous_size_distribution(self):
        sizes = [0, 1, 3, 5]
        got = M.exact_expected_random_jaccard_sizes(sizes, 5)
        expected = sum(
            M.exact_expected_random_jaccard(a, 5, b)
            for a, b in itertools.combinations(sizes, 2)
        ) / math.comb(len(sizes), 2)
        assert got == pytest.approx(expected)

    def test_exact_formula_rejects_impossible_subsets(self):
        with pytest.raises(ValueError, match="exceed"):
            M.exact_expected_random_jaccard(11, 10)
        with pytest.raises(ValueError, match="nonnegative"):
            M.exact_expected_random_jaccard(-1, 10)

    def test_core_noise_formula_matches_exhaustive_enumeration(self):
        for universe_size in range(1, 8):
            universe = set(range(universe_size))
            for core_size in range(universe_size + 1):
                core = frozenset(range(core_size))
                remainder = sorted(universe - core)
                for noise_size in range(len(remainder) + 1):
                    findings = [
                        core | frozenset(noise)
                        for noise in itertools.combinations(remainder, noise_size)
                    ]
                    brute = sum(
                        M.jaccard(a, b) for a in findings for b in findings
                    ) / len(findings) ** 2
                    exact = M.exact_expected_core_noise_jaccard(
                        core_size, noise_size, universe_size
                    )
                    assert exact == pytest.approx(brute, abs=1e-14)

    def test_core_noise_rejects_impossible_configuration(self):
        with pytest.raises(ValueError, match="fit"):
            M.exact_expected_core_noise_jaccard(8, 3, 10)


class TestPairwiseJackknife:
    def test_identical_sets_have_zero_width_interval(self):
        sets = [frozenset({1, 2, 3})] * 8
        assert M.jackknife_normal_ci_pairwise(sets, M.jaccard) == [1.0, 1.0]

    def test_matches_brute_force_delete_one_formula(self):
        sets = [
            frozenset({0, 1, 2}),
            frozenset({0, 1, 3}),
            frozenset({0, 4}),
            frozenset({5, 6}),
            frozenset({0, 1, 2, 3}),
            frozenset({1, 7}),
        ]
        point = M.mean_pairwise_jaccard(sets)
        leave_one = [
            M.mean_pairwise_jaccard(sets[:i] + sets[i + 1 :])
            for i in range(len(sets))
        ]
        pseudo = [
            len(sets) * point - (len(sets) - 1) * value
            for value in leave_one
        ]
        pseudo_mean = sum(pseudo) / len(pseudo)
        pseudo_variance = sum((x - pseudo_mean) ** 2 for x in pseudo)
        pseudo_variance /= len(pseudo) - 1
        standard_error = math.sqrt(pseudo_variance / len(pseudo))
        z = 1.9599639845400536
        expected = [
            max(0.0, point - z * standard_error),
            min(1.0, point + z * standard_error),
        ]
        assert M.jackknife_normal_ci_pairwise(sets, M.jaccard) == pytest.approx(
            expected
        )

    def test_degenerate_and_bounds_validation(self):
        assert M.jackknife_normal_ci_pairwise(["a", "b", "c"], lambda a, b: 0) is None
        with pytest.raises(ValueError, match="ordered"):
            M.jackknife_normal_ci_pairwise(
                [1, 2, 3, 4], lambda a, b: a - b, bounds=(1.0, 0.0)
            )

    def test_unbounded_interval_can_cross_metric_range(self):
        values = [0.0, 0.0, 0.0, 1.0]
        interval = M.jackknife_normal_ci_pairwise(
            values, lambda a, b: (a + b) / 2, bounds=None
        )
        assert interval[0] < 0.0 < interval[1]


class TestUnbiasedUVariance:
    def test_estimator_is_unbiased_on_exhaustive_binary_population(self):
        samples = list(itertools.product((0, 1), repeat=5))
        points = [
            sum(float(a == b) for a, b in itertools.combinations(sample, 2))
            / math.comb(len(sample), 2)
            for sample in samples
        ]
        target = sum(points) / len(points)
        actual_variance = sum((point - target) ** 2 for point in points) / len(points)
        estimated_variance = sum(
            M.unbiased_variance_u_pairwise(
                sample, lambda a, b: float(a == b)
            )
            for sample in samples
        ) / len(samples)
        assert estimated_variance == pytest.approx(actual_variance, abs=1e-15)

    def test_constant_kernel_has_zero_variance(self):
        values = [1, 2, 3, 4, 5, 6]
        assert M.unbiased_variance_u_pairwise(
            values, lambda a, b: 0.25
        ) == pytest.approx(0.0, abs=1e-15)

    def test_normal_interval_is_bounded_and_ordered(self):
        values = [0.0, 0.1, 0.4, 0.8, 1.0, 0.25, 0.6, 0.9]
        interval = M.u_normal_ci_pairwise(values, lambda a, b: (a + b) / 2)
        if interval is not None:
            assert 0.0 <= interval[0] <= interval[1] <= 1.0


class TestPairwiseBca:
    def test_identical_sets_have_exact_interval(self):
        sets = [frozenset({1, 2})] * 8
        assert M.bootstrap_bca_ci_pairwise(
            sets, M.jaccard, n_boot=100, seed=1
        ) == [1.0, 1.0]

    def test_seed_is_reproducible_and_interval_ordered(self):
        sets = [frozenset({0, i, i + 1}) for i in range(8)]
        first = M.bootstrap_bca_ci_pairwise(
            sets, M.jaccard, n_boot=200, seed=9
        )
        second = M.bootstrap_bca_ci_pairwise(
            sets, M.jaccard, n_boot=200, seed=9
        )
        assert first == second
        assert 0.0 <= first[0] <= first[1] <= 1.0

    def test_validation(self):
        assert M.bootstrap_bca_ci_pairwise([1, 2, 3], lambda a, b: 0.5) is None
        with pytest.raises(ValueError, match="n_boot"):
            M.bootstrap_bca_ci_pairwise(
                [1, 2, 3, 4], lambda a, b: 0.5, n_boot=10
            )


class TestPairedHoeffding:
    def test_estimator_and_interval_share_seeded_pairing(self):
        sets = [
            frozenset({0, i, i + 1})
            for i in range(8)
        ]
        point = M.paired_mean_pairwise(sets, M.jaccard, seed=17)
        interval = M.hoeffding_ci_pairwise(sets, M.jaccard, seed=17)
        assert interval[0] <= point <= interval[1]
        assert M.paired_mean_pairwise(sets, M.jaccard, seed=17) == point

    def test_finite_sample_half_width_matches_hoeffding_formula(self):
        values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        interval = M.hoeffding_ci_pairwise(
            values,
            lambda a, b: (a + b) / 2,
            seed=2,
            alpha=0.05,
        )
        point = M.paired_mean_pairwise(
            values, lambda a, b: (a + b) / 2, seed=2
        )
        half_width = math.sqrt(math.log(40.0) / (2 * 4))
        assert interval == pytest.approx(
            [max(0.0, point - half_width), min(1.0, point + half_width)]
        )

    def test_requires_pair_and_enforces_declared_bounds(self):
        assert M.hoeffding_ci_pairwise([1], lambda a, b: 0.5) is None
        with pytest.raises(ValueError, match="outside"):
            M.hoeffding_ci_pairwise([1, 2], lambda a, b: 2.0)
        with pytest.raises(ValueError, match="alpha"):
            M.hoeffding_ci_pairwise([1, 2], lambda a, b: 0.5, alpha=1.0)

    def test_pair_indices_are_reproducible_disjoint_and_drop_one(self):
        pairs = M.disjoint_pair_indices(9, seed=7)
        assert pairs == M.disjoint_pair_indices(9, seed=7)
        flat = [index for pair in pairs for index in pair]
        assert len(pairs) == 4
        assert len(flat) == len(set(flat)) == 8

    def test_generic_bounded_interval(self):
        interval = M.hoeffding_ci_bounded(
            [0.1, 0.2, 0.3, 0.4], alpha=0.05
        )
        assert interval[0] <= 0.25 <= interval[1]

    def test_difference_interval_uses_simultaneous_group_bounds(self):
        real = [frozenset({1, 2})] * 100
        null = [frozenset({i}) for i in range(100)]
        result = M.hoeffding_difference_pairwise(
            real, null, M.jaccard, real_seed=4, null_seed=5
        )
        assert result["estimate"] == 1.0
        assert result["ci"][0] > 0.0
        assert result["real_estimate"] == 1.0
        assert result["null_estimate"] == 0.0

    def test_cluster_interval_counts_independent_units(self):
        clusters = [
            [frozenset({cluster_id})] * 20
            for cluster_id in range(100)
        ]
        result = M.cluster_hoeffding_ci_pairwise(
            clusters, M.jaccard, seed=3
        )
        assert result["estimate"] == 0.0
        assert result["n_clusters"] == 100
        assert result["n_runs"] == 2000
        assert result["ci"][1] > 0.0

    def test_cluster_interval_rejects_empty_cluster(self):
        with pytest.raises(ValueError, match="nonempty"):
            M.cluster_hoeffding_ci_pairwise(
                [[frozenset({1})], []], M.jaccard
            )


class TestModalShareHoeffding:
    def test_stable_registered_labels_get_bounded_interval(self):
        labels = ["a"] * 100
        interval = M.modal_share_hoeffding_ci(
            labels, ["a", "b"], alpha=0.05
        )
        assert 0.8 < interval[0] <= interval[1] == 1.0

    def test_rejects_posthoc_or_invalid_class_space(self):
        with pytest.raises(ValueError, match="outside"):
            M.modal_share_hoeffding_ci(["x"], ["a", "b"])
        with pytest.raises(ValueError, match="unique"):
            M.modal_share_hoeffding_ci(["a"], ["a", "a"])


class TestNguyenConcentration:
    @staticmethod
    def _brute_variance_u(values):
        total = count = 0
        for indices in itertools.combinations(range(len(values)), 4):
            i, j, k, l = indices
            pairings = [
                ((i, j), (k, l)),
                ((i, k), (j, l)),
                ((i, l), (j, k)),
            ]
            for (a, b), (c, d) in pairings:
                left = (values[a] + values[b]) / 2
                right = (values[c] + values[d]) / 2
                total += (left - right) ** 2 / 2
                count += 1
        return total / count

    def test_fast_variance_u_matches_order_four_definition(self):
        values = [0.0, 0.1, 0.4, 0.8, 1.0, 0.25]
        pair_fn = lambda a, b: (a + b) / 2
        fast = M.pairwise_kernel_variance_u(values, pair_fn)
        assert fast == pytest.approx(self._brute_variance_u(values), abs=1e-15)

    def test_interval_contains_complete_u_statistic(self):
        sets = [frozenset({0, i, i + 1}) for i in range(8)]
        interval = M.nguyen_ci_pairwise(sets, M.jaccard)
        point = M.mean_pairwise_jaccard(sets)
        assert interval[0] <= point <= interval[1]
        assert 0.0 <= interval[0] <= interval[1] <= 1.0

    def test_constant_kernel_has_zero_variance_estimate(self):
        sets = [frozenset({1, 2})] * 10
        assert M.pairwise_kernel_variance_u(sets, M.jaccard) == pytest.approx(0.0)
        interval = M.nguyen_ci_pairwise(sets, M.jaccard)
        assert interval[1] == 1.0

    def test_validation(self):
        assert M.nguyen_ci_pairwise([1, 2, 3], lambda a, b: 0.5) is None
        with pytest.raises(ValueError, match="outside"):
            M.nguyen_ci_pairwise([1, 2, 3, 4], lambda a, b: -1.0)


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
