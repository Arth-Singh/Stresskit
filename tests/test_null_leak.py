"""Null-score leak statistics: classes, polarity, CI, degenerate inputs."""

import pytest

from stresskit import metrics as M
from stresskit.null_leak import (
    CLASS_DEGRADED,
    CLASS_MATCHES,
    CLASS_PARTIAL,
    bootstrap_ci_difference,
    classify_leak,
    leak_from_runs,
    leak_from_summaries,
)


def two_point(center, half_width, n):
    """``n`` scores alternating ``center -/+ half_width``: mean exactly
    ``center``, population sd exactly ``half_width``."""
    return [center + (half_width if i % 2 else -half_width) for i in range(n)]


REAL_STRONG = two_point(1.0, 0.05, 20)
NULL_WEAK = two_point(0.1, 0.05, 20)


def from_summaries_of(real, null, **kw):
    return leak_from_summaries(
        M.mean(real), M.std(real), len(real), M.mean(null), M.std(null), len(null), **kw
    )


class TestDegradedNull:
    def test_class_ci_and_retention(self):
        stats = leak_from_runs(REAL_STRONG, NULL_WEAK, polarity=1, scale="ratio")
        assert stats["leak_class"] == CLASS_DEGRADED
        assert stats["retention"] == pytest.approx(0.1)
        assert stats["d"] == pytest.approx(18.0)
        assert stats["z"] > 1.96
        lo, hi = stats["ci_difference"]
        assert lo > 0
        assert lo <= 0.9 <= hi

    def test_signed_scale_needs_no_retention(self):
        stats = leak_from_runs(REAL_STRONG, NULL_WEAK, polarity=1, scale="signed")
        assert stats["retention"] is None
        assert stats["leak_class"] == CLASS_DEGRADED


class TestNullEqualsReal:
    def test_matches_and_ci_covers_zero(self):
        stats = leak_from_runs(
            REAL_STRONG, list(REAL_STRONG), polarity=1, scale="ratio"
        )
        assert stats["leak_class"] == CLASS_MATCHES
        assert stats["d"] == 0.0
        assert stats["retention"] == pytest.approx(1.0)
        lo, hi = stats["ci_difference"]
        assert lo <= 0.0 <= hi

    def test_null_exceeding_real_matches(self):
        stats = leak_from_runs(NULL_WEAK, REAL_STRONG, polarity=1, scale="ratio")
        assert stats["d"] < 0
        assert stats["retention"] == pytest.approx(10.0)
        assert stats["leak_class"] == CLASS_MATCHES


class TestPolarity:
    def test_minus_one_flips_the_sign(self):
        low_real, high_null = two_point(0.1, 0.05, 20), two_point(1.0, 0.05, 20)
        higher_is_better = leak_from_runs(
            low_real, high_null, polarity=1, scale="signed"
        )
        lower_is_better = leak_from_runs(
            low_real, high_null, polarity=-1, scale="signed"
        )
        assert higher_is_better["d"] == pytest.approx(-18.0)
        assert higher_is_better["leak_class"] == CLASS_MATCHES
        assert lower_is_better["d"] == pytest.approx(18.0)
        assert lower_is_better["leak_class"] == CLASS_DEGRADED
        assert lower_is_better["ci_difference"][0] > 0
        assert higher_is_better["ci_difference"][1] < 0

    def test_ratio_scale_rejects_lower_is_better(self):
        with pytest.raises(ValueError, match="scale='signed'"):
            leak_from_runs(REAL_STRONG, NULL_WEAK, polarity=-1, scale="ratio")

    def test_invalid_polarity_and_scale(self):
        with pytest.raises(ValueError, match="polarity"):
            leak_from_runs(REAL_STRONG, NULL_WEAK, polarity=2, scale="ratio")
        with pytest.raises(ValueError, match="scale"):
            leak_from_runs(REAL_STRONG, NULL_WEAK, polarity=1, scale="log")


class TestSummariesAgreeWithRuns:
    @pytest.mark.parametrize(
        "real, null, expected",
        [
            (REAL_STRONG, NULL_WEAK, CLASS_DEGRADED),
            (REAL_STRONG, list(REAL_STRONG), CLASS_MATCHES),
        ],
    )
    def test_same_class_and_statistics(self, real, null, expected):
        runs = leak_from_runs(real, null, polarity=1, scale="ratio")
        summaries = from_summaries_of(real, null, polarity=1, scale="ratio")
        assert runs["leak_class"] == summaries["leak_class"] == expected
        for key in ("d", "z", "retention", "pooled_sd", "difference"):
            assert runs[key] == pytest.approx(summaries[key])
        assert summaries["ci_difference"] is None
        assert runs["ci_difference"] is not None


class TestClassBoundaries:
    def test_partial_between_the_bars(self):
        stats = leak_from_summaries(
            1.0, 0.1, 20, 0.93, 0.1, 20, polarity=1, scale="signed"
        )
        assert stats["d"] == pytest.approx(0.7)
        assert stats["leak_class"] == CLASS_PARTIAL

    def test_ratio_retention_overrides_a_large_d(self):
        stats = leak_from_summaries(
            1.0, 0.01, 20, 0.95, 0.01, 20, polarity=1, scale="signed"
        )
        assert stats["leak_class"] == CLASS_DEGRADED
        stats = leak_from_summaries(
            1.0, 0.01, 20, 0.95, 0.01, 20, polarity=1, scale="ratio"
        )
        assert stats["retention"] == pytest.approx(0.95)
        assert stats["leak_class"] == CLASS_MATCHES

    def test_ratio_retention_above_half_is_partial_despite_d(self):
        stats = leak_from_summaries(
            1.0, 0.05, 20, 0.7, 0.05, 20, polarity=1, scale="ratio"
        )
        assert stats["d"] > 1.0 and stats["z"] > 1.96
        assert stats["leak_class"] == CLASS_PARTIAL

    def test_small_samples_need_significance(self):
        stats = leak_from_summaries(
            1.0, 0.5, 2, 0.4, 0.5, 2, polarity=1, scale="signed"
        )
        assert stats["d"] == pytest.approx(1.2)
        assert stats["z"] < 1.96
        assert stats["leak_class"] == CLASS_PARTIAL

    def test_classify_is_pure(self):
        stats = leak_from_summaries(
            1.0, 0.1, 20, 0.1, 0.1, 20, polarity=1, scale="ratio"
        )
        assert classify_leak(stats) == stats["leak_class"] == CLASS_DEGRADED


class TestBootstrapCI:
    def test_none_below_four_per_group(self):
        assert bootstrap_ci_difference([1, 2, 3], [1, 2, 3, 4]) is None
        assert bootstrap_ci_difference([1, 2, 3, 4], [1, 2, 3]) is None
        assert bootstrap_ci_difference([1, 2, 3, 4], [1, 2, 3, 4]) is not None

    def test_deterministic_in_seed(self):
        a, b = two_point(1.0, 0.2, 10), two_point(0.5, 0.2, 10)
        assert bootstrap_ci_difference(a, b, seed=3) == bootstrap_ci_difference(
            a, b, seed=3
        )
        assert bootstrap_ci_difference(a, b, seed=3) != bootstrap_ci_difference(
            a, b, seed=4
        )

    def test_interval_brackets_the_difference(self):
        lo, hi = bootstrap_ci_difference(REAL_STRONG, NULL_WEAK)
        assert lo <= 0.9 <= hi


class TestZeroSpread:
    def test_identical_constants_match(self):
        stats = leak_from_runs([1.0] * 5, [1.0] * 5, polarity=1, scale="ratio")
        assert stats["d"] is None and stats["z"] is None
        assert stats["leak_class"] == CLASS_MATCHES

    def test_constant_null_at_zero_is_degraded(self):
        stats = leak_from_runs([1.0] * 5, [0.0] * 5, polarity=1, scale="ratio")
        assert stats["retention"] == 0.0
        assert stats["leak_class"] == CLASS_DEGRADED

    def test_constant_null_between_bars_is_partial(self):
        stats = leak_from_runs([1.0] * 5, [0.7] * 5, polarity=1, scale="ratio")
        assert stats["leak_class"] == CLASS_PARTIAL

    def test_signed_scale_uses_the_sign(self):
        weaker = leak_from_runs([1.0] * 5, [0.7] * 5, polarity=1, scale="signed")
        stronger = leak_from_runs([1.0] * 5, [1.2] * 5, polarity=1, scale="signed")
        assert weaker["leak_class"] == CLASS_DEGRADED
        assert stronger["leak_class"] == CLASS_MATCHES


class TestInputValidation:
    def test_fewer_than_two_scores(self):
        with pytest.raises(ValueError, match=">= 2"):
            leak_from_runs([1.0], [0.0, 0.1], polarity=1, scale="ratio")
        with pytest.raises(ValueError, match=">= 2"):
            leak_from_summaries(1.0, 0.1, 5, 0.0, 0.1, 1, polarity=1, scale="ratio")

    def test_non_finite_score(self):
        with pytest.raises(ValueError, match="non-finite"):
            leak_from_runs([1.0, float("nan")], [0.0, 0.1], polarity=1, scale="ratio")

    def test_negative_sd(self):
        with pytest.raises(ValueError, match="standard deviation"):
            leak_from_summaries(1.0, -0.1, 5, 0.0, 0.1, 5, polarity=1, scale="ratio")
