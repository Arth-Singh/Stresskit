"""Tests for rank-biased overlap and the Jacobian-lens adapter."""

import random

import pytest

from stresskit.adapters import jlens
from stresskit.metrics import pairwise_rbo, rbo


# ---------------------------------------------------------------------------
# rbo
# ---------------------------------------------------------------------------

def rbo_reference(a, b, p):
    """Direct transcription of Webber et al. (2010) eq. 32."""
    if len(a) > len(b):
        a, b = b, a
    s, l = len(a), len(b)  # noqa: E741
    def X(d):
        return len(set(a[:min(d, s)]) & set(b[:d]))
    xs, xl = X(s), X(l)
    total = sum(
        (X(d) / d + (xs * (d - s) / (s * d) if d > s else 0)) * p ** (d - 1)
        for d in range(1, l + 1)
    )
    return (1 - p) * total + ((xl - xs) / l + xs / s) * p ** l


def test_rbo_matches_reference_on_random_lists():
    rng = random.Random(7)
    for _ in range(300):
        a = rng.sample(range(15), rng.randint(1, 10))
        b = rng.sample(range(15), rng.randint(1, 10))
        for p in (0.5, 0.9, 0.98):
            assert rbo(a, b, p) == pytest.approx(rbo_reference(a, b, p), abs=1e-12)


def test_rbo_identical_and_disjoint():
    assert rbo([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert rbo(list("abc"), list("xyz")) == 0.0
    assert rbo([], []) == 1.0
    assert rbo([1], []) == 0.0


def test_rbo_weights_the_head():
    base = [1, 2, 3, 4, 5]
    tail_swap = rbo(base, [1, 2, 3, 5, 4])
    head_swap = rbo(base, [2, 1, 3, 4, 5])
    assert head_swap < tail_swap


def test_rbo_rejects_duplicates_and_bad_p():
    with pytest.raises(ValueError, match="duplicate"):
        rbo([1, 1, 2], [1, 2, 3])
    with pytest.raises(ValueError, match="p must be"):
        rbo([1], [1], p=1.0)


def test_pairwise_rbo():
    assert pairwise_rbo([[1, 2]]) is None
    assert pairwise_rbo([[1, 2], [1, 2]]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# jlens adapter
# ---------------------------------------------------------------------------

def test_junk_share():
    assert jlens.junk_share([" moon", "Ġgold", "▁ship"]) == 0.0
    assert jlens.junk_share(["?!", " x", "123", " moon"]) == pytest.approx(0.75)
    with pytest.raises(ValueError):
        jlens.junk_share([])


def test_readout_finding_keeps_ranking():
    f = jlens.readout_finding(
        [" moon", " lunar", "?!", " tide"], k=3, universe_size=50000, score=0.8
    )
    assert f.components == frozenset({" moon", " lunar", "?!"})
    assert f.claim == " moon"
    assert f.meta["ranked"][-1] == " tide"
    assert f.meta["junk_share"] == pytest.approx(0.25)
    rbo_val = jlens.pairwise_readout_rbo(
        [f, jlens.readout_finding([" moon", " lunar", " tide", "?!"], k=3)]
    )
    assert 0.5 < rbo_val < 1.0


def test_min_rank_over_layers():
    ranked = {
        4: [" cat", " dog", " moon"],
        5: [" moon", " cat", " dog"],
        6: [" dog", " cat", " tree"],
    }
    assert jlens.min_rank(ranked, "moon") == 1
    assert jlens.min_rank(ranked, "moon", layers=[4, 6]) == 3
    assert jlens.min_rank(ranked, "sun") is None
    assert jlens.min_rank(ranked, " MOON ") == 1          # normalized
    assert jlens.min_rank(ranked, " MOON ", normalize=False) is None


def test_band_layers():
    assert jlens.band_layers(36, "mid-third") == list(range(12, 24))
    assert jlens.band_layers(36, "mid-half") == list(range(9, 27))
    assert len(jlens.band_layers(36, "all")) == 36
    with pytest.raises(ValueError):
        jlens.band_layers(36, "bottom")
