"""Tests for the run cache."""

import pytest

import stresskit as sk


def make_counting_finder(calls):
    def finder(data, seed, config):
        calls.append((seed, tuple(sorted(config.items()))))
        import random
        rng = random.Random(seed)
        picked = sorted(rng.sample(sorted(data), 6))
        return sk.feature_set(
            picked, claim="stable", score=1.0 + config.get("t", 0),
            universe_size=100,
        )
    return finder


DATA = list(range(40))


def test_cache_requires_key(tmp_path):
    with pytest.raises(ValueError, match="cache_key"):
        sk.stress(make_counting_finder([]), DATA, cache_dir=str(tmp_path))


def test_cache_roundtrip_identical_results(tmp_path):
    calls1, calls2 = [], []
    kwargs = dict(
        battery=["seeds", "bootstrap", "hyperparams"],
        n_runs=4,
        config={"t": 0},
        hyperparams={"t": [1, 2]},
        cache_dir=str(tmp_path),
        cache_key="v1",
    )
    r1 = sk.stress(make_counting_finder(calls1), DATA, **kwargs)
    r2 = sk.stress(make_counting_finder(calls2), DATA, **kwargs)

    assert len(calls1) > 0
    assert calls2 == []                     # every run served from cache
    assert r2.grade == r1.grade
    assert r2.pooled["mean_pairwise_jaccard"] == r1.pooled["mean_pairwise_jaccard"]
    assert [r.finding.components for r in r2.runs] == \
        [r.finding.components for r in r1.runs]
    assert any("restored from cache" in n for n in r2.card.notes)


def test_cache_key_busts_cache(tmp_path):
    calls1, calls2 = [], []
    common = dict(battery=["seeds"], n_runs=3, cache_dir=str(tmp_path))
    sk.stress(make_counting_finder(calls1), DATA, cache_key="v1", **common)
    sk.stress(make_counting_finder(calls2), DATA, cache_key="v2", **common)
    assert len(calls2) == len(calls1)       # nothing reused across keys


def test_cache_handles_tuple_components(tmp_path):
    def finder(data, seed, config):
        return sk.feature_set(
            [(0, 1), (2, 3)], claim="edges", score=0.5, universe_size=10
        )
    common = dict(battery=["seeds"], n_runs=2,
                  cache_dir=str(tmp_path), cache_key="v1")
    r1 = sk.stress(finder, DATA, **common)
    r2 = sk.stress(finder, DATA, **common)
    assert r2.base.components == frozenset({(0, 1), (2, 3)})
    assert r2.base.components == r1.base.components


def test_null_control_cached_separately(tmp_path):
    calls = []
    kwargs = dict(battery=["seeds"], n_runs=3,
                  null_data=list(range(100, 140)),
                  cache_dir=str(tmp_path), cache_key="v1")
    sk.stress(make_counting_finder(calls), DATA, **kwargs)
    n_first = len(calls)
    sk.stress(make_counting_finder(calls), DATA, **kwargs)
    assert len(calls) == n_first            # real AND null runs all cached
