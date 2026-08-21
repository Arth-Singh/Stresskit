"""Tests for zero-friction entry points: from_findings, sae_lens.stability,
and the eap adapter."""

import json
import random

import numpy as np
import pytest

import stresskit as sk
from stresskit.adapters import eap, sae_lens


# ---------------------------------------------------------------------------
# from_findings (post-hoc mode)
# ---------------------------------------------------------------------------

def make_findings(n=8, jitter=1, seed=0, claim="late"):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        comps = set(range(10)) - set(rng.sample(range(10), jitter))
        out.append(sk.feature_set(comps, claim=claim, score=0.9 + 0.01 * rng.random(),
                                  universe_size=144))
    return out


def test_from_findings_produces_graded_card():
    result = sk.from_findings(make_findings(), model="gpt2", task="ioi")
    assert result.grade in "ABCD"
    assert result.pooled["mean_pairwise_jaccard"] > 0.7
    assert any("post-hoc" in n for n in result.card.notes)
    md = result.card.to_markdown()
    assert "Stability Card" in md


def test_from_findings_axis_labels():
    findings = make_findings(7)
    result = sk.from_findings(findings, axes=["seeds"] * 4 + ["templates"] * 2)
    assert set(result.axis_metrics) == {"seeds", "templates"}


def test_from_findings_axes_length_mismatch():
    with pytest.raises(ValueError, match="axes has"):
        sk.from_findings(make_findings(4), axes=["seeds"])


def test_from_findings_rejects_non_findings():
    with pytest.raises(TypeError, match="not a stresskit.Finding"):
        sk.from_findings([{"edges": [1, 2]}, {"edges": [2, 3]}])


def test_from_findings_needs_two():
    with pytest.raises(ValueError, match=">= 2"):
        sk.from_findings(make_findings(1))


def test_from_findings_null_specificity():
    real = make_findings(8, jitter=1)
    null = make_findings(8, jitter=6, seed=99)       # unstable on null
    result = sk.from_findings(real, null_findings=null)
    assert "specificity" in result.checks
    assert result.checks["specificity"]["value"] > 1.0
    with pytest.raises(ValueError, match="null_findings"):
        sk.from_findings(real, null_findings=null[:1])


def test_from_findings_verifies():
    from stresskit.card import verify_card_dict
    out = verify_card_dict(sk.from_findings(make_findings()).card.to_dict())
    assert out["ok"], out["problems"]


# ---------------------------------------------------------------------------
# sae_lens.stability
# ---------------------------------------------------------------------------

def make_decoder(n=60, d=16, seed=0, noise=0.0):
    rng = np.random.default_rng(seed)
    base = np.random.default_rng(123).standard_normal((n, d)).astype(np.float32)
    return base + noise * rng.standard_normal((n, d)).astype(np.float32)


class FakeTensor:
    """Mimics a torch tensor's detach().cpu().float().numpy() chain."""
    def __init__(self, arr):
        self._a = arr
    def detach(self):
        return self
    def cpu(self):
        return self
    def float(self):
        return self
    def numpy(self):
        return self._a


class FakeSAE:
    def __init__(self, arr):
        self.W_dec = FakeTensor(arr)


def test_sae_stability_consistent_seeds_grade_a():
    saes = [FakeSAE(make_decoder(seed=s, noise=0.05)) for s in range(3)]
    report = sae_lens.stability(saes, name="toy-release")
    assert report.grade == "A"
    assert report.checks["seed_consistency"]["passed"]
    assert report.checks["above_noise_floor"]["passed"]
    assert report.metrics["mean_mcc"] > 0.9
    assert "SAE Stability Report" in report.to_markdown()
    assert report.badge_dict()["label"] == "SAE stability"


def test_sae_stability_random_decoders_fail():
    saes = [np.random.default_rng(s).standard_normal((60, 16)) for s in range(3)]
    report = sae_lens.stability(saes)
    assert not report.checks["seed_consistency"]["passed"]
    assert not report.checks["above_noise_floor"]["passed"]
    assert report.grade in "CD"


def test_sae_stability_single_sae_redundancy_only():
    W = np.vstack([make_decoder(30, 16), make_decoder(30, 16)])  # exact dupes
    report = sae_lens.stability(W)
    assert set(report.checks) == {"redundancy"}
    assert not report.checks["redundancy"]["passed"]
    assert any("single SAE" in n for n in report.notes)


def test_sae_stability_max_features_subsample():
    saes = [make_decoder(seed=s, noise=0.05) for s in range(2)]
    report = sae_lens.stability(saes, max_features=40, seed=7)
    assert report.metrics["n_features"] == 40
    assert any("subsample" in n for n in report.notes)


def test_sae_stability_shape_errors():
    with pytest.raises(ValueError, match="d_model"):
        sae_lens.stability([make_decoder(20, 16), make_decoder(20, 8)])
    with pytest.raises(ValueError, match="decoder"):
        sae_lens.stability(np.zeros(5))


# ---------------------------------------------------------------------------
# eap adapter
# ---------------------------------------------------------------------------

class FakeEdge:
    def __init__(self, score, in_graph):
        self.score = score
        self.in_graph = in_graph


class FakeGraph:
    def __init__(self, edges):
        self.edges = edges


def fake_graph():
    return FakeGraph({
        "input->a0.h1": FakeEdge(0.1, False),
        "a9.h6->a10.h7<q>": FakeEdge(0.9, True),
        "a10.h7->logits": FakeEdge(0.8, True),
        "m11->logits": FakeEdge(0.7, True),
        "m2->a5.h0<k>": FakeEdge(0.05, False),
    })


def test_edge_layer_parsing():
    assert eap.edge_layer("a9.h6->a10.h7<q>") == 9
    assert eap.edge_layer("m11->logits") == 11
    assert eap.edge_layer("input->a0.h1") is None


def test_graph_to_finding():
    f = eap.graph_to_finding(fake_graph(), score=0.85, n_layers=12)
    assert f.components == {"a9.h6->a10.h7<q>", "a10.h7->logits", "m11->logits"}
    assert f.universe_size == 5
    assert f.claim == "late"          # median layer 10 of 12
    assert f.score == 0.85


def test_finding_from_json(tmp_path):
    d = {
        "cfg": {"n_layers": 12},
        "nodes": {},
        "edges": {
            "a9.h6->a10.h7<q>": {"score": 0.9, "in_graph": True},
            "m11->logits": {"score": 0.7, "in_graph": True},
            "input->a0.h1": {"score": 0.1, "in_graph": False},
        },
    }
    p = tmp_path / "graph.json"
    p.write_text(json.dumps(d))
    f = eap.finding_from_json(str(p), score=0.9)
    assert f.components == {"a9.h6->a10.h7<q>", "m11->logits"}
    assert f.claim == "late"
    assert f.meta["source"] == str(p)
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"nodes": {}}))
    with pytest.raises(ValueError, match="edges"):
        eap.finding_from_json(str(bad))


def test_finder_from_graph_fn_end_to_end():
    def graph_fn(data, seed, config):
        g = fake_graph()
        if seed % 2:                                  # mild instability
            g.edges["m2->a5.h0<k>"].in_graph = True
        return g

    finder = eap.finder_from_graph_fn(
        graph_fn, score_fn=lambda g: 0.8, n_layers=12)
    result = sk.stress(finder, list(range(20)), battery=["seeds"], n_runs=6)
    assert result.grade in "ABCD"
    assert result.base.claim == "late"
