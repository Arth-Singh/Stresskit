import numpy as np
import pytest

from stresskit.adapters import sae
from stresskit.adapters.transformer_lens import (
    edges_to_finding,
    layer_band_claim,
    select_edges,
)


class TestSAEMatching:
    def test_identical_decoders_mcc_one(self):
        rng = np.random.default_rng(0)
        W = rng.normal(size=(64, 16))
        out = sae.match_features(W, W.copy())
        assert out["mcc"] == pytest.approx(1.0)

    def test_permuted_and_flipped_decoder_mcc_one(self):
        rng = np.random.default_rng(1)
        W = rng.normal(size=(50, 32))
        perm = rng.permutation(50)
        flipped = -W[perm]  # sign flips + permutation should not matter
        out = sae.match_features(W, flipped)
        assert out["mcc"] == pytest.approx(1.0)

    def test_random_decoders_low_mcc(self):
        rng = np.random.default_rng(2)
        A = rng.normal(size=(100, 128))
        B = rng.normal(size=(100, 128))
        out = sae.match_features(A, B)
        assert out["mcc"] < 0.5

    def test_seed_consistency(self):
        rng = np.random.default_rng(3)
        W = rng.normal(size=(40, 16))
        res = sae.seed_consistency([W, W.copy(), -W])
        assert res["mean_mcc"] == pytest.approx(1.0)
        assert res["n_runs"] == 3

    def test_dim_mismatch_raises(self):
        with pytest.raises(ValueError, match="d_model mismatch"):
            sae.match_features(np.ones((4, 8)), np.ones((4, 9)))


class TestRedundancyAudit:
    def test_finds_planted_duplicates(self):
        rng = np.random.default_rng(4)
        W = rng.normal(size=(30, 64))
        W[10] = W[3] * 2.0          # exact duplicate direction
        W[11] = -W[3]               # sign-flipped duplicate
        out = sae.redundancy_audit(W, threshold=0.95)
        assert out["n_redundant_features"] >= 3
        assert out["largest_cluster"] >= 3
        assert out["n_features"] == 30

    def test_clean_decoder_no_duplicates(self):
        # orthogonal rows -> no redundancy
        W = np.eye(20)
        out = sae.redundancy_audit(W, threshold=0.9)
        assert out["n_redundant_features"] == 0
        assert out["redundant_fraction"] == 0.0

    def test_batched_matches_unbatched(self):
        rng = np.random.default_rng(5)
        W = rng.normal(size=(64, 32))
        W[50] = W[7]
        a = sae.redundancy_audit(W, threshold=0.95, batch_size=7)
        b = sae.redundancy_audit(W, threshold=0.95, batch_size=1000)
        assert a["n_duplicate_pairs"] == b["n_duplicate_pairs"]
        assert a["n_redundant_features"] == b["n_redundant_features"]


class TestTopFeaturesFinding:
    def test_recovers_informative_features(self):
        rng = np.random.default_rng(6)
        n, d = 400, 50
        acts = rng.normal(size=(n, d))
        y = rng.random(n) > 0.5
        informative = [3, 17, 42]
        for f in informative:
            acts[y, f] += 3.0
        finding = sae.top_features_finding(acts, y, k=3)
        assert finding.components == frozenset(informative)
        assert finding.score > 1.0
        assert finding.universe_size == d

    def test_single_class_raises(self):
        acts = np.zeros((10, 5))
        with pytest.raises(ValueError, match="both classes"):
            sae.top_features_finding(acts, np.ones(10), k=2)


class TestTransformerLensAdapter:
    SCORES = {f"blocks.{layer}.edge{i}": (layer + 1) * 0.1 + i * 0.01
              for layer in range(12) for i in range(4)}

    def test_select_top_k(self):
        comps = select_edges(self.SCORES, top_k=5)
        assert len(comps) == 5

    def test_select_threshold(self):
        comps = select_edges(self.SCORES, threshold=1.0)
        assert all(self.SCORES[c] >= 1.0 for c in comps)

    def test_select_requires_exactly_one(self):
        with pytest.raises(ValueError):
            select_edges(self.SCORES, top_k=5, threshold=0.1)
        with pytest.raises(ValueError):
            select_edges(self.SCORES)

    def test_layer_band_claim(self):
        layer_of = lambda e: int(e.split(".")[1])
        early = [f"blocks.{l}.edge0" for l in (0, 1, 2, 3)]
        late = [f"blocks.{l}.edge0" for l in (9, 10, 11)]
        assert layer_band_claim(early, layer_of, 12) == "early"
        assert layer_band_claim(late, layer_of, 12) == "late"
        assert layer_band_claim([], layer_of, 12) == "empty"

    def test_edges_to_finding_auto_claim(self):
        finding = edges_to_finding(
            self.SCORES,
            top_k=6,
            layer_of=lambda e: int(e.split(".")[1]),
            n_layers=12,
            score=0.7,
        )
        assert finding.size == 6
        assert finding.claim == "late"  # top scores live in late layers
        assert finding.universe_size == len(self.SCORES)
