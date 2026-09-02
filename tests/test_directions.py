"""Direction-valued findings: the |cosine| structural family end to end.

Mirrors the set-valued tests (test_metrics / test_battery / test_hardening)
for the direction path: exact small cases computed by hand, the analytic
random null against its Monte-Carlo form, sign invariance, the two fail-fast
errors (mixed kinds, mixed dimensions), and a card that survives a round trip
through save/load/verify but not through tampering.
"""

import json
import math
import os
import random

import pytest

import stresskit as sk
from stresskit import baselines as B
from stresskit import metrics as M
from stresskit.card import (
    SCHEMA_VERSION, StabilityCard, verify_card_dict, _vector_digest,
)

ROOT2 = math.sqrt(2.0) / 2.0
ROOT3_2 = math.sqrt(3.0) / 2.0

# Four unit directions in R^2 at 0, 60, 90 and 120 degrees. |cos| between two
# lines at angles a and b is |cos(a - b)|, so every pair is exact by hand:
#   (0,60) 0.5   (0,90) 0   (0,120) 0.5
#   (60,90) √3/2 (60,120) 0.5   (90,120) √3/2
PLANE = [
    (1.0, 0.0),
    (0.5, ROOT3_2),
    (0.0, 1.0),
    (-0.5, ROOT3_2),
]
PLANE_PAIRS = [0.5, 0.0, 0.5, ROOT3_2, 0.5, ROOT3_2]
PLANE_MEAN = sum(PLANE_PAIRS) / len(PLANE_PAIRS)


def clustered(n, dim=64, spread=0.12, seed=0, flip_every=0):
    """n directions scattered around one axis; every other one optionally
    sign-flipped so tests can prove the sign convention does not matter."""
    rng = random.Random(seed)
    core = [rng.gauss(0.0, 1.0) for _ in range(dim)]
    out = []
    for i in range(n):
        jitter = random.Random(seed * 1000 + i)
        v = [c + jitter.gauss(0.0, spread) for c in core]
        if flip_every and i % flip_every == 0:
            v = [-x for x in v]
        out.append(v)
    return out


# ---------------------------------------------------------------------------
# the constructor
# ---------------------------------------------------------------------------

class TestDirectionConstructor:
    def test_normalizes_on_construction(self):
        f = sk.direction([3.0, 4.0])
        assert f.vector == (0.6, 0.8)
        assert math.isclose(sum(x * x for x in f.vector), 1.0)

    def test_carries_claim_score_and_meta(self):
        f = sk.direction([1, 0], claim="mid-layer", score=0.9, layer=13)
        assert (f.claim, f.score, f.meta) == ("mid-layer", 0.9, {"layer": 13})

    def test_is_direction_kind_not_set_kind(self):
        f = sk.direction([1, 0])
        assert f.kind == "direction"
        assert f.dim == 2
        assert f.has_direction() is True
        # a direction has no component set, so the Jaccard path must not see
        # it as an empty set (J(∅,∅) = 1 would fake perfect stability)
        assert f.has_structure() is False
        assert f.size == 0

    def test_set_findings_are_unchanged(self):
        c = sk.circuit([(9, 6), (10, 7)])
        assert c.kind == "set" and c.dim is None and c.has_direction() is False
        assert sk.probe(0.7).kind == "none"

    @pytest.mark.parametrize("bad", [[], [1.0, float("nan")],
                                     [float("inf"), 0.0], [0.0, 0.0, 0.0]])
    def test_rejects_undirected_vectors(self, bad):
        with pytest.raises(ValueError):
            sk.direction(bad)

    def test_rejects_components_and_vector_together(self):
        with pytest.raises(ValueError, match="either components"):
            sk.Finding(components=[1, 2], vector=[1.0, 0.0])


# ---------------------------------------------------------------------------
# the metric
# ---------------------------------------------------------------------------

class TestCosineMetrics:
    def test_known_pairs(self):
        assert M.cosine_similarity([1, 0], [1, 0]) == 1.0
        assert M.cosine_similarity([1, 0], [-1, 0]) == -1.0
        assert M.cosine_similarity([1, 0], [0, 1]) == 0.0
        assert math.isclose(M.cosine_similarity([1, 0], [1, 1]), ROOT2)

    def test_scale_invariance(self):
        assert math.isclose(M.cosine_similarity([1, 2, 3], [2, 4, 6]), 1.0)
        assert math.isclose(M.abs_cosine([1, 2, 3], [-7, -14, -21]), 1.0)

    def test_abs_cosine_ignores_the_sign_convention(self):
        a, b = [1.0, 2.0, -1.0], [0.5, 2.5, 1.0]
        signed = M.cosine_similarity(a, b)
        assert M.abs_cosine(a, b) == abs(signed)
        assert M.abs_cosine([-x for x in a], b) == abs(signed)
        assert M.abs_cosine(a, [-x for x in b]) == abs(signed)

    def test_dimension_mismatch_is_an_error(self):
        with pytest.raises(ValueError, match="equal dimensions"):
            M.cosine_similarity([1, 0], [1, 0, 0])

    def test_zero_vector_is_an_error(self):
        with pytest.raises(ValueError, match="zero vector"):
            M.cosine_similarity([0, 0], [1, 0])

    def test_pairwise_matches_hand_computation(self):
        got = M.pairwise_abs_cosine(PLANE)
        assert len(got) == 6
        for g, want in zip(got, PLANE_PAIRS):
            assert math.isclose(g, want, abs_tol=1e-12)
        assert math.isclose(M.mean_pairwise_abs_cosine(PLANE), PLANE_MEAN)

    def test_mean_needs_two_vectors(self):
        assert M.mean_pairwise_abs_cosine([]) is None
        assert M.mean_pairwise_abs_cosine([(1.0, 0.0)]) is None

    def test_identical_directions_score_one(self):
        v = [0.3, -0.4, 0.5]
        assert math.isclose(M.mean_pairwise_abs_cosine([v, v, v]), 1.0)

    def test_mean_is_invariant_to_any_pattern_of_sign_flips(self):
        base = clustered(6, dim=16, seed=3)
        want = M.mean_pairwise_abs_cosine(base)
        for mask in (0b000001, 0b010101, 0b111111, 0b101010):
            flipped = [[-x for x in v] if (mask >> i) & 1 else v
                       for i, v in enumerate(base)]
            assert math.isclose(M.mean_pairwise_abs_cosine(flipped), want)


# ---------------------------------------------------------------------------
# the random null
# ---------------------------------------------------------------------------

class TestRandomDirectionNull:
    def test_analytic_closed_forms(self):
        # R^2: the angle is uniform, so E|cos| = 2/π.
        assert math.isclose(M.expected_random_abs_cosine(2), 2 / math.pi)
        # R^3: the cosine is uniform on [-1, 1], so E|cos| = 1/2.
        assert math.isclose(M.expected_random_abs_cosine(3), 0.5)
        # R^1: every unit vector is ±1.
        assert M.expected_random_abs_cosine(1) == 1.0
        assert M.expected_random_abs_cosine(0) is None

    @pytest.mark.parametrize("dim", [8, 64, 512])
    def test_analytic_approaches_sqrt_two_over_pi_d(self, dim):
        exact = M.expected_random_abs_cosine(dim)
        asymptotic = math.sqrt(2.0 / (math.pi * dim))
        assert abs(exact - asymptotic) / asymptotic < 4.0 / dim

    @pytest.mark.parametrize("dim", [3, 16, 128])
    def test_monte_carlo_matches_the_analytic_value(self, dim):
        mc = B.empirical_random_abs_cosine(8, dim, seed=0, repeats=300)
        exact = M.expected_random_abs_cosine(dim)
        assert abs(mc - exact) < 0.1 * exact

    def test_monte_carlo_is_seeded(self):
        a = B.empirical_random_abs_cosine(6, 32, seed=7, repeats=40)
        b = B.empirical_random_abs_cosine(6, 32, seed=7, repeats=40)
        c = B.empirical_random_abs_cosine(6, 32, seed=8, repeats=40)
        assert a == b and a != c

    def test_needs_two_vectors(self):
        assert B.empirical_random_abs_cosine(1, 8) is None
        with pytest.raises(ValueError):
            B.empirical_random_abs_cosine(4, 0)

    def test_random_directions_are_unit_direction_findings(self):
        fs = B.random_directions(5, 9, seed=2)
        assert [f.kind for f in fs] == ["direction"] * 5
        for f in fs:
            assert math.isclose(sum(x * x for x in f.vector), 1.0)
        assert [f.vector for f in fs] == [
            f.vector for f in B.random_directions(5, 9, seed=2)
        ]


# ---------------------------------------------------------------------------
# the battery
# ---------------------------------------------------------------------------

class TestDirectionBattery:
    def test_pooled_metrics_match_hand_computation(self):
        res = sk.from_findings([sk.direction(v) for v in PLANE])
        pooled = res.pooled
        assert pooled["n_direction_runs"] == 4
        assert pooled["direction_dim"] == 2
        assert math.isclose(pooled["mean_pairwise_abs_cosine"], PLANE_MEAN)
        assert pooled["min_pairwise_abs_cosine"] == 0.0
        assert math.isclose(pooled["expected_random_abs_cosine"], 2 / math.pi)
        assert math.isclose(
            pooled["abs_cosine_vs_random"], PLANE_MEAN / (2 / math.pi))
        # no set-valued structure was produced, so the Jaccard family stays empty
        assert pooled["mean_pairwise_jaccard"] is None
        assert pooled["n_structured_runs"] == 0

    def test_scattered_directions_are_graded_at_the_cosine_bar(self):
        res = sk.from_findings([sk.direction(v) for v in PLANE])
        check = res.checks["structural_stability"]
        assert check["threshold"] == sk.Thresholds().cosine == 0.8
        assert check["op"] == ">="
        assert "|cosine|" in check["description"]
        assert check["passed"] is False
        # at random in R^2 the battery cannot be better than D
        assert res.checks["beats_random"]["passed"] is False
        assert res.grade == "D"

    def test_clustered_directions_pass_and_beat_random(self):
        findings = [sk.direction(v, claim="one direction", score=0.9)
                    for v in clustered(10, dim=64, spread=0.1, seed=1)]
        res = sk.from_findings(findings)
        assert res.checks["structural_stability"]["passed"] is True
        assert res.checks["beats_random"]["value"] > 3.0
        assert res.pooled["mean_pairwise_abs_cosine_ci95"] is not None
        assert res.grade == "A"

    def test_grade_is_invariant_to_the_sign_convention(self):
        plain = clustered(10, dim=64, spread=0.1, seed=1)
        flipped = clustered(10, dim=64, spread=0.1, seed=1, flip_every=2)
        a = sk.from_findings([sk.direction(v) for v in plain])
        b = sk.from_findings([sk.direction(v) for v in flipped])
        assert math.isclose(a.pooled["mean_pairwise_abs_cosine"],
                            b.pooled["mean_pairwise_abs_cosine"])
        assert a.grade == b.grade
        assert (a.pooled["mean_pairwise_abs_cosine_ci95"]
                == b.pooled["mean_pairwise_abs_cosine_ci95"])

    def test_specificity_against_a_direction_null_control(self):
        real = [sk.direction(v, claim="one direction", score=0.9)
                for v in clustered(8, dim=64, spread=0.08, seed=4)]
        nulls = [sk.direction(f.vector, claim="none", score=0.1)
                 for f in B.random_directions(8, 64, seed=99)]
        res = sk.from_findings(real, null_findings=nulls)
        spec = res.checks["specificity"]
        assert spec["value"] > sk.Thresholds().specificity_ratio
        assert spec["passed"] is True
        assert res.null_summary["mean_pairwise_abs_cosine"] < 0.4
        assert res.pooled["specificity_ci95"] is not None

    def test_per_axis_breakdown_reports_cosine(self):
        findings = [sk.direction(v) for v in clustered(7, dim=32, seed=5)]
        res = sk.from_findings(findings, axes=["seeds"] * 3 + ["bootstrap"] * 3)
        for axis in ("seeds", "bootstrap"):
            assert res.axis_metrics[axis]["mean_pairwise_abs_cosine"] is not None
            assert res.axis_metrics[axis]["mean_pairwise_jaccard"] is None
        assert res.pooled["mean_pairwise_abs_cosine_axis_balanced"] is not None

    def test_cross_universe_directions_are_excluded_from_geometry(self):
        vectors = clustered(6, dim=32, seed=6)
        findings = [sk.direction(v, claim="c", universe="L12") for v in vectors[:4]]
        findings += [sk.direction(v, claim="c", universe="L20") for v in vectors[4:]]
        res = sk.from_findings(findings)
        assert res.pooled["n_direction_runs"] == 4
        assert res.pooled["n_cross_universe_excluded"] == 2
        assert res.pooled["n_runs"] == 6  # claims still pool over every run

    def test_stress_runs_a_direction_finder(self, tmp_path):
        core = [1.0, 0.0, 0.0, 0.0]

        def finder(data, seed, config):
            rng = random.Random(seed)
            v = [c + rng.gauss(0.0, config.get("noise", 0.05)) for c in core]
            return sk.direction(v, claim="x-axis", score=1.0 - 0.01 * (seed % 3))

        res = sk.stress(finder, list(range(20)), battery=["seeds", "bootstrap"],
                        n_runs=5, cache_dir=str(tmp_path), cache_key="v1")
        assert res.structure_kind == "direction"
        assert res.pooled["mean_pairwise_abs_cosine"] > 0.9
        assert "abs_cosine" in repr(res)

        cached = sk.stress(finder, list(range(20)),
                           battery=["seeds", "bootstrap"], n_runs=5,
                           cache_dir=str(tmp_path), cache_key="v1")
        assert (cached.pooled["mean_pairwise_abs_cosine"]
                == res.pooled["mean_pairwise_abs_cosine"])
        assert any("restored from cache" in n for n in cached.card.notes)

    def test_verdict_trace_works_for_directions(self):
        findings = [sk.direction(v, claim="one direction", score=0.9)
                    for v in clustered(8, dim=32, spread=0.08, seed=8)]
        trace = sk.verdict_trace(findings, sizes=[4, 6, 8], n_subsamples=4)
        assert trace["full_grade"] in ("A", "B", "C", "D")
        assert trace["sizes"] == [4, 6, 8]
        for k in trace["sizes"]:
            assert "structural_stability" in trace["per_size"][k]["check_pass_frac"]
        assert "Verdict-stability trace" in sk.verdict_trace_markdown(trace)


# ---------------------------------------------------------------------------
# fail fast on things that are not comparable
# ---------------------------------------------------------------------------

class TestIncomparableFindings:
    def test_mixed_kinds_in_one_battery(self):
        findings = [sk.direction([1.0, 0.0]), sk.circuit([(9, 6)]),
                    sk.direction([0.0, 1.0])]
        with pytest.raises(ValueError, match="mixes structural kinds"):
            sk.from_findings(findings)

    def test_mixed_kinds_across_real_and_null(self):
        real = [sk.direction(v) for v in clustered(4, dim=8, seed=1)]
        nulls = [sk.circuit([1, 2]), sk.circuit([2, 3])]
        with pytest.raises(ValueError, match="mixes structural kinds"):
            sk.from_findings(real, null_findings=nulls)

    def test_a_finder_that_switches_kind_on_the_null_control(self):
        # the null control runs the same finder on different data; a finder
        # that returns a direction there and a set here is a bug to surface,
        # not a null control to quietly skip
        def finder(data, seed, config):
            if max(data) < 100:
                rng = random.Random(seed)
                return sk.direction([1.0 + rng.gauss(0, 0.05), rng.gauss(0, 0.05)],
                                    claim="x", score=1.0)
            return sk.feature_set([1, 2, 3], claim="x", score=1.0,
                                  universe_size=10)

        with pytest.raises(ValueError, match="null control together"):
            sk.stress(finder, list(range(20)), battery=["seeds"], n_runs=4,
                      null_data=list(range(100, 120)))

    def test_mixed_direction_dimensions(self):
        findings = [sk.direction([1.0, 0.0]), sk.direction([1.0, 0.0, 0.0])]
        with pytest.raises(ValueError, match="mixes direction dimensions"):
            sk.from_findings(findings)

    def test_score_only_findings_join_either_group(self):
        findings = [sk.direction([1.0, 0.0], score=0.9),
                    sk.direction([0.9, 0.1], score=0.8),
                    sk.probe(0.7)]
        res = sk.from_findings(findings)
        assert res.pooled["n_direction_runs"] == 2
        assert res.pooled["n_runs"] == 3


# ---------------------------------------------------------------------------
# the card
# ---------------------------------------------------------------------------

def direction_card_dict(with_null=True, seed=11):
    real = [sk.direction(v, claim="one direction", score=0.9 + 0.001 * i)
            for i, v in enumerate(clustered(8, dim=32, spread=0.08, seed=seed))]
    nulls = ([sk.direction(f.vector, claim="scatter", score=0.1)
              for f in B.random_directions(8, 32, seed=seed + 500)]
             if with_null else None)
    res = sk.from_findings(real, null_findings=nulls, model="toy",
                           task="toy direction", method="difference in means")
    return json.loads(json.dumps(res.card.to_dict(), default=str))


class TestDirectionCard:
    def test_card_shape(self):
        d = direction_card_dict()
        assert d["schema_version"] == SCHEMA_VERSION == "0.4"
        assert d["battery"]["structure_kind"] == "direction"
        assert d["verdict"]["thresholds"]["cosine"] == 0.8
        block = d["directions"]
        assert block["dim"] == 32
        assert block["embedded"] is True
        assert len(block["abs_cosine"]) == 8
        assert len(block["null_abs_cosine"]) == 8
        assert block["order"][0] == "base"
        assert set(block["bootstrap"]) == {"n_boot", "alpha", "seed"}

    def test_runs_carry_dimension_and_digest_not_raw_vectors(self):
        d = direction_card_dict()
        assert len(d["runs"]) == 8
        for row in d["runs"]:
            assert row["direction_dim"] == 32
            assert len(row["direction_sha256"]) == 64
            assert "vector" not in row
            assert row["structure_present"] is False

    def test_digest_matches_the_unit_direction(self):
        f = sk.direction([3.0, 4.0])
        res = sk.from_findings([f, sk.direction([3.0, 4.1])])
        row = res.card.runs[0]
        assert row["direction_sha256"] == _vector_digest((0.6, 0.8))

    def test_set_cards_gain_nothing(self):
        res = sk.stress(
            lambda data, seed, config: sk.feature_set(
                sorted(random.Random(seed).sample(sorted(data), 8)),
                claim="late", score=0.9, universe_size=100),
            list(range(30)), battery=["seeds"], n_runs=5)
        d = json.loads(json.dumps(res.card.to_dict(), default=str))
        assert "directions" not in d
        assert "structure_kind" not in d["battery"]
        assert "cosine" not in d["verdict"]["thresholds"]
        assert all("direction_dim" not in r for r in d["runs"])
        assert verify_card_dict(d)["ok"]

    def test_round_trip_through_save_load_verify(self, tmp_path):
        d = direction_card_dict()
        path = os.path.join(str(tmp_path), "card.json")
        StabilityCard.from_dict(d).save(path)
        reloaded = sk.load_card(path)
        assert reloaded.directions == d["directions"]
        assert reloaded.to_dict() == d
        result = verify_card_dict(reloaded.to_dict())
        assert result["ok"], result["problems"]
        assert result["recomputed_grade"] == d["verdict"]["grade"]

    def test_markdown_render_speaks_cosine(self):
        card = StabilityCard.from_dict(direction_card_dict())
        md = card.to_markdown()
        # pipes inside a markdown table cell have to be escaped
        assert r"| mean pairwise \|cos\| |" in md
        assert r"| random-null \|cos\| in R^d |" in md
        assert "mean pairwise Jaccard" not in md
        assert "structured runs" not in md
        assert r"| axis | runs | \|cos\| | flip rate |" in md
        # every |cos| inside a table cell is escaped; the notes below the
        # tables are prose and may use the notation bare
        tables = md.split("## Notes")[0]
        assert "|cos|" not in tables.replace(r"\|cos\|", "")

    def test_verifies_without_a_null_control(self):
        d = direction_card_dict(with_null=False)
        assert "null_abs_cosine" not in d["directions"]
        assert verify_card_dict(d)["ok"]


class TestTamperedDirectionCard:
    def _problems(self, mutate):
        d = direction_card_dict()
        mutate(d)
        return verify_card_dict(d)["problems"]

    def test_inflated_matrix_entry_is_caught(self):
        def mutate(d):
            d["directions"]["abs_cosine"][0][1] = 1.0
            d["directions"]["abs_cosine"][1][0] = 1.0
        problems = self._problems(mutate)
        assert any("does not recompute" in p for p in problems)

    def test_inflated_pooled_metric_is_caught(self):
        def mutate(d):
            d["metrics"]["pooled"]["mean_pairwise_abs_cosine"] = 0.999
            d["verdict"]["checks"]["structural_stability"]["value"] = 0.999
        problems = self._problems(mutate)
        assert any("mean_pairwise_abs_cosine" in p and "recompute" in p
                   for p in problems)

    def test_inflated_check_value_alone_is_caught(self):
        def mutate(d):
            d["verdict"]["checks"]["structural_stability"]["value"] = 0.99
        problems = self._problems(mutate)
        assert any("!= pooled mean_pairwise_abs_cosine" in p for p in problems)

    def test_forged_confidence_interval_is_caught(self):
        def mutate(d):
            d["metrics"]["pooled"]["mean_pairwise_abs_cosine_ci95"] = [0.9, 0.95]
            d["verdict"]["checks"]["structural_stability"]["ci"] = [0.9, 0.95]
        problems = self._problems(mutate)
        assert any("ci95" in p and "recompute" in p for p in problems)

    def test_asymmetric_matrix_is_caught(self):
        def mutate(d):
            d["directions"]["abs_cosine"][2][3] = 0.123
        problems = self._problems(mutate)
        assert any("not symmetric" in p for p in problems)

    def test_broken_diagonal_is_caught(self):
        def mutate(d):
            d["directions"]["abs_cosine"][1][1] = 0.5
        problems = self._problems(mutate)
        assert any("diagonal" in p for p in problems)

    def test_out_of_range_entry_is_caught(self):
        def mutate(d):
            d["directions"]["abs_cosine"][0][1] = 1.4
            d["directions"]["abs_cosine"][1][0] = 1.4
        problems = self._problems(mutate)
        assert any("outside [0, 1]" in p for p in problems)

    def test_mismatched_run_dimension_is_caught(self):
        def mutate(d):
            d["runs"][0]["direction_dim"] = 4096
        problems = self._problems(mutate)
        assert any("dimension other than" in p for p in problems)

    def test_stripped_matrix_is_caught(self):
        def mutate(d):
            d.pop("directions")
        problems = self._problems(mutate)
        assert any("carries no 'directions' block" in p for p in problems)

    def test_forged_null_control_is_caught(self):
        def mutate(d):
            d["metrics"]["null_control"]["mean_pairwise_abs_cosine"] = 0.001
            j = d["metrics"]["pooled"]["mean_pairwise_abs_cosine"]
            d["verdict"]["checks"]["specificity"]["value"] = j / 0.001
        problems = self._problems(mutate)
        assert any("null matrix" in p for p in problems)
