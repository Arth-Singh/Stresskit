import json
import random

import pytest

import stresskit as sk
from stresskit.card import StabilityCard, validate_card_dict


def make_result():
    def finder(data, seed, config):
        rng = random.Random(seed)
        return sk.circuit(
            frozenset(range(8)),
            claim="late",
            score=0.8 + rng.uniform(-0.01, 0.01),
            universe_size=1000,
        )

    return sk.stress(
        finder,
        list(range(30)),
        n_runs=4,
        claim_statement="The task is implemented by 8 late-layer edges",
        model="toy-model",
        task="toy-task",
        method="toy-EAP",
    )


class TestRoundTrip:
    def test_save_load(self, tmp_path):
        result = make_result()
        path = tmp_path / "card.json"
        result.card.save(str(path))
        loaded = sk.load_card(str(path))
        assert loaded.grade == result.grade
        assert loaded.claim["statement"] == "The task is implemented by 8 late-layer edges"
        assert loaded.metrics["pooled"]["mean_pairwise_jaccard"] == 1.0

    def test_json_is_valid(self, tmp_path):
        result = make_result()
        path = tmp_path / "card.json"
        result.card.save(str(path))
        with open(path) as f:
            d = json.load(f)
        validate_card_dict(d)  # should not raise


class TestValidation:
    def test_missing_fields_rejected(self):
        with pytest.raises(ValueError, match="missing required fields"):
            validate_card_dict({"schema_version": "0.1"})

    def test_bad_grade_rejected(self):
        result = make_result()
        d = result.card.to_dict()
        d["verdict"]["grade"] = "Z"
        with pytest.raises(ValueError, match="grade"):
            validate_card_dict(d)


class TestRenders:
    def test_markdown_contains_essentials(self):
        result = make_result()
        md = result.to_markdown()
        assert "Stability Card" in md
        assert "descriptive grade **A**" in md
        assert "does not issue a confirmatory verdict" in md
        assert "toy-model" in md
        assert "mean pairwise Jaccard" in md
        assert "| structural stability |" in md

    def test_badge_shape(self):
        result = make_result()
        badge = result.card.badge_dict()
        assert badge["schemaVersion"] == 1
        assert badge["label"] == "diagnostic stability"
        assert badge["message"].startswith("A · J=")
        assert badge["color"] == "brightgreen"

    def test_badge_color_tracks_grade(self):
        result = make_result()
        result.card.verdict["grade"] = "D"
        assert result.card.badge_dict()["color"] == "red"
