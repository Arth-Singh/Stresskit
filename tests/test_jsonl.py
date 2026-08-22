"""JSONL entry point: findings_from_jsonl and from_jsonl."""

import json

import pytest

import stresskit as sk
from stresskit import findings_from_jsonl, from_jsonl


def write_jsonl(path, records):
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")


STABLE = [
    {"components": [[9, 6], [9, 9], [10, 0]], "claim": "late",
     "score": 3.1, "universe_size": 144},
    {"components": [[9, 6], [9, 9], [10, 0]], "claim": "late",
     "score": 3.0, "axis": "seeds"},
    {"components": [[9, 6], [9, 9], [10, 7]], "claim": "late",
     "score": 3.2, "axis": "seeds"},
    {"components": [[9, 6], [9, 9], [10, 0]], "claim": "late",
     "score": 3.1, "axis": "bootstrap"},
]


class TestLoader:
    def test_roundtrip(self, tmp_path):
        p = tmp_path / "sweep.jsonl"
        write_jsonl(p, STABLE)
        findings = findings_from_jsonl(str(p))
        assert len(findings) == 4
        assert findings[0].components == frozenset({(9, 6), (9, 9), (10, 0)})
        assert findings[0].claim == "late"
        assert findings[0].universe_size == 144
        assert findings[1].meta["axis"] == "seeds"
        assert "axis" not in findings[0].meta

    def test_blank_lines_skipped(self, tmp_path):
        p = tmp_path / "sweep.jsonl"
        p.write_text(json.dumps(STABLE[0]) + "\n\n" + json.dumps(STABLE[1]) + "\n")
        assert len(findings_from_jsonl(str(p))) == 2

    def test_key_remapping(self, tmp_path):
        p = tmp_path / "sweep.jsonl"
        write_jsonl(p, [{"edges": ["a", "b"], "faithfulness": 0.9},
                        {"edges": ["a", "c"], "faithfulness": 0.8}])
        findings = findings_from_jsonl(
            str(p), components_key="edges", score_key="faithfulness")
        assert findings[0].components == frozenset({"a", "b"})
        assert findings[0].score == 0.9
        assert findings[0].claim is None

    def test_partial_fields_ok(self, tmp_path):
        p = tmp_path / "sweep.jsonl"
        write_jsonl(p, [{"score": 0.9}, {"score": 0.8}])
        findings = findings_from_jsonl(str(p))
        assert not findings[0].has_structure()
        assert findings[0].score == 0.9

    def test_bad_json_names_line(self, tmp_path):
        p = tmp_path / "sweep.jsonl"
        p.write_text(json.dumps(STABLE[0]) + "\n{oops\n")
        with pytest.raises(ValueError, match=":2:"):
            findings_from_jsonl(str(p))

    def test_non_object_line_rejected(self, tmp_path):
        p = tmp_path / "sweep.jsonl"
        p.write_text("[1, 2]\n")
        with pytest.raises(ValueError, match="JSON object"):
            findings_from_jsonl(str(p))

    def test_empty_file_rejected(self, tmp_path):
        p = tmp_path / "sweep.jsonl"
        p.write_text("\n")
        with pytest.raises(ValueError, match="empty"):
            findings_from_jsonl(str(p))


class TestFromJsonl:
    def test_grades_a_stable_log(self, tmp_path):
        p = tmp_path / "sweep.jsonl"
        write_jsonl(p, STABLE)
        result = from_jsonl(str(p), model="gpt2", task="IOI")
        assert result.grade in "ABCD"
        assert result.card.claim["model"] == "gpt2"
        # axis labels flow through to the per-axis breakdown
        assert set(result.axis_metrics) == {"seeds", "bootstrap"}

    def test_card_verifies(self, tmp_path):
        p = tmp_path / "sweep.jsonl"
        write_jsonl(p, STABLE)
        result = from_jsonl(str(p))
        check = sk.verify_card_dict(result.card.to_dict())
        assert check["ok"], check["problems"]

    def test_null_path_enables_specificity(self, tmp_path):
        p = tmp_path / "sweep.jsonl"
        write_jsonl(p, STABLE)
        null = tmp_path / "null.jsonl"
        write_jsonl(null, [
            {"components": [[0, i], [1, i + 1]], "universe_size": 144}
            for i in range(4)
        ])
        result = from_jsonl(str(p), null_path=str(null))
        assert "specificity" in result.checks

    def test_partial_axis_labels_rejected(self, tmp_path):
        p = tmp_path / "sweep.jsonl"
        records = [dict(r) for r in STABLE]
        del records[3]["axis"]
        write_jsonl(p, records)
        with pytest.raises(ValueError, match="axis"):
            from_jsonl(str(p))

    def test_unlabeled_log_pools_under_runs(self, tmp_path):
        p = tmp_path / "sweep.jsonl"
        write_jsonl(p, [{k: v for k, v in r.items() if k != "axis"}
                        for r in STABLE])
        result = from_jsonl(str(p))
        assert set(result.axis_metrics) == {"runs"}
