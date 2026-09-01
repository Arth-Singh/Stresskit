import json
import random

import stresskit as sk
from stresskit.cli import main
from stresskit.report import CHECKLIST_FIELDS, generate_checklist, missing_fields


def _make_card(tmp_path):
    def finder(data, seed, config):
        rng = random.Random(seed)
        return sk.circuit(frozenset(range(5)), claim="early",
                          score=0.5 + rng.uniform(-0.01, 0.01), universe_size=100)

    result = sk.stress(finder, list(range(20)), n_runs=3, model="m", task="t")
    path = tmp_path / "card.json"
    result.card.save(str(path))
    return path


class TestChecklist:
    def test_all_reported(self):
        answers = {key: "x" for key, _, _ in CHECKLIST_FIELDS}
        md = generate_checklist(answers)
        assert "All fields reported" in md
        assert "**NOT REPORTED**" not in md  # the table marker; intro prose may mention the phrase

    def test_missing_flagged(self):
        md = generate_checklist({"model": "gpt2"})
        assert "NOT REPORTED" in md
        assert "gpt2" in md

    def test_missing_fields_helper(self):
        missing = missing_fields({"model": "gpt2", "task": "IOI"})
        assert "model" not in missing
        assert "n_seeds" in missing


class TestCLI:
    def test_render(self, tmp_path, capsys):
        path = _make_card(tmp_path)
        assert main(["render", str(path)]) == 0
        out = capsys.readouterr().out
        assert "Stability Card" in out

    def test_badge_stdout(self, tmp_path, capsys):
        path = _make_card(tmp_path)
        assert main(["badge", str(path)]) == 0
        badge = json.loads(capsys.readouterr().out)
        assert badge["label"] == "diagnostic stability"

    def test_badge_file(self, tmp_path, capsys):
        path = _make_card(tmp_path)
        out_path = tmp_path / "badge.json"
        assert main(["badge", str(path), "-o", str(out_path)]) == 0
        badge = json.loads(out_path.read_text())
        assert badge["schemaVersion"] == 1

    def test_report_flags(self, capsys):
        assert main(["report", "--model", "gpt2-small", "--task", "IOI"]) == 0
        out = capsys.readouterr().out
        assert "gpt2-small" in out
        assert "NOT REPORTED" in out

    def test_version(self, capsys):
        assert main(["version"]) == 0
        assert sk.__version__ in capsys.readouterr().out

    def test_no_command_shows_help(self, capsys):
        assert main([]) == 1
