"""Scoreboard generation: row collection, markdown safety, determinism."""

import json
import os

import pytest

from stresskit.cli import main as cli_main
from stresskit.scoreboard import collect_rows, scoreboard_markdown, write_scoreboard


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFERENCES = os.path.join(REPO_ROOT, "references")

needs_references = pytest.mark.skipif(
    not os.path.isdir(REFERENCES), reason="reference cards not present")


@needs_references
class TestCollectRows:
    def test_finds_all_reference_artifacts(self):
        rows = collect_rows([REFERENCES])
        kinds = {r["kind"] for r in rows}
        assert len(rows) == 8
        assert kinds == {"stability card", "oracle report"}

    def test_stability_cards_sort_first(self):
        rows = collect_rows([REFERENCES])
        kinds = [r["kind"] for r in rows]
        assert kinds == sorted(kinds, key=lambda k: k != "stability card")

    def test_rows_carry_verdict_fields(self):
        for r in collect_rows([REFERENCES]):
            assert r["grade"] in "ABCD"
            assert r["confidence"] in ("high", "low", "unknown")
            assert "/" in r["checks"]

    def test_non_artifact_json_skipped(self, tmp_path):
        (tmp_path / "junk.json").write_text(json.dumps({"a": 1}))
        (tmp_path / "broken.json").write_text("{not json")
        assert collect_rows([str(tmp_path)]) == []


@needs_references
class TestMarkdown:
    def test_pipes_in_method_are_escaped(self):
        # the IOI cards' method string contains literal "|attribution|"
        md = scoreboard_markdown(collect_rows([REFERENCES]))
        assert "|attribution|" not in md
        assert "\\|attribution\\|" in md
        header = next(l for l in md.splitlines() if l.startswith("| finding"))
        n_cols = header.count("|")
        for line in md.splitlines():
            if line.startswith("| ") and "**" in line:
                assert line.count("|") - line.count("\\|") == n_cols, line

    def test_links_prefer_markdown_render(self):
        md = scoreboard_markdown(collect_rows([REFERENCES]),
                                 relative_to=REPO_ROOT)
        assert "references/cards/ioi_gpt2_small.md" in md
        assert "ioi_gpt2_small.json" not in md

    def test_deterministic(self):
        rows = collect_rows([REFERENCES])
        assert scoreboard_markdown(rows) == scoreboard_markdown(
            collect_rows([REFERENCES]))


@needs_references
class TestCliScoreboard:
    def test_write_and_freshness_roundtrip(self, tmp_path):
        out = tmp_path / "SCOREBOARD.md"
        n = write_scoreboard([REFERENCES], str(out))
        assert n == 8
        first = out.read_text()
        write_scoreboard([REFERENCES], str(out))
        assert out.read_text() == first  # regenerating is byte-identical

    def test_cli_stdout(self, capsys):
        rc = cli_main(["scoreboard", REFERENCES])
        out = capsys.readouterr().out
        assert rc == 0
        assert "# Stability Scoreboard" in out

    def test_repo_scoreboard_is_fresh(self):
        # the committed SCOREBOARD.md must match what the cards generate —
        # the same invariant CI enforces
        committed = os.path.join(REPO_ROOT, "SCOREBOARD.md")
        if not os.path.exists(committed):
            pytest.skip("SCOREBOARD.md not generated yet")
        rows = collect_rows([REFERENCES])
        expected = scoreboard_markdown(rows, relative_to=REPO_ROOT) + "\n"
        with open(committed, encoding="utf-8") as f:
            assert f.read() == expected
