"""Scoreboard generation: row collection, markdown safety, determinism."""

import json
import os

import pytest

from stresskit.cli import main as cli_main
from stresskit.scoreboard import (collect_rows, load_papers, paper_rows,
                                  registered_paper_rows, scoreboard_markdown,
                                  write_scoreboard)
from stresskit.site import build_site


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFERENCES = os.path.join(REPO_ROOT, "references")

needs_references = pytest.mark.skipif(
    not os.path.isdir(REFERENCES), reason="reference cards not present")


@needs_references
class TestCollectRows:
    def test_finds_all_reference_artifacts(self):
        rows = collect_rows([REFERENCES])
        kinds = {r["kind"] for r in rows}
        assert rows
        assert len({r["path"] for r in rows}) == len(rows)
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
        assert n == len(collect_rows([REFERENCES]))
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
        papers = registered_paper_rows([REFERENCES], rows)
        expected = scoreboard_markdown(rows, relative_to=REPO_ROOT,
                                       papers=papers) + "\n"
        with open(committed, encoding="utf-8") as f:
            assert f.read() == expected


def _fake_card(path, grade="A", confidence="high"):
    path.write_text(json.dumps({
        "schema_version": "0.4",
        "claim": {"task": "t", "model": "m", "method": "meth"},
        "battery": {"n_runs_total": 3},
        "metrics": {"pooled": {"confidence": confidence, "mean_pairwise_jaccard": 0.9}},
        "verdict": {"grade": grade, "checks": {
            "structural_stability": {"passed": True, "robust": True},
            "specificity": {"passed": grade == "A", "robust": confidence == "high",
                            "value": 2.0}}},
    }))


@needs_references
class TestPaperRegistry:
    REGISTRY = os.path.join(REFERENCES, "papers.json")

    def test_every_graded_artifact_is_registered_once(self):
        rows = collect_rows([REFERENCES])
        prows = paper_rows(load_papers(self.REGISTRY), rows)
        listed = [r["path"] for p in prows for r in p["rows"]]
        assert sorted(listed) == sorted(r["path"] for r in rows)
        assert len(listed) == len(set(listed))
        for p in prows:
            assert p["checks_total"] >= p["checks_passed"] >= 0
            assert p["n_runs"] > 0

    def test_registry_order_is_leaderboard_order(self):
        papers = load_papers(self.REGISTRY)
        prows = paper_rows(papers, collect_rows([REFERENCES]))
        assert [p["title"] for p in prows] == [p["title"] for p in papers]

    def test_scoreboard_has_papers_table_before_findings(self):
        rows = collect_rows([REFERENCES])
        md = scoreboard_markdown(rows, relative_to=REPO_ROOT,
                                 papers=paper_rows(load_papers(self.REGISTRY), rows))
        assert md.index("## Papers") < md.index("## All graded findings")
        assert "https://arxiv.org/abs/2608.02486" in md
        assert "references/cards/folkmotif_llama3p1_8b.md" in md

    def test_site_index_has_papers_panel(self, tmp_path):
        build_site([REFERENCES], str(tmp_path))
        index = (tmp_path / "index.html").read_text()
        assert "<h2 style=\"font-size:1.1rem;margin-top:0\">Papers</h2>" in index
        assert "papers audited" in index
        assert 'href="folkmotif_llama3p1_8b.html"' in index
        assert "https://arxiv.org/abs/2608.05578" in index


class TestPaperRegistryFailsFast:
    def _registry(self, tmp_path, papers):
        reg = tmp_path / "papers.json"
        reg.write_text(json.dumps({"papers": papers}))
        return str(reg)

    def test_missing_card_is_an_error(self, tmp_path):
        reg = self._registry(tmp_path, [{
            "title": "P", "models": "m", "cards": ["cards/nope.json"],
            "reproduced": "-", "result": "-", "audited": "2026-09-02"}])
        with pytest.raises(FileNotFoundError):
            load_papers(reg)

    def test_unregistered_card_is_an_error(self, tmp_path):
        (tmp_path / "cards").mkdir()
        _fake_card(tmp_path / "cards" / "a.json")
        _fake_card(tmp_path / "cards" / "b.json", grade="C", confidence="low")
        reg = self._registry(tmp_path, [{
            "title": "P", "models": "m", "cards": ["cards/a.json"],
            "reproduced": "-", "result": "-", "audited": "2026-09-02"}])
        rows = collect_rows([str(tmp_path)])
        with pytest.raises(ValueError, match="missing from the paper registry"):
            paper_rows(load_papers(reg), rows)

    def test_card_in_two_papers_is_an_error(self, tmp_path):
        (tmp_path / "cards").mkdir()
        _fake_card(tmp_path / "cards" / "a.json")
        entry = {"title": "P", "models": "m", "cards": ["cards/a.json"],
                 "reproduced": "-", "result": "-", "audited": "2026-09-02"}
        reg = self._registry(tmp_path, [entry, {**entry, "title": "Q"}])
        with pytest.raises(ValueError, match="listed under both"):
            paper_rows(load_papers(reg), collect_rows([str(tmp_path)]))

    def test_registry_without_arxiv_and_low_confidence_marker(self, tmp_path):
        (tmp_path / "cards").mkdir()
        _fake_card(tmp_path / "cards" / "a.json", grade="B", confidence="low")
        reg = self._registry(tmp_path, [{
            "title": "P", "arxiv": None, "models": "m", "cards": ["cards/a.json"],
            "reproduced": "-", "result": "r | with pipe", "audited": "2026-09-02"}])
        rows = collect_rows([str(tmp_path)])
        md = scoreboard_markdown(rows, papers=paper_rows(load_papers(reg), rows))
        line = next(ln for ln in md.splitlines() if ln.startswith("| P |"))
        assert "arxiv.org" not in line
        assert "**B**](" in line and "†" in line
        assert "r \\| with pipe" in line

    def test_no_registry_keeps_plain_scoreboard(self, tmp_path):
        (tmp_path / "cards").mkdir()
        _fake_card(tmp_path / "cards" / "a.json")
        out = tmp_path / "SCOREBOARD.md"
        assert write_scoreboard([str(tmp_path)], str(out)) == 1
        assert "## Papers" not in out.read_text()
