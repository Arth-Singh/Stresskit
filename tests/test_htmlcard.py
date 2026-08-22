"""HTML card renderer: content, escaping, determinism, CLI."""

import json
import os

import pytest

import stresskit as sk
from stresskit.cli import main as cli_main
from stresskit.htmlcard import card_html


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_card_dict(statement="toy claim"):
    findings = [sk.circuit({("a", i) for i in range(10)}, claim="late",
                           score=1.0 + 0.01 * j, universe_size=100)
                for j in range(5)]
    return sk.from_findings(findings, claim_statement=statement,
                            model="gpt2", task="toy").card.to_dict()


class TestCardHtml:
    def test_stability_card_renders(self):
        page = card_html(make_card_dict())
        assert page.startswith("<!doctype html>")
        assert "toy claim" in page
        assert "structural stability" in page
        assert 'class="thr"' in page          # threshold tick drawn

    def test_oracle_report_renders(self):
        path = os.path.join(REPO_ROOT,
                            "references/cards/ao_qwen3_full-mixture.json")
        if not os.path.exists(path):
            pytest.skip("reference cards not present")
        with open(path, encoding="utf-8") as f:
            page = card_html(json.load(f))
        assert "Oracle Reliability Report" in page
        assert "null hallucination" in page

    def test_low_confidence_banner(self):
        path = os.path.join(REPO_ROOT, "references/cards/ioi_gpt2_small.json")
        if not os.path.exists(path):
            pytest.skip("reference cards not present")
        with open(path, encoding="utf-8") as f:
            page = card_html(json.load(f))
        assert "LOW CONFIDENCE" in page
        assert "undecided" in page

    def test_html_escaping(self):
        d = make_card_dict(statement='<script>alert("x")</script> & <b>')
        page = card_html(d)
        assert "<script>" not in page
        assert "&lt;script&gt;" in page

    def test_deterministic(self):
        d = make_card_dict()
        assert card_html(d) == card_html(json.loads(json.dumps(d)))

    def test_non_artifact_rejected(self):
        with pytest.raises(ValueError):
            card_html({"schemaVersion": 1})


class TestRenderCli:
    def test_html_to_file(self, tmp_path, capsys):
        card = tmp_path / "card.json"
        card.write_text(json.dumps(make_card_dict()))
        out = tmp_path / "card.html"
        rc = cli_main(["render", str(card), "--html", "-o", str(out)])
        assert rc == 0
        assert out.read_text().startswith("<!doctype html>")

    def test_markdown_still_default(self, tmp_path, capsys):
        card = tmp_path / "card.json"
        card.write_text(json.dumps(make_card_dict()))
        rc = cli_main(["render", str(card)])
        out = capsys.readouterr().out
        assert rc == 0
        assert "<!doctype" not in out
        assert "Stability Card" in out or "grade" in out.lower()

    def test_markdown_to_file(self, tmp_path, capsys):
        card = tmp_path / "card.json"
        card.write_text(json.dumps(make_card_dict()))
        out = tmp_path / "card.md"
        rc = cli_main(["render", str(card), "-o", str(out)])
        assert rc == 0
        assert out.read_text().strip()
