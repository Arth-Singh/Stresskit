"""Trace charts, the demo, and the site generator."""

import json
import os
import xml.etree.ElementTree as ET

import pytest

from stresskit.card import GRADE_ORDER
from stresskit.cli import main as cli_main
from stresskit.demo import run_demo
from stresskit.site import build_site
from stresskit.tracechart import trace_svg


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFERENCES = os.path.join(REPO_ROOT, "references")
IOI_TRACE = os.path.join(REFERENCES, "cards", "ioi_gpt2_small.trace.json")

needs_references = pytest.mark.skipif(
    not os.path.isdir(REFERENCES), reason="reference cards not present")


def load_ioi_trace():
    with open(IOI_TRACE, encoding="utf-8") as f:
        return json.load(f)


@needs_references
class TestTraceChart:
    def test_valid_svg_with_annotations(self):
        svg = trace_svg(load_ioi_trace())
        root = ET.fromstring(svg)  # well-formed XML
        assert root.tag.endswith("svg")
        assert "coin flip" in svg           # IOI's n=6 story is annotated
        assert "settles at n = 45" in svg
        assert "<title>" in svg             # native tooltips per segment

    def test_deterministic(self):
        t = load_ioi_trace()
        assert trace_svg(t) == trace_svg(json.loads(json.dumps(t)))

    def test_title_escaped(self):
        svg = trace_svg(load_ioi_trace(), title='<script>"x"&</script>')
        ET.fromstring(svg)
        assert "<script>" not in svg

    def test_settled_trace_has_no_coinflip_note(self):
        with open(os.path.join(REFERENCES, "cards",
                               "greater_than_gpt2_small.trace.json"),
                  encoding="utf-8") as f:
            svg = trace_svg(json.load(f))
        assert "settles at n = 6" in svg
        assert "coin flip" not in svg       # GT settles immediately

    def test_cli_trace(self, tmp_path, capsys):
        out = tmp_path / "trace.svg"
        rc = cli_main(["trace", IOI_TRACE, "-o", str(out)])
        assert rc == 0
        ET.fromstring(out.read_text())


class TestDemo:
    def test_demo_separates_real_from_noise(self):
        lines = []
        results = run_demo(echo=lines.append)
        real, null = results["real"], results["null"]
        assert real.grade == "A"
        assert GRADE_ORDER.index(null.grade) > GRADE_ORDER.index(real.grade)
        text = "\n".join(lines)
        assert "PURE NOISE" in text
        assert "Side by side" in text

    def test_demo_html_cards(self, tmp_path):
        run_demo(html_dir=str(tmp_path), echo=lambda *_: None)
        for slug in ("real_effect", "pure_noise"):
            page = (tmp_path / f"demo_{slug}.html").read_text()
            assert page.startswith("<!doctype html>")

    def test_cli_demo(self, capsys):
        assert cli_main(["demo"]) == 0
        assert "grade A" in capsys.readouterr().out


@needs_references
class TestSite:
    def test_build_from_references(self, tmp_path):
        n = build_site([REFERENCES], str(tmp_path))
        assert n
        index = (tmp_path / "index.html").read_text()
        assert "Do interpretability findings survive re-running?" in index
        assert "<svg" in index                       # hero trace chart
        assert "coin flip" in index                  # IOI is the hero
        pages = sorted(p.name for p in tmp_path.glob("*.html"))
        assert len(pages) == n + 1                   # one page per card, plus index
        assert len(list(tmp_path.glob("*.json"))) == n   # cards copied

    def test_card_page_embeds_trace_and_audit(self, tmp_path):
        build_site([REFERENCES], str(tmp_path))
        page = (tmp_path / "ioi_gpt2_small.html").read_text()
        assert "Verdict-stability trace" in page
        assert "stresskit verify ioi_gpt2_small.json" in page
        assert 'href="index.html"' in page           # back link

    def test_deterministic(self, tmp_path):
        a, b = tmp_path / "a", tmp_path / "b"
        build_site([REFERENCES], str(a))
        build_site([REFERENCES], str(b))
        for pa in sorted(a.iterdir()):
            assert pa.read_bytes() == (b / pa.name).read_bytes()

    def test_empty_dir_rejected(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(ValueError, match="no cards"):
            build_site([str(empty)], str(tmp_path / "out"))

    def test_cli_site(self, tmp_path, capsys):
        out_dir = tmp_path / "s"
        rc = cli_main(["site", REFERENCES, "-o", str(out_dir)])
        assert rc == 0
        n_cards = len(list(out_dir.glob("*.html"))) - 1   # index is not a card
        assert f"{n_cards} card pages" in capsys.readouterr().out
