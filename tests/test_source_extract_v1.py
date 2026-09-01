import json

import pytest

from stresskit.cli import main
from stresskit.source_extract import extract_source_bytes, render_notebook_text


def _notebook():
    return {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {"hidden": "metadata-secret"},
                "source": ["Claim heading\n", "Claim body α"],
            },
            {
                "cell_type": "code",
                "execution_count": 42,
                "metadata": {"tags": ["remove-cell"]},
                "outputs": [{"text": "output-secret"}],
                "source": "result = intervention(model)\n",
            },
            {"cell_type": "raw", "metadata": {}, "source": ""},
        ],
        "metadata": {"kernelspec": {"display_name": "metadata-secret"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def test_notebook_render_is_ordered_and_source_only(tmp_path):
    notebook_path = tmp_path / "claim.ipynb"
    notebook_path.write_text(json.dumps(_notebook()), encoding="utf-8")

    extracted = extract_source_bytes(notebook_path)

    assert extracted == (
        b'<stresskit-cell type="markdown">\n'
        b"Claim heading\nClaim body \xce\xb1\n"
        b"</stresskit-cell>\n"
        b'<stresskit-cell type="code">\n'
        b"result = intervention(model)\n"
        b"</stresskit-cell>\n"
        b'<stresskit-cell type="raw">\n'
        b"\n</stresskit-cell>\n"
    )
    assert b"metadata-secret" not in extracted
    assert b"output-secret" not in extracted
    assert b"execution_count" not in extracted


def test_notebook_output_ignores_non_source_state_and_is_byte_stable(tmp_path):
    first = _notebook()
    second = _notebook()
    second["metadata"] = {"changed": True}
    second["cells"][0]["metadata"] = {"changed": True}
    second["cells"][1]["execution_count"] = None
    second["cells"][1]["outputs"] = [{"data": {"text/plain": ["different"]}}]
    first_path = tmp_path / "first.ipynb"
    second_path = tmp_path / "second.ipynb"
    first_path.write_text(json.dumps(first, indent=1), encoding="utf-8")
    second_path.write_text(json.dumps(second, separators=(",", ":")), encoding="utf-8")

    assert extract_source_bytes(first_path) == extract_source_bytes(second_path)
    assert render_notebook_text(first) == render_notebook_text(second)


def test_plain_utf8_file_round_trips_exact_bytes(tmp_path):
    source = tmp_path / "paper.txt"
    raw = b"\xef\xbb\xbfPr\xc3\xa9face\r\nClaim \xce\xb1.\r\n"
    source.write_bytes(raw)

    assert extract_source_bytes(source) == raw


@pytest.mark.parametrize(
    "cell, message",
    [
        (None, "must be an object"),
        ({"source": "x"}, "must have cell_type"),
        ({"cell_type": "heading", "source": "x"}, "must have cell_type"),
        ({"cell_type": "code"}, "missing source"),
        ({"cell_type": "code", "source": ["x", 1]}, "string or list"),
        ({"cell_type": "code", "source": {"text": "x"}}, "string or list"),
    ],
)
def test_malformed_notebook_cells_are_rejected(tmp_path, cell, message):
    path = tmp_path / "bad.ipynb"
    path.write_text(json.dumps({"cells": [cell]}), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        extract_source_bytes(path)


def test_extract_source_cli_writes_output(tmp_path):
    input_path = tmp_path / "claim.ipynb"
    output_path = tmp_path / "claim.txt"
    input_path.write_text(json.dumps(_notebook()), encoding="utf-8")

    assert main([
        "audit", "extract-source", str(input_path), "-o", str(output_path)
    ]) == 0
    assert output_path.read_bytes() == extract_source_bytes(input_path)


def test_extract_source_rejects_non_utf8_and_invalid_notebook_json(tmp_path):
    binary = tmp_path / "binary.txt"
    binary.write_bytes(b"\xff")
    with pytest.raises(ValueError, match="must be UTF-8"):
        extract_source_bytes(binary)

    notebook = tmp_path / "bad.ipynb"
    notebook.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="valid notebook JSON"):
        extract_source_bytes(notebook)
