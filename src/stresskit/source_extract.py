"""Deterministic, outcome-blind text extraction for source documents."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Union


_NOTEBOOK_SUFFIX = ".ipynb"
_CELL_TYPES = frozenset(("code", "markdown", "raw"))


def _decode_utf8(data: bytes, path: Path) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path} must be UTF-8") from exc


def _cell_source(cell: Mapping[str, Any], index: int) -> str:
    cell_type = cell.get("cell_type")
    if not isinstance(cell_type, str) or cell_type not in _CELL_TYPES:
        raise ValueError(
            f"notebook cell {index} must have cell_type code, markdown, or raw"
        )
    if "source" not in cell:
        raise ValueError(f"notebook cell {index} is missing source")
    source = cell["source"]
    if isinstance(source, str):
        return source
    if isinstance(source, list) and all(isinstance(part, str) for part in source):
        return "".join(source)
    raise ValueError(f"notebook cell {index} source must be a string or list of strings")


def render_notebook_text(notebook: Mapping[str, Any]) -> str:
    """Render ordered notebook cell types and sources into canonical UTF-8 text.

    Notebook metadata, cell metadata, execution counts, and outputs are omitted.
    Cell sources are not normalized or interpreted; list-form sources are joined
    using the Jupyter notebook format's concatenation rule.
    """
    cells = notebook.get("cells")
    if not isinstance(cells, list):
        raise ValueError("notebook cells must be a list")

    blocks = []
    for index, cell in enumerate(cells):
        if not isinstance(cell, Mapping):
            raise ValueError(f"notebook cell {index} must be an object")
        cell_type = cell.get("cell_type")
        source = _cell_source(cell, index)
        block = f'<stresskit-cell type="{cell_type}">\n{source}'
        if not source.endswith("\n"):
            block += "\n"
        blocks.append(block + "</stresskit-cell>\n")
    return "".join(blocks)


def extract_source_bytes(path: Union[str, Path]) -> bytes:
    """Return deterministic UTF-8 source text bytes for a file or notebook.

    Ordinary UTF-8 files are returned byte-for-byte. ``.ipynb`` files are parsed
    and rendered with :func:`render_notebook_text` so non-source notebook state
    cannot affect claim evidence.
    """
    source_path = Path(path)
    raw = source_path.read_bytes()
    text = _decode_utf8(raw, source_path)
    if source_path.suffix.lower() != _NOTEBOOK_SUFFIX:
        return raw

    try:
        notebook = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{source_path} must contain valid notebook JSON") from exc
    if not isinstance(notebook, Mapping):
        raise ValueError("notebook root must be an object")
    rendered = render_notebook_text(notebook)
    try:
        return rendered.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("notebook source must contain valid Unicode") from exc


def write_extracted_source(
    input_path: Union[str, Path], output_path: Union[str, Path]
) -> None:
    """Extract ``input_path`` and write stable UTF-8 bytes to ``output_path``."""
    Path(output_path).write_bytes(extract_source_bytes(input_path))
