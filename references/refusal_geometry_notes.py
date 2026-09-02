"""Re-derive the direction-geometry notes of a refusal-direction card.

The card's graded checks come from `stresskit verify`; the geometry notes are
descriptive text computed from the per-run manifest (`<card>.runs.json`) and
the saved unit directions in the raw directory. This script recomputes them
with `run_refusal_direction_card.geometry_notes`, replaces the previous
geometry lines in the card JSON, and re-renders the markdown.

Usage:
    python references/refusal_geometry_notes.py references/cards/refusal_direction_<slug>.json
"""

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_refusal_direction_card as rd  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("card")
    args = ap.parse_args()

    manifest_path = args.card.replace(".json", ".runs.json")
    with open(manifest_path) as f:
        manifest = json.load(f)
    raw_dir = os.path.join(os.path.dirname(args.card), manifest["raw_dir"])
    with open(args.card) as f:
        card = json.load(f)
    rows = manifest["runs"]
    if not rows[0].get("components"):
        # manifests written before components were recorded: take them from the
        # card's own run rows, which are emitted in the same order
        real = [r for r in rows if r["group"] == "real"]
        if len(card["runs"]) < len(real):
            raise RuntimeError(f"{args.card}: fewer card runs than manifest real runs")
        for row, card_row in zip(real, card["runs"]):
            if row["axis"] != card_row["axis"] or row["variant"] != card_row["variant"]:
                raise RuntimeError(f"{args.card}: run order differs from the manifest")
            row["components"] = card_row["components"]
    notes = rd.geometry_notes(rows, raw_dir)
    kept = [n for n in card["notes"] if not n.startswith(rd.GEOMETRY_NOTE_PREFIXES)]
    # keep the random-direction sanity line last
    tail = [n for n in kept if n.startswith("random-direction sanity")]
    head = [n for n in kept if not n.startswith("random-direction sanity")]
    card["notes"] = head + notes + tail
    with open(args.card, "w") as f:
        json.dump(card, f, indent=2, default=str)
        f.write("\n")

    md = args.card.replace(".json", ".md")
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, PYTHONPATH=os.path.join(repo, "src"))
    subprocess.run([sys.executable, "-m", "stresskit.cli", "render", args.card, "-o", md],
                   check=True, env=env)
    print(f"updated {args.card} and {md} ({len(notes)} geometry notes)")


if __name__ == "__main__":
    main()
