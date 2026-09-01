"""Outcome-blind round-robin scheduler for a frozen StressKit v1 registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping

from stresskit.integrity import digest_json


_FORBIDDEN_OUTCOME_KEYS = {
    "result", "grade", "verdict", "passed", "failed", "effect_size",
    "audit_status",
}


def _keys(value: Any) -> set:
    found = set()
    if isinstance(value, Mapping):
        found.update(value)
        for child in value.values():
            found.update(_keys(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_keys(child))
    return found


def schedule(registry: Mapping[str, Any]) -> Dict[str, Any]:
    """Order frozen claims by uncovered stratum then cheapest compute tier."""
    if registry.get("artifact") != "stresskit_release_registry" or \
            registry.get("schema_version") != "1.0" or \
            registry.get("status") != "frozen" or \
            registry.get("outcome_blind") is not True:
        raise ValueError("scheduler accepts only frozen v1 release registry")
    forbidden = _keys(registry) & _FORBIDDEN_OUTCOME_KEYS
    if forbidden:
        raise ValueError(f"registry contains outcome keys: {sorted(forbidden)}")
    claims = registry.get("claims")
    if not isinstance(claims, list) or not claims:
        raise ValueError("frozen registry needs claims")
    claim_ids = [row.get("claim_id") for row in claims if isinstance(row, Mapping)]
    if len(claim_ids) != len(claims) or not all(
            isinstance(value, str) and value for value in claim_ids):
        raise ValueError("every frozen registry row needs claim_id")
    if len(claim_ids) != len(set(claim_ids)):
        raise ValueError("frozen registry claim IDs must be unique")
    groups: Dict[str, List[Mapping[str, Any]]] = {}
    for row in claims:
        if row.get("disposition") != "eligible":
            continue
        stratum = row.get("stratum")
        if not isinstance(stratum, str) or not stratum:
            raise ValueError("eligible claim needs method-family stratum")
        tier = row.get("compute_tier")
        if not isinstance(tier, int) or tier < 0:
            raise ValueError("eligible claim compute_tier must be non-negative integer")
        groups.setdefault(stratum, []).append(row)
    for rows in groups.values():
        rows.sort(key=lambda row: (row["compute_tier"], row["claim_id"]))
    ordered = []
    round_index = 0
    while any(groups.values()):
        active = sorted(
            (name for name, rows in groups.items() if rows),
            key=lambda name: (groups[name][0]["compute_tier"], name),
        )
        for name in active:
            row = groups[name].pop(0)
            ordered.append({
                "position": len(ordered) + 1,
                "round": round_index + 1,
                "claim_id": row["claim_id"],
                "stratum": name,
                "compute_tier": row["compute_tier"],
            })
        round_index += 1
    return {
        "artifact": "stresskit_outcome_blind_schedule",
        "schema_version": "1.0",
        "release_id": registry.get("release_id"),
        "registry_digest": digest_json(registry),
        "rule": "round-robin uncovered stratum, then lowest compute tier, then claim_id",
        "outcome_blind": True,
        "jobs": ordered,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("registry", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    result = schedule(json.loads(args.registry.read_text()))
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.write_text(text)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
