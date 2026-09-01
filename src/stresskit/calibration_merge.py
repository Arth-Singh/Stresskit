"""Validated merger for deterministic calibration-array JSON shards."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .calibration import CalibrationResult


ADDITIVE_FIELDS = (
    "trials_requested",
    "estimate_count",
    "estimate_sum",
    "squared_error_sum",
    "interval_valid_count",
    "coverage_count",
    "interval_width_sum",
    "pass_count",
    "fail_count",
    "inconclusive_count",
)
STATIC_FIELDS = (
    "scenario",
    "interval_method",
    "n_runs",
    "confidence_level",
    "threshold",
    "master_seed",
    "bootstrap_replicates",
)


def merge_calibration_payloads(
    payloads: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Merge nonoverlapping shards after checking cell and source identity."""
    if not payloads:
        raise ValueError("at least one calibration payload is required")
    if any(payload.get("schema_version") != "0.1" for payload in payloads):
        raise ValueError("all payloads must use calibration schema_version 0.1")
    source_digests = {
        payload.get("provenance", {}).get("source_sha256") for payload in payloads
    }
    if None in source_digests or len(source_digests) != 1:
        raise ValueError("all shards must have the same nonempty source_sha256")

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for payload in payloads:
        for row in payload.get("results", []):
            key = json.dumps(
                {name: row[name] for name in STATIC_FIELDS},
                sort_keys=True,
                separators=(",", ":"),
            )
            grouped[key].append(row)
    if not grouped:
        raise ValueError("payloads contain no calibration results")

    result_field_names = {field.name for field in fields(CalibrationResult)}
    merged_results = []
    cell_ranges = {}
    for key, rows in grouped.items():
        ranges = sorted(
            (row["trial_start"], row["trial_start"] + row["trials_requested"])
            for row in rows
        )
        for previous, current in zip(ranges, ranges[1:]):
            if current[0] < previous[1]:
                raise ValueError(f"overlapping trial ranges for cell {key}")
        base = {name: rows[0][name] for name in result_field_names}
        base["trial_start"] = ranges[0][0]
        for name in ADDITIVE_FIELDS:
            base[name] = sum(row[name] for row in rows)
        result = CalibrationResult(**base)
        merged_results.append(result.to_dict())
        cell_ranges[key] = [[start, end] for start, end in ranges]

    merged_results.sort(
        key=lambda row: (
            row["scenario"]["name"], row["interval_method"], row["n_runs"]
        )
    )
    return {
        "schema_version": "0.1",
        "study": payloads[0].get("study"),
        "status": "merged_calibration_shards",
        "provenance": {
            "source_sha256": next(iter(source_digests)),
            "input_payload_count": len(payloads),
        },
        "cell_trial_ranges": cell_ranges,
        "results": merged_results,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Merge StressKit calibration shards")
    parser.add_argument("inputs", nargs="+", type=Path)
    args = parser.parse_args(argv)
    payloads = [json.loads(path.read_text()) for path in args.inputs]
    print(json.dumps(merge_calibration_payloads(payloads), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
