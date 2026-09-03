"""Validated merger for deterministic calibration-array JSON shards.

Shards of one study are grouped by their static cell identity, checked for
disjoint trial ranges, and summed field by field.  Which fields are static and
which are additive is decided by the payload's ``study``; every payload in a
merge must belong to the same study.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .battery_calibration import BatteryCalibrationResult
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


@dataclass(frozen=True)
class MergeSpec:
    """How one study's result rows split into identity and additive parts.

    ``additive_mappings`` are dict-valued fields whose entries are summed
    key-wise (``counts`` and ``sums`` of the battery study).
    """

    result_type: type
    static_fields: Tuple[str, ...]
    additive_fields: Tuple[str, ...]
    additive_mappings: Tuple[str, ...]
    sort_key: Callable[[Dict[str, Any]], Tuple[Any, ...]]


MERGE_SPECS: Dict[str, MergeSpec] = {
    "structural_known_truth_pilot": MergeSpec(
        result_type=CalibrationResult,
        static_fields=STATIC_FIELDS,
        additive_fields=ADDITIVE_FIELDS,
        additive_mappings=(),
        sort_key=lambda row: (
            row["scenario"]["name"],
            row["interval_method"],
            row["n_runs"],
        ),
    ),
    "battery_known_truth": MergeSpec(
        result_type=BatteryCalibrationResult,
        static_fields=(
            "scenario",
            "n_runs",
            "thresholds",
            "master_seed",
            "truth_grade_v03",
            "truth_grade_v04",
            "truths",
        ),
        additive_fields=("trials_requested",),
        additive_mappings=("counts", "sums"),
        sort_key=lambda row: (row["scenario"]["name"], row["n_runs"]),
    ),
}


def merge_calibration_payloads(
    payloads: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Merge nonoverlapping shards after checking cell and source identity."""
    if not payloads:
        raise ValueError("at least one calibration payload is required")
    if any(payload.get("schema_version") != "0.1" for payload in payloads):
        raise ValueError("all payloads must use calibration schema_version 0.1")
    studies = {payload.get("study") for payload in payloads}
    if len(studies) != 1:
        raise ValueError(
            f"all shards must belong to one study, got {sorted(map(str, studies))}"
        )
    study = next(iter(studies))
    spec = MERGE_SPECS.get(study)
    if spec is None:
        raise ValueError(
            f"no merge rule for study {study!r}; expected one of {sorted(MERGE_SPECS)}"
        )
    source_digests = {
        payload.get("provenance", {}).get("source_sha256") for payload in payloads
    }
    if None in source_digests or len(source_digests) != 1:
        raise ValueError("all shards must have the same nonempty source_sha256")

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for payload in payloads:
        for row in payload.get("results", []):
            key = json.dumps(
                {name: row[name] for name in spec.static_fields},
                sort_keys=True,
                separators=(",", ":"),
            )
            grouped[key].append(row)
    if not grouped:
        raise ValueError("payloads contain no calibration results")

    result_field_names = {field.name for field in fields(spec.result_type)}
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
        for name in spec.additive_fields:
            base[name] = sum(row[name] for row in rows)
        for name in spec.additive_mappings:
            keys = sorted({entry for row in rows for entry in row[name]})
            base[name] = {
                entry: sum(row[name].get(entry, 0) for row in rows) for entry in keys
            }
        result = spec.result_type(**base)
        merged_results.append(result.to_dict())
        cell_ranges[key] = [[start, end] for start, end in ranges]

    merged_results.sort(key=spec.sort_key)
    return {
        "schema_version": "0.1",
        "study": study,
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
