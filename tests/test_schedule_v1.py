"""Outcome-blind scheduler covers strata before second claims."""

import importlib.util
from pathlib import Path

import pytest


PATH = Path(__file__).parents[1] / "benchmark" / "schedule_audits.py"
SPEC = importlib.util.spec_from_file_location("schedule_audits", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_round_robin_then_compute_cost():
    registry = {
        "artifact": "stresskit_release_registry",
        "schema_version": "1.0",
        "status": "frozen",
        "outcome_blind": True,
        "release_id": "r1",
        "claims": [
            {"claim_id": "cot-expensive", "stratum": "CoT", "compute_tier": 3,
             "disposition": "eligible"},
            {"claim_id": "cot-cheap", "stratum": "CoT", "compute_tier": 1,
             "disposition": "eligible"},
            {"claim_id": "probe", "stratum": "probes", "compute_tier": 2,
             "disposition": "eligible"},
        ],
    }
    result = MODULE.schedule(registry)
    assert result["registry_digest"].startswith("sha256:")
    jobs = result["jobs"]
    assert [row["claim_id"] for row in jobs] == [
        "cot-cheap", "probe", "cot-expensive"
    ]
    assert [row["round"] for row in jobs] == [1, 1, 2]


def test_scheduler_rejects_outcomes_and_unfrozen_input():
    with pytest.raises(ValueError, match="frozen"):
        MODULE.schedule({"artifact": "candidate"})
    registry = {
        "artifact": "stresskit_release_registry",
        "schema_version": "1.0",
        "status": "frozen",
        "outcome_blind": True,
        "claims": [{"claim_id": "x", "stratum": "CoT", "compute_tier": 1,
                    "disposition": "eligible", "verdict": "pass"}],
    }
    with pytest.raises(ValueError, match="outcome keys"):
        MODULE.schedule(registry)
