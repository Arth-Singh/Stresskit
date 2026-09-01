"""Fresh-seed known-truth calibration for v1 bounded audit profiles."""

from __future__ import annotations

import argparse
import json
import random
from typing import Any, Dict, List

from .audit_profiles import (
    PROFILE_REGISTRY,
    PROFILE_REGISTRY_DIGEST,
    hoeffding_interval,
    threshold_check,
)
from .integrity import digest_json


CALIBRATION_METHOD = {
    "name": "stresskit-v1-bounded-profile-calibration",
    "version": "1",
    "coverage_scenarios": [0.1, 0.5, 0.9],
    "invalid_scenarios": [
        {"direction": "higher", "truth": 0.8, "threshold": 0.8},
        {"direction": "lower", "truth": 0.1, "threshold": 0.1},
    ],
    "valid_scenarios": [
        {"direction": "higher", "truth": 1.0, "threshold": 0.9},
        {"direction": "lower", "truth": 0.0, "threshold": 0.1},
    ],
    "signed_specificity_scenario": {
        "bounds": [-1.0, 1.0],
        "coverage_probabilities": [0.8, 0.2],
        "invalid_threshold": 0.6,
        "valid_probabilities": [1.0, 0.0],
        "valid_threshold": 0.8
    },
}


def _draw(rng: random.Random, probability: float, count: int) -> List[float]:
    return [float(rng.random() < probability) for _ in range(count)]


def _one_seed(*, trials: int, units: int, seed: int) -> Dict[str, Any]:
    rng = random.Random(seed)
    coverage = {}
    for truth in CALIBRATION_METHOD["coverage_scenarios"]:
        covered = 0
        for _ in range(trials):
            interval = hoeffding_interval(_draw(rng, truth, units))
            covered += int(interval[0] <= truth <= interval[1])
        coverage[str(truth)] = {
            "covered": covered,
            "trials": trials,
            "rate": covered / trials,
        }
    invalid = {}
    for index, scenario in enumerate(CALIBRATION_METHOD["invalid_scenarios"]):
        passed = 0
        for _ in range(trials):
            check = threshold_check(
                _draw(rng, scenario["truth"], units),
                threshold=scenario["threshold"],
                direction=scenario["direction"],
                minimum_units=units,
            )
            passed += int(check["state"] == "pass")
        invalid[str(index)] = {
            **scenario, "false_passes": passed, "trials": trials,
            "false_pass_rate": passed / trials,
        }
    valid = {}
    for index, scenario in enumerate(CALIBRATION_METHOD["valid_scenarios"]):
        passed = 0
        for _ in range(trials):
            check = threshold_check(
                _draw(rng, scenario["truth"], units),
                threshold=scenario["threshold"],
                direction=scenario["direction"],
                minimum_units=units,
            )
            passed += int(check["state"] == "pass")
        valid[str(index)] = {
            **scenario, "passes": passed, "trials": trials,
            "power": passed / trials,
        }
    signed = CALIBRATION_METHOD["signed_specificity_scenario"]
    left, right = signed["coverage_probabilities"]
    truth = left - right
    signed_covered = 0
    signed_false_passes = 0
    signed_power = 0
    for _ in range(trials):
        differences = [a - b for a, b in zip(
            _draw(rng, left, units), _draw(rng, right, units)
        )]
        interval = hoeffding_interval(differences, bounds=(-1.0, 1.0))
        signed_covered += int(interval[0] <= truth <= interval[1])
        invalid_check = threshold_check(
            differences, threshold=signed["invalid_threshold"],
            direction="higher", bounds=(-1.0, 1.0), minimum_units=units,
        )
        signed_false_passes += int(invalid_check["state"] == "pass")
        valid_differences = [a - b for a, b in zip(
            _draw(rng, signed["valid_probabilities"][0], units),
            _draw(rng, signed["valid_probabilities"][1], units),
        )]
        valid_check = threshold_check(
            valid_differences, threshold=signed["valid_threshold"],
            direction="higher", bounds=(-1.0, 1.0), minimum_units=units,
        )
        signed_power += int(valid_check["state"] == "pass")
    return {
        "seed": seed,
        "coverage": coverage,
        "known_invalid": invalid,
        "known_valid_power": valid,
        "signed_specificity": {
            "truth": truth,
            "coverage": signed_covered / trials,
            "covered": signed_covered,
            "false_pass_rate_at_boundary": signed_false_passes / trials,
            "false_passes_at_boundary": signed_false_passes,
            "valid_power": signed_power / trials,
            "valid_passes": signed_power,
            "trials": trials,
        },
    }


def run_profile_calibration(
    *, trials: int = 2000, units: int = 200, seed: int = 20260901
) -> Dict[str, Any]:
    """Run frozen core scenarios and disjoint fresh-seed replication."""
    if trials < 100:
        raise ValueError("calibration needs at least 100 trials")
    if units < 2:
        raise ValueError("calibration needs at least 2 independent units")
    primary = _one_seed(trials=trials, units=units, seed=seed)
    replication = _one_seed(trials=trials, units=units, seed=seed + 1)
    all_coverage = [
        row["rate"] for result in (primary, replication)
        for row in result["coverage"].values()
    ]
    all_false_pass = [
        row["false_pass_rate"] for result in (primary, replication)
        for row in result["known_invalid"].values()
    ]
    all_coverage.extend(
        result["signed_specificity"]["coverage"]
        for result in (primary, replication)
    )
    all_false_pass.extend(
        result["signed_specificity"]["false_pass_rate_at_boundary"]
        for result in (primary, replication)
    )
    artifact = {
        "artifact": "stresskit_v1_profile_calibration",
        "schema_version": "1.0",
        "method": CALIBRATION_METHOD,
        "method_digest": digest_json(CALIBRATION_METHOD),
        "profile_registry_digest": PROFILE_REGISTRY_DIGEST,
        "profiles": {
            key: {
                "profile_digest": profile.digest,
                "minimum_independent_units": profile.minimum_independent_units,
                "minimum_control_units": profile.minimum_control_units,
                "reported_minimum_detectable_margin": profile.minimum_detectable_margin,
            }
            for key, profile in sorted(PROFILE_REGISTRY.items())
        },
        "trials_per_scenario": trials,
        "independent_units_per_trial": units,
        "primary": primary,
        "fresh_seed_replication": replication,
        "acceptance": {
            "minimum_coverage": 0.93,
            "maximum_known_invalid_false_pass_rate": 0.05,
            "observed_minimum_coverage": min(all_coverage),
            "observed_maximum_known_invalid_false_pass_rate": max(all_false_pass),
            "passed": min(all_coverage) >= 0.93 and max(all_false_pass) <= 0.05,
        },
        "interpretation": (
            "Calibration covers bounded independent-unit inference shared by all "
            "registered reducers. It does not validate claim construction, null "
            "construct validity, executor isolation, or external-task choice."
        ),
    }
    return artifact


def main(argv=None) -> int:
    """Print deterministic calibration JSON."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=2000)
    parser.add_argument("--units", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260901)
    args = parser.parse_args(argv)
    artifact = run_profile_calibration(
        trials=args.trials, units=args.units, seed=args.seed
    )
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0 if artifact["acceptance"]["passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
