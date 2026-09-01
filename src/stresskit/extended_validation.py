"""Known-truth validation scenarios S6--S9.

S1--S5 live in :mod:`stresskit.calibration`. This module covers axis
interactions, real/null specificity, score-metric applicability, and clustered
dependence. Outputs contain additive counts and deterministic trial ranges so
frozen studies can be sharded without changing any draw.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .battery import decision_state
from .calibration import StructuralScenario
from .metrics import (
    cluster_hoeffding_ci_pairwise,
    hoeffding_ci_pairwise,
    hoeffding_difference_pairwise,
    jaccard,
    mean_pairwise_jaccard,
    paired_mean_pairwise,
    score_variation_assessment,
)
from .specification import SpecificationSpace


def _seed(master_seed: int, study: str, cell: MappingLike, trial: int) -> int:
    value = {
        "master_seed": master_seed,
        "study": study,
        "cell": cell,
        "trial": trial,
    }
    digest = hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big")


MappingLike = Dict[str, Any]


def _mcse(successes: int, trials: int) -> float:
    probability = successes / trials
    return math.sqrt(probability * (1.0 - probability) / trials)


def s6_interaction_counterexample() -> Dict[str, Any]:
    """Exact hidden-interaction example: OAT J=1 while crossed J=1/2."""
    space = SpecificationSpace(axes={"a": [0, 1], "b": [0, 1]})
    stable = frozenset({1, 2, 3})
    interaction = frozenset({4, 5, 6})

    def finding(configuration):
        return interaction if configuration == {"a": 1, "b": 1} else stable

    oat = [
        finding(row["configuration"])
        for row in space.diagnostic_oat_manifest({"a": 0, "b": 0})
    ]
    crossed = [
        finding(row["configuration"])
        for row in space.enumerate_manifest()
    ]
    return {
        "scenario": "S6_hidden_axis_interaction",
        "oat_run_count": len(oat),
        "crossed_run_count": len(crossed),
        "oat_mean_pairwise_jaccard": mean_pairwise_jaccard(oat),
        "crossed_mean_pairwise_jaccard": mean_pairwise_jaccard(crossed),
        "interaction_detected_only_when_crossed": True,
    }


def _specificity_cell(
    real: StructuralScenario,
    null: StructuralScenario,
    *,
    n_runs: int,
    n_trials: int,
    trial_start: int,
    master_seed: int,
    alpha: float,
    threshold: float,
) -> Dict[str, Any]:
    truth = real.true_mean_jaccard() - null.true_mean_jaccard()
    identity: MappingLike = {
        "real": real.to_dict(),
        "null": null.to_dict(),
        "n_runs": n_runs,
        "alpha": alpha,
        "threshold": threshold,
    }
    coverage = valid = pass_count = fail_count = inconclusive_count = 0
    width_sum = estimate_sum = 0.0
    for trial in range(trial_start, trial_start + n_trials):
        rng = random.Random(_seed(master_seed, "S7_specificity", identity, trial))
        real_sets = [real.draw(rng) for _ in range(n_runs)]
        null_sets = [null.draw(rng) for _ in range(n_runs)]
        result = hoeffding_difference_pairwise(
            real_sets,
            null_sets,
            jaccard,
            real_seed=_seed(master_seed, "S7_real_pairing", identity, trial),
            null_seed=_seed(master_seed, "S7_null_pairing", identity, trial),
            alpha=alpha,
        )
        assert result is not None
        interval = result["ci"]
        estimate = float(result["estimate"])
        valid += 1
        coverage += int(interval[0] <= truth <= interval[1])
        width_sum += interval[1] - interval[0]
        estimate_sum += estimate
        state = decision_state(estimate, threshold, ">=", interval)
        if state == "pass":
            pass_count += 1
        elif state == "fail":
            fail_count += 1
        else:
            inconclusive_count += 1
    return {
        "scenario": f"S7_{real.name}_vs_{null.name}",
        "true_difference": truth,
        "n_runs_per_group": n_runs,
        "alpha": alpha,
        "threshold": threshold,
        "trial_start": trial_start,
        "trials": n_trials,
        "coverage_count": coverage,
        "coverage": coverage / valid,
        "coverage_mcse": _mcse(coverage, valid),
        "mean_width": width_sum / valid,
        "mean_estimate": estimate_sum / valid,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "inconclusive_count": inconclusive_count,
        "false_pass_rate": pass_count / n_trials if truth < threshold else None,
        "false_fail_rate": fail_count / n_trials if truth >= threshold else None,
    }


def s8_score_applicability() -> List[Dict[str, Any]]:
    """Deterministic applicability controls for coefficient of variation."""
    scenarios = [
        ("positive_ratio", [8.0, 10.0, 12.0], "ratio", True),
        ("near_zero", [0.001, 0.002, 0.003], "ratio", False),
        ("signed", [-1.0, 0.0, 1.0], "signed", False),
        ("negative_ratio", [-1.0, 2.0, 3.0], "ratio", False),
        ("heavy_tailed_ratio", [1.0, 1.0, 1.0, 100.0], "ratio", True),
    ]
    rows = []
    for name, values, scale_type, expected in scenarios:
        result = score_variation_assessment(
            values, scale_type=scale_type, minimum_abs_mean=0.1
        )
        rows.append(
            {
                "scenario": f"S8_{name}",
                "expected_applicable": expected,
                **result,
                "matched_expectation": result["applicable"] is expected,
            }
        )
    return rows


def _cluster_cell(
    *,
    n_clusters: int,
    repeats_per_cluster: int,
    n_trials: int,
    trial_start: int,
    master_seed: int,
    alpha: float,
    mixture_probability: float,
    threshold: float,
) -> Dict[str, Any]:
    left = frozenset(range(20))
    right = frozenset(range(20, 40))
    truth = mixture_probability ** 2 + (1.0 - mixture_probability) ** 2
    identity: MappingLike = {
        "n_clusters": n_clusters,
        "repeats_per_cluster": repeats_per_cluster,
        "alpha": alpha,
        "mixture_probability": mixture_probability,
        "threshold": threshold,
    }
    cluster_cover = naive_cover = 0
    cluster_width = naive_width = 0.0
    cluster_false_pass = naive_false_pass = 0
    for trial in range(trial_start, trial_start + n_trials):
        rng = random.Random(_seed(master_seed, "S9_clusters", identity, trial))
        clusters = [
            [left if rng.random() < mixture_probability else right]
            * repeats_per_cluster
            for _ in range(n_clusters)
        ]
        flattened = [value for cluster in clusters for value in cluster]
        pairing_seed = _seed(master_seed, "S9_pairing", identity, trial)
        cluster_result = cluster_hoeffding_ci_pairwise(
            clusters, jaccard, seed=pairing_seed, alpha=alpha
        )
        naive_interval = hoeffding_ci_pairwise(
            flattened, jaccard, seed=pairing_seed, alpha=alpha
        )
        assert cluster_result is not None and naive_interval is not None
        cluster_interval = cluster_result["ci"]
        cluster_estimate = float(cluster_result["estimate"])
        naive_estimate = paired_mean_pairwise(
            flattened, jaccard, seed=pairing_seed
        )
        assert naive_estimate is not None
        cluster_cover += int(cluster_interval[0] <= truth <= cluster_interval[1])
        naive_cover += int(naive_interval[0] <= truth <= naive_interval[1])
        cluster_width += cluster_interval[1] - cluster_interval[0]
        naive_width += naive_interval[1] - naive_interval[0]
        cluster_false_pass += int(
            decision_state(cluster_estimate, threshold, ">=", cluster_interval)
            == "pass"
        )
        naive_false_pass += int(
            decision_state(naive_estimate, threshold, ">=", naive_interval)
            == "pass"
        )
    return {
        "scenario": "S9_repeated_runs_within_independent_cluster",
        "true_mean_jaccard": truth,
        "n_clusters": n_clusters,
        "repeats_per_cluster": repeats_per_cluster,
        "n_runs": n_clusters * repeats_per_cluster,
        "alpha": alpha,
        "threshold": threshold,
        "trial_start": trial_start,
        "trials": n_trials,
        "cluster_coverage_count": cluster_cover,
        "cluster_coverage": cluster_cover / n_trials,
        "cluster_coverage_mcse": _mcse(cluster_cover, n_trials),
        "cluster_mean_width": cluster_width / n_trials,
        "cluster_false_pass_rate": cluster_false_pass / n_trials,
        "naive_run_coverage_count": naive_cover,
        "naive_run_coverage": naive_cover / n_trials,
        "naive_run_coverage_mcse": _mcse(naive_cover, n_trials),
        "naive_run_mean_width": naive_width / n_trials,
        "naive_run_false_pass_rate": naive_false_pass / n_trials,
    }


def extended_validation_source_digest() -> str:
    package = Path(__file__).parent
    digest = hashlib.sha256()
    for name in (
        "battery.py",
        "calibration.py",
        "extended_validation.py",
        "metrics.py",
        "specification.py",
    ):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update((package / name).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def run_extended_validation(
    *,
    n_trials: int,
    trial_start: int = 0,
    run_counts: Sequence[int] = (100, 200, 400),
    cluster_counts: Sequence[int] = (20, 50, 100),
    repeats_per_cluster: int = 20,
    confidence_level: float = 0.95,
    threshold: float = 0.8,
    specificity_threshold: float = 0.2,
    master_seed: int = 20260824,
) -> Dict[str, Any]:
    if n_trials <= 0 or trial_start < 0:
        raise ValueError("n_trials must be positive and trial_start nonnegative")
    alpha = 1.0 - confidence_level
    stable = StructuralScenario("stable", "stable", 100, 20)
    uniform = StructuralScenario("uniform", "uniform", 100, 20)
    specificity = []
    for n_runs in run_counts:
        specificity.extend(
            [
                _specificity_cell(
                    stable,
                    stable,
                    n_runs=n_runs,
                    n_trials=n_trials,
                    trial_start=trial_start,
                    master_seed=master_seed,
                    alpha=alpha,
                    threshold=specificity_threshold,
                ),
                _specificity_cell(
                    stable,
                    uniform,
                    n_runs=n_runs,
                    n_trials=n_trials,
                    trial_start=trial_start,
                    master_seed=master_seed,
                    alpha=alpha,
                    threshold=specificity_threshold,
                ),
            ]
        )
    dependence = [
        _cluster_cell(
            n_clusters=count,
            repeats_per_cluster=repeats_per_cluster,
            n_trials=n_trials,
            trial_start=trial_start,
            master_seed=master_seed,
            alpha=alpha,
            mixture_probability=0.7,
            threshold=threshold,
        )
        for count in cluster_counts
    ]
    return {
        "schema_version": "0.1",
        "study": "known_truth_S6_S9",
        "configuration": {
            "n_trials": n_trials,
            "trial_start": trial_start,
            "run_counts": list(run_counts),
            "cluster_counts": list(cluster_counts),
            "repeats_per_cluster": repeats_per_cluster,
            "confidence_level": confidence_level,
            "threshold": threshold,
            "specificity_threshold": specificity_threshold,
            "master_seed": master_seed,
        },
        "provenance": {
            "source_sha256": extended_validation_source_digest(),
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        },
        "S6_interaction": s6_interaction_counterexample(),
        "S7_specificity": specificity,
        "S8_scores": s8_score_applicability(),
        "S9_dependence": dependence,
    }


def _int_list(value: str) -> List[int]:
    result = [int(item) for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("expected comma-separated integers")
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run StressKit S6--S9 validation")
    parser.add_argument("--trials", type=int, default=2000)
    parser.add_argument("--trial-start", type=int, default=0)
    parser.add_argument("--runs", type=_int_list, default=[100, 200, 400])
    parser.add_argument("--clusters", type=_int_list, default=[20, 50, 100])
    parser.add_argument("--repeats-per-cluster", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260824)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            run_extended_validation(
                n_trials=args.trials,
                trial_start=args.trial_start,
                run_counts=args.runs,
                cluster_counts=args.clusters,
                repeats_per_cluster=args.repeats_per_cluster,
                master_seed=args.seed,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
