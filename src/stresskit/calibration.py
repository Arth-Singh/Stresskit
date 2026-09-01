"""Known-truth calibration for pairwise structural-agreement intervals.

This module deliberately separates calibration from benchmark evaluation.
Every scenario has an exact finite-population target, and every trial seed is
derived from its full cell identity so jobs can be deterministically sharded.
The first release covers structural scenarios S1--S5 from
``docs/VALIDATION_PLAN.md``; later modules cover specificity, scores,
interactions, and dependence.
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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .battery import decision_state
from .metrics import (
    bootstrap_ci_pairwise,
    bootstrap_bca_ci_pairwise,
    exact_expected_core_noise_jaccard,
    exact_expected_random_jaccard,
    hoeffding_ci_pairwise,
    jackknife_normal_ci_pairwise,
    jaccard,
    mean_pairwise_jaccard,
    nguyen_ci_pairwise,
    paired_mean_pairwise,
    u_normal_ci_pairwise,
)


INTERVAL_METHODS = (
    "bootstrap_percentile",
    "bootstrap_bca",
    "jackknife_normal",
    "paired_hoeffding",
    "nguyen_concentration",
    "u_normal_unbiased_variance",
)
SCENARIO_KINDS = (
    "stable",
    "uniform",
    "core_noise",
    "two_mode",
    "heterogeneous_uniform",
)


@dataclass(frozen=True)
class StructuralScenario:
    """Specification for a structural known-truth data-generating process."""

    name: str
    kind: str
    universe_size: int
    finding_size: int
    core_size: int = 0
    alternate_size: Optional[int] = None
    mixture_probability: float = 0.5
    mode_overlap: int = 0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("scenario name must be nonempty")
        if self.kind not in SCENARIO_KINDS:
            raise ValueError(
                f"unknown scenario kind {self.kind!r}; expected one of {SCENARIO_KINDS}"
            )
        if self.universe_size <= 0:
            raise ValueError("universe_size must be positive")
        if not 0 <= self.finding_size <= self.universe_size:
            raise ValueError("finding_size must lie in [0, universe_size]")
        if not 0.0 <= self.mixture_probability <= 1.0:
            raise ValueError("mixture_probability must lie in [0, 1]")
        if self.kind == "core_noise" and not 0 <= self.core_size <= self.finding_size:
            raise ValueError("core_size must lie in [0, finding_size]")
        if self.kind == "two_mode":
            if not 0 <= self.mode_overlap <= self.finding_size:
                raise ValueError("mode_overlap must lie in [0, finding_size]")
            union_size = 2 * self.finding_size - self.mode_overlap
            if union_size > self.universe_size:
                raise ValueError(
                    "two mode sets do not fit in universe: "
                    f"union_size={union_size}, universe_size={self.universe_size}"
                )
        if self.kind == "heterogeneous_uniform":
            if self.alternate_size is None:
                raise ValueError("heterogeneous_uniform requires alternate_size")
            if not 0 <= self.alternate_size <= self.universe_size:
                raise ValueError("alternate_size must lie in [0, universe_size]")

    def true_mean_jaccard(self) -> float:
        """Exact target for two independent draws from this scenario."""
        if self.kind == "stable":
            return 1.0
        if self.kind == "uniform":
            value = exact_expected_random_jaccard(
                self.finding_size, self.universe_size
            )
        elif self.kind == "core_noise":
            value = exact_expected_core_noise_jaccard(
                self.core_size,
                self.finding_size - self.core_size,
                self.universe_size,
            )
        elif self.kind == "two_mode":
            p = self.mixture_probability
            between = self.mode_overlap / (
                2 * self.finding_size - self.mode_overlap
            ) if self.finding_size else 1.0
            return (p * p + (1.0 - p) ** 2) + 2.0 * p * (1.0 - p) * between
        else:
            assert self.alternate_size is not None
            p = self.mixture_probability
            same_primary = exact_expected_random_jaccard(
                self.finding_size, self.universe_size
            )
            cross = exact_expected_random_jaccard(
                self.finding_size, self.universe_size, self.alternate_size
            )
            same_alternate = exact_expected_random_jaccard(
                self.alternate_size, self.universe_size
            )
            assert same_primary is not None and cross is not None
            assert same_alternate is not None
            return (
                p * p * same_primary
                + 2.0 * p * (1.0 - p) * cross
                + (1.0 - p) ** 2 * same_alternate
            )
        assert value is not None
        return value

    def draw(self, rng: random.Random) -> frozenset:
        """Draw one finding using only supplied deterministic RNG state."""
        if self.kind == "stable":
            return frozenset(range(self.finding_size))
        if self.kind == "uniform":
            return _uniform_subset(rng, self.universe_size, self.finding_size)
        if self.kind == "core_noise":
            core = frozenset(range(self.core_size))
            noise = rng.sample(
                range(self.core_size, self.universe_size),
                self.finding_size - self.core_size,
            )
            return core | frozenset(noise)
        if self.kind == "two_mode":
            if rng.random() < self.mixture_probability:
                return frozenset(range(self.finding_size))
            shared = range(self.mode_overlap)
            distinct = range(
                self.finding_size,
                2 * self.finding_size - self.mode_overlap,
            )
            return frozenset(shared) | frozenset(distinct)
        assert self.alternate_size is not None
        size = (
            self.finding_size
            if rng.random() < self.mixture_probability
            else self.alternate_size
        )
        return _uniform_subset(rng, self.universe_size, size)

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["true_mean_jaccard"] = self.true_mean_jaccard()
        return result


@dataclass(frozen=True)
class CalibrationResult:
    """Sufficient aggregates and derived diagnostics for one calibration cell."""

    scenario: Dict[str, Any]
    interval_method: str
    n_runs: int
    confidence_level: float
    threshold: Optional[float]
    master_seed: int
    trial_start: int
    trials_requested: int
    bootstrap_replicates: Optional[int]
    estimate_count: int
    estimate_sum: float
    squared_error_sum: float
    interval_valid_count: int
    coverage_count: int
    interval_width_sum: float
    pass_count: int
    fail_count: int
    inconclusive_count: int

    def to_dict(self) -> Dict[str, Any]:
        raw = asdict(self)
        truth = float(self.scenario["true_mean_jaccard"])
        raw["mean_estimate"] = (
            self.estimate_sum / self.estimate_count if self.estimate_count else None
        )
        raw["bias"] = (
            raw["mean_estimate"] - truth
            if raw["mean_estimate"] is not None
            else None
        )
        raw["rmse"] = (
            math.sqrt(self.squared_error_sum / self.estimate_count)
            if self.estimate_count
            else None
        )
        raw["interval_valid_rate"] = (
            self.interval_valid_count / self.trials_requested
            if self.trials_requested
            else None
        )
        raw["empirical_coverage"] = (
            self.coverage_count / self.interval_valid_count
            if self.interval_valid_count
            else None
        )
        raw["coverage_mcse"] = _binomial_mcse(
            self.coverage_count, self.interval_valid_count
        )
        raw["mean_interval_width"] = (
            self.interval_width_sum / self.interval_valid_count
            if self.interval_valid_count
            else None
        )
        for state in ("pass", "fail", "inconclusive"):
            raw[f"{state}_rate"] = (
                getattr(self, f"{state}_count") / self.trials_requested
                if self.threshold is not None and self.trials_requested
                else None
            )
        if self.threshold is None:
            raw["false_pass_rate"] = None
            raw["false_fail_rate"] = None
        elif truth < self.threshold:
            raw["false_pass_rate"] = self.pass_count / self.trials_requested
            raw["false_fail_rate"] = None
        else:
            raw["false_pass_rate"] = None
            raw["false_fail_rate"] = self.fail_count / self.trials_requested
        return raw


def _uniform_subset(
    rng: random.Random, universe_size: int, size: int
) -> frozenset:
    return frozenset(rng.sample(range(universe_size), size))


def _binomial_mcse(successes: int, trials: int) -> Optional[float]:
    if trials <= 0:
        return None
    p = successes / trials
    return math.sqrt(p * (1.0 - p) / trials)


def _derived_seed(
    master_seed: int,
    scenario: StructuralScenario,
    n_runs: int,
    trial_index: int,
    stream: str,
) -> int:
    identity = {
        "master_seed": master_seed,
        "scenario": asdict(scenario),
        "n_runs": n_runs,
        "trial_index": trial_index,
        "stream": stream,
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big")


def _estimate_and_interval(
    findings: Sequence[frozenset],
    method: str,
    alpha: float,
    seed: int,
    bootstrap_replicates: int,
) -> Tuple[Optional[float], Optional[List[float]]]:
    if method == "bootstrap_percentile":
        return (
            mean_pairwise_jaccard(findings),
            bootstrap_ci_pairwise(
                findings,
                jaccard,
                n_boot=bootstrap_replicates,
                seed=seed,
                alpha=alpha,
            ),
        )
    if method == "bootstrap_bca":
        return (
            mean_pairwise_jaccard(findings),
            bootstrap_bca_ci_pairwise(
                findings,
                jaccard,
                n_boot=bootstrap_replicates,
                seed=seed,
                alpha=alpha,
            ),
        )
    if method == "jackknife_normal":
        return (
            mean_pairwise_jaccard(findings),
            jackknife_normal_ci_pairwise(findings, jaccard, alpha=alpha),
        )
    if method == "paired_hoeffding":
        return (
            paired_mean_pairwise(findings, jaccard, seed=seed),
            hoeffding_ci_pairwise(findings, jaccard, seed=seed, alpha=alpha),
        )
    if method == "nguyen_concentration":
        return (
            mean_pairwise_jaccard(findings),
            nguyen_ci_pairwise(findings, jaccard, alpha=alpha),
        )
    if method == "u_normal_unbiased_variance":
        return (
            mean_pairwise_jaccard(findings),
            u_normal_ci_pairwise(findings, jaccard, alpha=alpha),
        )
    raise ValueError(f"unknown interval method {method!r}; expected {INTERVAL_METHODS}")


def run_calibration_cell(
    scenario: StructuralScenario,
    interval_method: str,
    n_runs: int,
    n_trials: int,
    *,
    confidence_level: float = 0.95,
    threshold: Optional[float] = None,
    master_seed: int = 0,
    trial_start: int = 0,
    bootstrap_replicates: int = 500,
) -> CalibrationResult:
    """Run one reproducible, shardable known-truth calibration cell."""
    if interval_method not in INTERVAL_METHODS:
        raise ValueError(
            f"unknown interval method {interval_method!r}; expected {INTERVAL_METHODS}"
        )
    if n_runs < 4:
        raise ValueError("n_runs must be at least 4 for current interval methods")
    if n_trials <= 0:
        raise ValueError("n_trials must be positive")
    if trial_start < 0:
        raise ValueError("trial_start must be nonnegative")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie in (0, 1)")
    if interval_method in ("bootstrap_percentile", "bootstrap_bca") and bootstrap_replicates < 20:
        raise ValueError("bootstrap_replicates must be at least 20")

    truth = scenario.true_mean_jaccard()
    alpha = 1.0 - confidence_level
    estimate_sum = squared_error_sum = interval_width_sum = 0.0
    estimate_count = interval_valid_count = coverage_count = 0
    state_counts = {"pass": 0, "fail": 0, "inconclusive": 0}

    for trial_index in range(trial_start, trial_start + n_trials):
        sample_seed = _derived_seed(
            master_seed, scenario, n_runs, trial_index, "sample"
        )
        interval_seed = _derived_seed(
            master_seed, scenario, n_runs, trial_index, interval_method
        )
        rng = random.Random(sample_seed)
        findings = [scenario.draw(rng) for _ in range(n_runs)]
        estimate, interval = _estimate_and_interval(
            findings,
            interval_method,
            alpha,
            interval_seed,
            bootstrap_replicates,
        )
        assert estimate is not None
        estimate_count += 1
        estimate_sum += estimate
        squared_error_sum += (estimate - truth) ** 2
        if interval is not None:
            interval_valid_count += 1
            interval_width_sum += interval[1] - interval[0]
            coverage_count += int(interval[0] <= truth <= interval[1])
        if threshold is not None:
            state = decision_state(estimate, threshold, ">=", interval)
            state_counts[state] += 1

    return CalibrationResult(
        scenario=scenario.to_dict(),
        interval_method=interval_method,
        n_runs=n_runs,
        confidence_level=confidence_level,
        threshold=threshold,
        master_seed=master_seed,
        trial_start=trial_start,
        trials_requested=n_trials,
        bootstrap_replicates=(
            bootstrap_replicates
            if interval_method in ("bootstrap_percentile", "bootstrap_bca")
            else None
        ),
        estimate_count=estimate_count,
        estimate_sum=estimate_sum,
        squared_error_sum=squared_error_sum,
        interval_valid_count=interval_valid_count,
        coverage_count=coverage_count,
        interval_width_sum=interval_width_sum,
        pass_count=state_counts["pass"],
        fail_count=state_counts["fail"],
        inconclusive_count=state_counts["inconclusive"],
    )


def default_structural_scenarios() -> List[StructuralScenario]:
    """Small representative S1--S5 set for smoke tests and local pilots."""
    return [
        StructuralScenario("S1_stable", "stable", 100, 20),
        StructuralScenario("S2_uniform", "uniform", 100, 20),
        StructuralScenario("S3_core_noise", "core_noise", 100, 20, core_size=15),
        StructuralScenario(
            "S4_two_mode",
            "two_mode",
            100,
            20,
            mixture_probability=0.7,
            mode_overlap=5,
        ),
        StructuralScenario(
            "S5_heterogeneous",
            "heterogeneous_uniform",
            100,
            20,
            alternate_size=5,
            mixture_probability=0.5,
        ),
    ]


def run_calibration_grid(
    scenarios: Iterable[StructuralScenario],
    interval_methods: Sequence[str],
    run_counts: Sequence[int],
    n_trials: int,
    **kwargs: Any,
) -> List[CalibrationResult]:
    """Run Cartesian product of scenarios, methods, and run counts."""
    return [
        run_calibration_cell(
            scenario,
            method,
            n_runs,
            n_trials,
            **kwargs,
        )
        for scenario in scenarios
        for method in interval_methods
        for n_runs in run_counts
    ]


def _parse_int_list(value: str) -> List[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected comma-separated integers")
    return values


def calibration_source_digest() -> str:
    """Content digest of source files that determine calibration output."""
    package_dir = Path(__file__).parent
    digest = hashlib.sha256()
    for filename in ("battery.py", "calibration.py", "metrics.py"):
        path = package_dir / filename
        digest.update(filename.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run deterministic known-truth structural calibration"
    )
    parser.add_argument("--runs", type=_parse_int_list, default=[5, 10, 20])
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--trial-start", type=int, default=0)
    parser.add_argument("--bootstrap-replicates", type=int, default=300)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument(
        "--methods",
        default=",".join(INTERVAL_METHODS),
        help=f"comma-separated subset of {INTERVAL_METHODS}",
    )
    args = parser.parse_args(argv)
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    results = run_calibration_grid(
        default_structural_scenarios(),
        methods,
        args.runs,
        args.trials,
        confidence_level=args.confidence_level,
        threshold=args.threshold,
        master_seed=args.seed,
        trial_start=args.trial_start,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    payload = {
        "schema_version": "0.1",
        "study": "structural_known_truth_pilot",
        "configuration": {
            "run_counts": args.runs,
            "trials": args.trials,
            "trial_start": args.trial_start,
            "bootstrap_replicates": args.bootstrap_replicates,
            "confidence_level": args.confidence_level,
            "threshold": args.threshold,
            "master_seed": args.seed,
            "interval_methods": methods,
        },
        "provenance": {
            "source_sha256": calibration_source_digest(),
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        },
        "results": [result.to_dict() for result in results],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
