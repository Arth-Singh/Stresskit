"""End-to-end known-truth calibration of the whole StressKit battery.

:mod:`stresskit.calibration` calibrates the structural interval in isolation.
This module plants exact truths for all five checks -- structural stability,
claim stability, score stability, beats-random, specificity -- draws Finding
lists from them, runs the lists through :func:`stresskit.from_findings`, and
counts how often each check's interval covers its truth, how often the point
rule and the three-state rule decide wrongly, and how often the letter grade
under grade rules v0.3 and v0.4 equals the truth grade.

Every trial seed derives from the full cell identity, so shards over disjoint
trial ranges reproduce the same draws as one long run and merge additively.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .battery import Thresholds, from_findings, grade_checks, make_check
from .calibration import StructuralScenario, _binomial_mcse
from .extended_validation import _seed
from .finding import Finding, feature_set
from .metrics import exact_expected_random_jaccard


CHECKS = (
    "structural_stability",
    "claim_stability",
    "score_stability",
    "beats_random",
    "specificity",
)
CHECK_OPS = {
    "structural_stability": ">=",
    "claim_stability": ">=",
    "score_stability": "<=",
    "beats_random": ">=",
    "specificity": ">=",
}
CHECK_THRESHOLD_FIELDS = {
    "structural_stability": "jaccard",
    "claim_stability": "modal_share",
    "score_stability": "score_cv",
    "beats_random": "random_margin",
    "specificity": "specificity_ratio",
}
STATES = ("pass", "fail", "inconclusive")
GRADES = ("A", "B", "C", "D")
CONFIDENCES = ("high", "low", "unknown")
GRADE_RULE_TAGS = {"v0.3": "v03", "v0.4": "v04"}
STUDY = "battery_known_truth"
SCHEMA_VERSION = "0.1"
DEFAULT_RUN_COUNTS = (6, 10, 20, 40, 100)
RANDOM_KINDS = ("uniform", "heterogeneous_uniform")


def truth_checks_from(
    truths: Dict[str, Optional[float]], thresholds: Thresholds
) -> Dict[str, Dict[str, Any]]:
    """Decided check dicts built from exact truths.

    A truth is exact, so its check is decided by the point comparison: the
    state is ``pass`` or ``fail``, never ``inconclusive``.  Checks whose truth
    is ``None`` (no null control, hence no specificity) are absent, exactly as
    they are absent from a battery that was never given null findings.
    """
    checks: Dict[str, Dict[str, Any]] = {}
    for name in CHECKS:
        truth = truths.get(name)
        if truth is None:
            continue
        bar = getattr(thresholds, CHECK_THRESHOLD_FIELDS[name])
        check = make_check(truth, bar, CHECK_OPS[name], "", ci=None)
        check["state"] = "pass" if check["passed"] else "fail"
        checks[name] = check
    return checks


@dataclass(frozen=True)
class BatteryScenario:
    """Known-truth data-generating process for one whole-battery cell.

    ``structure`` fixes the component sets of the real runs; the claim of each
    run is ``"c0"`` with probability ``claim_modal_probability`` and otherwise
    uniform over the remaining ``n_claim_classes - 1`` labels; the score is
    Normal(``score_mean``, (``score_cv`` * ``score_mean``)^2).  ``null`` and
    the ``null_*`` fields play the same roles for the null-control runs.
    """

    name: str
    structure: StructuralScenario
    claim_modal_probability: float = 1.0
    n_claim_classes: int = 3
    score_mean: float = 1.0
    score_cv: float = 0.0
    null: Optional[StructuralScenario] = None
    null_claim_modal_probability: float = 1.0
    null_score_mean: float = 1.0
    null_score_cv: float = 0.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("battery scenario name must be nonempty")
        if self.n_claim_classes < 2:
            raise ValueError(
                "n_claim_classes must be at least 2 so the modal share has a "
                f"population value, got {self.n_claim_classes}"
            )
        for label, probability in (
            ("claim_modal_probability", self.claim_modal_probability),
            ("null_claim_modal_probability", self.null_claim_modal_probability),
        ):
            if not 0.0 <= probability <= 1.0:
                raise ValueError(f"{label} must lie in [0, 1], got {probability}")
        for label, cv in (
            ("score_cv", self.score_cv),
            ("null_score_cv", self.null_score_cv),
        ):
            if cv < 0.0:
                raise ValueError(f"{label} must be nonnegative, got {cv}")
        for label, mean in (
            ("score_mean", self.score_mean),
            ("null_score_mean", self.null_score_mean),
        ):
            if mean == 0.0:
                raise ValueError(
                    f"{label} must be nonzero: the coefficient of variation is "
                    "undefined at mean zero"
                )
        if (
            self.null is not None
            and self.null.universe_size != self.structure.universe_size
        ):
            raise ValueError(
                "null and real structures must share one universe_size, got "
                f"{self.null.universe_size} and {self.structure.universe_size}"
            )

    def truths(self) -> Dict[str, Optional[float]]:
        """Exact population value of every check's statistic.

        ``beats_random`` is the structural truth over the exact size-matched
        random expectation for the fixed-size kinds.  For the uniform kinds
        the scenario *is* the size-matched random null: every pair of runs is
        a pair of independent uniform subsets of exactly the sizes the harness
        conditions its Monte-Carlo null on, so the population ratio is 1.
        """
        structural = self.structure.true_mean_jaccard()
        p = self.claim_modal_probability
        claim = max(p, (1.0 - p) / (self.n_claim_classes - 1))
        if self.structure.kind in RANDOM_KINDS:
            beats_random = 1.0
        else:
            random_null = exact_expected_random_jaccard(
                self.structure.finding_size, self.structure.universe_size
            )
            if random_null is None or random_null <= 0.0:
                raise ValueError(
                    "size-matched random expectation is undefined for "
                    f"k={self.structure.finding_size}, "
                    f"N={self.structure.universe_size}"
                )
            beats_random = structural / random_null
        specificity = (
            structural / self.null.true_mean_jaccard()
            if self.null is not None
            else None
        )
        return {
            "structural_stability": structural,
            "claim_stability": claim,
            "score_stability": self.score_cv,
            "beats_random": beats_random,
            "specificity": specificity,
        }

    def truth_checks(self, thresholds: Thresholds) -> Dict[str, Dict[str, Any]]:
        return truth_checks_from(self.truths(), thresholds)

    def truth_grade(self, rule: str, thresholds: Thresholds) -> str:
        return grade_checks(
            self.truth_checks(thresholds),
            rule=rule,
            random_floor=thresholds.random_floor,
        )

    def draw(
        self, rng: random.Random, n_runs: int, *, null: bool = False
    ) -> List[Finding]:
        """Draw ``n_runs`` findings from the real (or null) process."""
        if null:
            if self.null is None:
                raise ValueError(
                    f"scenario {self.name!r} has no null structure to draw from"
                )
            structure = self.null
            modal_probability = self.null_claim_modal_probability
            score_mean = self.null_score_mean
            score_cv = self.null_score_cv
        else:
            structure = self.structure
            modal_probability = self.claim_modal_probability
            score_mean = self.score_mean
            score_cv = self.score_cv
        findings = []
        for _ in range(n_runs):
            components = structure.draw(rng)
            if rng.random() < modal_probability:
                claim = "c0"
            else:
                claim = f"c{rng.randrange(1, self.n_claim_classes)}"
            score = rng.gauss(score_mean, score_cv * abs(score_mean))
            findings.append(
                feature_set(
                    components,
                    claim=claim,
                    score=score,
                    universe_size=structure.universe_size,
                )
            )
        return findings

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "structure": self.structure.to_dict(),
            "claim_modal_probability": self.claim_modal_probability,
            "n_claim_classes": self.n_claim_classes,
            "score_mean": self.score_mean,
            "score_cv": self.score_cv,
            "null": self.null.to_dict() if self.null is not None else None,
            "null_claim_modal_probability": self.null_claim_modal_probability,
            "null_score_mean": self.null_score_mean,
            "null_score_cv": self.null_score_cv,
        }


def _rate(successes: int, trials: int) -> Optional[float]:
    return successes / trials if trials else None


@dataclass(frozen=True)
class BatteryCalibrationResult:
    """Additive counters and sums for one (scenario, n_runs) cell.

    ``counts`` and ``sums`` are flat, key-wise additive, and always carry the
    full key set, so disjoint trial shards merge without knowing the schema.
    Every rate is derived from them in :meth:`to_dict`; none is stored.
    """

    scenario: Dict[str, Any]
    n_runs: int
    thresholds: Dict[str, float]
    master_seed: int
    trial_start: int
    trials_requested: int
    truth_grade_v03: str
    truth_grade_v04: str
    truths: Dict[str, Optional[float]]
    counts: Dict[str, int]
    sums: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        raw = asdict(self)
        raw["rates"] = self._rates()
        return raw

    def _rates(self) -> Dict[str, Any]:
        counts = self.counts
        done = counts["trials_done"]
        truth_checks = truth_checks_from(self.truths, Thresholds(**self.thresholds))
        rates: Dict[str, Any] = {
            "error_rate": _rate(counts["errors"], self.trials_requested),
            "confidence": {
                level: _rate(counts[f"conf:{level}"], done) for level in CONFIDENCES
            },
            "grade": {},
            "checks": {},
        }
        for tag in GRADE_RULE_TAGS.values():
            correct = counts[f"correct_{tag}"]
            rates["grade"][tag] = {
                "truth": getattr(self, f"truth_grade_{tag}"),
                "accuracy": _rate(correct, done),
                "accuracy_mcse": _binomial_mcse(correct, done),
                "distribution": {
                    grade: _rate(counts[f"grade_{tag}:{grade}"], done)
                    for grade in GRADES
                },
                "wrong_by_confidence": {
                    level: _rate(counts[f"wrong_{tag}&conf:{level}"], done)
                    for level in CONFIDENCES
                },
            }
        for name in CHECKS:
            truth = self.truths.get(name)
            applicable = counts[f"{name}:applicable"]
            finite = applicable - counts[f"{name}:nonfinite_value"]
            ci_available = counts[f"{name}:ci_available"]
            covered = counts[f"{name}:covered"]
            inconclusive = counts[f"{name}:state:inconclusive"]
            block: Dict[str, Any] = {
                "truth": truth,
                "applicable_rate": _rate(applicable, done),
                "ci_available_rate": _rate(ci_available, applicable),
                "coverage": _rate(covered, ci_available),
                "coverage_mcse": _binomial_mcse(covered, ci_available),
                "pass_point_rate": _rate(counts[f"{name}:passed_point"], applicable),
                "state_rates": {
                    state: _rate(counts[f"{name}:state:{state}"], applicable)
                    for state in STATES
                },
                "inconclusive_mcse": _binomial_mcse(inconclusive, applicable),
                "nonfinite_rate": _rate(counts[f"{name}:nonfinite_value"], applicable),
                "mean_estimate": (
                    self.sums[f"{name}:estimate_sum"] / finite if finite else None
                ),
                "mean_width": (
                    self.sums[f"{name}:width_sum"] / ci_available
                    if ci_available
                    else None
                ),
            }
            block["bias"] = (
                block["mean_estimate"] - truth
                if block["mean_estimate"] is not None and truth is not None
                else None
            )
            block["rmse"] = (
                math.sqrt(self.sums[f"{name}:squared_error_sum"] / finite)
                if finite and truth is not None
                else None
            )
            truth_check = truth_checks.get(name)
            for rule in ("point", "state"):
                for outcome in ("false_pass", "false_fail"):
                    wrong = counts[f"{name}:{outcome}_{rule}"]
                    defined = truth_check is not None and (
                        truth_check["passed"] == (outcome == "false_fail")
                    )
                    block[f"{outcome}_{rule}_rate"] = (
                        _rate(wrong, applicable) if defined else None
                    )
                    block[f"{outcome}_{rule}_mcse"] = (
                        _binomial_mcse(wrong, applicable) if defined else None
                    )
            rates["checks"][name] = block
        return rates


def _empty_counts() -> Dict[str, int]:
    counts: Dict[str, int] = {"trials_done": 0, "errors": 0}
    for tag in GRADE_RULE_TAGS.values():
        for grade in GRADES:
            counts[f"grade_{tag}:{grade}"] = 0
        counts[f"correct_{tag}"] = 0
        for level in CONFIDENCES:
            counts[f"wrong_{tag}&conf:{level}"] = 0
    for level in CONFIDENCES:
        counts[f"conf:{level}"] = 0
    for name in CHECKS:
        counts[f"{name}:applicable"] = 0
        counts[f"{name}:ci_available"] = 0
        counts[f"{name}:covered"] = 0
        counts[f"{name}:passed_point"] = 0
        counts[f"{name}:nonfinite_value"] = 0
        for state in STATES:
            counts[f"{name}:state:{state}"] = 0
        for outcome in ("false_pass", "false_fail"):
            for rule in ("point", "state"):
                counts[f"{name}:{outcome}_{rule}"] = 0
    return counts


def _empty_sums() -> Dict[str, float]:
    sums: Dict[str, float] = {}
    for name in CHECKS:
        sums[f"{name}:estimate_sum"] = 0.0
        sums[f"{name}:squared_error_sum"] = 0.0
        sums[f"{name}:width_sum"] = 0.0
    return sums


def run_battery_cell(
    scenario: BatteryScenario,
    n_runs: int,
    n_trials: int,
    *,
    thresholds: Optional[Thresholds] = None,
    master_seed: int = 0,
    trial_start: int = 0,
) -> BatteryCalibrationResult:
    """Run one reproducible, shardable whole-battery known-truth cell.

    Per trial the sample seed drives the real draws and then the null draws
    from one stream, and a separate battery seed goes to ``from_findings`` so
    the Monte-Carlo random null and every bootstrap move per trial exactly as
    they do in a real battery.  A trial whose battery raises ``ValueError`` is
    counted under ``errors`` and contributes nothing else.
    """
    if n_runs < 2:
        raise ValueError(f"n_runs must be at least 2 for from_findings, got {n_runs}")
    if n_trials <= 0:
        raise ValueError(f"n_trials must be positive, got {n_trials}")
    if trial_start < 0:
        raise ValueError(f"trial_start must be nonnegative, got {trial_start}")
    thresholds = thresholds or Thresholds()
    threshold_dict = asdict(thresholds)
    cell_identity = {
        "scenario": scenario.to_dict(),
        "n_runs": n_runs,
        "thresholds": threshold_dict,
    }
    truths = scenario.truths()
    truth_checks = scenario.truth_checks(thresholds)
    truth_grades = {
        rule: scenario.truth_grade(rule, thresholds) for rule in GRADE_RULE_TAGS
    }
    counts = _empty_counts()
    sums = _empty_sums()

    for trial in range(trial_start, trial_start + n_trials):
        sample_seed = _seed(master_seed, "battery_calibration", cell_identity, trial)
        battery_seed = _seed(master_seed, "battery_seed", cell_identity, trial)
        rng = random.Random(sample_seed)
        real = scenario.draw(rng, n_runs)
        null = (
            scenario.draw(rng, n_runs, null=True) if scenario.null is not None else None
        )
        try:
            result = from_findings(
                real, null_findings=null, thresholds=thresholds, seed=battery_seed
            )
        except ValueError:
            counts["errors"] += 1
            continue
        counts["trials_done"] += 1

        grades = {
            "v0.3": grade_checks(
                result.checks, rule="v0.3", random_floor=thresholds.random_floor
            ),
            "v0.4": result.grade,
        }
        confidence = result.pooled["confidence"]
        counts[f"conf:{confidence}"] += 1
        for rule, tag in GRADE_RULE_TAGS.items():
            counts[f"grade_{tag}:{grades[rule]}"] += 1
            if grades[rule] == truth_grades[rule]:
                counts[f"correct_{tag}"] += 1
            else:
                counts[f"wrong_{tag}&conf:{confidence}"] += 1

        for name in CHECKS:
            check = result.checks.get(name)
            if check is None:
                continue
            counts[f"{name}:applicable"] += 1
            value = check["value"]
            ci = check["ci"]
            state = check["state"]
            counts[f"{name}:state:{state}"] += 1
            if check["passed"]:
                counts[f"{name}:passed_point"] += 1
            if ci is not None:
                counts[f"{name}:ci_available"] += 1
                sums[f"{name}:width_sum"] += ci[1] - ci[0]
            if math.isfinite(value):
                sums[f"{name}:estimate_sum"] += value
            else:
                counts[f"{name}:nonfinite_value"] += 1
            truth = truths[name]
            if truth is None:
                continue
            if math.isfinite(value):
                sums[f"{name}:squared_error_sum"] += (value - truth) ** 2
            if ci is not None and ci[0] <= truth <= ci[1]:
                counts[f"{name}:covered"] += 1
            truth_passes = truth_checks[name]["passed"]
            if check["passed"] and not truth_passes:
                counts[f"{name}:false_pass_point"] += 1
            if not check["passed"] and truth_passes:
                counts[f"{name}:false_fail_point"] += 1
            if state == "pass" and not truth_passes:
                counts[f"{name}:false_pass_state"] += 1
            if state == "fail" and truth_passes:
                counts[f"{name}:false_fail_state"] += 1

    return BatteryCalibrationResult(
        scenario=scenario.to_dict(),
        n_runs=n_runs,
        thresholds=threshold_dict,
        master_seed=master_seed,
        trial_start=trial_start,
        trials_requested=n_trials,
        truth_grade_v03=truth_grades["v0.3"],
        truth_grade_v04=truth_grades["v0.4"],
        truths=truths,
        counts=counts,
        sums=sums,
    )


def _two_mode(mixture_probability: float, mode_overlap: int) -> StructuralScenario:
    return StructuralScenario(
        f"two_mode_p{mixture_probability}_overlap{mode_overlap}",
        "two_mode",
        500,
        20,
        mixture_probability=mixture_probability,
        mode_overlap=mode_overlap,
    )


def default_battery_scenarios() -> List[BatteryScenario]:
    """The 16 registered cells: N = 500, k = 20, K = 3, score mean 1.0.

    Structured real and null processes are two-mode mixtures rather than
    fixed-core sets: a fixed core with one or two resampled items makes almost
    every pairwise Jaccard identical, so the bootstrap interval collapses to a
    point that sits a hair off the exact truth and coverage measures that
    degeneracy instead of the interval.  Two modes with probability 0.5 give
    every pair a real chance of landing on either value.
    """
    stable = StructuralScenario("stable", "stable", 500, 20)
    uniform = StructuralScenario("uniform", "uniform", 500, 20)
    heterogeneous = StructuralScenario(
        "heterogeneous_20_12_p0.5",
        "heterogeneous_uniform",
        500,
        20,
        alternate_size=12,
        mixture_probability=0.5,
    )
    near_stable = _two_mode(0.5, 18)
    pass_edge = _two_mode(0.5, 16)
    fail_edge = _two_mode(0.5, 14)
    two_mode = _two_mode(0.7, 5)
    mixed = _two_mode(0.5, 8)
    at_bar_null = _two_mode(0.49, 7)

    def cell(
        name: str,
        structure: StructuralScenario,
        claim_p: float,
        score_cv: float,
        null: StructuralScenario,
    ) -> BatteryScenario:
        return BatteryScenario(
            name,
            structure,
            claim_modal_probability=claim_p,
            n_claim_classes=3,
            score_mean=1.0,
            score_cv=score_cv,
            null=null,
            null_claim_modal_probability=1.0,
            null_score_mean=1.0,
            null_score_cv=0.0,
        )

    return [
        cell("A_control", stable, 1.0, 0.02, uniform),
        cell("A_noisy", near_stable, 0.9, 0.10, uniform),
        cell("J_pass_edge", pass_edge, 0.95, 0.05, uniform),
        cell("J_fail_edge", fail_edge, 0.95, 0.05, uniform),
        cell("J_two_mode", two_mode, 0.95, 0.05, uniform),
        cell("claim_pass_edge", near_stable, 0.85, 0.05, uniform),
        cell("claim_at_bar", near_stable, 0.80, 0.05, uniform),
        cell("claim_fail_edge", near_stable, 0.70, 0.05, uniform),
        cell("cv_pass_edge", near_stable, 0.95, 0.20, uniform),
        cell("cv_fail_edge", near_stable, 0.95, 0.35, uniform),
        cell("spec_same_structure", near_stable, 0.95, 0.05, near_stable),
        cell("spec_at_bar", near_stable, 0.95, 0.05, at_bar_null),
        cell("spec_ranker", near_stable, 0.95, 0.05, pass_edge),
        cell("random_D", uniform, 0.95, 0.05, uniform),
        cell("mixed_C", mixed, 0.5, 0.5, mixed),
        cell("sizes_D", heterogeneous, 0.95, 0.05, uniform),
    ]


def run_battery_grid(
    scenarios: Iterable[BatteryScenario],
    run_counts: Sequence[int],
    n_trials: int,
    **kwargs: Any,
) -> List[BatteryCalibrationResult]:
    """Run the Cartesian product of scenarios and run counts."""
    return [
        run_battery_cell(scenario, n_runs, n_trials, **kwargs)
        for scenario in scenarios
        for n_runs in run_counts
    ]


def battery_calibration_source_digest() -> str:
    """Content digest of the source files that determine this study's output."""
    package_dir = Path(__file__).parent
    digest = hashlib.sha256()
    for filename in (
        "battery.py",
        "baselines.py",
        "calibration.py",
        "battery_calibration.py",
        "finding.py",
        "metrics.py",
        "card.py",
    ):
        path = package_dir / filename
        digest.update(filename.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _parse_int_list(value: str) -> List[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected comma-separated integers")
    return values


def select_cells(spec: str) -> List[BatteryScenario]:
    """Resolve ``all`` or a comma-separated list of registered cell names."""
    scenarios = default_battery_scenarios()
    if spec.strip() == "all":
        return scenarios
    by_name = {scenario.name: scenario for scenario in scenarios}
    requested = [item.strip() for item in spec.split(",") if item.strip()]
    if not requested:
        raise ValueError(
            "--cells must be 'all' or a comma-separated list of cell names"
        )
    unknown = [name for name in requested if name not in by_name]
    if unknown:
        raise ValueError(
            f"unknown cell name(s) {unknown}; expected a subset of {sorted(by_name)}"
        )
    return [by_name[name] for name in requested]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run deterministic whole-battery known-truth calibration"
    )
    parser.add_argument(
        "--runs", type=_parse_int_list, default=list(DEFAULT_RUN_COUNTS)
    )
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--trial-start", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument(
        "--cells",
        default="all",
        help="'all' or a comma-separated subset of the registered cell names",
    )
    parser.add_argument(
        "--list-cells",
        action="store_true",
        help="print the registered cell names, one per line, and exit",
    )
    parser.add_argument(
        "--time-one",
        action="store_true",
        help="time one trial per run count on the first requested cell and exit",
    )
    args = parser.parse_args(argv)
    if args.list_cells:
        for scenario in default_battery_scenarios():
            print(scenario.name)
        return 0
    try:
        scenarios = select_cells(args.cells)
    except ValueError as error:
        parser.error(str(error))
    thresholds = Thresholds()
    if args.time_one:
        timings: Dict[str, float] = {}
        for n_runs in args.runs:
            started = time.perf_counter()
            run_battery_cell(
                scenarios[0],
                n_runs,
                1,
                thresholds=thresholds,
                master_seed=args.seed,
                trial_start=args.trial_start,
            )
            timings[str(n_runs)] = time.perf_counter() - started
        print(json.dumps(timings, indent=2, sort_keys=True))
        return 0
    results = run_battery_grid(
        scenarios,
        args.runs,
        args.trials,
        thresholds=thresholds,
        master_seed=args.seed,
        trial_start=args.trial_start,
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "study": STUDY,
        "configuration": {
            "run_counts": args.runs,
            "trials": args.trials,
            "trial_start": args.trial_start,
            "master_seed": args.seed,
            "cells": [scenario.name for scenario in scenarios],
            "thresholds": asdict(thresholds),
        },
        "provenance": {
            "source_sha256": battery_calibration_source_digest(),
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "results": [result.to_dict() for result in results],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
