"""The stress() engine: run a discovery function across a perturbation
multiverse and grade the stability of what comes back.

Design: one-at-a-time (OAT) sweeps around a base configuration. For each
axis in the battery, ONLY that axis varies while everything else stays at
base. This keeps run counts linear, keeps attribution honest ("the claim
flips when the prompt template changes"), and matches how the variance
papers isolate their effects. A fully crossed grid is future work.

Axes
----
- ``seeds``      vary the discovery seed on identical data/config
- ``bootstrap``  resample the dataset with replacement (finite-sample noise)
- ``templates``  swap in alternative datasets (prompt templates, paraphrases,
                 corpora) supplied as ``{label: dataset}``
- ``hyperparams`` sweep method knobs supplied as ``{param: [values]}``
                 (thresholds, ablation types, metrics, ...), one param at a
                 time against the base config
"""

from __future__ import annotations

import random
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from . import metrics as M
from .finding import Finding
from .card import StabilityCard, GRADE_ORDER

FindingFn = Callable[[Any, int, Dict[str, Any]], Finding]

DEFAULT_BATTERY: Tuple[str, ...] = ("seeds", "bootstrap")
KNOWN_AXES = ("seeds", "bootstrap", "templates", "hyperparams")


@dataclass
class Thresholds:
    """Pass bars for the stability checks.

    Defaults follow published proposals: Jaccard ≥ 0.8 under resampling
    (arXiv:2510.00845) and modal share π* ≥ 0.8, i.e. filability at the
    loosest tolerance α = 0.2 (arXiv:2608.13754). ``random_margin`` requires
    structural overlap to beat the size-matched random null by that factor.
    """

    jaccard: float = 0.8
    modal_share: float = 0.8
    score_cv: float = 0.25
    random_margin: float = 3.0


@dataclass
class RunRecord:
    axis: str          # "base" | "seeds" | "bootstrap" | "templates" | "hyperparams"
    variant: str       # human-readable descriptor, e.g. "seed=3", "threshold=0.1"
    seed: int
    config: Dict[str, Any]
    finding: Finding


@dataclass
class StressResult:
    base: Finding
    runs: List[RunRecord]
    axis_metrics: Dict[str, Dict[str, Any]]
    pooled: Dict[str, Any]
    checks: Dict[str, Any]
    grade: str
    card: StabilityCard = None  # type: ignore[assignment]

    def to_markdown(self) -> str:
        return self.card.to_markdown()

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        j = self.pooled.get("mean_pairwise_jaccard")
        f = self.pooled.get("flip_rate")
        return (
            f"StressResult(grade={self.grade!r}, runs={len(self.runs)}, "
            f"jaccard={j if j is None else round(j, 3)}, "
            f"flip_rate={f if f is None else round(f, 3)})"
        )


def _summarize(findings: Sequence[Finding]) -> Dict[str, Any]:
    """Structural / claim / score summary for one group of findings."""
    structured = [f.components for f in findings if f.has_structure()]
    labels = [f.claim for f in findings if f.claim is not None]
    scores = [f.score for f in findings if f.score is not None]
    sizes = [f.size for f in findings if f.has_structure()]

    out: Dict[str, Any] = {
        "n_runs": len(findings),
        "mean_pairwise_jaccard": M.mean_pairwise_jaccard(structured),
        "min_pairwise_jaccard": (
            min(M.pairwise_jaccard(structured)) if len(structured) >= 2 else None
        ),
        "flip_rate": M.flip_rate(labels) if len(labels) >= 2 else None,
        "modal_share": M.modal_share(labels) if labels else None,
        "n_claim_classes": M.n_claim_classes(labels) if labels else 0,
        "claim_counts": dict(
            sorted(
                ((c, labels.count(c)) for c in set(labels)),
                key=lambda kv: -kv[1],
            )
        ) if labels else {},
        "score_mean": M.mean(scores),
        "score_std": M.std(scores),
        "score_cv": M.coefficient_of_variation(scores),
        "median_size": (statistics.median(sizes) if sizes else None),
    }
    return out


def _resolve_universe(findings: Sequence[Finding]) -> Optional[int]:
    for f in findings:
        if f.universe_size:
            return int(f.universe_size)
    return None


def stress(
    finding_fn: FindingFn,
    data: Any = None,
    *,
    battery: Sequence[str] = DEFAULT_BATTERY,
    n_runs: int = 10,
    seed: int = 0,
    config: Optional[Mapping[str, Any]] = None,
    templates: Optional[Mapping[str, Any]] = None,
    hyperparams: Optional[Mapping[str, Sequence[Any]]] = None,
    thresholds: Optional[Thresholds] = None,
    claim_statement: Optional[str] = None,
    model: Optional[str] = None,
    task: Optional[str] = None,
    method: Optional[str] = None,
    verbose: bool = False,
) -> StressResult:
    """Stress-test an interpretability finding.

    Parameters
    ----------
    finding_fn:
        ``finding_fn(data, seed, config) -> Finding``. Your discovery method,
        wrapped. It must be a pure function of its three arguments (fix all
        other randomness with the seed) or the harness measures your bugs,
        not your finding.
    data:
        The dataset the discovery runs on (any sequence for ``bootstrap``;
        anything your ``finding_fn`` accepts otherwise).
    battery:
        Which axes to sweep. Any of: "seeds", "bootstrap", "templates",
        "hyperparams". Axes without required inputs are skipped with a note.
    n_runs:
        Runs per axis for seeds/bootstrap.
    config:
        Base method configuration passed to every run.
    templates:
        ``{label: dataset}`` alternatives for the templates axis. The base
        dataset participates via the base run.
    hyperparams:
        ``{param_name: [alternative values]}`` for the hyperparams axis.
    claim_statement / model / task / method:
        Metadata recorded on the Stability Card.
    """
    thresholds = thresholds or Thresholds()
    base_config: Dict[str, Any] = dict(config or {})
    battery = tuple(battery)
    for ax in battery:
        if ax not in KNOWN_AXES:
            raise ValueError(f"Unknown battery axis {ax!r}. Known: {KNOWN_AXES}")

    notes: List[str] = []
    runs: List[RunRecord] = []
    t0 = time.time()

    def run_one(axis: str, variant: str, run_seed: int, run_data: Any,
                run_config: Dict[str, Any]) -> Finding:
        finding = finding_fn(run_data, run_seed, dict(run_config))
        if not isinstance(finding, Finding):
            raise TypeError(
                f"finding_fn must return a stresskit.Finding, got "
                f"{type(finding).__name__}. Wrap your output with "
                f"stresskit.circuit(...), feature_set(...) or probe(...)."
            )
        runs.append(RunRecord(axis, variant, run_seed, dict(run_config), finding))
        if verbose:  # pragma: no cover - console output
            print(f"[stresskit] {axis:<12} {variant:<24} "
                  f"size={finding.size} claim={finding.claim} score={finding.score}")
        return finding

    # --- base run -----------------------------------------------------------
    base = run_one("base", "base", seed, data, base_config)

    # --- seeds axis ----------------------------------------------------------
    if "seeds" in battery:
        for i in range(1, n_runs + 1):
            run_one("seeds", f"seed={seed + i}", seed + i, data, base_config)

    # --- bootstrap axis ------------------------------------------------------
    if "bootstrap" in battery:
        try:
            n_items = len(data)  # type: ignore[arg-type]
        except TypeError:
            n_items = 0
        if n_items < 2:
            notes.append("bootstrap axis skipped: data is not a sized sequence")
        else:
            rng = random.Random(seed ^ 0xB007)
            items = list(data)  # type: ignore[arg-type]
            for i in range(1, n_runs + 1):
                sample = [items[rng.randrange(n_items)] for _ in range(n_items)]
                run_one("bootstrap", f"resample={i}", seed, sample, base_config)

    # --- templates axis ------------------------------------------------------
    if "templates" in battery:
        if not templates:
            notes.append("templates axis skipped: no templates= provided")
        else:
            for label, tdata in templates.items():
                run_one("templates", f"template={label}", seed, tdata, base_config)

    # --- hyperparams axis ----------------------------------------------------
    if "hyperparams" in battery:
        if not hyperparams:
            notes.append("hyperparams axis skipped: no hyperparams= provided")
        else:
            for param, values in hyperparams.items():
                for value in values:
                    if param in base_config and base_config[param] == value:
                        continue  # identical to base run
                    cfg = dict(base_config)
                    cfg[param] = value
                    run_one("hyperparams", f"{param}={value}", seed, data, cfg)

    # --- vacuous seeds-axis detection ------------------------------------------
    seed_findings = [r.finding for r in runs if r.axis == "seeds"]
    if len(seed_findings) >= 2 and all(
        f.components == base.components
        and f.claim == base.claim
        and f.score == base.score
        for f in seed_findings
    ):
        notes.append(
            "seeds axis: every run returned a bit-identical finding — your "
            "finding_fn may ignore its seed (deterministic given data). The "
            "axis then measures nothing; use the seed inside your method "
            "(subsampling, init, tie-breaking) or drop 'seeds' from the battery."
        )

    # --- metrics --------------------------------------------------------------
    axis_names = sorted({r.axis for r in runs} - {"base"})
    axis_metrics: Dict[str, Dict[str, Any]] = {}
    score_groups: Dict[str, List[float]] = {}
    for axis in axis_names:
        group = [base] + [r.finding for r in runs if r.axis == axis]
        axis_metrics[axis] = _summarize(group)
        scores = [f.score for f in group if f.score is not None]
        if len(scores) >= 2:
            score_groups[axis] = scores

    all_findings = [r.finding for r in runs]
    pooled = _summarize(all_findings)
    pooled["variance_shares"] = M.variance_shares(score_groups) if score_groups else {}

    # --- size-matched random null ----------------------------------------------
    universe = _resolve_universe(all_findings)
    random_null = None
    if universe and pooled["median_size"]:
        random_null = M.expected_random_jaccard(pooled["median_size"], universe)
    pooled["expected_random_jaccard"] = random_null
    if random_null and pooled["mean_pairwise_jaccard"] is not None and random_null > 0:
        pooled["jaccard_vs_random"] = pooled["mean_pairwise_jaccard"] / random_null
    else:
        pooled["jaccard_vs_random"] = None

    # --- checks & grade ----------------------------------------------------------
    checks: Dict[str, Any] = {}
    j = pooled["mean_pairwise_jaccard"]
    if j is not None:
        checks["structural_stability"] = {
            "value": j,
            "threshold": thresholds.jaccard,
            "passed": j >= thresholds.jaccard,
            "description": "mean pairwise Jaccard across all perturbed runs",
        }
    ms = pooled["modal_share"]
    if ms is not None and pooled["n_runs"] >= 2 and pooled["n_claim_classes"] >= 1:
        checks["claim_stability"] = {
            "value": ms,
            "threshold": thresholds.modal_share,
            "passed": ms >= thresholds.modal_share,
            "description": "modal claim share π* (filability at α=0.2)",
        }
    cv = pooled["score_cv"]
    if cv is not None:
        checks["score_stability"] = {
            "value": cv,
            "threshold": thresholds.score_cv,
            "passed": cv <= thresholds.score_cv,
            "description": "coefficient of variation of the quality score",
        }
    vr = pooled["jaccard_vs_random"]
    if vr is not None:
        checks["beats_random"] = {
            "value": vr,
            "threshold": thresholds.random_margin,
            "passed": vr >= thresholds.random_margin,
            "description": "structural overlap vs size-matched random null (×)",
        }

    if not checks:
        raise ValueError(
            "Nothing to grade: no components, claims, or scores were present in "
            "any Finding. Give your findings at least one of components=, "
            "claim=, or score=."
        )

    grade = _grade(checks)

    result = StressResult(
        base=base,
        runs=runs,
        axis_metrics=axis_metrics,
        pooled=pooled,
        checks=checks,
        grade=grade,
    )
    result.card = StabilityCard.from_stress(
        result,
        battery=list(battery),
        n_runs=len(runs),
        seed=seed,
        base_config=base_config,
        thresholds=thresholds,
        claim_statement=claim_statement,
        model=model,
        task=task,
        method=method,
        notes=notes,
        wall_seconds=round(time.time() - t0, 3),
    )
    return result


def _grade(checks: Dict[str, Any]) -> str:
    """A: all applicable checks pass. B: at least half pass. C: at least one
    passes. D: none pass (or overlap is at the random null)."""
    passed = sum(1 for c in checks.values() if c["passed"])
    total = len(checks)
    br = checks.get("beats_random")
    at_random = br is not None and br["value"] is not None and br["value"] <= 1.5
    if passed == total:
        return "A"
    if at_random and passed == 0:
        return "D"
    if passed * 2 >= total:
        return "B"
    if passed >= 1:
        return "C"
    return "D"


assert set(GRADE_ORDER) == {"A", "B", "C", "D"}
