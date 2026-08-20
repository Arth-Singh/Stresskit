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

import hashlib
import json
import os
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


# --------------------------------------------------------------------------
# Run cache: skip finder calls whose (axis, variant, seed, config) already
# ran under the same user-asserted cache_key. The key is the user's promise
# that data and finding_fn are unchanged — StressKit cannot verify that, so
# it refuses to cache without one.
# --------------------------------------------------------------------------

def _cache_path(cache_dir: str, cache_key: str, axis: str, variant: str,
                run_seed: int, run_config: Mapping[str, Any]) -> str:
    payload = json.dumps(
        [cache_key, axis, variant, run_seed, dict(run_config)],
        sort_keys=True, default=str,
    )
    digest = hashlib.sha256(payload.encode()).hexdigest()[:24]
    return os.path.join(cache_dir, f"run_{digest}.json")


def _thaw(c: Any) -> Any:
    return tuple(_thaw(x) for x in c) if isinstance(c, list) else c


def _freeze(c: Any) -> Any:
    return [_freeze(x) for x in c] if isinstance(c, tuple) else c


def _cache_load(path: str) -> Optional[Finding]:
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return Finding(
        components=frozenset(_thaw(c) for c in d["components"]),
        claim=d["claim"],
        score=d["score"],
        universe_size=d["universe_size"],
        meta=d.get("meta", {}),
    )


def _cache_store(path: str, finding: Finding) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(
            {
                "components": sorted((_freeze(c) for c in finding.components),
                                     key=repr),
                "claim": finding.claim,
                "score": finding.score,
                "universe_size": finding.universe_size,
                "meta": finding.meta,
            },
            f, default=str,
        )
    os.replace(tmp, path)


@dataclass
class Thresholds:
    """Pass bars for the stability checks.

    Defaults follow published proposals: Jaccard ≥ 0.8 under resampling
    (arXiv:2510.00845) and modal share π* ≥ 0.8, i.e. filability at the
    loosest tolerance α = 0.2 (arXiv:2608.13754). ``random_margin`` requires
    structural overlap to beat the size-matched random null by that factor.
    ``specificity_ratio`` is the required degradation factor on null-control
    data (Adebayo-style sanity check: a method as stable on null data as on
    real data is finding its own artifacts, not the model's structure).
    """

    jaccard: float = 0.8
    modal_share: float = 0.8
    score_cv: float = 0.25
    random_margin: float = 3.0
    specificity_ratio: float = 1.5


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
    null_summary: Optional[Dict[str, Any]] = None
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


_UNSET = object()


def _universe_of(f: Finding) -> Any:
    """Optional component-universe label (``meta['universe']``).

    Jaccard between findings drawn from different universes (different
    datasets, different item namespaces) is undefined, not zero. Findings
    whose universe label differs from the base run's are excluded from
    structural comparison; claims and scores still pool.
    """
    return f.meta.get("universe")


def _summarize(findings: Sequence[Finding], claim_equiv=None,
               universe: Any = _UNSET) -> Dict[str, Any]:
    """Structural / claim / score summary for one group of findings.

    ``claim_equiv(a, b) -> bool`` optionally treats semantically equivalent
    claim phrasings as one claim class (essential for natural-language
    claims from oracles/verbalizers; see stresskit.judges). When
    ``universe`` is given, structural metrics use only findings with that
    universe label (see ``_universe_of``).
    """
    if universe is _UNSET:
        struct_f = [f for f in findings if f.has_structure()]
        n_cross_universe = 0
    else:
        struct_f = [f for f in findings
                    if f.has_structure() and _universe_of(f) == universe]
        n_cross_universe = sum(
            1 for f in findings if f.has_structure()) - len(struct_f)
    structured = [f.components for f in struct_f]
    labels = [f.claim for f in findings if f.claim is not None]
    scores = [f.score for f in findings if f.score is not None]
    sizes = [f.size for f in struct_f]

    if labels and claim_equiv is not None:
        ids = M.cluster_labels(labels, claim_equiv)
        reps: Dict[int, str] = {}
        for lab, cid in zip(labels, ids):
            reps.setdefault(cid, lab)
        class_labels: List[str] = [str(reps[cid]) for cid in ids]
    else:
        class_labels = list(labels)

    out: Dict[str, Any] = {
        "n_runs": len(findings),
        "mean_pairwise_jaccard": M.mean_pairwise_jaccard(structured),
        "min_pairwise_jaccard": (
            min(M.pairwise_jaccard(structured)) if len(structured) >= 2 else None
        ),
        "flip_rate": M.flip_rate(class_labels) if len(class_labels) >= 2 else None,
        "modal_share": M.modal_share(class_labels) if class_labels else None,
        "n_claim_classes": M.n_claim_classes(class_labels) if class_labels else 0,
        "claim_counts": dict(
            sorted(
                ((c, class_labels.count(c)) for c in set(class_labels)),
                key=lambda kv: -kv[1],
            )
        ) if class_labels else {},
        "score_mean": M.mean(scores),
        "score_std": M.std(scores),
        "score_cv": M.coefficient_of_variation(scores),
        "median_size": (statistics.median(sizes) if sizes else None),
    }
    if universe is not _UNSET and n_cross_universe:
        out["n_cross_universe_excluded"] = n_cross_universe
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
    claim_equiv=None,
    null_data: Any = None,
    claim_statement: Optional[str] = None,
    model: Optional[str] = None,
    task: Optional[str] = None,
    method: Optional[str] = None,
    cache_dir: Optional[str] = None,
    cache_key: Optional[str] = None,
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
    claim_equiv:
        Optional ``(a, b) -> bool`` judge treating equivalent claim phrasings
        as one class (see ``stresskit.judges``). Use this whenever claims are
        natural language rather than fixed labels.
    null_data:
        Optional control dataset where the effect should NOT exist (shuffled
        labels, scrambled prompts, unrelated corpus). StressKit re-runs the
        seeds/bootstrap axes on it and adds a *specificity* check: structural
        stability on real data must exceed stability on null data by
        ``thresholds.specificity_ratio``. A finder that is equally "stable"
        on null data is discovering its own artifacts (dead salmon).
    claim_statement / model / task / method:
        Metadata recorded on the Stability Card.
    cache_dir / cache_key:
        Enable the run cache: completed runs are written to ``cache_dir``
        and re-runs with the same (cache_key, axis, variant, seed, config)
        skip the finder call. ``cache_key`` is your assertion that the data
        and finding_fn are unchanged — bump it whenever either changes.
        Cached findings must have JSON-representable components (strings,
        numbers, or tuples thereof).
    """
    thresholds = thresholds or Thresholds()
    if cache_dir is not None and not cache_key:
        raise ValueError(
            "cache_dir requires cache_key= — a string that changes whenever "
            "your data or finding_fn changes. StressKit cannot fingerprint "
            "those for you, so it refuses to guess."
        )
    base_config: Dict[str, Any] = dict(config or {})
    battery = tuple(battery)
    for ax in battery:
        if ax not in KNOWN_AXES:
            raise ValueError(f"Unknown battery axis {ax!r}. Known: {KNOWN_AXES}")

    notes: List[str] = []
    runs: List[RunRecord] = []
    t0 = time.time()

    cache_hits = 0

    def run_one(axis: str, variant: str, run_seed: int, run_data: Any,
                run_config: Dict[str, Any]) -> Finding:
        nonlocal cache_hits
        path = None
        if cache_dir is not None:
            path = _cache_path(cache_dir, cache_key, axis, variant,
                               run_seed, run_config)
            cached = _cache_load(path)
            if cached is not None:
                cache_hits += 1
                runs.append(RunRecord(axis, variant, run_seed,
                                      dict(run_config), cached))
                if verbose:  # pragma: no cover - console output
                    print(f"[stresskit] {axis:<12} {variant:<24} (cached)")
                return cached
        finding = finding_fn(run_data, run_seed, dict(run_config))
        if not isinstance(finding, Finding):
            raise TypeError(
                f"finding_fn must return a stresskit.Finding, got "
                f"{type(finding).__name__}. Wrap your output with "
                f"stresskit.circuit(...), feature_set(...) or probe(...)."
            )
        if path is not None:
            _cache_store(path, finding)
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

    if cache_hits:
        notes.append(
            f"{cache_hits}/{len(runs)} runs restored from cache "
            f"(cache_key={cache_key!r})"
        )

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
    base_universe = _universe_of(base)
    for axis in axis_names:
        group = [base] + [r.finding for r in runs if r.axis == axis]
        axis_metrics[axis] = _summarize(group, claim_equiv, universe=base_universe)
        scores = [f.score for f in group if f.score is not None]
        if len(scores) >= 2:
            score_groups[axis] = scores

    all_findings = [r.finding for r in runs]
    pooled = _summarize(all_findings, claim_equiv, universe=base_universe)
    if pooled.get("n_cross_universe_excluded"):
        notes.append(
            f"{pooled['n_cross_universe_excluded']} run(s) from a different "
            "component universe (meta['universe']) excluded from structural "
            "comparison — Jaccard across universes is undefined; their claims "
            "and scores still count"
        )
    pooled["variance_shares"] = M.variance_shares(score_groups) if score_groups else {}

    # Guard the pooled structural metric against size-mismatched runs (a
    # top-k sweep from 200 to 800 edges mechanically bounds Jaccard): the
    # graded value uses only runs within 2x of the base finding's size.
    if base.has_structure():
        comparable = [
            f.components for f in all_findings
            if f.has_structure() and _universe_of(f) == base_universe
            and base.size / 2 <= f.size <= base.size * 2
        ]
        n_excluded = sum(
            1 for f in all_findings
            if f.has_structure() and _universe_of(f) == base_universe
        ) - len(comparable)
        if n_excluded > 0:
            pooled["mean_pairwise_jaccard_all_sizes"] = pooled["mean_pairwise_jaccard"]
            pooled["mean_pairwise_jaccard"] = M.mean_pairwise_jaccard(comparable)
            pooled["n_size_mismatched_excluded"] = n_excluded
            notes.append(
                f"structural stability graded on {len(comparable)} size-comparable "
                f"runs; {n_excluded} run(s) with >2x size difference excluded "
                f"(pooled-over-all value kept as mean_pairwise_jaccard_all_sizes)"
            )
    else:
        comparable = []

    # Bootstrap CIs on the headline stability metrics (resampling runs,
    # the unit used in arXiv:2608.13754).
    structured_for_ci = comparable if comparable else [
        f.components for f in all_findings
        if f.has_structure() and _universe_of(f) == base_universe
    ]
    pooled["mean_pairwise_jaccard_ci95"] = M.bootstrap_ci(
        structured_for_ci, M.mean_pairwise_jaccard, seed=seed
    )
    ci_labels = [f.claim for f in all_findings if f.claim is not None]
    if claim_equiv is not None and ci_labels:
        ci_labels = M.cluster_labels(ci_labels, claim_equiv)
    pooled["flip_rate_ci95"] = M.bootstrap_ci(ci_labels, M.flip_rate, seed=seed)

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

    # --- null-control (specificity) --------------------------------------------
    null_summary: Optional[Dict[str, Any]] = None
    if null_data is not None:
        null_axes = tuple(ax for ax in battery if ax in ("seeds", "bootstrap")) or ("seeds",)
        try:
            null_result = stress(
                finding_fn,
                null_data,
                battery=null_axes,
                n_runs=n_runs,
                seed=seed ^ 0x5EC,
                config=base_config,
                thresholds=thresholds,
                claim_equiv=claim_equiv,
                cache_dir=cache_dir,
                cache_key=f"{cache_key}|null" if cache_key else None,
                verbose=verbose,
            )
            null_summary = {
                k: null_result.pooled.get(k)
                for k in (
                    "n_runs", "mean_pairwise_jaccard", "flip_rate",
                    "modal_share", "score_mean", "score_cv", "median_size",
                )
            }
        except (ValueError, TypeError) as e:
            notes.append(f"null control skipped: finder failed on null_data ({e})")

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
    if null_summary is not None and j is not None:
        null_j = null_summary.get("mean_pairwise_jaccard")
        if null_j is not None:
            ratio = (j / null_j) if null_j > 1e-9 else float("inf")
            checks["specificity"] = {
                "value": ratio,
                "threshold": thresholds.specificity_ratio,
                "passed": ratio >= thresholds.specificity_ratio,
                "description": "structural stability on real vs null-control data (×)",
            }

    if not checks:
        raise ValueError(
            "Nothing to grade: no components, claims, or scores were present in "
            "any Finding. Give your findings at least one of components=, "
            "claim=, or score=."
        )

    grade = grade_checks(checks)

    result = StressResult(
        base=base,
        runs=runs,
        axis_metrics=axis_metrics,
        pooled=pooled,
        checks=checks,
        grade=grade,
        null_summary=null_summary,
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


def grade_checks(checks: Dict[str, Any]) -> str:
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
