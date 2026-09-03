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
import math
import os
import random
import statistics
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from . import metrics as M
from . import baselines as B
from .finding import Finding
from .card import StabilityCard, GRADE_ORDER, GRADE_RULE, GRADE_RULES

FindingFn = Callable[[Any, int, Dict[str, Any]], Finding]

DEFAULT_BATTERY: Tuple[str, ...] = ("seeds", "bootstrap")
KNOWN_AXES = ("seeds", "bootstrap", "templates", "hyperparams")

# The pairwise |cosine| matrix of a direction battery is embedded on the card
# (making the structural metric, its CI, and the grade recomputable offline)
# up to this many pairs; beyond that only per-run dimensions and SHA-256
# digests are kept.
MAX_EMBED_DIRECTION_PAIRS = 20_000


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
        structure_present=d.get("structure_present"),
        vector=d.get("vector"),
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
                "structure_present": finding.has_structure(),
                **({"vector": list(finding.vector)}
                   if finding.has_direction() else {}),
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

    ``cosine`` is the structural bar for direction-valued findings (mean
    pairwise |cosine|), and it is a separate policy from ``jaccard`` even
    though it carries the same number. It is separate because the two grade
    different quantities and must be revisable independently. It is 0.8
    because |cos| is the direct analogue of Jaccard here — the fraction of
    one unit direction recovered by projecting it onto the other, as Jaccard
    is the fraction of a set shared — so the registered structural bar of
    arXiv:2510.00845 transfers unchanged rather than inventing a second
    number. A second
    reading of the same bar: at |cos| = 0.8 the residual disagreement
    ‖a − b‖ = √(2 − 2·0.8) ≈ 0.63 is already 63% of the direction's own
    length. The value was fixed by this analogy, not calibrated on any card.

    ``random_floor`` is the at-random floor: structural overlap at or below
    this multiple of the size-matched random null is graded D outright,
    whatever the other checks say. It was a literal 1.5 inside the grader
    before grade rule v0.4 made it a registered threshold.
    """

    jaccard: float = 0.8
    modal_share: float = 0.8
    score_cv: float = 0.25
    random_margin: float = 3.0
    specificity_ratio: float = 1.5
    cosine: float = 0.8
    random_floor: float = 1.5


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
    null_runs: Optional[List[RunRecord]] = None
    card: StabilityCard = None  # type: ignore[assignment]
    structure_kind: str = "set"

    def to_markdown(self) -> str:
        return self.card.to_markdown()

    def verdict_trace(self, **kwargs) -> Dict[str, Any]:
        """Verdict-stability trace over this result's own runs (see
        ``stresskit.verdict_trace``); null runs propagate when present."""
        return verdict_trace(
            [r.finding for r in self.runs],
            null_findings=(
                [r.finding for r in self.null_runs] if self.null_runs else None
            ),
            **kwargs,
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        f = self.pooled.get("flip_rate")
        if self.structure_kind == "direction":
            label = "abs_cosine"
            j = self.pooled.get("mean_pairwise_abs_cosine")
        else:
            label = "jaccard"
            j = self.pooled.get("mean_pairwise_jaccard")
        return (
            f"StressResult(grade={self.grade!r}, runs={len(self.runs)}, "
            f"{label}={j if j is None else round(j, 3)}, "
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


def _structure_kind(findings: Sequence[Finding], what: str) -> str:
    """The one structural kind a group of findings shares: ``"set"``,
    ``"direction"``, or ``"none"``.

    Sets are compared with Jaccard and directions with |cosine|; the two are
    not commensurable, so a group carrying both is a modelling error, not a
    number to average. Directions must also agree on dimension — vectors of
    different length live in different spaces and have no cosine at all.
    Findings that are purely claim- or score-valued carry no structure and
    join either group.
    """
    kinds = sorted({f.kind for f in findings} - {"none"})
    if len(kinds) > 1:
        counts = ", ".join(
            f"{k}: {sum(1 for f in findings if f.kind == k)}" for k in kinds
        )
        raise ValueError(
            f"{what} mixes structural kinds ({counts}). Set-valued findings "
            "are compared with Jaccard and direction-valued findings with "
            "|cosine|; there is no meaningful overlap between the two. Grade "
            "them as separate batteries."
        )
    kind = kinds[0] if kinds else "none"
    if kind == "direction":
        dims = sorted({f.dim for f in findings if f.has_direction()})
        if len(dims) > 1:
            raise ValueError(
                f"{what} mixes direction dimensions {dims}. Cosine is "
                "undefined between vectors of different length — these runs "
                "produced directions in different spaces (different models, "
                "or a changed residual width), so they cannot be pooled."
            )
    return kind


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
        dir_f = [f for f in findings if f.has_direction()]
        n_cross_universe = 0
    else:
        struct_f = [f for f in findings
                    if f.has_structure() and _universe_of(f) == universe]
        dir_f = [f for f in findings
                 if f.has_direction() and _universe_of(f) == universe]
        n_cross_universe = sum(
            1 for f in findings
            if f.has_structure() or f.has_direction()
        ) - len(struct_f) - len(dir_f)
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
        "n_structured_runs": len(struct_f),
        "n_empty_findings": sum(f.size == 0 for f in struct_f),
        "empty_finding_rate": (
            sum(f.size == 0 for f in struct_f) / len(struct_f)
            if struct_f else None
        ),
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
    if dir_f:
        pc = M.pairwise_abs_cosine([f.vector for f in dir_f])
        out["n_direction_runs"] = len(dir_f)
        out["direction_dim"] = dir_f[0].dim
        out["mean_pairwise_abs_cosine"] = (sum(pc) / len(pc)) if pc else None
        out["min_pairwise_abs_cosine"] = min(pc) if pc else None
    if universe is not _UNSET and n_cross_universe:
        out["n_cross_universe_excluded"] = n_cross_universe
    return out


def _direction_summary_keys(pooled: Mapping[str, Any]) -> Dict[str, Any]:
    """The direction-valued entries of a summary, empty for a set battery.

    Kept conditional so a set-valued card's null-control block is byte-for-byte
    what it has always been.
    """
    return {
        k: pooled[k] for k in
        ("n_direction_runs", "direction_dim",
         "mean_pairwise_abs_cosine", "min_pairwise_abs_cosine")
        if k in pooled
    }


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
        and f.vector == base.vector
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

    # --- vacuous bootstrap-axis detection --------------------------------------
    # Every resample runs at the base seed by design, so that the axis
    # isolates data variation. A finder that ignores its data therefore
    # repeats the base finding here and inflates every pooled stability
    # metric with identical runs.
    boot_findings = [r.finding for r in runs if r.axis == "bootstrap"]
    if len(boot_findings) >= 2 and all(
        f.components == base.components
        and f.vector == base.vector
        and f.claim == base.claim
        and f.score == base.score
        for f in boot_findings
    ):
        notes.append(
            "bootstrap axis: every resample returned a bit-identical finding — "
            "your finding_fn may ignore its data (the axis runs every resample "
            "at the base seed so that it isolates data variation). The axis "
            "then measures nothing and its runs inflate the pooled stability "
            "metrics; make the method read the data it is given or drop "
            "'bootstrap' from the battery."
        )

    # --- null-control (specificity) --------------------------------------------
    null_summary: Optional[Dict[str, Any]] = None
    null_components: Optional[List[frozenset]] = None
    null_vectors: Optional[List[Tuple[float, ...]]] = None
    null_runs: Optional[List[RunRecord]] = None
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
            null_summary.update(_direction_summary_keys(null_result.pooled))
            null_components = _graded_components(
                [r.finding for r in null_result.runs], null_result.base
            )
            null_vectors = _graded_vectors(
                [r.finding for r in null_result.runs], null_result.base
            )
            null_runs = null_result.runs
        except (ValueError, TypeError) as e:
            notes.append(f"null control skipped: finder failed on null_data ({e})")
    if null_runs is not None:
        # outside the try: a finder that returns a set on real data and a
        # direction on null data is a bug to surface, not a null to skip
        _structure_kind(
            [r.finding for r in runs] + [r.finding for r in null_runs],
            "the real runs and the null control together",
        )

    result = _analyze(
        runs, base,
        thresholds=thresholds, claim_equiv=claim_equiv, notes=notes,
        seed=seed, battery=battery, n_runs=n_runs, base_config=base_config,
        null_summary=null_summary, null_components=null_components,
        null_vectors=null_vectors,
        claim_statement=claim_statement,
        model=model, task=task, method=method, t0=t0,
    )
    result.null_runs = null_runs
    return result


def _graded_components(
    findings: Sequence[Finding], base: Finding
) -> List[frozenset]:
    """Component sets that structural grading actually uses: structured,
    same universe as the base, and (when the base has structure) within the
    2x size guard — falling back to all same-universe structured sets when
    the guard would leave nothing, mirroring ``_analyze``."""
    bu = _universe_of(base)
    structs = [
        f.components for f in findings
        if f.has_structure() and _universe_of(f) == bu
    ]
    if base.has_structure():
        sized = [c for c in structs
                 if base.size / 2 <= len(c) <= base.size * 2]
        return sized if sized else structs
    return structs


def _abs_cosine_matrix(
    vectors: Sequence[Sequence[float]]
) -> List[List[float]]:
    """Full symmetric |cosine| matrix with a unit diagonal.

    The card embeds this instead of the raw high-dimensional vectors: it is
    quadratic in run count rather than linear in model width, and everything
    the verdict rests on — the mean, the minimum, and the bootstrap CI, which
    resamples runs — is a function of it alone.
    """
    n = len(vectors)
    matrix = [[1.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            v = M.abs_cosine(vectors[i], vectors[j])
            matrix[i][j] = matrix[j][i] = v
    return matrix


def _graded_vectors(
    findings: Sequence[Finding], base: Finding
) -> List[Tuple[float, ...]]:
    """Direction vectors that structural grading actually uses: findings that
    carry a direction and share the base run's universe label. Directions
    have no size, so the 2x size guard of ``_graded_components`` has no
    analogue here."""
    bu = _universe_of(base)
    return [f.vector for f in findings
            if f.has_direction() and _universe_of(f) == bu]


def _analyze(
    runs: List[RunRecord],
    base: Finding,
    *,
    thresholds: Thresholds,
    claim_equiv,
    notes: List[str],
    seed: int,
    battery: Sequence[str],
    n_runs: int,
    base_config: Dict[str, Any],
    null_summary: Optional[Dict[str, Any]],
    null_components: Optional[List[frozenset]] = None,
    null_vectors: Optional[List[Tuple[float, ...]]] = None,
    claim_statement: Optional[str],
    model: Optional[str],
    task: Optional[str],
    method: Optional[str],
    t0: float,
) -> StressResult:
    """Metrics -> checks -> grade -> card, from completed run records.

    Shared by ``stress`` (which executes the runs) and ``from_findings``
    (post-hoc mode over findings the caller already has).
    """
    # --- metrics --------------------------------------------------------------
    kind = _structure_kind([r.finding for r in runs], "this battery")
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
    # absolute score variance per axis — shares alone hide whether "78% of
    # variance" is 78% of something real or 78% of near-zero noise.
    pooled["variance_absolute"] = {
        axis: (M.std(list(xs)) or 0.0) ** 2 for axis, xs in score_groups.items()
    }

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
    pooled["mean_pairwise_jaccard_ci95"] = M.bootstrap_ci_pairwise(
        structured_for_ci, M.jaccard, seed=seed
    )
    ci_labels = [f.claim for f in all_findings if f.claim is not None]
    if claim_equiv is not None and ci_labels:
        ci_labels = M.cluster_labels(ci_labels, claim_equiv)
    pooled["flip_rate_ci95"] = M.bootstrap_ci_pairwise(
        ci_labels, lambda a, b: float(a != b), seed=seed)
    pooled["modal_share_ci95"] = M.bootstrap_ci(ci_labels, M.modal_share, seed=seed)
    ci_scores = [f.score for f in all_findings if f.score is not None]
    pooled["score_cv_ci95"] = M.bootstrap_ci(
        ci_scores, M.coefficient_of_variation, seed=seed)

    # --- size-matched random null ----------------------------------------------
    # Prefer the Monte-Carlo null over the observed size distribution (exact
    # in expectation, heterogeneous sizes); keep the analytic k/(2N-k) as a
    # reported cross-check.
    universe = _resolve_universe(all_findings)
    graded_sizes = [len(s) for s in (comparable if comparable else structured_for_ci)]
    random_null = None
    if universe and graded_sizes:
        random_null = B.empirical_random_jaccard(graded_sizes, universe, seed=seed)
    pooled["expected_random_jaccard"] = random_null
    pooled["expected_random_jaccard_analytic"] = (
        M.expected_random_jaccard(pooled["median_size"], universe)
        if universe and pooled["median_size"] is not None else None
    )
    if random_null and pooled["mean_pairwise_jaccard"] is not None and random_null > 0:
        pooled["jaccard_vs_random"] = pooled["mean_pairwise_jaccard"] / random_null
    else:
        pooled["jaccard_vs_random"] = None

    # Axis-balanced Jaccard: mean of per-axis values, so a 20-run seeds axis
    # cannot drown a 2-run templates axis in the pooled number. Reported
    # alongside; a large gap to the pooled value means the battery's run
    # counts, not the finding, are shaping the headline metric.
    axis_js = [m["mean_pairwise_jaccard"] for m in axis_metrics.values()
               if m.get("mean_pairwise_jaccard") is not None]
    pooled["mean_pairwise_jaccard_axis_balanced"] = (
        sum(axis_js) / len(axis_js) if axis_js else None
    )
    pj, bj = pooled["mean_pairwise_jaccard"], pooled["mean_pairwise_jaccard_axis_balanced"]
    if pj is not None and bj is not None and abs(pj - bj) > 0.1:
        notes.append(
            f"pooled Jaccard ({pj:.3f}) and axis-balanced Jaccard ({bj:.3f}) "
            "diverge by more than 0.1 — per-axis run counts are shaping the "
            "pooled number; read the per-axis breakdown before citing it"
        )

    # --- direction-valued structure ------------------------------------------
    # Directions are graded with mean pairwise |cosine| instead of Jaccard.
    # The random null is the exact analytic E[|cos|] between independent
    # uniform unit vectors in R^d (metrics.expected_random_abs_cosine), which
    # is the expectation of the mean-pairwise statistic at any run count;
    # baselines.empirical_random_abs_cosine is its Monte-Carlo form and is
    # reported as the cross-check, the reverse of the Jaccard case where the
    # closed form is the approximation.
    graded_vectors: List[Tuple[float, ...]] = []
    cosines: List[List[float]] = []
    null_cosines: List[List[float]] = []
    directions_block: Optional[Dict[str, Any]] = None
    if kind == "direction":
        graded_vectors = _graded_vectors(all_findings, base)
        # Every interval below resamples runs, so it is a function of the
        # pairwise matrix alone. Building it once and indexing into it keeps
        # the bootstrap independent of model width — recomputing a 4096-dim
        # cosine inside 500 replicates costs minutes for the same numbers —
        # and is exactly what an auditor recomputes from the card.
        cosines = _abs_cosine_matrix(graded_vectors)
        null_cosines = (_abs_cosine_matrix(null_vectors)
                        if null_vectors else [])
        pooled["mean_pairwise_abs_cosine_ci95"] = M.bootstrap_ci_pairwise(
            range(len(cosines)), lambda a, b: cosines[a][b], seed=seed
        )
        dim = len(graded_vectors[0]) if graded_vectors else None
        random_cos = (
            M.expected_random_abs_cosine(dim) if dim is not None else None
        )
        pooled["expected_random_abs_cosine"] = random_cos
        ac = pooled.get("mean_pairwise_abs_cosine")
        if random_cos and ac is not None and random_cos > 0:
            pooled["abs_cosine_vs_random"] = ac / random_cos
        else:
            pooled["abs_cosine_vs_random"] = None
        axis_cs = [m["mean_pairwise_abs_cosine"] for m in axis_metrics.values()
                   if m.get("mean_pairwise_abs_cosine") is not None]
        pooled["mean_pairwise_abs_cosine_axis_balanced"] = (
            sum(axis_cs) / len(axis_cs) if axis_cs else None
        )
        bc = pooled["mean_pairwise_abs_cosine_axis_balanced"]
        if ac is not None and bc is not None and abs(ac - bc) > 0.1:
            notes.append(
                f"pooled |cos| ({ac:.3f}) and axis-balanced |cos| ({bc:.3f}) "
                "diverge by more than 0.1 — per-axis run counts are shaping "
                "the pooled number; read the per-axis breakdown before citing it"
            )
        if len(graded_vectors) >= 2:
            n_pairs = len(graded_vectors) * (len(graded_vectors) - 1) // 2
            directions_block = {
                "dim": dim,
                "order": [
                    r.variant for r in runs
                    if r.finding.has_direction()
                    and _universe_of(r.finding) == base_universe
                ],
                "bootstrap": {"n_boot": 500, "alpha": 0.05, "seed": seed},
                "embedded": n_pairs <= MAX_EMBED_DIRECTION_PAIRS,
            }
            if directions_block["embedded"]:
                directions_block["abs_cosine"] = cosines
                if len(null_cosines) >= 2:
                    directions_block["null_abs_cosine"] = null_cosines

    # --- checks & grade ----------------------------------------------------------
    _mk = make_check
    checks: Dict[str, Any] = {}
    j = pooled["mean_pairwise_jaccard"]
    if j is not None:
        checks["structural_stability"] = _mk(
            j, thresholds.jaccard, ">=",
            "mean pairwise Jaccard across all perturbed runs",
            ci=pooled.get("mean_pairwise_jaccard_ci95"))
    elif pooled.get("mean_pairwise_abs_cosine") is not None:
        checks["structural_stability"] = _mk(
            pooled["mean_pairwise_abs_cosine"], thresholds.cosine, ">=",
            "mean pairwise |cosine| across all perturbed runs",
            ci=pooled.get("mean_pairwise_abs_cosine_ci95"))
    ms = pooled["modal_share"]
    if ms is not None and pooled["n_runs"] >= 2 and pooled["n_claim_classes"] >= 1:
        checks["claim_stability"] = _mk(
            ms, thresholds.modal_share, ">=",
            "modal claim share π* (filability at α=0.2)",
            ci=pooled.get("modal_share_ci95"))
    cv = pooled["score_cv"]
    if cv is not None:
        checks["score_stability"] = _mk(
            cv, thresholds.score_cv, "<=",
            "coefficient of variation of the quality score",
            ci=pooled.get("score_cv_ci95"))
    vr = pooled["jaccard_vs_random"]
    if vr is not None:
        # The MC null is an expectation over a fully known distribution
        # (random size-matched subsets), so it enters as a constant; the
        # ratio's uncertainty is the real-Jaccard bootstrap, rescaled.
        jci = pooled.get("mean_pairwise_jaccard_ci95")
        vr_ci = ([jci[0] / random_null, jci[1] / random_null]
                 if jci and random_null else None)
        pooled["jaccard_vs_random_ci95"] = vr_ci
        checks["beats_random"] = _mk(
            vr, thresholds.random_margin, ">=",
            "structural overlap vs size-matched random null (×)",
            ci=vr_ci)
    if kind == "direction":
        ac = pooled.get("mean_pairwise_abs_cosine")
        cvr = pooled.get("abs_cosine_vs_random")
        if cvr is not None:
            # The null is an exact expectation over a fully known
            # distribution (independent uniform unit vectors), so it enters
            # as a constant and the ratio inherits the |cos| bootstrap.
            null_cos = pooled.get("expected_random_abs_cosine")
            aci = pooled.get("mean_pairwise_abs_cosine_ci95")
            cvr_ci = ([aci[0] / null_cos, aci[1] / null_cos]
                      if aci and null_cos else None)
            pooled["abs_cosine_vs_random_ci95"] = cvr_ci
            checks["beats_random"] = _mk(
                cvr, thresholds.random_margin, ">=",
                "direction overlap vs random unit vectors in R^d (×)",
                ci=cvr_ci)
        if null_summary is not None and ac is not None:
            null_ac = null_summary.get("mean_pairwise_abs_cosine")
            if null_ac is not None:
                ratio = (ac / null_ac) if null_ac > 1e-9 else float("inf")
                spec_ci = (M.bootstrap_ci_ratio_pairwise(
                    [("real", i) for i in range(len(cosines))],
                    [("null", i) for i in range(len(null_cosines))],
                    lambda a, b: (cosines if a[0] == "real"
                                  else null_cosines)[a[1]][b[1]],
                    seed=seed)
                    if null_cosines else None)
                pooled["specificity_ci95"] = spec_ci
                checks["specificity"] = _mk(
                    ratio, thresholds.specificity_ratio, ">=",
                    "direction stability on real vs null-control data (×)",
                    ci=spec_ci)

    if null_summary is not None and j is not None:
        null_j = null_summary.get("mean_pairwise_jaccard")
        if null_j is not None:
            ratio = (j / null_j) if null_j > 1e-9 else float("inf")
            # Both sides of the ratio are estimated from runs, so the CI
            # resamples real and null-control runs independently.
            spec_ci = (M.bootstrap_ci_ratio_pairwise(
                structured_for_ci, null_components, M.jaccard, seed=seed)
                if null_components else None)
            pooled["specificity_ci95"] = spec_ci
            checks["specificity"] = _mk(
                ratio, thresholds.specificity_ratio, ">=",
                "structural stability on real vs null-control data (×)",
                ci=spec_ci)

    if not checks:
        raise ValueError(
            "Nothing to grade: no components, claims, or scores were present in "
            "any Finding. Give your findings at least one of components=, "
            "claim=, or score=."
        )

    grade = grade_checks(
        checks, rule=GRADE_RULE, random_floor=thresholds.random_floor
    )

    # Confidence: does the evidence actually resolve the checks — in either
    # direction? A straddling CI on a fail means the grade may be too harsh;
    # on a pass, too generous. Both are undecided, both cost confidence.
    borderline = [name for name, c in checks.items()
                  if c.get("robust") is False]
    resolvable = [c for c in checks.values() if c.get("robust") is not None]
    if not resolvable:
        confidence = "unknown"
    elif borderline:
        confidence = "low"
    else:
        confidence = "high"
    pooled["confidence"] = confidence
    pooled["borderline_checks"] = borderline
    if borderline:
        detail = ", ".join(
            f"{name} ({'pass' if checks[name]['passed'] else 'fail'})"
            for name in sorted(borderline)
        )
        notes.append(
            f"underpowered verdict: the 95% CI straddles the bar for {detail} "
            f"at n_runs={n_runs} — these verdict components are not decided "
            f"by the data. Treat the grade as provisional and raise n_runs "
            f"(or widen the battery) before reporting it."
        )

    result = StressResult(
        base=base,
        runs=runs,
        axis_metrics=axis_metrics,
        pooled=pooled,
        checks=checks,
        grade=grade,
        null_summary=null_summary,
        structure_kind=kind,
    )
    result.card = StabilityCard.from_stress(
        result,
        directions=directions_block,
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
        claim_equiv_used=claim_equiv is not None,
    )
    return result


def from_findings(
    findings: Sequence[Finding],
    *,
    axes: Optional[Sequence[str]] = None,
    null_findings: Optional[Sequence[Finding]] = None,
    thresholds: Optional[Thresholds] = None,
    claim_equiv=None,
    seed: int = 0,
    claim_statement: Optional[str] = None,
    model: Optional[str] = None,
    task: Optional[str] = None,
    method: Optional[str] = None,
) -> StressResult:
    """Post-hoc stability card from findings you ALREADY have.

    No finding_fn, no re-running: hand over the per-run findings from
    result files, sweep logs, or old pickles and get the same graded card
    ``stress`` produces. The first finding is treated as the base run.

    Parameters
    ----------
    findings:
        One Finding per completed run (>= 2). Convert raw runs with
        ``stresskit.circuit`` / ``feature_set`` / ``probe``.
    axes:
        Optional per-run axis labels for runs after the first (e.g.
        ``["seeds", "seeds", "templates"]``), enabling the per-axis
        breakdown. Without labels every run is pooled under ``"runs"``
        and the card says so — axis attribution is only as good as the
        labels you supply.
    null_findings:
        Findings from the same method on a null control (shuffled labels,
        scrambled prompts). Enables the specificity check.
    """
    findings = list(findings)
    if len(findings) < 2:
        raise ValueError(f"from_findings needs >= 2 findings, got {len(findings)}")
    for i, f in enumerate(findings):
        if not isinstance(f, Finding):
            raise TypeError(
                f"findings[{i}] is {type(f).__name__}, not a stresskit.Finding. "
                "Wrap each run with stresskit.circuit(...), feature_set(...) "
                "or probe(...)."
            )
    rest = findings[1:]
    if axes is not None:
        axes = list(axes)
        if len(axes) != len(rest):
            raise ValueError(
                f"axes has {len(axes)} labels for {len(rest)} non-base findings"
            )
    else:
        axes = ["runs"] * len(rest)

    thresholds = thresholds or Thresholds()
    t0 = time.time()
    notes = [
        "post-hoc mode: findings were supplied directly, not produced by a "
        "controlled battery"
        + ("" if any(a != "runs" for a in axes)
           else " — no axis labels given, so all runs pool under 'runs'")
    ]

    base = findings[0]
    runs: List[RunRecord] = [RunRecord("base", "base", seed, {}, base)]
    counts: Dict[str, int] = {}
    for axis, f in zip(axes, rest):
        counts[axis] = counts.get(axis, 0) + 1
        runs.append(RunRecord(axis, f"{axis}={counts[axis]}", seed, {}, f))

    null_summary: Optional[Dict[str, Any]] = None
    null_components: Optional[List[frozenset]] = None
    null_vectors: Optional[List[Tuple[float, ...]]] = None
    if null_findings is not None:
        null_findings = list(null_findings)
        if len(null_findings) < 2:
            raise ValueError("null_findings needs >= 2 findings for a stability estimate")
        _structure_kind(
            list(findings) + null_findings,
            "the supplied findings and null_findings together",
        )
        null_universe = _universe_of(null_findings[0])
        null_pooled = _summarize(null_findings, claim_equiv, universe=null_universe)
        null_components = _graded_components(null_findings, null_findings[0])
        null_vectors = _graded_vectors(null_findings, null_findings[0])
        # The structural null summary is pooled over the same size-guarded
        # sets as the specificity interval, exactly as stress() pools its
        # null battery; otherwise the specificity point estimate and its CI
        # would sit on different estimands in post-hoc mode.
        null_jaccard = null_pooled["mean_pairwise_jaccard"]
        n_null_structured = sum(
            1 for f in null_findings
            if f.has_structure() and _universe_of(f) == null_universe
        )
        if null_components and len(null_components) < n_null_structured:
            null_jaccard = M.mean_pairwise_jaccard(null_components)
        null_summary = {
            "n_runs": len(null_findings),
            "mean_pairwise_jaccard": null_jaccard,
            "flip_rate": null_pooled["flip_rate"],
            "modal_share": null_pooled["modal_share"],
            "score_mean": null_pooled["score_mean"],
            "score_cv": null_pooled["score_cv"],
            "median_size": null_pooled["median_size"],
        }
        null_summary.update(_direction_summary_keys(null_pooled))

    return _analyze(
        runs, base,
        thresholds=thresholds, claim_equiv=claim_equiv, notes=notes,
        seed=seed, battery=sorted(set(axes)), n_runs=len(findings),
        base_config={}, null_summary=null_summary,
        null_components=null_components, null_vectors=null_vectors,
        claim_statement=claim_statement, model=model, task=task,
        method=method, t0=t0,
    )


def from_jsonl(
    path: str,
    *,
    null_path: Optional[str] = None,
    **kwargs: Any,
) -> StressResult:
    """Post-hoc stability card straight from a JSONL sweep log.

    One JSON object per line (see :func:`stresskit.findings_from_jsonl` for
    the field names and how to remap them). ``axis`` fields, when present,
    drive the per-axis breakdown; the first line is the base run. All other
    keyword arguments pass through to :func:`from_findings`::

        result = sk.from_jsonl("sweep.jsonl", null_path="sweep_null.jsonl",
                               model="gpt2-small", task="IOI")
        print(result.to_markdown())
    """
    from .finding import findings_from_jsonl

    loader_keys = ("components_key", "claim_key", "score_key",
                   "universe_size_key", "axis_key")
    loader_kwargs = {k: kwargs.pop(k) for k in loader_keys if k in kwargs}
    findings = findings_from_jsonl(path, **loader_kwargs)
    axes = [f.meta["axis"] for f in findings[1:] if "axis" in f.meta]
    if axes and len(axes) != len(findings) - 1:
        raise ValueError(
            f"{path}: {len(axes)} of {len(findings) - 1} non-base lines "
            "carry an 'axis' field — label every non-base line or none"
        )
    if null_path is not None and "null_findings" not in kwargs:
        kwargs["null_findings"] = findings_from_jsonl(null_path, **loader_kwargs)
    return from_findings(findings, axes=axes or None, **kwargs)


def verdict_trace(
    findings: Sequence[Finding],
    *,
    null_findings: Optional[Sequence[Finding]] = None,
    sizes: Optional[Sequence[int]] = None,
    n_subsamples: int = 30,
    thresholds: Optional[Thresholds] = None,
    claim_equiv=None,
    seed: int = 0,
) -> Dict[str, Any]:
    """How stable is the verdict itself as a function of run count?

    Papers typically report stability from 5–10 runs. This asks whether a
    verdict at that n means anything: it draws random size-k subsets of the
    supplied findings (and matching subsets of the null-control findings,
    when given), grades every subset with the full analysis, and reports
    the distribution of grades and per-check outcomes at each k.

    ``settled_n`` is the smallest k from which every k' >= k has a modal
    grade equal to the full-sample grade with subset agreement >= 0.9 —
    the run count at which the verdict stops being a coin flip. None means
    the verdict never settles within the runs supplied, which is itself
    the finding.

    Costs no new runs: subsets are regraded from the findings you already
    have (each regrade includes its bootstrap CIs, so expect roughly a
    minute of CPU per hundred subsets at n around 20).
    """
    findings = list(findings)
    n = len(findings)
    if n < 5:
        raise ValueError(f"verdict_trace needs >= 5 findings, got {n}")
    if sizes is None:
        sizes = sorted({k for k in (4, 6, 8, 10, 14, 20, 28, n) if 4 <= k <= n})
    else:
        sizes = sorted({int(k) for k in sizes})
        bad = [k for k in sizes if k < 4 or k > n]
        if bad:
            raise ValueError(
                f"subset sizes must lie in [4, {n}] (4 is the minimum for "
                f"bootstrap CIs); got {bad}"
            )

    nulls = list(null_findings) if null_findings is not None else None
    full = from_findings(
        findings, null_findings=nulls, thresholds=thresholds,
        claim_equiv=claim_equiv, seed=seed,
    )

    rng = random.Random(seed)
    per_size: Dict[int, Dict[str, Any]] = {}
    for k in sizes:
        grades: List[str] = []
        confidences: List[str] = []
        check_passes: Dict[str, List[bool]] = {}
        check_decided: Dict[str, List[bool]] = {}
        n_draws = 1 if k == n else n_subsamples
        for draw in range(n_draws):
            sub = (findings if k == n
                   else [findings[i] for i in rng.sample(range(n), k)])
            nsub = None
            if nulls:
                km = min(k, len(nulls))
                if km >= 2:
                    nsub = [nulls[i] for i in rng.sample(range(len(nulls)), km)]
            try:
                r = from_findings(
                    sub, null_findings=nsub, thresholds=thresholds,
                    claim_equiv=claim_equiv, seed=seed + draw,
                )
            except ValueError:
                continue
            grades.append(r.grade)
            confidences.append(r.pooled["confidence"])
            for name, c in r.checks.items():
                check_passes.setdefault(name, []).append(bool(c["passed"]))
                check_decided.setdefault(name, []).append(c.get("state") == "pass")
        if not grades:
            continue
        dist = {g: grades.count(g) / len(grades) for g in sorted(set(grades))}
        modal = max(dist, key=lambda g: dist[g])
        per_size[k] = {
            "n_subsamples": len(grades),
            "grade_dist": dist,
            "modal_grade": modal,
            "modal_grade_share": dist[modal],
            "low_confidence_share": confidences.count("low") / len(confidences),
            "check_pass_frac": {
                name: sum(v) / len(v)
                for name, v in sorted(check_passes.items())
            },
            "check_decided_pass_frac": {
                name: sum(v) / len(v)
                for name, v in sorted(check_decided.items())
            },
        }

    settled_n = None
    usable = [k for k in sizes if k in per_size]
    for i, k in enumerate(usable):
        if all(
            per_size[k2]["modal_grade"] == full.grade
            and per_size[k2]["modal_grade_share"] >= 0.9
            for k2 in usable[i:]
        ):
            settled_n = k
            break

    return {
        "n_total": n,
        "full_grade": full.grade,
        "full_confidence": full.pooled["confidence"],
        "grade_rule": GRADE_RULE,
        "seed": seed,
        "n_subsamples": n_subsamples,
        "thresholds": asdict(thresholds or Thresholds()),
        "sizes": usable,
        "per_size": per_size,
        "settled_n": settled_n,
    }


def verdict_trace_markdown(trace: Dict[str, Any]) -> str:
    """Render a verdict_trace() result as a markdown table."""
    header = (
        f"## Verdict-stability trace — grade **{trace['full_grade']}** "
        f"({trace['full_confidence']} confidence) at n = {trace['n_total']}"
    )
    if "grade_rule" in trace:
        header += (
            f" — grade rule {trace['grade_rule']}, seed {trace['seed']}, "
            f"{trace['n_subsamples']} subsets per size"
        )
    lines = [
        header,
        "",
        "| n runs | grade distribution | modal | low-confidence | flakiest check |",
        "|---|---|---|---|---|",
    ]
    for k in trace["sizes"]:
        row = trace["per_size"][k]
        dist = " · ".join(
            f"{g} {share:.0%}" for g, share in sorted(row["grade_dist"].items())
        )
        flaky = ""
        if row["check_pass_frac"]:
            name, frac = min(
                row["check_pass_frac"].items(),
                key=lambda kv: abs(kv[1] - 0.5),
            )
            flaky = f"{name} (pass {frac:.0%})"
        lines.append(
            f"| {k} | {dist} | {row['modal_grade']} "
            f"| {row['low_confidence_share']:.0%} | {flaky} |"
        )
    lines.append("")
    if trace["settled_n"] is not None:
        lines.append(
            f"Verdict settles at **n = {trace['settled_n']}**: from there on, "
            f"the modal grade matches the full-sample grade with >= 90% "
            f"subset agreement."
        )
    else:
        lines.append(
            f"The verdict does **not settle** within n = {trace['n_total']} "
            f"runs — subsets keep disagreeing about the grade. Any single "
            f"report at these run counts is a coin flip."
        )
    return "\n".join(lines)


def make_check(
    value: float,
    threshold: float,
    op: str,
    description: str,
    ci: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """One graded check: point estimate vs threshold, CI-resolved or not.

    ``passed`` compares the point estimate. ``robust`` (when a CI exists)
    records whether the CI actually *decides* the verdict — entirely on the
    passing side for a pass, entirely on the failing side for a fail. A
    straddling CI (robust=False) means the data did not resolve this check
    in either direction, and it lowers the verdict's confidence whether the
    point estimate happened to land above or below the bar.
    """
    passed = value >= threshold if op == ">=" else value <= threshold
    robust = None
    if ci is not None:
        if op == ">=":
            robust = ci[0] >= threshold if passed else ci[1] < threshold
        else:
            robust = ci[1] <= threshold if passed else ci[0] > threshold
    return {"value": value, "threshold": threshold, "passed": passed,
            "op": op, "ci": list(ci) if ci is not None else None,
            "robust": robust,
            "state": decision_state(value, threshold, op, ci),
            "description": description}


def decision_state(
    value: float,
    threshold: float,
    op: str,
    ci: Optional[Sequence[float]],
    *,
    minimum_n_met: bool = True,
) -> str:
    """Normative three-state decision from a confidence interval.

    ``pass`` and ``fail`` require the entire interval to lie on the
    corresponding side of the registered boundary.  An unavailable or
    boundary-crossing interval, or an unmet calibrated sample-size rule, is
    ``inconclusive``.  ``value`` remains an explicit argument so callers bind
    the state to the same estimate recorded on the check; interval position,
    not point-estimate position, determines the state.
    """
    if op not in (">=", "<="):
        raise ValueError(f"op must be '>=' or '<=', got {op!r}")
    if not minimum_n_met or ci is None or len(ci) != 2:
        return "inconclusive"
    lo, hi = float(ci[0]), float(ci[1])
    if not (math.isfinite(value) and math.isfinite(threshold)
            and math.isfinite(lo) and math.isfinite(hi)):
        return "inconclusive"
    if lo > hi:
        raise ValueError(f"confidence interval must be ordered, got {ci!r}")
    if op == ">=":
        if lo >= threshold:
            return "pass"
        if hi < threshold:
            return "fail"
    else:
        if hi <= threshold:
            return "pass"
        if lo > threshold:
            return "fail"
    return "inconclusive"


def confirmatory_verdict(
    checks: Mapping[str, Mapping[str, Any]],
    *,
    required: Optional[Sequence[str]] = None,
) -> str:
    """Combine required three-state checks without majority voting.

    Any required failure fails the audit.  If none fail but at least one is
    unavailable or inconclusive, the audit is inconclusive.  Only unanimous
    required passes produce a pass.  This is the normative confirmatory
    decision; :func:`grade_checks` remains the legacy descriptive grade.
    """
    names = list(checks) if required is None else list(required)
    if not names:
        raise ValueError("confirmatory verdict requires at least one check")
    states = []
    for name in names:
        check = checks.get(name)
        state = check.get("state") if check is not None else None
        states.append(state if state in ("pass", "fail", "inconclusive")
                      else "inconclusive")
    if "fail" in states:
        return "fail"
    if all(state == "pass" for state in states):
        return "pass"
    return "inconclusive"


def grade_checks(
    checks: Dict[str, Any],
    *,
    rule: str,
    random_floor: float = 1.5,
) -> str:
    """Letter grade from the graded checks under a named grade rule.

    ``v0.3`` (point rule): every check whose point estimate clears its bar
    counts as a pass. A: all pass. B: at least half. C: at least one. D:
    none. ``v0.4`` (decided rule): only a check whose whole 95% interval
    clears the bar (``state == "pass"``) counts; the same letter bands
    apply, then a decided specificity fail caps the letter at C (stability
    the method also shows on null data is a property of the method, not of
    the data) and a battery without a specificity check caps it at B (an
    untested null is not a passed one). Under both rules structural overlap
    at or below ``random_floor`` times the size-matched random null is D
    outright.
    """
    if rule not in GRADE_RULES:
        raise ValueError(f"grade rule must be one of {GRADE_RULES}, got {rule!r}")
    br = checks.get("beats_random")
    if br is not None and br.get("value") is not None and br["value"] <= random_floor:
        return "D"
    total = len(checks)
    if rule == "v0.3":
        passed = sum(1 for c in checks.values() if c["passed"])
        cap = "A"
    else:
        passed = sum(1 for c in checks.values() if c.get("state") == "pass")
        specificity = checks.get("specificity")
        if specificity is None:
            cap = "B"
        elif specificity.get("state") == "fail":
            cap = "C"
        else:
            cap = "A"
    if passed == total:
        grade = "A"
    elif passed * 2 >= total:
        grade = "B"
    elif passed >= 1:
        grade = "C"
    else:
        grade = "D"
    return max(grade, cap, key=GRADE_ORDER.index)


assert set(GRADE_ORDER) == {"A", "B", "C", "D"}
