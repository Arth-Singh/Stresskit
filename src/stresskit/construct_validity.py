"""Construct validity: which degenerate finders does the battery catch?

StressKit grades a finding with five checks and a letter grade. That grade
is only evidence of anything if a method that cheats, memorises, or ignores
its input receives a worse letter than an honest method. This module freezes
that question into a table: nine finders, two battery designs, with and
without a null control, graded under both grade rules.

Finders
-------
- ``constant``         fixed set, fixed claim, fixed score; ignores data and seed
- ``index_ranker``     keeps the first k feature indices; ranking ignores data
- ``random_subset``    seeded random k-subset, claim and score; ignores data
- ``planted_leak``     returns the planted answer on any data (memorised truth)
- ``size_inflating``   planted answer padded with every even index (105 of 200)
- ``fixed_direction``  the basis vector e_3 in R^64; ignores data and seed
- ``random_direction`` seeded random unit vector in R^64; ignores data
- ``demo_positive``    the demo's top-k correlation finder on real data
- ``demo_on_noise``    the same finder with pure noise as its "real" data

Arms
----
- ``default``    seeds + bootstrap + hyperparams (k in {6, 12}), 8 runs per axis
- ``seeds_only`` seeds alone, 16 runs

Direction finders have no ``k``, so their arms drop the hyperparams axis and
keep everything else.

Every finder that ignores its data repeats the base finding on the bootstrap
axis, because that axis runs every resample at the base seed to isolate data
variation. Those identical runs inflate the pooled stability metrics, so a
random-subset finder can grade above D under the default arm. The engine
flags the vacuous axis in the card notes; the flags are reported here
alongside the grade so that the two can be compared.

One caveat on the seeds flag: a finder that recovers the same set on every
seed because the effect is that clear (``demo_positive`` here) is flagged
too; the detector cannot tell "ignores the seed" from "perfectly stable
given the seed".

Run ``python -m stresskit.construct_validity --markdown`` for the table.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import __version__
from . import battery as _battery
from .battery import FindingFn, Thresholds, grade_checks
from .card import GRADE_ORDER, GRADE_RULE, GRADE_RULES
from .demo import N_FEATURES, TRUE, make_data
from .demo import finder as demo_finder
from .finding import Finding, direction, feature_set

DIRECTION_DIM = 64
CONSTANT_SET = frozenset(range(100, 108))
INFLATED_SET = TRUE | frozenset(range(0, N_FEATURES, 2))

CLAIM_STATEMENT = "The behavior is driven by 8 specific features (degenerate finder)"


def _half_claim(components: Sequence[int], k: int) -> str:
    return (
        "first-half"
        if sum(1 for i in components if i < 100) >= k / 2
        else "second-half"
    )


def _recovered(components: Sequence[int]) -> float:
    return len(frozenset(components) & TRUE) / len(TRUE)


def constant(data: Any, seed: int, config: Dict[str, Any]) -> Finding:
    """Fixed set {100..107}, fixed claim, fixed score; ignores data and seed."""
    return feature_set(
        CONSTANT_SET, claim="second-half", score=0.5, universe_size=N_FEATURES
    )


def index_ranker(data: Any, seed: int, config: Dict[str, Any]) -> Finding:
    """Keeps the first k feature indices; the ranking never reads the data."""
    k = config["k"]
    top = list(range(k))
    return feature_set(
        top, claim=_half_claim(top, k), score=_recovered(top), universe_size=N_FEATURES
    )


def random_subset(data: Any, seed: int, config: Dict[str, Any]) -> Finding:
    """Seeded uniform random k-subset with a random claim and score; ignores data."""
    rng = random.Random(seed)
    top = rng.sample(range(N_FEATURES), config["k"])
    claim = rng.choice(("first-half", "second-half"))
    return feature_set(top, claim=claim, score=rng.random(), universe_size=N_FEATURES)


def planted_leak(data: Any, seed: int, config: Dict[str, Any]) -> Finding:
    """Returns the planted answer on any data: a method that memorised the truth."""
    return feature_set(TRUE, claim="first-half", score=1.0, universe_size=N_FEATURES)


def size_inflating(data: Any, seed: int, config: Dict[str, Any]) -> Finding:
    """Planted answer padded with every even index below N_FEATURES (105 components)."""
    return feature_set(
        INFLATED_SET, claim="first-half", score=1.0, universe_size=N_FEATURES
    )


def fixed_direction(data: Any, seed: int, config: Dict[str, Any]) -> Finding:
    """The basis vector e_3 in R^64; ignores data and seed."""
    vector = [0.0] * DIRECTION_DIM
    vector[3] = 1.0
    return direction(vector, claim="one direction", score=1.0)


def random_direction(data: Any, seed: int, config: Dict[str, Any]) -> Finding:
    """Seeded random unit vector in R^64 with a fixed claim and score; ignores data."""
    rng = random.Random(seed)
    vector = [rng.gauss(0.0, 1.0) for _ in range(DIRECTION_DIM)]
    return direction(vector, claim="one direction", score=1.0)


@dataclass
class FinderSpec:
    name: str
    fn: FindingFn
    kind: str
    real_data: str
    null_data: str
    ignores_data: bool
    description: str


@dataclass
class Arm:
    name: str
    battery: Tuple[str, ...]
    n_runs: int
    hyperparams: Optional[Dict[str, List[Any]]]


FINDERS: Tuple[FinderSpec, ...] = (
    FinderSpec(
        name="constant",
        fn=constant,
        kind="set",
        real_data="real",
        null_data="noise",
        ignores_data=True,
        description="fixed set {100..107}, claim 'second-half', score 0.5",
    ),
    FinderSpec(
        name="index_ranker",
        fn=index_ranker,
        kind="set",
        real_data="real",
        null_data="noise",
        ignores_data=True,
        description="first k feature indices; data-independent ranking",
    ),
    FinderSpec(
        name="random_subset",
        fn=random_subset,
        kind="set",
        real_data="real",
        null_data="noise",
        ignores_data=True,
        description="seeded random k-subset, random claim, random score",
    ),
    FinderSpec(
        name="planted_leak",
        fn=planted_leak,
        kind="set",
        real_data="real",
        null_data="noise",
        ignores_data=True,
        description="the planted TRUE set on any data (memorised answer)",
    ),
    FinderSpec(
        name="size_inflating",
        fn=size_inflating,
        kind="set",
        real_data="real",
        null_data="noise",
        ignores_data=True,
        description="TRUE plus every even index below 200 (105 components)",
    ),
    FinderSpec(
        name="fixed_direction",
        fn=fixed_direction,
        kind="direction",
        real_data="real",
        null_data="noise",
        ignores_data=True,
        description="basis vector e_3 in R^64",
    ),
    FinderSpec(
        name="random_direction",
        fn=random_direction,
        kind="direction",
        real_data="real",
        null_data="noise",
        ignores_data=True,
        description="seeded random unit vector in R^64",
    ),
    FinderSpec(
        name="demo_positive",
        fn=demo_finder,
        kind="set",
        real_data="real",
        null_data="noise",
        ignores_data=False,
        description=(
            "demo top-k correlation finder on the real-effect data (positive control)"
        ),
    ),
    FinderSpec(
        name="demo_on_noise",
        fn=demo_finder,
        kind="set",
        real_data="noise",
        null_data="noise_alt",
        ignores_data=False,
        description="demo top-k correlation finder with pure noise as its real data",
    ),
)

ARMS: Tuple[Arm, ...] = (
    Arm("default", ("seeds", "bootstrap", "hyperparams"), 8, {"k": [6, 12]}),
    Arm("seeds_only", ("seeds",), 16, None),
)

POSITIVE_CONTROL = "demo_positive"

_GRADE_KEYS = {"v0.3": "grade_v03", "v0.4": "grade_v04"}


def _datasets(seed: int) -> Dict[str, Any]:
    return {
        "real": make_data(400, noise=0.5, seed=seed),
        "noise": make_data(100, noise=1.0, seed=seed, signal=0.0),
        "noise_alt": make_data(100, noise=1.0, seed=seed + 1, signal=0.0),
    }


def _stress_row(
    spec: FinderSpec, arm: Arm, with_null: bool, datasets: Dict[str, Any], seed: int
) -> Dict[str, Any]:
    if spec.kind == "direction":
        battery = tuple(ax for ax in arm.battery if ax != "hyperparams")
        hyperparams = None
        config: Dict[str, Any] = {}
    else:
        battery = arm.battery
        hyperparams = arm.hyperparams
        config = {"k": 8}
    result = _battery.stress(
        spec.fn,
        datasets[spec.real_data],
        battery=battery,
        n_runs=arm.n_runs,
        seed=seed,
        config=config,
        hyperparams=hyperparams,
        null_data=datasets[spec.null_data] if with_null else None,
        claim_statement=CLAIM_STATEMENT,
        model="toy-linear-model",
        task="synthetic-attribution",
        method=spec.name,
    )
    grades = {rule: grade_checks(result.checks, rule=rule) for rule in GRADE_RULES}
    if grades[GRADE_RULE] != result.grade:
        raise RuntimeError(
            f"{spec.name}/{arm.name}: engine grade {result.grade!r} disagrees with "
            f"grade_checks(rule={GRADE_RULE!r}) = {grades[GRADE_RULE]!r}"
        )
    notes = result.card.notes
    return {
        "finder": spec.name,
        "arm": arm.name,
        "with_null": with_null,
        "battery": list(battery),
        "n_runs": arm.n_runs,
        "grade_v04": grades["v0.4"],
        "grade_v03": grades["v0.3"],
        "confidence": result.pooled["confidence"],
        "checks": {
            name: {"value": c["value"], "state": c["state"], "passed": c["passed"]}
            for name, c in result.checks.items()
        },
        "notes_flags": {
            "vacuous_seeds": any("seeds axis:" in n for n in notes),
            "vacuous_bootstrap": any("bootstrap axis:" in n for n in notes),
        },
    }


def run_degenerate_matrix(*, seed: int = 0) -> List[Dict[str, Any]]:
    """One row per (finder, arm, with_null), in FINDERS x ARMS x (False, True) order."""
    datasets = _datasets(seed)
    return [
        _stress_row(spec, arm, with_null, datasets, seed)
        for spec in FINDERS
        for arm in ARMS
        for with_null in (False, True)
    ]


def _grade_rank(grade: str) -> int:
    return GRADE_ORDER.index(grade)


def uncaught_rows(rows: Sequence[Dict[str, Any]], *, rule: str) -> List[Dict[str, Any]]:
    """Degenerate rows graded no worse than the positive control in the same
    (arm, with_null) cell under ``rule``. A battery with construct validity
    returns an empty list."""
    if rule not in _GRADE_KEYS:
        raise ValueError(
            f"grade rule must be one of {tuple(_GRADE_KEYS)}, got {rule!r}"
        )
    key = _GRADE_KEYS[rule]
    control = {
        (r["arm"], r["with_null"]): r[key]
        for r in rows
        if r["finder"] == POSITIVE_CONTROL
    }
    out = []
    for r in rows:
        if r["finder"] == POSITIVE_CONTROL:
            continue
        cell = (r["arm"], r["with_null"])
        if cell not in control:
            raise ValueError(
                f"no positive-control row for arm={cell[0]!r}, with_null={cell[1]!r}"
            )
        if _grade_rank(r[key]) <= _grade_rank(control[cell]):
            out.append(
                {
                    "finder": r["finder"],
                    "arm": r["arm"],
                    "with_null": r["with_null"],
                    "grade": r[key],
                    "positive_control_grade": control[cell],
                }
            )
    return out


def degenerate_matrix_document(
    rows: Sequence[Dict[str, Any]], *, seed: int
) -> Dict[str, Any]:
    """The frozen artifact: rows plus the configuration that produced them and
    the list of degenerate finders each grade rule fails to separate from the
    positive control."""
    return {
        "seed": seed,
        "stresskit_version": __version__,
        "thresholds": asdict(Thresholds()),
        "grade_rules": list(GRADE_RULES),
        "engine_grade_rule": GRADE_RULE,
        "arms": [
            {
                "name": arm.name,
                "battery": list(arm.battery),
                "n_runs": arm.n_runs,
                "hyperparams": arm.hyperparams,
            }
            for arm in ARMS
        ],
        "finders": [
            {
                "name": spec.name,
                "kind": spec.kind,
                "real_data": spec.real_data,
                "null_data": spec.null_data,
                "ignores_data": spec.ignores_data,
                "description": spec.description,
            }
            for spec in FINDERS
        ],
        "datasets": {
            "real": "make_data(400, noise=0.5, seed=seed)",
            "noise": "make_data(100, noise=1.0, seed=seed, signal=0.0)",
            "noise_alt": "make_data(100, noise=1.0, seed=seed + 1, signal=0.0)",
        },
        "rows": list(rows),
        "uncaught_v03": uncaught_rows(rows, rule="v0.3"),
        "uncaught_v04": uncaught_rows(rows, rule="v0.4"),
    }


def _fmt_value(value: Any) -> str:
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return f"{value:.2f}"


def _compact_checks(checks: Dict[str, Dict[str, Any]]) -> str:
    parts = [
        f"{name} {c['state']} ({_fmt_value(c['value'])})"
        for name, c in checks.items()
        if c["state"] != "pass"
    ]
    return "; ".join(parts) if parts else "—"


def _compact_flags(flags: Dict[str, bool]) -> str:
    names = [name.replace("vacuous_", "") for name, on in flags.items() if on]
    return "vacuous " + ", ".join(names) if names else "—"


def degenerate_matrix_markdown(rows: Sequence[Dict[str, Any]]) -> str:
    """One table row per configuration, then the rows each grade rule fails
    to separate from the positive control."""
    lines = [
        "| finder | arm | null | v0.3 | v0.4 | confidence "
        "| checks failed/undecided | flags |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['finder']} | {r['arm']} | {'yes' if r['with_null'] else 'no'} "
            f"| {r['grade_v03']} | {r['grade_v04']} | {r['confidence']} "
            f"| {_compact_checks(r['checks'])} | {_compact_flags(r['notes_flags'])} |"
        )
    for rule in GRADE_RULES:
        uncaught = uncaught_rows(rows, rule=rule)
        lines.append("")
        lines.append(
            f"Degenerate finders graded no worse than `{POSITIVE_CONTROL}` in the "
            f"same cell under {rule}: {len(uncaught)}"
        )
        for u in uncaught:
            lines.append(
                f"- {u['finder']} / {u['arm']} / null={'yes' if u['with_null'] else 'no'}: "
                f"{u['grade']} (positive control {u['positive_control_grade']})"
            )
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Grade degenerate finders under both grade rules and battery arms"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="print the markdown table instead of JSON",
    )
    parser.add_argument("--out", type=Path, help="write the JSON document to this path")
    parser.add_argument(
        "--markdown-out", type=Path, help="write the markdown table to this path"
    )
    args = parser.parse_args(argv)
    rows = run_degenerate_matrix(seed=args.seed)
    document = degenerate_matrix_document(rows, seed=args.seed)
    rendered_json = json.dumps(document, indent=2, allow_nan=False) + "\n"
    rendered_markdown = degenerate_matrix_markdown(rows) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered_json, encoding="utf-8")
    if args.markdown_out is not None:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(rendered_markdown, encoding="utf-8")
    print(rendered_markdown if args.markdown else rendered_json, end="")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
