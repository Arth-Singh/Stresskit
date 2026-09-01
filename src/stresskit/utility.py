"""Downstream-utility axis: did the finding buy anything outside interpretability?

Stability answers "would this finding survive a different defensible analysis".
It does not answer "is this finding worth anything", and a method can be
perfectly stable and useless — a sparse autoencoder that reconstructs the model
better is more accurate, not more useful.

So a Stability Card carries a second, independent axis: the method is put on a
task stated in ordinary language, and scored against a baseline that never
touches model internals. If an interpretability method cannot beat prompting or
a classifier over outputs, it has not yet earned its complexity, whatever its
grade on the stability axis.

The non-internals baseline is mandatory here. A comparison against another
interpretability method measures which internal technique wins; only a
comparison against something that ignores internals measures whether looking
inside helped at all.

    >>> block = utility_block(
    ...     task="flag support replies that contradict the order record",
    ...     metric="precision at 50 flags",
    ...     with_method=0.71,
    ...     baselines=[Baseline("keyword rules over the reply text", 0.44, uses_internals=False),
    ...                Baseline("logit lens on the final token", 0.66, uses_internals=True)],
    ...     n=400,
    ... )
    >>> utility_check(block)["state"]
    'pass'
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .integrity import digest_json, require_sha256_digest

# Terms that make a task statement a description of a technique rather than of
# something someone wanted done. A task phrased this way is not disqualified —
# it is flagged, because it is easier to fool yourself with.
INTERPRETABILITY_VOCABULARY = (
    "activation", "ablation", "attention head", "attribution", "circuit",
    "crosscoder", "direction", "faithfulness", "feature", "latent", "lens",
    "logit", "neuron", "patching", "probe", "reconstruction", "residual stream",
    "sae", "sparse autoencoder", "steering", "superposition", "transcoder",
)

_WORD = re.compile(r"[a-z]+")


def interpretability_phrasing(task: str) -> List[str]:
    """Interpretability terms appearing in a task statement, lowercased.

    Non-empty means the task is phrased in the language of the technique being
    tested, which is exactly the framing that makes a result hard to falsify.
    """
    text = " ".join(_WORD.findall(task.lower()))
    return [term for term in INTERPRETABILITY_VOCABULARY if term in text]


@dataclass(frozen=True)
class Baseline:
    """A comparison point on the downstream task.

    ``uses_internals`` is the load-bearing field: False means the baseline
    reaches its score without reading anything inside the model.
    """

    name: str
    value: float
    uses_internals: bool

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "value": self.value,
                "uses_internals": self.uses_internals}


@dataclass(frozen=True)
class UtilityMetricSpec:
    """Registered external-task metric and independent-unit interpretation.

    ``bounds`` apply to a metric value for one independent unit.  Metrics such
    as precision, F1, and AUROC are computed within each independent unit and
    then averaged; treating their individual examples as additive would be a
    different estimand and is rejected.
    """

    name: str
    direction: str
    bounds: Tuple[float, float]
    independent_unit: str
    practical_margin: float
    generalization_split: str
    nondecomposable_policy: str = "mean_over_independent_units"
    minimum_independent_units: int = 200
    positive_label: Any = 1

    def __post_init__(self) -> None:
        if self.name not in _METRIC_DIRECTIONS:
            raise ValueError(f"unsupported utility metric {self.name!r}")
        if self.direction != _METRIC_DIRECTIONS[self.name]:
            raise ValueError(
                f"metric {self.name!r} has registered direction "
                f"{_METRIC_DIRECTIONS[self.name]!r}, not {self.direction!r}"
            )
        lo, hi = self.bounds
        if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
            raise ValueError("utility metric bounds must be finite and ordered")
        if self.name in _BOUNDED_CLASSIFICATION_METRICS and self.bounds != (0.0, 1.0):
            raise ValueError(f"metric {self.name!r} has registered bounds (0.0, 1.0)")
        if not self.independent_unit.strip():
            raise ValueError("utility metric needs a named independent_unit")
        if not self.generalization_split.strip():
            raise ValueError("utility metric needs a held-out generalization_split")
        if not math.isfinite(self.practical_margin) or self.practical_margin < 0:
            raise ValueError("utility practical_margin must be finite and non-negative")
        if self.practical_margin >= hi - lo:
            raise ValueError("utility practical_margin must be smaller than metric range")
        if self.minimum_independent_units < 2:
            raise ValueError("utility needs at least 2 independent units")
        if self.non_decomposable and self.nondecomposable_policy != \
                "mean_over_independent_units":
            raise ValueError(
                "nondecomposable metrics require mean_over_independent_units policy"
            )

    @property
    def non_decomposable(self) -> bool:
        """Return whether metric cannot be expressed as per-example scores."""
        return self.name in _NONDECOMPOSABLE_METRICS

    def to_dict(self) -> Dict[str, Any]:
        """Return JSON representation."""
        return {
            "name": self.name,
            "direction": self.direction,
            "bounds": list(self.bounds),
            "independent_unit": self.independent_unit,
            "practical_margin": self.practical_margin,
            "generalization_split": self.generalization_split,
            "nondecomposable_policy": self.nondecomposable_policy,
            "minimum_independent_units": self.minimum_independent_units,
            "positive_label": self.positive_label,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "UtilityMetricSpec":
        """Build a metric specification from JSON."""
        bounds = payload.get("bounds", [])
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ValueError("utility metric bounds must be [lo, hi]")
        return cls(
            name=str(payload.get("name", "")),
            direction=str(payload.get("direction", "")),
            bounds=(float(bounds[0]), float(bounds[1])),
            independent_unit=str(payload.get("independent_unit", "")),
            practical_margin=float(payload.get("practical_margin", 0.0)),
            generalization_split=str(payload.get("generalization_split", "")),
            nondecomposable_policy=str(
                payload.get("nondecomposable_policy", "mean_over_independent_units")
            ),
            minimum_independent_units=int(
                payload.get("minimum_independent_units", 200)
            ),
            positive_label=payload.get("positive_label", 1),
        )


@dataclass(frozen=True)
class PredictionBaseline:
    """Raw predictions plus frozen, auditable baseline provenance."""

    name: str
    predictions: Sequence[Any]
    uses_internals: bool
    implementation_digest: str
    input_manifest_digest: str
    allowed_input_kinds: Sequence[str]
    access_policy: Mapping[str, str]

    def __post_init__(self) -> None:
        canonical_baseline_provenance({
            "name": self.name,
            "uses_internals": self.uses_internals,
            "implementation_digest": self.implementation_digest,
            "input_manifest_digest": self.input_manifest_digest,
            "allowed_input_kinds": list(self.allowed_input_kinds),
            "access_policy": dict(self.access_policy),
        })

    def provenance_dict(self) -> Dict[str, Any]:
        """Return fields frozen in the ClaimRecord baseline registry."""
        return canonical_baseline_provenance({
            "name": self.name,
            "uses_internals": self.uses_internals,
            "implementation_digest": self.implementation_digest,
            "input_manifest_digest": self.input_manifest_digest,
            "allowed_input_kinds": list(self.allowed_input_kinds),
            "access_policy": dict(self.access_policy),
        })

    def to_dict(self) -> Dict[str, Any]:
        """Return JSON representation."""
        return {
            **self.provenance_dict(),
            "predictions": list(self.predictions),
        }


BASELINE_INPUT_KINDS = frozenset({
    "external_features",
    "input_text",
    "labels",
    "loss",
    "metadata",
    "model_output",
    "output_probability",
    "activation",
    "gradient",
    "weight",
    "internal_state",
})
INTERNAL_BASELINE_INPUT_KINDS = frozenset({
    "activation", "gradient", "weight", "internal_state",
})
_BASELINE_PROVENANCE_KEYS = frozenset({
    "name",
    "uses_internals",
    "implementation_digest",
    "input_manifest_digest",
    "allowed_input_kinds",
    "access_policy",
})
_BASELINE_EVIDENCE_KEYS = _BASELINE_PROVENANCE_KEYS | {"predictions"}


def canonical_baseline_provenance(row: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate and canonicalize one frozen utility-baseline registry row.

    This proves declared implementation/input identity and access policy. It
    cannot prove semantic absence of covert leakage inside audited code.
    """
    if not isinstance(row, Mapping) or set(row) != _BASELINE_PROVENANCE_KEYS:
        raise ValueError(
            "utility baseline provenance needs name, uses_internals, "
            "implementation_digest, input_manifest_digest, allowed_input_kinds, "
            "and access_policy"
        )
    name = row["name"]
    if not isinstance(name, str) or not name.strip():
        raise ValueError("utility baseline name must not be empty")
    uses_internals = row["uses_internals"]
    if not isinstance(uses_internals, bool):
        raise ValueError("utility baseline uses_internals must be boolean")
    implementation_digest = require_sha256_digest(
        row["implementation_digest"], "utility baseline implementation_digest"
    )
    input_manifest_digest = require_sha256_digest(
        row["input_manifest_digest"], "utility baseline input_manifest_digest"
    )
    allowed = row["allowed_input_kinds"]
    if not isinstance(allowed, list) or not allowed or any(
            not isinstance(value, str) or value not in BASELINE_INPUT_KINDS
            for value in allowed):
        raise ValueError("utility baseline allowed_input_kinds are unsupported or empty")
    if len(allowed) != len(set(allowed)) or allowed != sorted(allowed):
        raise ValueError("utility baseline allowed_input_kinds must be unique and sorted")
    policy = row["access_policy"]
    if not isinstance(policy, Mapping) or set(policy) != {
            "network", "mounted_inputs", "model_internals"}:
        raise ValueError(
            "utility baseline access_policy needs network, mounted_inputs, and model_internals"
        )
    expected_policy = {
        "network": "disabled",
        "mounted_inputs": "manifest_only",
        "model_internals": "allowed" if uses_internals else "forbidden",
    }
    if dict(policy) != expected_policy:
        raise ValueError("utility baseline access_policy is not canonical")
    if not uses_internals and INTERNAL_BASELINE_INPUT_KINDS & set(allowed):
        raise ValueError(
            "non-internals baseline forbids activation, gradient, weight, and internal_state inputs"
        )
    return {
        "name": name,
        "uses_internals": uses_internals,
        "implementation_digest": implementation_digest,
        "input_manifest_digest": input_manifest_digest,
        "allowed_input_kinds": list(allowed),
        "access_policy": expected_policy,
    }


def baseline_registry_from_evidence(
    baselines: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Extract exact canonical provenance from raw prediction rows."""
    registry = []
    for row in baselines:
        if not isinstance(row, Mapping) or set(row) != _BASELINE_EVIDENCE_KEYS:
            raise ValueError(
                "each utility baseline needs exact frozen provenance and raw predictions"
            )
        registry.append(canonical_baseline_provenance({
            key: row[key] for key in _BASELINE_PROVENANCE_KEYS
        }))
    return registry


def validate_utility_input_manifest(
    payload: Mapping[str, Any], allowed_input_kinds: Sequence[str]
) -> None:
    """Validate one content-addressed manifest of inputs mounted for a baseline."""
    if not isinstance(payload, Mapping) or set(payload) != {
            "artifact", "schema_version", "inputs"} or \
            payload.get("artifact") != "stresskit_utility_input_manifest" or \
            payload.get("schema_version") != "1.0":
        raise ValueError("utility input manifest has invalid header or fields")
    inputs = payload.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        raise ValueError("utility input manifest needs non-empty inputs")
    pairs = []
    for item in inputs:
        if not isinstance(item, Mapping) or set(item) != {"kind", "digest"}:
            raise ValueError("utility input manifest rows need kind and digest")
        kind = item.get("kind")
        if not isinstance(kind, str) or kind not in BASELINE_INPUT_KINDS:
            raise ValueError("utility input manifest contains unsupported input kind")
        digest = require_sha256_digest(
            item.get("digest"), "utility input manifest digest"
        )
        pairs.append((kind, digest))
    if len(pairs) != len(set(pairs)):
        raise ValueError("utility input manifest contains duplicate inputs")
    if sorted({kind for kind, _ in pairs}) != list(allowed_input_kinds):
        raise ValueError(
            "utility input manifest kinds differ from frozen allowed_input_kinds"
        )


_METRIC_DIRECTIONS = {
    "accuracy": "higher",
    "exact_match": "higher",
    "precision": "higher",
    "recall": "higher",
    "f1": "higher",
    "auroc": "higher",
    "error_rate": "lower",
    "mean_absolute_error": "lower",
    "mean_squared_error": "lower",
}

_UNIT_INTERVAL_METRICS = {"accuracy", "exact_match", "error_rate"}
_NONDECOMPOSABLE_METRICS = {"precision", "recall", "f1", "auroc"}
_BOUNDED_CLASSIFICATION_METRICS = _UNIT_INTERVAL_METRICS | _NONDECOMPOSABLE_METRICS


def _same(left: Any, right: Any) -> bool:
    return type(left) is type(right) and left == right


def _binary(value: Any, positive_label: Any) -> bool:
    return _same(value, positive_label)


def _metric_value(
    name: str,
    predictions: Sequence[Any],
    labels: Sequence[Any],
    *,
    positive_label: Any,
) -> float:
    if len(predictions) != len(labels) or not predictions:
        raise ValueError("metric needs equal non-empty predictions and labels")
    if name in ("accuracy", "exact_match"):
        return sum(_same(prediction, label)
                   for prediction, label in zip(predictions, labels)) / len(labels)
    if name == "error_rate":
        return sum(not _same(prediction, label)
                   for prediction, label in zip(predictions, labels)) / len(labels)
    if name in ("mean_absolute_error", "mean_squared_error"):
        errors = []
        for prediction, label in zip(predictions, labels):
            if not isinstance(prediction, (int, float)) or \
                    not isinstance(label, (int, float)):
                raise ValueError(f"{name} needs numeric predictions and labels")
            delta = float(prediction) - float(label)
            if not math.isfinite(delta):
                raise ValueError(f"{name} received non-finite input")
            errors.append(abs(delta) if name == "mean_absolute_error" else delta * delta)
        return sum(errors) / len(errors)
    if name in ("precision", "recall", "f1"):
        predicted = [_binary(value, positive_label) for value in predictions]
        actual = [_binary(value, positive_label) for value in labels]
        true_positive = sum(p and a for p, a in zip(predicted, actual))
        predicted_positive = sum(predicted)
        actual_positive = sum(actual)
        if name == "precision":
            if predicted_positive == 0:
                raise ValueError("precision undefined: no predicted positives in unit")
            return true_positive / predicted_positive
        if name == "recall":
            if actual_positive == 0:
                raise ValueError("recall undefined: no positive labels in unit")
            return true_positive / actual_positive
        denominator = predicted_positive + actual_positive
        if denominator == 0:
            raise ValueError("F1 undefined: unit has no predicted or actual positives")
        return 2.0 * true_positive / denominator
    if name == "auroc":
        scores = []
        actual = []
        for prediction, label in zip(predictions, labels):
            if not isinstance(prediction, (int, float)) or \
                    not math.isfinite(float(prediction)):
                raise ValueError("AUROC needs finite numeric prediction scores")
            scores.append(float(prediction))
            actual.append(_binary(label, positive_label))
        positives = [score for score, flag in zip(scores, actual) if flag]
        negatives = [score for score, flag in zip(scores, actual) if not flag]
        if not positives or not negatives:
            raise ValueError("AUROC needs positive and negative labels in each unit")
        wins = 0.0
        for positive in positives:
            for negative in negatives:
                wins += 1.0 if positive > negative else 0.5 if positive == negative else 0.0
        return wins / (len(positives) * len(negatives))
    raise ValueError(f"unsupported utility metric {name!r}")


def _raw_utility_payload(block: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: value for key, value in block.items()
        if key not in ("derived", "raw_digest")
    }


def _group_indices(unit_ids: Sequence[Any]) -> List[Tuple[str, List[int]]]:
    groups: Dict[str, List[int]] = {}
    for index, raw in enumerate(unit_ids):
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("utility unit_ids must be non-empty strings")
        groups.setdefault(raw, []).append(index)
    return [(key, groups[key]) for key in sorted(groups)]


def _recompute_raw_utility(block: Mapping[str, Any]) -> Dict[str, Any]:
    spec = UtilityMetricSpec.from_dict(block.get("metric_spec", {}))
    task = block.get("task")
    if not isinstance(task, str) or not task.strip():
        raise ValueError("utility evidence needs an external task statement")
    phrasing = interpretability_phrasing(task)
    if phrasing:
        raise ValueError(
            "utility external task must use ordinary task language, not "
            "interpretability-method jargon: " + ", ".join(phrasing)
        )
    split = block.get("split")
    if split != spec.generalization_split:
        raise ValueError(
            "utility evidence split does not match registered generalization_split"
        )
    labels = block.get("labels")
    unit_ids = block.get("unit_ids")
    method = block.get("method")
    baselines = block.get("baselines")
    if not isinstance(labels, list) or not isinstance(unit_ids, list):
        raise ValueError("utility evidence needs raw labels and unit_ids lists")
    if len(labels) != len(unit_ids) or not labels:
        raise ValueError("utility labels and unit_ids must have equal non-zero length")
    if not isinstance(method, Mapping) or not isinstance(method.get("predictions"), list):
        raise ValueError("utility evidence needs method.predictions list")
    if method.get("uses_internals") is not True:
        raise ValueError("utility method must explicitly declare uses_internals=true")
    if len(method["predictions"]) != len(labels):
        raise ValueError("method predictions length does not match labels")
    if not isinstance(baselines, list) or not baselines:
        raise ValueError("utility evidence needs at least one baseline")
    clean_baselines = []
    baseline_names = set()
    registry = baseline_registry_from_evidence(baselines)
    for row, provenance in zip(baselines, registry):
        if not isinstance(row.get("predictions"), list):
            raise ValueError("each utility baseline needs raw predictions")
        row = {**provenance, "predictions": list(row["predictions"])}
        if row["name"] in baseline_names:
            raise ValueError(f"duplicate utility baseline name {row['name']!r}")
        baseline_names.add(row["name"])
        if len(row["predictions"]) != len(labels):
            raise ValueError(f"baseline {row['name']!r} length does not match labels")
        clean_baselines.append(row)
    non_internals = [row for row in clean_baselines if not row["uses_internals"]]
    if not non_internals:
        raise ValueError("utility evidence needs baseline with uses_internals=false")

    groups = _group_indices(unit_ids)
    lo, hi = spec.bounds

    def unit_scores(predictions: Sequence[Any]) -> List[float]:
        scores = []
        for _, indices in groups:
            score = _metric_value(
                spec.name,
                [predictions[index] for index in indices],
                [labels[index] for index in indices],
                positive_label=spec.positive_label,
            )
            if not math.isfinite(score) or not lo <= score <= hi:
                raise ValueError(
                    f"metric value {score} lies outside registered bounds [{lo}, {hi}]"
                )
            scores.append(score)
        return scores

    method_units = unit_scores(method["predictions"])
    baseline_units = {
        str(row["name"]): unit_scores(row["predictions"])
        for row in clean_baselines
    }
    baseline_values = {
        name: sum(values) / len(values) for name, values in baseline_units.items()
    }
    if spec.direction == "higher":
        reference = max(
            non_internals,
            key=lambda row: (baseline_values[str(row["name"])], str(row["name"])),
        )
    else:
        reference = min(
            non_internals,
            key=lambda row: (baseline_values[str(row["name"])], str(row["name"])),
        )
    reference_name = str(reference["name"])
    reference_units = baseline_units[reference_name]
    if spec.direction == "higher":
        deltas = [left - right for left, right in zip(method_units, reference_units)]
    else:
        deltas = [right - left for left, right in zip(method_units, reference_units)]
    method_value = sum(method_units) / len(method_units)
    oriented_delta = sum(deltas) / len(deltas)
    from .audit_profiles import hoeffding_interval

    interval = None
    state = "inconclusive"
    p_value = 1.0
    if len(groups) >= spec.minimum_independent_units:
        width = hi - lo
        interval = hoeffding_interval(
            deltas, bounds=(-width, width), alpha=0.05
        )
        if interval[0] > spec.practical_margin:
            state = "pass"
        elif interval[1] <= spec.practical_margin:
            state = "fail"
        distance = max(0.0, oriented_delta - spec.practical_margin)
        p_value = min(1.0, math.exp(
            -2.0 * len(groups) * (distance / (2.0 * width)) ** 2
        ))
    return {
        "method_value": method_value,
        "baseline_values": baseline_values,
        "reference_baseline": reference_name,
        "oriented_delta": oriented_delta,
        "delta_interval95": interval,
        "state": state,
        "p_value": p_value,
        "n_independent": len(groups),
        "minimum_independent_units": spec.minimum_independent_units,
        "metric_direction": spec.direction,
        "metric_bounds": list(spec.bounds),
        "practical_margin": spec.practical_margin,
        "independent_unit": spec.independent_unit,
        "generalization_split": spec.generalization_split,
        "nondecomposable_policy": spec.nondecomposable_policy,
    }


def build_utility_evidence(
    *,
    task: str,
    metric_spec: UtilityMetricSpec,
    labels: Sequence[Any],
    unit_ids: Sequence[str],
    method_predictions: Sequence[Any],
    baselines: Sequence[PredictionBaseline],
    split: str,
    method_name: str = "interpretability method",
) -> Dict[str, Any]:
    """Build v1 utility evidence from raw predictions and labels only."""
    block: Dict[str, Any] = {
        "evidence_version": "1.0",
        "task": task,
        "metric_spec": metric_spec.to_dict(),
        "split": split,
        "labels": list(labels),
        "unit_ids": list(unit_ids),
        "method": {
            "name": method_name,
            "uses_internals": True,
            "predictions": list(method_predictions),
        },
        "baselines": [baseline.to_dict() for baseline in baselines],
    }
    block["raw_digest"] = digest_json(_raw_utility_payload(block))
    block["derived"] = _recompute_raw_utility(block)
    return block


def _equivalent_stored(stored: Any, recomputed: Any, path: str,
                       problems: List[str]) -> None:
    if isinstance(recomputed, float):
        if not isinstance(stored, (int, float)) or \
                not math.isfinite(float(stored)) or \
                abs(float(stored) - recomputed) > 1e-12:
            problems.append(f"{path}: stored {stored!r}, recomputed {recomputed!r}")
        return
    if isinstance(recomputed, Mapping):
        if not isinstance(stored, Mapping):
            problems.append(f"{path}: stored value is not an object")
            return
        if set(stored) != set(recomputed):
            problems.append(f"{path}: stored keys differ from recomputed keys")
            return
        for key in recomputed:
            _equivalent_stored(stored[key], recomputed[key], f"{path}.{key}", problems)
        return
    if isinstance(recomputed, list):
        if not isinstance(stored, list) or len(stored) != len(recomputed):
            problems.append(f"{path}: stored list differs from recomputed list")
            return
        for index, value in enumerate(recomputed):
            _equivalent_stored(stored[index], value, f"{path}[{index}]", problems)
        return
    if stored != recomputed:
        problems.append(f"{path}: stored {stored!r}, recomputed {recomputed!r}")


def verify_utility_evidence(block: Mapping[str, Any]) -> Dict[str, Any]:
    """Recompute v1 utility metric, baseline, delta, interval, and state.

    Stored summaries are treated as claims to check, never as evidence.  Any
    contradiction makes evidence invalid and prevents publication.
    """
    problems: List[str] = []
    if not isinstance(block, Mapping) or block.get("evidence_version") != "1.0":
        return {
            "valid": False,
            "state": "abstain",
            "problems": ["utility evidence_version must be '1.0'"],
        }
    try:
        raw_digest = digest_json(_raw_utility_payload(block))
        if block.get("raw_digest") != raw_digest:
            problems.append(
                f"raw_digest: stored {block.get('raw_digest')!r}, recomputed {raw_digest!r}"
            )
        recomputed = _recompute_raw_utility(block)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        return {"valid": False, "state": "abstain", "problems": [str(exc)]}
    _equivalent_stored(block.get("derived"), recomputed, "derived", problems)
    return {
        "valid": not problems,
        "problems": problems,
        **recomputed,
        **({"state": "abstain"} if problems else {}),
    }


def best_non_internals(baselines: Sequence[Baseline]) -> Baseline:
    """The strongest baseline that never reads model internals."""
    candidates = [b for b in baselines if not b.uses_internals]
    if not candidates:
        raise ValueError(
            "a utility claim needs at least one baseline with "
            "uses_internals=False: without one, the result cannot show that "
            "looking inside the model helped at all"
        )
    return max(candidates, key=lambda b: b.value)


def bootstrap_delta_ci(
    paired_deltas: Sequence[float],
    *,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> List[float]:
    """Percentile bootstrap CI for a mean paired delta (method minus baseline).

    ``paired_deltas`` holds one delta per evaluation item, so the resample
    respects the pairing. Fewer than two items cannot support an interval.
    """
    deltas = [float(d) for d in paired_deltas]
    if len(deltas) < 2:
        raise ValueError("bootstrap needs at least 2 paired deltas")
    rng = random.Random(seed)
    n = len(deltas)
    means = []
    for _ in range(n_boot):
        means.append(sum(deltas[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[min(int((1 - alpha / 2) * n_boot), n_boot - 1)]
    return [lo, hi]


def utility_block(
    *,
    task: str,
    metric: str,
    with_method: float,
    baselines: Sequence[Baseline],
    n: int,
    paired_deltas: Optional[Sequence[float]] = None,
    n_boot: int = 2000,
    seed: int = 0,
) -> Dict[str, Any]:
    """Build the card's ``utility`` block, or fail with the reason it cannot.

    ``paired_deltas`` are per-item differences against the best non-internals
    baseline; supplying them adds a confidence interval, without which the
    check can only report ``inconclusive``.
    """
    if not task.strip():
        raise ValueError("utility claim needs a task statement")
    if not metric.strip():
        raise ValueError("utility claim needs a named metric")
    if n < 1:
        raise ValueError(f"utility claim needs n >= 1 evaluation items, got {n}")
    reference = best_non_internals(baselines)

    block: Dict[str, Any] = {
        "task": task,
        "metric": metric,
        "n": n,
        "with_method": float(with_method),
        "baselines": [b.to_dict() for b in baselines],
        "reference_baseline": reference.name,
        "delta_vs_non_internals": float(with_method) - reference.value,
    }
    flagged = interpretability_phrasing(task)
    if flagged:
        block["task_phrasing_warning"] = (
            "task is stated in interpretability vocabulary "
            f"({', '.join(flagged)}); a task phrased outside the technique is "
            "harder to fool yourself with"
        )
    if paired_deltas is not None:
        if len(paired_deltas) != n:
            raise ValueError(
                f"paired_deltas has {len(paired_deltas)} items but n is {n}"
            )
        block["delta_ci95"] = bootstrap_delta_ci(
            paired_deltas, n_boot=n_boot, seed=seed)
        block["bootstrap"] = {"n_boot": n_boot, "seed": seed}
    return block


def utility_check(block: Dict[str, Any], *, min_delta: float = 0.0) -> Dict[str, Any]:
    """Grade one utility block against its non-internals baseline.

    ``pass`` when the whole interval clears ``min_delta``, ``fail`` when the
    whole interval sits at or below it, ``inconclusive`` when the interval
    straddles it or no interval was supplied.
    """
    if block.get("evidence_version") == "1.0":
        result = verify_utility_evidence(block)
        registered = block.get("metric_spec", {}).get("practical_margin")
        if min_delta != 0.0 and registered != min_delta:
            return {
                **result,
                "valid": False,
                "state": "abstain",
                "problems": list(result.get("problems", [])) + [
                    "caller min_delta contradicts frozen utility practical_margin"
                ],
            }
        return result
    delta = block["delta_vs_non_internals"]
    ci = block.get("delta_ci95")
    detail = {
        "delta_vs_non_internals": delta,
        "reference_baseline": block.get("reference_baseline"),
        "min_delta": min_delta,
    }
    if ci is None:
        return {
            "state": "inconclusive",
            "reason": "no confidence interval: supply paired_deltas to resolve "
                      "whether the margin survives resampling",
            **detail,
        }
    detail["delta_ci95"] = ci
    if ci[0] > min_delta:
        return {"state": "pass", **detail}
    if ci[1] <= min_delta:
        return {"state": "fail", **detail}
    return {
        "state": "inconclusive",
        "reason": "interval straddles the margin",
        **detail,
    }


def attach_utility(card: Any, block: Dict[str, Any]) -> Any:
    """Validate a utility block and attach it to an existing Stability Card.

    Kept off the ``stress`` signature on purpose: the frozen calibration
    manifest hashes ``battery.py``, so threading a card-only field through the
    run path would invalidate a completed method-validation study for no gain.
    """
    validate_utility_block(block)
    card.utility = block
    return card


def validate_utility_block(block: Any) -> None:
    """Structural validation for a ``utility`` block read back from a card."""
    if not isinstance(block, dict):
        raise ValueError("Stability Card utility must be an object")
    if block.get("evidence_version") == "1.0":
        result = verify_utility_evidence(block)
        if not result["valid"]:
            raise ValueError(
                "invalid raw utility evidence: " + "; ".join(result["problems"])
            )
        return
    for key in ("task", "metric", "with_method", "baselines", "n",
                "delta_vs_non_internals"):
        if key not in block:
            raise ValueError(f"Stability Card utility missing {key!r}")
    baselines = block["baselines"]
    if not isinstance(baselines, list) or not baselines:
        raise ValueError("Stability Card utility.baselines must be a non-empty list")
    for row in baselines:
        if not isinstance(row, dict) or not {"name", "value", "uses_internals"} <= set(row):
            raise ValueError(
                "each utility baseline needs name, value and uses_internals"
            )
        if not isinstance(row["uses_internals"], bool):
            raise ValueError("utility baseline uses_internals must be a boolean")
    if not any(not row["uses_internals"] for row in baselines):
        raise ValueError(
            "Stability Card utility.baselines must include at least one "
            "baseline with uses_internals=false"
        )
    ci = block.get("delta_ci95")
    if ci is not None:
        if (not isinstance(ci, list) or len(ci) != 2
                or not all(isinstance(x, (int, float)) for x in ci)
                or ci[0] > ci[1]):
            raise ValueError(
                "Stability Card utility.delta_ci95 must be an ordered [lo, hi] pair"
            )
