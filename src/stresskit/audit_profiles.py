"""Frozen claim profiles and multiplicity-aware bounded inference for v1 audits."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .integrity import digest_json


@dataclass(frozen=True)
class ThresholdProfile:
    """One registered reducer, estimand, threshold set, and calibration policy."""

    profile_id: str
    finding_type: str
    reducer_name: str
    statistic: str
    bounds: Tuple[float, float]
    stability_min: float
    positive_control_min: float
    negative_control_max: float
    generalization_min: float
    practical_margin: float
    minimum_independent_units: int
    minimum_control_units: int
    alpha: float = 0.05

    def to_dict(self) -> Dict[str, Any]:
        """Return canonical registry row."""
        return {
            "profile_id": self.profile_id,
            "finding_type": self.finding_type,
            "reducer_name": self.reducer_name,
            "statistic": self.statistic,
            "bounds": list(self.bounds),
            "thresholds": {
                "stability_min": self.stability_min,
                "positive_control_min": self.positive_control_min,
                "negative_control_max": self.negative_control_max,
                "generalization_min": self.generalization_min,
                "practical_margin": self.practical_margin,
            },
            "minimum_independent_units": self.minimum_independent_units,
            "minimum_control_units": self.minimum_control_units,
            "alpha": self.alpha,
            "interval": "two-sided Hoeffding bound over independent bounded units",
            "coverage_guarantee": 1.0 - self.alpha,
            "independent_unit_requirement": (
                "dependency_id, not pair count or repeated output count"
            ),
            "power": {
                "reported_as": "minimum detectable margin at registered n",
                "minimum_detectable_margin": self.minimum_detectable_margin,
            },
        }

    @property
    def digest(self) -> str:
        """Return canonical profile digest."""
        return digest_json(self.to_dict())

    @property
    def minimum_detectable_margin(self) -> float:
        """Return two-sided Hoeffding half-width at registered sample size."""
        width = self.bounds[1] - self.bounds[0]
        return width * math.sqrt(
            math.log(2.0 / self.alpha)
            / (2.0 * self.minimum_independent_units)
        )


def _profile(
    profile_id: str,
    finding_type: str,
    reducer_name: str,
    statistic: str,
    stability_min: float,
    positive_control_min: float,
    negative_control_max: float,
    generalization_min: float,
    practical_margin: float = 0.0,
) -> ThresholdProfile:
    return ThresholdProfile(
        profile_id=profile_id,
        finding_type=finding_type,
        reducer_name=reducer_name,
        statistic=statistic,
        bounds=(0.0, 1.0),
        stability_min=stability_min,
        positive_control_min=positive_control_min,
        negative_control_max=negative_control_max,
        generalization_min=generalization_min,
        practical_margin=practical_margin,
        minimum_independent_units=200,
        minimum_control_units=200,
    )


_PROFILES = (
    _profile("set_graph_v1", "set_graph", "set_graph", "Jaccard agreement",
             0.80, 0.90, 0.10, 0.75),
    _profile("categorical_v1", "categorical", "categorical", "modal agreement",
             0.80, 0.90, 0.10, 0.75),
    _profile("scalar_effect_v1", "scalar_effect", "scalar_effect",
             "normalized scalar agreement", 0.80, 0.90, 0.10, 0.75),
    _profile("vector_direction_v1", "vector_direction", "vector_direction",
             "mapped cosine agreement", 0.80, 0.90, 0.10, 0.75),
    _profile("ranked_output_v1", "ranked_output", "ranked_output",
             "rank-biased overlap", 0.75, 0.90, 0.10, 0.70),
    _profile("utility_v1", "utility", "utility_predictions",
             "oriented external-task improvement", 0.80, 0.90, 0.10, 0.75,
             practical_margin=0.02),
    _profile("cot_trajectory_v1", "cot_trajectory", "cot_trajectory",
             "answer-and-trajectory agreement", 0.75, 0.90, 0.10, 0.70),
)

PROFILE_REGISTRY: Dict[str, ThresholdProfile] = {
    profile.profile_id: profile for profile in _PROFILES
}
PROFILE_REGISTRY_DIGEST = digest_json(
    [PROFILE_REGISTRY[key].to_dict() for key in sorted(PROFILE_REGISTRY)]
)


_REDUCER_DEFINITIONS = {
    "set_graph": (
        "v1: frozen universe digest/size/namespace; canonical finite node/edge "
        "token set; Jaccard; empty-empty=1"
    ),
    "categorical": "v1: exact JSON scalar in frozen class set; equality agreement",
    "scalar_effect": (
        "v1: finite scalar inside frozen [lo,hi]; one minus normalized distance"
    ),
    "vector_direction": (
        "v1: frozen dimension finite nonzero vector; cosine mapped from [-1,1] to [0,1]"
    ),
    "ranked_output": (
        "v1: unique JSON scalar ranking within frozen maximum length; "
        "rank-biased overlap p=0.9"
    ),
    "utility_predictions": (
        "v1: ordinary-language external task, metric spec, and exact baseline "
        "implementation/input/access provenance; raw predictions/labels; verified "
        "result-state targets"
    ),
    "cot_trajectory": (
        "v1: frozen maximum events; exact final answer plus normalized "
        "Levenshtein similarity over the ordered event sequence with "
        "multiplicity preserved"
    ),
}


def reducer_digest(name: str) -> str:
    """Return frozen semantic digest for a built-in deterministic reducer."""
    try:
        definition = _REDUCER_DEFINITIONS[name]
    except KeyError as exc:
        raise ValueError(f"unsupported reducer {name!r}") from exc
    return digest_json({"name": name, "definition": definition})


def validate_reducer_config(profile_id: str, config: Mapping[str, Any]) -> None:
    """Validate claim-type metadata that raw outputs cannot choose post hoc."""
    profile = get_profile(profile_id)
    name = profile.reducer_name
    if not isinstance(config, Mapping):
        raise ValueError("reducer config must be an object")
    if name == "set_graph":
        if set(config) != {
            "component_universe_digest", "component_universe_size", "namespace"
        }:
            raise ValueError(
                "set_graph reducer config must contain only universe digest, size, and namespace"
            )
        for key in ("component_universe_digest", "component_universe_size", "namespace"):
            if key not in config:
                raise ValueError(f"set_graph reducer config needs {key}")
        from .integrity import require_sha256_digest

        require_sha256_digest(
            config["component_universe_digest"], "component_universe_digest"
        )
        if not isinstance(config["component_universe_size"], int) or \
                isinstance(config["component_universe_size"], bool) or \
                config["component_universe_size"] < 1:
            raise ValueError("component_universe_size must be a positive integer")
        if not isinstance(config["namespace"], str) or not config["namespace"].strip():
            raise ValueError("component universe namespace must not be empty")
    elif name == "categorical":
        if set(config) != {"classes"}:
            raise ValueError("categorical reducer config must contain only classes")
        classes = config.get("classes")
        if not isinstance(classes, list) or len(classes) < 2:
            raise ValueError("categorical reducer config needs at least two classes")
        tokens = [_json_token(value) for value in classes]
        if len(set(tokens)) != len(tokens):
            raise ValueError("categorical classes must be unique")
    elif name == "scalar_effect":
        if set(config) != {"effect_bounds"}:
            raise ValueError("scalar reducer config must contain only effect_bounds")
        bounds = config.get("effect_bounds")
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise ValueError("scalar reducer config needs effect_bounds [lo, hi]")
        lo, hi = float(bounds[0]), float(bounds[1])
        if not math.isfinite(lo) or not math.isfinite(hi) or not lo < hi:
            raise ValueError("scalar effect_bounds must be finite and ordered")
    elif name == "vector_direction":
        if set(config) != {"dimension"}:
            raise ValueError("vector reducer config must contain only dimension")
        dimension = config.get("dimension")
        if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension < 1:
            raise ValueError("vector reducer config needs positive dimension")
    elif name == "ranked_output":
        if set(config) != {"maximum_length"}:
            raise ValueError("ranked reducer config must contain only maximum_length")
        maximum = config.get("maximum_length")
        if not isinstance(maximum, int) or isinstance(maximum, bool) or maximum < 1:
            raise ValueError("ranked reducer config needs positive maximum_length")
    elif name == "cot_trajectory":
        if set(config) != {"maximum_events"}:
            raise ValueError("CoT reducer config must contain only maximum_events")
        maximum = config.get("maximum_events")
        if not isinstance(maximum, int) or isinstance(maximum, bool) or maximum < 1:
            raise ValueError("CoT reducer config needs positive maximum_events")
    elif name == "utility_predictions":
        if set(config) != {"external_task", "metric_spec", "baseline_registry"}:
            raise ValueError(
                "utility reducer config needs only external_task, metric_spec, and baseline_registry"
            )
        task = config.get("external_task")
        if not isinstance(task, str) or not task.strip():
            raise ValueError("utility external_task must not be empty")
        metric_payload = config.get("metric_spec")
        if not isinstance(metric_payload, Mapping):
            raise ValueError("utility reducer config needs metric_spec object")
        from .utility import (
            UtilityMetricSpec,
            canonical_baseline_provenance,
            interpretability_phrasing,
        )

        phrasing = interpretability_phrasing(task)
        if phrasing:
            raise ValueError(
                "utility external_task must use ordinary task language, not "
                "interpretability-method jargon: " + ", ".join(phrasing)
            )

        metric = UtilityMetricSpec.from_dict(metric_payload)
        if metric.to_dict() != dict(metric_payload):
            raise ValueError("utility metric_spec must use canonical registered fields")
        if metric.practical_margin != profile.practical_margin:
            raise ValueError("utility practical_margin differs from utility_v1 profile")
        if metric.minimum_independent_units < profile.minimum_independent_units:
            raise ValueError("utility independent-unit minimum is below utility_v1 profile")
        baselines = config.get("baseline_registry")
        if not isinstance(baselines, list) or not baselines:
            raise ValueError("utility baseline_registry must be a non-empty list")
        names = []
        for row in baselines:
            canonical = canonical_baseline_provenance(row)
            if canonical != dict(row):
                raise ValueError("utility baseline provenance must be canonical")
            names.append(canonical["name"])
        if len(names) != len(set(names)):
            raise ValueError("utility baseline names must be unique")
        if not any(row["uses_internals"] is False for row in baselines):
            raise ValueError("utility needs a registered non-internals baseline")


def validate_expected_target(
    profile_id: str,
    expected: Any,
    config: Mapping[str, Any],
    *,
    component_universe: Optional[Sequence[str]] = None,
) -> None:
    """Reject invalid claim/control targets before they can become findings."""
    validate_reducer_config(profile_id, config)
    name = get_profile(profile_id).reducer_name
    if name == "set_graph":
        if not isinstance(expected, list):
            raise ValueError("set_graph expected target must be a list")
        tokens = [_json_token(value) for value in expected]
        if len(tokens) != len(set(tokens)):
            raise ValueError("set_graph expected target entries must be unique")
        if len(tokens) > config["component_universe_size"]:
            raise ValueError("set_graph expected target exceeds component universe size")
        if component_universe is not None and not set(tokens) <= set(component_universe):
            raise ValueError("set_graph expected target lies outside component universe")
    elif name == "categorical":
        if _json_token(expected) not in {_json_token(value) for value in config["classes"]}:
            raise ValueError("categorical expected target is outside frozen class set")
    elif name == "scalar_effect":
        if not isinstance(expected, Mapping) or set(expected) != {"value", "bounds"}:
            raise ValueError("scalar expected target needs only value and bounds")
        bounds = [float(value) for value in expected["bounds"]]
        value = float(expected["value"])
        if bounds != [float(item) for item in config["effect_bounds"]] or \
                not math.isfinite(value) or not bounds[0] <= value <= bounds[1]:
            raise ValueError("scalar expected target differs from frozen bounds")
    elif name == "vector_direction":
        if not isinstance(expected, list) or len(expected) != config["dimension"]:
            raise ValueError("vector expected target has invalid dimension")
        clean = [float(value) for value in expected]
        if not all(math.isfinite(value) for value in clean) or \
                sum(value * value for value in clean) <= 0.0:
            raise ValueError("vector expected target must be finite and nonzero")
    elif name == "ranked_output":
        if not isinstance(expected, list) or not expected or \
                len(expected) > config["maximum_length"]:
            raise ValueError("ranked expected target has invalid length")
        clean = [_json_token(value) for value in expected]
        if len(clean) != len(set(clean)):
            raise ValueError("ranked expected target entries must be unique")
    elif name == "cot_trajectory":
        if not isinstance(expected, Mapping) or set(expected) != {
            "final_answer", "trajectory"
        }:
            raise ValueError("CoT expected target needs only final_answer and trajectory")
        trajectory = expected["trajectory"]
        if expected["final_answer"] is None or not isinstance(trajectory, list) or \
                len(trajectory) > config["maximum_events"]:
            raise ValueError("CoT expected target is invalid or exceeds maximum_events")
    elif name == "utility_predictions":
        if not isinstance(expected, Mapping) or set(expected) != {"state"} or \
                expected["state"] not in {"pass", "fail", "inconclusive"}:
            raise ValueError(
                "utility expected target must be {'state': pass|fail|inconclusive}"
            )


def get_profile(profile_id: str) -> ThresholdProfile:
    """Return a registered profile or force caller to abstain."""
    try:
        return PROFILE_REGISTRY[profile_id]
    except KeyError as exc:
        raise ValueError(f"unsupported audit profile {profile_id!r}; abstention required") from exc


def hoeffding_interval(
    values: Sequence[float],
    *,
    bounds: Tuple[float, float] = (0.0, 1.0),
    alpha: float = 0.05,
) -> List[float]:
    """Two-sided Hoeffding interval for independent bounded observations."""
    if not values:
        raise ValueError("Hoeffding interval needs at least one observation")
    lo, hi = bounds
    if not lo < hi:
        raise ValueError("Hoeffding bounds must be ordered and non-degenerate")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1")
    clean = [float(value) for value in values]
    if not all(math.isfinite(value) and lo <= value <= hi for value in clean):
        raise ValueError(f"observations must be finite and inside [{lo}, {hi}]")
    mean = sum(clean) / len(clean)
    epsilon = (hi - lo) * math.sqrt(math.log(2.0 / alpha) / (2.0 * len(clean)))
    return [max(lo, mean - epsilon), min(hi, mean + epsilon)]


def threshold_check(
    values: Sequence[float],
    *,
    threshold: float,
    direction: str,
    bounds: Tuple[float, float] = (0.0, 1.0),
    alpha: float = 0.05,
    minimum_units: int = 1,
) -> Dict[str, Any]:
    """Recompute estimate, interval, p-value, and three-state decision."""
    if direction not in ("higher", "lower"):
        raise ValueError("threshold direction must be 'higher' or 'lower'")
    clean = [float(value) for value in values]
    estimate = sum(clean) / len(clean) if clean else None
    result: Dict[str, Any] = {
        "estimate": estimate,
        "threshold": float(threshold),
        "direction": direction,
        "bounds": list(bounds),
        "n_independent": len(clean),
        "minimum_independent_units": minimum_units,
        "alpha": alpha,
    }
    if len(clean) < minimum_units:
        result.update({
            "state": "inconclusive",
            "interval": None,
            "p_value": 1.0,
            "reason": "fewer independent units than registered minimum",
        })
        return result
    interval = hoeffding_interval(clean, bounds=bounds, alpha=alpha)
    if direction == "higher":
        if interval[0] > threshold:
            state = "pass"
        elif interval[1] <= threshold:
            state = "fail"
        else:
            state = "inconclusive"
        distance = max(0.0, float(estimate) - threshold)
    else:
        if interval[1] < threshold:
            state = "pass"
        elif interval[0] >= threshold:
            state = "fail"
        else:
            state = "inconclusive"
        distance = max(0.0, threshold - float(estimate))
    width = bounds[1] - bounds[0]
    p_value = min(1.0, math.exp(-2.0 * len(clean) * (distance / width) ** 2))
    result.update({"state": state, "interval": interval, "p_value": p_value})
    return result


def holm_bonferroni(
    p_values: Mapping[str, float], *, alpha: float = 0.05
) -> Dict[str, Dict[str, Any]]:
    """Apply deterministic Holm-Bonferroni correction to one global family."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1")
    rows: List[Tuple[str, float]] = []
    for name, raw in p_values.items():
        value = float(raw)
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError(f"p-value for {name!r} must lie in [0, 1]")
        rows.append((name, value))
    rows.sort(key=lambda row: (row[1], row[0]))
    total = len(rows)
    output: Dict[str, Dict[str, Any]] = {}
    previous_adjusted = 0.0
    still_rejecting = True
    for index, (name, raw) in enumerate(rows):
        divisor = total - index
        cutoff = alpha / divisor
        adjusted = max(previous_adjusted, min(1.0, divisor * raw))
        previous_adjusted = adjusted
        rejected = still_rejecting and raw <= cutoff
        if not rejected:
            still_rejecting = False
        output[name] = {
            "raw_p_value": raw,
            "adjusted_p_value": adjusted,
            "cutoff": cutoff,
            "rank": index + 1,
            "family_size": total,
            "rejected": rejected,
        }
    return output


def _json_token(value: Any) -> str:
    import json

    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def reduce_raw_output(
    profile_id: str,
    raw: Mapping[str, Any],
    *,
    reducer_config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Reduce one raw output through a frozen, non-agentic profile reducer."""
    profile = get_profile(profile_id)
    name = profile.reducer_name
    config = dict(reducer_config or {})
    validate_reducer_config(profile_id, config)
    if name == "set_graph":
        if raw.get("universe_digest") != config["component_universe_digest"]:
            raise ValueError("set_graph raw output universe_digest differs from ClaimRecord")
        values = raw.get("components", raw.get("edges"))
        if not isinstance(values, list):
            raise ValueError("set_graph raw output needs components or edges list")
        tokens = sorted(set(_json_token(value) for value in values))
        if len(tokens) > config["component_universe_size"]:
            raise ValueError("set_graph finding is larger than component universe")
        return {"components": tokens, "universe_digest": raw["universe_digest"]}
    if name == "categorical":
        label = raw.get("label")
        if not isinstance(label, (str, int, float, bool)) or label is None:
            raise ValueError("categorical raw output needs a JSON scalar label")
        token = _json_token(label)
        if token not in {_json_token(value) for value in config["classes"]}:
            raise ValueError("categorical label is outside frozen class set")
        return {"label": token}
    if name == "scalar_effect":
        value = raw.get("effect")
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError("scalar_effect raw output needs finite effect")
        if "effect_bounds" in raw and raw["effect_bounds"] != config["effect_bounds"]:
            raise ValueError("raw scalar bounds differ from frozen reducer config")
        lo, hi = config["effect_bounds"]
        lo, hi = float(lo), float(hi)
        if not lo < hi or not lo <= float(value) <= hi:
            raise ValueError("scalar effect and bounds are inconsistent")
        return {"effect": float(value), "effect_bounds": [lo, hi]}
    if name == "vector_direction":
        vector = raw.get("vector")
        if not isinstance(vector, list) or not vector:
            raise ValueError("vector_direction raw output needs a non-empty vector")
        clean = [float(value) for value in vector]
        if len(clean) != config["dimension"]:
            raise ValueError("vector dimension differs from frozen reducer config")
        if not all(math.isfinite(value) for value in clean):
            raise ValueError("vector contains a non-finite value")
        if sum(value * value for value in clean) <= 0.0:
            raise ValueError("vector must be nonzero")
        return {"vector": clean}
    if name == "ranked_output":
        ranking = raw.get("ranking")
        if not isinstance(ranking, list) or not ranking:
            raise ValueError("ranked_output raw output needs a non-empty ranking")
        clean = [_json_token(value) for value in ranking]
        if len(clean) > config["maximum_length"]:
            raise ValueError("ranking exceeds frozen maximum_length")
        if len(set(clean)) != len(clean):
            raise ValueError("ranking entries must be unique")
        return {"ranking": clean}
    if name == "cot_trajectory":
        answer = raw.get("final_answer")
        trajectory = raw.get("trajectory")
        if answer is None or not isinstance(trajectory, list):
            raise ValueError("cot_trajectory needs final_answer and trajectory list")
        if len(trajectory) > config["maximum_events"]:
            raise ValueError("trajectory exceeds frozen maximum_events")
        return {
            "final_answer": _json_token(answer),
            "trajectory": [_json_token(value) for value in trajectory],
        }
    if name == "utility_predictions":
        evidence = raw.get("utility_evidence")
        if not isinstance(evidence, Mapping):
            raise ValueError("utility raw output needs utility_evidence object")
        if evidence.get("task") != config["external_task"]:
            raise ValueError("utility external task differs from frozen reducer config")
        if evidence.get("metric_spec") != config["metric_spec"]:
            raise ValueError("utility metric_spec differs from frozen reducer config")
        baselines = evidence.get("baselines")
        if not isinstance(baselines, list):
            raise ValueError("utility evidence needs baseline list")
        from .utility import baseline_registry_from_evidence

        actual_registry = baseline_registry_from_evidence(baselines)
        if actual_registry != config["baseline_registry"]:
            raise ValueError("utility baselines differ from frozen baseline_registry")
        return {"utility_evidence": dict(evidence)}
    raise ValueError(f"unsupported reducer {name!r}")


def _jaccard(left: Sequence[str], right: Sequence[str]) -> float:
    a, b = set(left), set(right)
    union = a | b
    return 1.0 if not union else len(a & b) / len(union)


def _sequence_similarity(left: Sequence[str], right: Sequence[str]) -> float:
    """Return normalized edit similarity while preserving order and repeats."""
    if not left and not right:
        return 1.0
    previous = list(range(len(right) + 1))
    for left_index, left_value in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_value in enumerate(right, start=1):
            current.append(min(
                current[-1] + 1,
                previous[right_index] + 1,
                previous[right_index - 1] + int(left_value != right_value),
            ))
        previous = current
    distance = previous[-1]
    return 1.0 - distance / max(len(left), len(right))


def finding_similarity(profile_id: str, left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    """Return bounded similarity between two reduced findings."""
    name = get_profile(profile_id).reducer_name
    if name == "set_graph":
        if left.get("universe_digest") != right.get("universe_digest"):
            raise ValueError("set/graph findings use different component universes")
        return _jaccard(left["components"], right["components"])
    if name == "categorical":
        return float(left["label"] == right["label"])
    if name == "scalar_effect":
        if left["effect_bounds"] != right["effect_bounds"]:
            raise ValueError("scalar findings use different registered bounds")
        lo, hi = left["effect_bounds"]
        return max(0.0, 1.0 - abs(left["effect"] - right["effect"]) / (hi - lo))
    if name == "vector_direction":
        a, b = left["vector"], right["vector"]
        if len(a) != len(b):
            raise ValueError("vector findings have different dimensions")
        dot = sum(x * y for x, y in zip(a, b))
        norm = math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))
        return max(0.0, min(1.0, (dot / norm + 1.0) / 2.0))
    if name == "ranked_output":
        p = 0.9
        a, b = left["ranking"], right["ranking"]
        depth = max(len(a), len(b))
        score = 0.0
        for index in range(1, depth + 1):
            agreement = len(set(a[:index]) & set(b[:index])) / index
            score += (1.0 - p) * (p ** (index - 1)) * agreement
        return score + (p ** depth) * len(set(a) & set(b)) / max(len(a), len(b))
    if name == "cot_trajectory":
        answer = float(left["final_answer"] == right["final_answer"])
        trajectory = _sequence_similarity(left["trajectory"], right["trajectory"])
        return (answer + trajectory) / 2.0
    if name == "utility_predictions":
        from .utility import UtilityMetricSpec, verify_utility_evidence

        left_evidence = left["utility_evidence"]
        right_evidence = right["utility_evidence"]
        for key in ("task", "metric_spec", "split"):
            if left_evidence.get(key) != right_evidence.get(key):
                raise ValueError(f"utility findings use different {key}")
        from .utility import baseline_registry_from_evidence

        left_registry = baseline_registry_from_evidence(
            left_evidence.get("baselines", [])
        )
        right_registry = baseline_registry_from_evidence(
            right_evidence.get("baselines", [])
        )
        if left_registry != right_registry:
            raise ValueError("utility findings use different baseline registries")
        left_result = verify_utility_evidence(left_evidence)
        right_result = verify_utility_evidence(right_evidence)
        if not left_result["valid"] or not right_result["valid"]:
            raise ValueError("utility similarity needs two verified evidence blocks")
        metric = UtilityMetricSpec.from_dict(left_evidence["metric_spec"])
        width = metric.bounds[1] - metric.bounds[0]
        distance = abs(
            float(left_result["oriented_delta"])
            - float(right_result["oriented_delta"])
        )
        return max(0.0, 1.0 - distance / (2.0 * width))
    raise ValueError(f"unsupported reducer {name!r}")


def claim_support(profile_id: str, finding: Mapping[str, Any], expected: Any) -> float:
    """Score a reduced finding against a frozen claim/control target."""
    name = get_profile(profile_id).reducer_name
    if name == "set_graph":
        if not isinstance(expected, list):
            raise ValueError("set_graph expected target must be a list")
        target = sorted(set(_json_token(value) for value in expected))
        return _jaccard(finding["components"], target)
    if name == "categorical":
        return float(finding["label"] == _json_token(expected))
    if name == "scalar_effect":
        if not isinstance(expected, Mapping) or "value" not in expected:
            raise ValueError("scalar expected target needs value and bounds")
        bounds = list(expected.get("bounds", finding["effect_bounds"]))
        value = float(expected["value"])
        if bounds != finding["effect_bounds"] or not bounds[0] <= value <= bounds[1]:
            raise ValueError("scalar expected target differs from frozen bounds")
        target = {"effect": value, "effect_bounds": bounds}
        return finding_similarity(profile_id, finding, target)
    if name == "vector_direction":
        if not isinstance(expected, list):
            raise ValueError("vector expected target must be a list")
        clean = [float(value) for value in expected]
        if len(clean) != len(finding["vector"]) or \
                not all(math.isfinite(value) for value in clean) or \
                sum(value * value for value in clean) <= 0:
            raise ValueError("vector expected target has invalid dimension or norm")
        target = {"vector": clean}
        return finding_similarity(profile_id, finding, target)
    if name == "ranked_output":
        if not isinstance(expected, list):
            raise ValueError("ranked expected target must be a list")
        clean = [_json_token(value) for value in expected]
        if not clean or len(clean) != len(set(clean)):
            raise ValueError("ranked expected target must be non-empty and unique")
        target = {"ranking": clean}
        return finding_similarity(profile_id, finding, target)
    if name == "cot_trajectory":
        if not isinstance(expected, Mapping):
            raise ValueError("CoT expected target must be an object")
        if expected.get("final_answer") is None or not isinstance(
                expected.get("trajectory", []), list):
            raise ValueError("CoT expected target needs final_answer and trajectory")
        target = {
            "final_answer": _json_token(expected["final_answer"]),
            "trajectory": [
                _json_token(value) for value in expected.get("trajectory", [])
            ],
        }
        return finding_similarity(profile_id, finding, target)
    if name == "utility_predictions":
        from .utility import verify_utility_evidence

        result = verify_utility_evidence(finding["utility_evidence"])
        if not result["valid"]:
            raise ValueError("invalid utility evidence: " + "; ".join(result["problems"]))
        if not isinstance(expected, Mapping) or expected.get("state") not in {
            "pass", "fail", "inconclusive"
        }:
            raise ValueError("utility expected target needs registered state")
        return float(result["state"] == expected["state"])
    raise ValueError(f"unsupported reducer {name!r}")
