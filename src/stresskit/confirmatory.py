"""Conservative confirmatory audits over preregistered IID specifications.

This module intentionally does not reuse diagnostic OAT pooling. It accepts a
manifest sampled IID from a declared specification distribution, treats runs
as independent experimental units, and uses only finite-sample bounded
concentration intervals. The initial profile sacrifices efficiency for an
auditable coverage guarantee.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from . import metrics as M
from .battery import confirmatory_verdict, decision_state, make_check
from .finding import Finding


CONFIRMATORY_ARTIFACT = "stresskit_confirmatory_card"
CONFIRMATORY_SCHEMA_VERSION = "0.1"
INFERENCE_METHOD = "paired_hoeffding_bonferroni_v1"
MINIMUM_CALIBRATED_RUNS = 200


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _derived_seed(master_seed: int, label: str) -> int:
    return int.from_bytes(
        hashlib.sha256(f"{master_seed}:{label}".encode("utf-8")).digest()[:8],
        "big",
    )


def _encode_component(value: Any) -> Any:
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise TypeError("confirmatory components cannot contain non-finite floats")
        return value
    if isinstance(value, tuple):
        return {"__tuple__": [_encode_component(item) for item in value]}
    raise TypeError(
        "confirmatory components must be JSON scalars or nested tuples; "
        f"got {type(value).__name__}"
    )


def _decode_component(value: Any) -> Any:
    if isinstance(value, dict) and set(value) == {"__tuple__"}:
        return tuple(_decode_component(item) for item in value["__tuple__"])
    if value is None or type(value) in (bool, int, float, str):
        return value
    raise ValueError(f"invalid encoded confirmatory component {value!r}")


def _encoded_components(finding: Finding) -> List[Any]:
    encoded = [_encode_component(value) for value in finding.components]
    return sorted(encoded, key=_canonical_json)


def _validate_manifest(
    manifest: Sequence[Mapping[str, Any]], n_runs: int, label: str
) -> List[Dict[str, Any]]:
    rows = [dict(row) for row in manifest]
    if len(rows) != n_runs:
        raise ValueError(
            f"{label} manifest has {len(rows)} rows for {n_runs} findings"
        )
    if any(row.get("design") != "iid_specification_sample" for row in rows):
        raise ValueError(
            f"{label} confirmatory inference requires design="
            "'iid_specification_sample'; diagnostic OAT and crossed enumeration "
            "need different estimands"
        )
    indices = [row.get("draw_index") for row in rows]
    if indices != list(range(n_runs)):
        raise ValueError(f"{label} manifest draw_index must be 0..n_runs-1")
    try:
        _canonical_json(rows)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} manifest must be canonical JSON") from error
    return rows


def _validate_findings(findings: Sequence[Finding], label: str) -> None:
    if len(findings) < 2:
        raise ValueError(f"{label} needs at least two findings")
    if any(not isinstance(finding, Finding) for finding in findings):
        raise TypeError(f"every {label} item must be a stresskit.Finding")
    structural = [finding.has_structure() for finding in findings]
    if any(structural) and not all(structural):
        raise ValueError(
            f"{label} mixes structural findings with runs lacking structural output"
        )
    if all(structural):
        universe_sizes = {finding.universe_size for finding in findings}
        universes = {finding.meta.get("universe") for finding in findings}
        if len(universe_sizes) != 1 or None in universe_sizes:
            raise ValueError(
                f"{label} structural findings need one explicit universe_size"
            )
        universe_size = int(next(iter(universe_sizes)))
        if universe_size <= 0 or any(finding.size > universe_size for finding in findings):
            raise ValueError(f"{label} findings must fit their positive universe_size")
        if len(universes) != 1:
            raise ValueError(f"{label} findings use different component universes")


def _run_rows(
    findings: Sequence[Finding], manifest: Sequence[Mapping[str, Any]]
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for finding, manifest_row in zip(findings, manifest):
        components = _encoded_components(finding) if finding.has_structure() else []
        rows.append(
            {
                "manifest": dict(manifest_row),
                "structure_present": finding.has_structure(),
                "components": components,
                "components_sha256": _digest(components),
                "claim": finding.claim,
                "score": finding.score,
                "universe_size": finding.universe_size,
                "universe": finding.meta.get("universe"),
            }
        )
    return rows


def _findings_from_rows(rows: Sequence[Mapping[str, Any]]) -> List[Finding]:
    findings = []
    for index, row in enumerate(rows):
        encoded = row.get("components")
        if not isinstance(encoded, list):
            raise ValueError(f"run {index}: components must be an array")
        if _digest(encoded) != row.get("components_sha256"):
            raise ValueError(f"run {index}: components do not match sha256")
        components = frozenset(_decode_component(value) for value in encoded)
        findings.append(
            Finding(
                components=components,
                claim=row.get("claim"),
                score=row.get("score"),
                universe_size=row.get("universe_size"),
                meta={"universe": row.get("universe")},
                structure_present=bool(row.get("structure_present")),
            )
        )
    return findings


def _check(
    value: float,
    threshold: float,
    interval: Sequence[float],
    description: str,
    *,
    metric: str,
    alpha: float,
    minimum_n_met: bool,
    pairing_seed: Optional[int] = None,
) -> Dict[str, Any]:
    check = make_check(value, threshold, ">=", description, ci=interval)
    check["state"] = decision_state(
        value, threshold, ">=", interval, minimum_n_met=minimum_n_met
    )
    check["metric"] = metric
    check["alpha"] = alpha
    check["minimum_n_met"] = minimum_n_met
    check["estimator"] = INFERENCE_METHOD
    if pairing_seed is not None:
        check["pairing_seed"] = pairing_seed
    return check


def _compute(
    findings: Sequence[Finding],
    null_findings: Optional[Sequence[Finding]],
    *,
    thresholds: Mapping[str, float],
    claim_classes: Optional[Sequence[str]],
    confidence_level: float,
    minimum_runs: int,
    master_seed: int,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]], Dict[str, int]]:
    structural = all(finding.has_structure() for finding in findings)
    expected_names: List[str] = []
    if structural:
        expected_names.extend(("structural_stability", "beats_random"))
    if "claim_stability" in thresholds:
        expected_names.append("claim_stability")
    if null_findings is not None:
        expected_names.append("specificity")
    if not expected_names:
        raise ValueError("confirmatory audit has no registered checks")
    if set(thresholds) != set(expected_names):
        raise ValueError(
            f"threshold keys must equal applicable checks {expected_names}, "
            f"got {sorted(thresholds)}"
        )

    familywise_alpha = 1.0 - confidence_level
    check_alpha = familywise_alpha / len(expected_names)
    minimum_n_met = len(findings) >= minimum_runs
    seeds = {
        name: _derived_seed(master_seed, name)
        for name in (
            "structural_stability",
            "beats_random",
            "specificity_real",
            "specificity_null",
        )
    }
    metrics: Dict[str, Any] = {
        "n_runs": len(findings),
        "n_independent_units": len(findings),
        "familywise_confidence_level": confidence_level,
        "per_check_alpha": check_alpha,
    }
    checks: Dict[str, Dict[str, Any]] = {}

    if structural:
        sets = [finding.components for finding in findings]
        structural_seed = seeds["structural_stability"]
        point = M.paired_mean_pairwise(sets, M.jaccard, seed=structural_seed)
        interval = M.hoeffding_ci_pairwise(
            sets, M.jaccard, seed=structural_seed, alpha=check_alpha
        )
        assert point is not None and interval is not None
        metrics.update(
            {
                "paired_mean_jaccard": point,
                "paired_mean_jaccard_ci": interval,
                "complete_mean_pairwise_jaccard_descriptive": (
                    M.mean_pairwise_jaccard(sets)
                ),
                "empty_finding_rate": sum(not value for value in sets) / len(sets),
            }
        )
        checks["structural_stability"] = _check(
            point,
            thresholds["structural_stability"],
            interval,
            "mean Jaccard for disjoint IID run pairs",
            metric="paired_mean_jaccard",
            alpha=check_alpha,
            minimum_n_met=minimum_n_met,
            pairing_seed=structural_seed,
        )

        universe_size = int(findings[0].universe_size)

        def adjusted(left: frozenset, right: frozenset) -> float:
            null = M.exact_expected_random_jaccard(
                len(left), universe_size, len(right)
            )
            assert null is not None
            return M.jaccard(left, right) - null

        random_seed = seeds["beats_random"]
        random_point = M.paired_mean_pairwise(sets, adjusted, seed=random_seed)
        random_interval = M.hoeffding_ci_pairwise(
            sets,
            adjusted,
            seed=random_seed,
            alpha=check_alpha,
            bounds=(-1.0, 1.0),
        )
        assert random_point is not None and random_interval is not None
        metrics["jaccard_minus_exact_size_matched_random"] = random_point
        metrics["jaccard_minus_exact_size_matched_random_ci"] = random_interval
        checks["beats_random"] = _check(
            random_point,
            thresholds["beats_random"],
            random_interval,
            "Jaccard minus exact pair-size-matched uniform-set expectation",
            metric="jaccard_minus_exact_size_matched_random",
            alpha=check_alpha,
            minimum_n_met=minimum_n_met,
            pairing_seed=random_seed,
        )

    if "claim_stability" in thresholds:
        if not claim_classes:
            raise ValueError("claim_stability requires preregistered claim_classes")
        labels = [finding.claim for finding in findings]
        if any(label is None for label in labels):
            raise ValueError("claim_stability requires one claim label per run")
        label_values = [str(label) for label in labels]
        classes = [str(label) for label in claim_classes]
        interval = M.modal_share_hoeffding_ci(
            label_values, classes, alpha=check_alpha
        )
        point = M.modal_share(label_values)
        assert point is not None and interval is not None
        metrics["modal_share"] = point
        metrics["modal_share_ci"] = interval
        metrics["claim_counts"] = {
            label: label_values.count(label) for label in classes
        }
        checks["claim_stability"] = _check(
            point,
            thresholds["claim_stability"],
            interval,
            "population modal share over preregistered claim classes",
            metric="modal_share",
            alpha=check_alpha,
            minimum_n_met=minimum_n_met,
        )

    if null_findings is not None:
        if not structural or not all(finding.has_structure() for finding in null_findings):
            raise ValueError("specificity requires structural real and null findings")
        real_sets = [finding.components for finding in findings]
        null_sets = [finding.components for finding in null_findings]
        result = M.hoeffding_difference_pairwise(
            real_sets,
            null_sets,
            M.jaccard,
            real_seed=seeds["specificity_real"],
            null_seed=seeds["specificity_null"],
            alpha=check_alpha,
        )
        assert result is not None
        specificity_minimum_met = minimum_n_met and len(null_sets) >= minimum_runs
        metrics["specificity_difference"] = result["estimate"]
        metrics["specificity_difference_ci"] = result["ci"]
        metrics["specificity_real"] = result["real_estimate"]
        metrics["specificity_null"] = result["null_estimate"]
        checks["specificity"] = _check(
            float(result["estimate"]),
            thresholds["specificity"],
            result["ci"],
            "real minus null-control mean Jaccard",
            metric="specificity_difference",
            alpha=check_alpha,
            minimum_n_met=specificity_minimum_met,
        )
        checks["specificity"]["real_pairing_seed"] = seeds["specificity_real"]
        checks["specificity"]["null_pairing_seed"] = seeds["specificity_null"]

    return metrics, checks, seeds


@dataclass
class ConfirmatoryCard:
    """Machine-verifiable finite-sample confirmatory audit artifact."""

    payload: Dict[str, Any]

    @property
    def state(self) -> str:
        return str(self.payload["verdict"]["state"])

    def to_dict(self) -> Dict[str, Any]:
        return json.loads(json.dumps(self.payload))

    def save(self, path: str) -> None:
        target = Path(path)
        with target.open("w", encoding="utf-8") as handle:
            json.dump(self.payload, handle, indent=2, sort_keys=True)
            handle.write("\n")

    def to_markdown(self) -> str:
        verdict = self.payload["verdict"]
        claim = self.payload["claim"]
        checks = self.payload["checks"]
        lines = [
            f"# Confirmatory Stability Card — **{verdict['state']}**",
            "",
            f"> **Claim:** {claim['statement']}",
            "> Finite-sample paired-Hoeffding profile; all required checks use "
            "a familywise confidence budget.",
            "",
            "| required check | estimate | simultaneous CI | threshold | state |",
            "|---|---:|---|---:|---|",
        ]
        for name in verdict["required_checks"]:
            check = checks[name]
            lines.append(
                f"| {name.replace('_', ' ')} | {check['value']:.3f} | "
                f"[{check['ci'][0]:.3f}, {check['ci'][1]:.3f}] | "
                f"≥ {check['threshold']:.3f} | {check['state']} |"
            )
        lines.extend(
            [
                "",
                f"Runs: {self.payload['design']['n_runs']} independent IID "
                f"specification draws; minimum: "
                f"{self.payload['inference']['minimum_runs']}.",
                "",
                "Failure means this registered claim did not clear at least one "
                "registered gate; it does not grade a paper or method family.",
            ]
        )
        return "\n".join(lines)


@dataclass
class ConfirmatoryResult:
    metrics: Dict[str, Any]
    checks: Dict[str, Dict[str, Any]]
    state: str
    card: ConfirmatoryCard

    def to_markdown(self) -> str:
        return self.card.to_markdown()


def confirmatory_from_findings(
    findings: Sequence[Finding],
    manifest: Sequence[Mapping[str, Any]],
    *,
    claim_statement: str,
    thresholds: Mapping[str, float],
    threshold_justifications: Mapping[str, str],
    claim_classes: Optional[Sequence[str]] = None,
    null_findings: Optional[Sequence[Finding]] = None,
    null_manifest: Optional[Sequence[Mapping[str, Any]]] = None,
    confidence_level: float = 0.95,
    minimum_runs: int = MINIMUM_CALIBRATED_RUNS,
    seed: int = 0,
    model: Optional[str] = None,
    task: Optional[str] = None,
    method: Optional[str] = None,
    claim_id: Optional[str] = None,
    provenance: Optional[Mapping[str, Any]] = None,
) -> ConfirmatoryResult:
    """Audit preregistered IID run outputs with finite-sample guarantees.

    ``thresholds`` uses applicable check names: ``structural_stability`` and
    ``beats_random`` are mandatory for structural findings; ``specificity`` is
    mandatory when a null is supplied; ``claim_stability`` is optional but then
    requires a finite preregistered ``claim_classes`` set. Every threshold needs
    a nonempty, public justification.
    """
    findings = list(findings)
    nulls = list(null_findings) if null_findings is not None else None
    _validate_findings(findings, "real")
    manifest_rows = _validate_manifest(manifest, len(findings), "real")
    if not claim_statement.strip():
        raise ValueError("claim_statement must be nonempty")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie in (0, 1)")
    if minimum_runs < MINIMUM_CALIBRATED_RUNS:
        raise ValueError(
            f"minimum_runs cannot be below calibrated floor "
            f"{MINIMUM_CALIBRATED_RUNS}"
        )
    threshold_values = {name: float(value) for name, value in thresholds.items()}
    if any(not math.isfinite(value) for value in threshold_values.values()):
        raise ValueError("thresholds must be finite")
    justifications = {name: str(value).strip() for name, value in threshold_justifications.items()}
    if set(justifications) != set(threshold_values) or not all(justifications.values()):
        raise ValueError("every threshold needs exactly one nonempty justification")

    null_manifest_rows = None
    if nulls is not None:
        _validate_findings(nulls, "null")
        if null_manifest is None:
            raise ValueError("null_findings require null_manifest")
        null_manifest_rows = _validate_manifest(null_manifest, len(nulls), "null")
        real_universe = (
            findings[0].universe_size,
            findings[0].meta.get("universe"),
        )
        null_universe = (nulls[0].universe_size, nulls[0].meta.get("universe"))
        if real_universe != null_universe:
            raise ValueError("real and null findings must use the same component universe")
    elif null_manifest is not None:
        raise ValueError("null_manifest supplied without null_findings")

    metrics, checks, pairing_seeds = _compute(
        findings,
        nulls,
        thresholds=threshold_values,
        claim_classes=claim_classes,
        confidence_level=confidence_level,
        minimum_runs=minimum_runs,
        master_seed=seed,
    )
    required = list(checks)
    state = confirmatory_verdict(checks, required=required)
    real_rows = _run_rows(findings, manifest_rows)
    null_rows = _run_rows(nulls, null_manifest_rows) if nulls is not None else []
    from . import __version__

    payload: Dict[str, Any] = {
        "artifact": CONFIRMATORY_ARTIFACT,
        "schema_version": CONFIRMATORY_SCHEMA_VERSION,
        "stresskit_version": __version__,
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "claim": {
            "claim_id": claim_id,
            "statement": claim_statement,
            "model": model,
            "task": task,
            "method": method,
        },
        "design": {
            "profile": "confirmatory",
            "sampling_design": "iid_specification_sample",
            "independent_unit": "run",
            "n_runs": len(findings),
            "n_null_runs": len(nulls) if nulls is not None else 0,
            "manifest_sha256": _digest(manifest_rows),
            "null_manifest_sha256": (
                _digest(null_manifest_rows) if null_manifest_rows is not None else None
            ),
        },
        "inference": {
            "method": INFERENCE_METHOD,
            "confidence_level": confidence_level,
            "familywise_alpha": 1.0 - confidence_level,
            "minimum_runs": minimum_runs,
            "master_seed": seed,
            "pairing_seeds": pairing_seeds,
            "claim_classes": list(claim_classes) if claim_classes is not None else None,
        },
        "thresholds": {
            name: {"value": threshold_values[name], "justification": justifications[name]}
            for name in threshold_values
        },
        "metrics": metrics,
        "checks": checks,
        "verdict": {"state": state, "required_checks": required},
        "runs": real_rows,
        "null_runs": null_rows,
        "provenance": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            **dict(provenance or {}),
        },
    }
    card = ConfirmatoryCard(payload)
    return ConfirmatoryResult(metrics, checks, state, card)


def _close(left: Any, right: Any, tolerance: float = 1e-12) -> bool:
    if isinstance(left, (int, float)) and not isinstance(left, bool) and isinstance(right, (int, float)) and not isinstance(right, bool):
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_close(a, b, tolerance) for a, b in zip(left, right))
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(_close(left[key], right[key], tolerance) for key in left)
    return left == right


def verify_confirmatory_card_dict(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Recompute manifest hashes, raw-run metrics, intervals, states, and verdict."""
    if payload.get("artifact") != CONFIRMATORY_ARTIFACT:
        raise ValueError(f"not a {CONFIRMATORY_ARTIFACT!r} artifact")
    if payload.get("schema_version") != CONFIRMATORY_SCHEMA_VERSION:
        raise ValueError("unsupported confirmatory card schema_version")
    problems: List[str] = []
    try:
        runs = payload.get("runs")
        null_runs = payload.get("null_runs") or []
        if not isinstance(runs, list) or not runs:
            raise ValueError("runs must be a nonempty array")
        if not isinstance(null_runs, list):
            raise ValueError("null_runs must be an array")
        manifest = [row["manifest"] for row in runs]
        _validate_manifest(manifest, len(runs), "real")
        null_manifest = [row["manifest"] for row in null_runs]
        if null_runs:
            _validate_manifest(null_manifest, len(null_runs), "null")
        design = payload.get("design") or {}
        expected_design = {
            "profile": "confirmatory",
            "sampling_design": "iid_specification_sample",
            "independent_unit": "run",
            "n_runs": len(runs),
            "n_null_runs": len(null_runs),
        }
        for key, expected in expected_design.items():
            if design.get(key) != expected:
                problems.append(
                    f"design.{key} stored {design.get(key)!r}, expected {expected!r}"
                )
        if _digest(manifest) != design.get("manifest_sha256"):
            problems.append("real manifest does not match manifest_sha256")
        expected_null_digest = _digest(null_manifest) if null_runs else None
        if expected_null_digest != design.get("null_manifest_sha256"):
            problems.append("null manifest does not match null_manifest_sha256")
        findings = _findings_from_rows(runs)
        null_findings = _findings_from_rows(null_runs) if null_runs else None
        _validate_findings(findings, "real")
        if null_findings is not None:
            _validate_findings(null_findings, "null")
        inference = payload.get("inference") or {}
        if inference.get("method") != INFERENCE_METHOD:
            problems.append("inference method is not the registered method")
        confidence_level = float(inference["confidence_level"])
        minimum_runs = int(inference["minimum_runs"])
        if minimum_runs < MINIMUM_CALIBRATED_RUNS:
            problems.append("minimum_runs is below calibrated floor")
        master_seed = int(inference["master_seed"])
        threshold_rows = payload.get("thresholds") or {}
        thresholds = {
            name: float(row["value"]) for name, row in threshold_rows.items()
        }
        if any(not str(row.get("justification", "")).strip() for row in threshold_rows.values()):
            problems.append("every threshold needs a nonempty justification")
        metrics, checks, seeds = _compute(
            findings,
            null_findings,
            thresholds=thresholds,
            claim_classes=inference.get("claim_classes"),
            confidence_level=confidence_level,
            minimum_runs=minimum_runs,
            master_seed=master_seed,
        )
        if not _close(inference.get("pairing_seeds"), seeds):
            problems.append("pairing_seeds do not derive from master_seed")
        expected_familywise_alpha = 1.0 - confidence_level
        if not _close(inference.get("familywise_alpha"), expected_familywise_alpha):
            problems.append("familywise_alpha does not match confidence_level")
        if not _close(payload.get("metrics"), metrics):
            problems.append("metrics do not recompute from raw runs")
        if not _close(payload.get("checks"), checks):
            problems.append("checks or intervals do not recompute from raw runs")
        required = list(checks)
        stored_verdict = payload.get("verdict") or {}
        if stored_verdict.get("required_checks") != required:
            problems.append("required_checks do not equal all registered checks")
        state = confirmatory_verdict(checks, required=required)
        if stored_verdict.get("state") != state:
            problems.append(
                f"verdict state stored {stored_verdict.get('state')!r}, "
                f"recomputed {state!r}"
            )
    except (KeyError, TypeError, ValueError) as error:
        problems.append(str(error))
        state = "inconclusive"
    return {
        "ok": not problems,
        "problems": problems,
        "recomputed_state": state,
    }
