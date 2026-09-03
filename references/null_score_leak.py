"""Null-score leak over every reference card with a null control.

The specificity check compares the finder's structural stability on real data
against its stability on a null control; it never looks at the null's SCORE,
which every card records unchecked in ``metrics.null_control`` (``score_mean``,
``score_cv``, ``n_runs``). A null that still scores as well as the real data
has not removed the signal, so a specificity failure on that card says "the
null is too soft", not "the method is non-specific". This script measures the
gap with :mod:`stresskit.null_leak` for every graded card that has a null
block and crosses the result with the card's specificity verdict and its null
family (``NULL_FAMILY`` in ``make_summary_figs.py``).

Real per-run scores come from the card's ``runs``. Null per-run scores come
from the ``<stem>.runs.json`` sidecar when it has them (rows with
``group == "null"``, or a top-level ``null`` list); otherwise only the summary
pair on the card is available and the bootstrap interval is left empty.
Sidecar null scores are checked against the card's ``null_control`` summary
before use.

``Finding.score`` declares no polarity, so ``SCORE_POLARITY`` and
``SCORE_SCALE`` below state, per card-name prefix, whether a higher or a
lower score is the stronger finding and whether a retention ratio null/real
is meaningful. ``SCORE_EVIDENCE`` cites the runner docstring each choice was
read from. An unmapped card is an error, never a silent default.

Each battery counts once: a ``<stem>.directions`` card re-analyses its base
card's runs and null summary with direction structure, so it is checked
against the base, listed under ``duplicates_of_base_cards`` and left out of
the crosstab and the headline counts. The markdown closes with the caveats a
reader needs before quoting a class (null-variance artifacts, predicted
non-effects, soft nulls in the signal family); their numbers come from the
records.

No model is run; everything derives from the stored artifacts. Writes
``artifacts/self_audit/null-score-leak.json`` and ``.md`` and prints the
markdown. Deterministic: no timestamps, stable ordering, digest of the inputs.

Usage:
    PYTHONPATH=src python references/null_score_leak.py [--json-only] [--only CARD]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from make_summary_figs import null_family, recorded_state  # noqa: E402
from stresskit.metrics import mean, std  # noqa: E402
from stresskit.null_leak import (  # noqa: E402
    CLASS_DEGRADED,
    CLASS_MATCHES,
    CLASSES,
    THRESHOLDS,
    leak_from_runs,
    leak_from_summaries,
)
from stresskit.scoreboard import collect_rows  # noqa: E402

REPO = os.path.dirname(HERE)
OUT_DIR = os.path.join(REPO, "artifacts", "self_audit")
JSON_NAME = "null-score-leak.json"
MD_NAME = "null-score-leak.md"
SCHEMA_VERSION = "0.1"

FAMILIES = ("signal", "structure")
STATES = ("pass", "incon", "fail")
STATE_LABEL = {"pass": "pass", "incon": "inconclusive", "fail": "fail"}
MEAN_TOLERANCE = 1e-6
DIRECTIONS_SUFFIX = ".directions"

# +1: a higher score is a stronger finding; -1: a lower score is. Keyed by
# card-name prefix like NULL_FAMILY; first matching prefix wins.
SCORE_POLARITY = {
    "ams": 1,
    "diff_mining": 1,
    "folkmotif": 1,
    "swd": -1,
    "greater_than": 1,
    "ioi": 1,
    "coax": 1,
    "reins": 1,
    "sae_causal": 1,
    "faithfulness": 1,
    "sycophancy": 1,
    "communication_map": 1,
    "homonym": -1,
    "lens_baseline": 1,
    "jlens": 1,
    "impossibility_truth": 1,
    "harc": 1,
    "mechtomo": 1,
    "refusal_direction": 1,
}

# "ratio": non-negative score whose no-signal value is 0, so null/real is a
# retention. "signed": a difference, a cosine, an R^2, a lower-is-stronger
# score, or a score with a chance floor (AUC and accuracy sit at ~0.5 under
# the null, so null/real would overstate retention).
SCORE_SCALE = {
    "ams": "signed",
    "diff_mining": "ratio",
    "folkmotif": "ratio",
    "swd": "signed",
    "greater_than": "ratio",
    "ioi": "ratio",
    "coax": "signed",
    "reins": "signed",
    "sae_causal": "ratio",
    "faithfulness": "signed",
    "sycophancy": "signed",
    "communication_map": "ratio",
    "homonym": "signed",
    "lens_baseline": "ratio",
    "jlens": "ratio",
    "impossibility_truth": "signed",
    "harc": "signed",
    "mechtomo": "signed",
    "refusal_direction": "ratio",
}

# Where each choice was read: the runner's "- score:" docstring line and its
# null-control description.
SCORE_EVIDENCE = {
    "ams": (
        "leave-one-out accuracy over 14 models (run_ams_scanner_card.py:44); "
        "chance floor ~0.5, so signed; null = half the pair labels swapped"
    ),
    "diff_mining": (
        "top-100 domain share (run_diff_mining_card.py:50); a share with floor 0; "
        "null = scrambled LoRA adapter"
    ),
    "folkmotif": (
        "DecodingSuppressed share of the 270 cells (run_folkmotif_card.py:52); "
        "floor 0; null = culture labels permuted"
    ),
    "swd": (
        "replacement CE delta in nats, replacement minus dense CE "
        "(run_swd_card.py:18,66): a loss, lower is stronger; "
        "null = random-token calibration blocks"
    ),
    "greater_than": (
        "denoising faithfulness, fraction of the clean-vs-corrupted gap recovered "
        "(run_greater_than_gpt2_card.py:6-8); floor 0; null = random scoring threshold"
    ),
    "ioi": (
        "denoising faithfulness, fraction of the logit-diff gap recovered "
        "(run_ioi_gpt2_card.py:7-9); floor 0; null = random answer names"
    ),
    "coax": (
        "CoAx ROC-AUC (run_coax_backup_card.py:44); chance 0.5, so signed; "
        "null = third-name giver prompts"
    ),
    "reins": (
        "harmful open rate minus matched-safe open rate (run_reins_gate_card.py:47); "
        "a difference; null = calibration labels permuted"
    ),
    "sae_causal": (
        "pooled causally-inert rate among recovered pairs "
        "(run_sae_causal_inertness_card.py:49): the claim asserts inertness, so higher "
        "is stronger; floor 0; null = feature-to-probe pairing permuted, which makes "
        "every atom trivially inert"
    ),
    "faithfulness": (
        "mean off-diagonal cross-cue cosine at the reference layer "
        "(run_faithfulness_steering_card.py:76); a cosine can be negative, so signed; "
        "null = within-cue half-split noise vectors"
    ),
    "sycophancy": (
        "transfer drop = in-domain AUC minus transfer AUC "
        "(run_sycophancy_probe_card.py:72,82); a difference; null = shuffle_labels"
    ),
    "communication_map": (
        "mean pooled far-from-chance share over seven models "
        "(run_communication_map_card.py:51); floor 0; null = Haar-rotated writer factors"
    ),
    "homonym": (
        "reconvergence ratio r = final-band distance / peak distance; late "
        "reconvergence iff r <= 0.9 (run_homonym_reconvergence_card.py:31,37): lower is "
        "stronger, no-signal value ~1; null = item pairing permuted"
    ),
    "lens_baseline": (
        "hit rate len(hits)/len(sample) (run_lens_baselines_qwen.py:187); floor 0; "
        "null = derangement of targets"
    ),
    "jlens": (
        "hit rate len(hits)/len(sample) (run_jlens_stability_qwen.py:152); floor 0; "
        "null = derangement of targets"
    ),
    "impossibility_truth": (
        "double-dissociation index, mean in-axis AUC minus off-axis excess over chance "
        "(run_impossibility_truth_card.py:54-57); chance value 0.5, so signed; "
        "null = condition labels permuted"
    ),
    "harc": (
        "mean prompt-side coupling gain over the band, cos(harc) - cos(base) "
        "(run_harc_card.py:79,703-715); a difference; null = labels permuted"
    ),
    "mechtomo": (
        "held-out aggregate R^2 (run_mechtomo_omp_card.py:34); can be negative; "
        "null = measurement-to-response pairing permuted"
    ),
    "refusal_direction": (
        "fraction of held-out non-compliance converted into coherent compliance by "
        "ablation, clipped at 0 (run_refusal_direction_card.py:22,464); floor 0; "
        "null = harmful/harmless labels permuted"
    ),
}


def prefixed(mapping: Dict[str, Any], name: str, what: str) -> Any:
    for prefix, value in mapping.items():
        if name.startswith(prefix):
            return value
    raise SystemExit(
        f"{name}: no {what} mapped for this card; add its prefix to the map"
    )


def sha256_of(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def rel(path: str) -> str:
    return os.path.relpath(path, REPO).replace(os.sep, "/")


def check_mean(label: str, scores: List[float], recorded: float) -> None:
    observed = mean(scores)
    if abs(observed - recorded) > MEAN_TOLERANCE * max(1.0, abs(recorded)):
        raise SystemExit(
            f"{label}: mean of the {len(scores)} per-run scores is {observed!r}, "
            f"the card records {recorded!r}"
        )


def real_scores(card: Dict[str, Any], name: str) -> List[float]:
    scores = []
    for i, run in enumerate(card["runs"]):
        score = run.get("score")
        if not isinstance(score, (int, float)):
            raise SystemExit(f"{name}: runs[{i}] has no numeric score ({score!r})")
        scores.append(float(score))
    check_mean(f"{name} real", scores, card["metrics"]["pooled"]["score_mean"])
    return scores


def sidecar_null_scores(
    card_path: str, name: str, null_control: Dict[str, Any]
) -> Tuple[Optional[List[float]], Optional[str]]:
    sidecar = card_path[: -len(".json")] + ".runs.json"
    if not os.path.exists(sidecar):
        return None, None
    with open(sidecar, encoding="utf-8") as handle:
        side = json.load(handle)
    if isinstance(side, dict) and isinstance(side.get("runs"), list):
        rows = [row for row in side["runs"] if row.get("group") == "null"]
    elif isinstance(side, dict) and isinstance(side.get("null"), list):
        rows = side["null"]
    else:
        raise SystemExit(
            f"{rel(sidecar)}: expected a 'runs' list with group='null' rows or a "
            "top-level 'null' list"
        )
    if not rows:
        return None, sidecar
    scores = [float(row["score"]) for row in rows]
    if len(scores) != null_control["n_runs"]:
        raise SystemExit(
            f"{name}: sidecar has {len(scores)} null rows, the card records "
            f"n_runs={null_control['n_runs']}"
        )
    check_mean(f"{name} null", scores, null_control["score_mean"])
    return scores, sidecar


def structural_metric(card: Dict[str, Any]) -> str:
    if card.get("battery", {}).get("structure_kind") == "direction":
        return "mean_pairwise_abs_cosine"
    return "mean_pairwise_jaccard"


def analyze_card(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The per-card record, or None for a card without a null control."""
    path = row["path"]
    name = os.path.basename(path)[: -len(".json")]
    with open(path, encoding="utf-8") as handle:
        card = json.load(handle)
    null_control = card["metrics"].get("null_control")
    if null_control is None:
        return None
    check = card["verdict"]["checks"].get("specificity")
    if check is None:
        raise SystemExit(f"{name}: has a null control but no specificity check")
    family = null_family(name)
    if family is None:
        raise SystemExit(f"{name}: no null family in make_summary_figs.NULL_FAMILY")
    polarity = prefixed(SCORE_POLARITY, name, "score polarity")
    scale = prefixed(SCORE_SCALE, name, "score scale")

    real = real_scores(card, name)
    null, sidecar = sidecar_null_scores(path, name, null_control)
    if null is not None:
        leak = leak_from_runs(real, null, polarity=polarity, scale=scale)
        source = "sidecar"
    else:
        if null_control["score_cv"] is None:
            raise SystemExit(
                f"{name}: null score_cv is None (|mean| ~ 0), so the null sd cannot be "
                "recovered from the summary; per-run null scores are needed"
            )
        null_sd = null_control["score_cv"] * abs(null_control["score_mean"])
        leak = leak_from_summaries(
            mean(real),
            std(real),
            len(real),
            null_control["score_mean"],
            null_sd,
            null_control["n_runs"],
            polarity=polarity,
            scale=scale,
        )
        source = "summary"

    metric = structural_metric(card)
    return {
        "card": name,
        "path": rel(path),
        "sha256": sha256_of(path),
        "schema_version": card["schema_version"],
        "family": family,
        "specificity": {
            "state": STATE_LABEL[recorded_state(check)],
            "value": check.get("value"),
            "threshold": check.get("threshold"),
        },
        "structural_metric": metric,
        "structural_real": card["metrics"]["pooled"].get(metric),
        "structural_null": null_control.get(metric),
        "null_scores_source": source,
        "sidecar": rel(sidecar) if sidecar else None,
        "sidecar_sha256": sha256_of(sidecar) if sidecar else None,
        "leak": leak,
    }


def crosstab(cards: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, int]]]:
    table = {
        family: {STATE_LABEL[state]: {cls: 0 for cls in CLASSES} for state in STATES}
        for family in FAMILIES
    }
    for card in cards:
        table[card["family"]][card["specificity"]["state"]][
            card["leak"]["leak_class"]
        ] += 1
    return table


def headline(table: Dict[str, Dict[str, Dict[str, int]]]) -> Dict[str, int]:
    fail = table["structure"]["fail"]
    return {
        "structure_fail_null_matches_or_exceeds": fail[CLASS_MATCHES],
        "structure_fail_null_degraded": fail[CLASS_DEGRADED],
    }


def cards_digest(cards: List[Dict[str, Any]]) -> str:
    pairs = sorted((card["path"], card["sha256"]) for card in cards)
    return hashlib.sha256(json.dumps(pairs).encode("utf-8")).hexdigest()


def fmt(x: Optional[float], spec: str = ".4g") -> str:
    return "—" if x is None else format(x, spec)


def fmt_group(mean_: float, sd: float, n: int) -> str:
    return f"{mean_:.4g} ± {sd:.2g} (n={n})"


def fmt_ci(ci: Optional[List[float]]) -> str:
    if ci is None:
        return "— (summary only)"
    return f"[{ci[0]:.3g}, {ci[1]:.3g}]"


def caveats(report: Dict[str, Any]) -> List[str]:
    """Readings the classes alone would get wrong; every number comes from the
    records, so a caveat disappears with the card it is about."""
    by_name = {card["card"]: card for card in report["cards"]}
    out: List[str] = []
    ioi = [
        by_name[name]
        for name in ("ioi_gpt2_small", "ioi_gpt2_medium", "ioi_gpt2_large")
        if name in by_name
    ]
    if ioi:
        detail = "; ".join(
            f"{card['card'][len('ioi_gpt2_') :]}: null sd {card['leak']['null_sd']:.2g} vs "
            f"real {card['leak']['real_sd']:.2g}, null mean "
            f"{card['leak']['null_mean'] / card['leak']['real_mean']:.2f}× real"
            for card in ioi
        )
        out.append(
            "The IOI `null_matches_or_exceeds` classes are null-variance artifacts, not "
            "soft nulls: under random answer names the faithfulness denominator (clean − "
            "corrupted logit difference) is near zero and the null score explodes "
            f"({detail}). A pooled sd that large pushes d towards 0 whatever the means. "
            "The rule is unchanged; read these rows by their null mean and sd, not by "
            "their class."
        )
    duplicates = report["duplicates_of_base_cards"]
    if duplicates:
        out.append(
            f"The {len(duplicates)} `refusal_direction_*.directions` cards re-express the "
            "same runs and null summary as their base cards with direction structure; "
            "they are analysed (see Duplicates) but excluded from the crosstab and the "
            "headline so each battery counts once."
        )
    gemma = by_name.get("sycophancy_gemma3_12b_it")
    if gemma:
        leak = gemma["leak"]
        out.append(
            f"`sycophancy_gemma3_12b_it` `{leak['leak_class']}` is what the paper "
            "predicts: on Gemma the transfer drop is small (claim label 'shared', drop < "
            f"0.15; real {leak['real_mean']:.3f} vs null {leak['null_mean']:.3f}), so a "
            "null with no drop reproduces the finding. The score cannot separate a shared "
            "representation from none, which is a limit of the score, not a soft null."
        )
    soft = []
    ams = by_name.get("ams_safety_scanner")
    if ams:
        soft.append(
            f"`ams_safety_scanner` (null LOO accuracy {ams['leak']['null_mean']:.3f} vs "
            f"real {ams['leak']['real_mean']:.3f}: swapping half the pair labels keeps "
            "most of the accuracy)"
        )
    swd = by_name.get("swd_gpt2")
    if swd:
        soft.append(
            f"`swd_gpt2` (CI on polarity·(real − null) "
            f"{fmt_ci(swd['leak']['ci_difference'])}: random-token calibration blocks "
            "match the real CE delta)"
        )
    if soft:
        out.append(
            "Soft nulls are not confined to the structure-preserving family: "
            + "; ".join(soft)
            + "."
        )
    return out


def markdown(report: Dict[str, Any]) -> str:
    cards = report["cards"]
    duplicates = report["duplicates_of_base_cards"]
    table = report["crosstab"]
    head = report["headline"]
    lines = [
        "# Null-score leak",
        "",
        "Does each card's null control still SCORE like the real data? The specificity",
        "check compares structural stability only; `metrics.null_control.score_mean` is",
        "recorded but never checked. `d` is the signed standardized difference",
        "`polarity · (real − null) / pooled sd`; retention is `null / real` on ratio-scale",
        "scores; the CI is a percentile bootstrap on `polarity · (real − null)` when",
        "per-run null scores exist. Classes (thresholds are choices, see",
        "`stresskit/null_leak.py`): `null_matches_or_exceeds` when d ≤ "
        f"{THRESHOLDS['d_matches_max']} or retention ≥ {THRESHOLDS['retention_matches_min']};",
        f"`null_degraded` when d ≥ {THRESHOLDS['d_degraded_min']}, z ≥ "
        f"{THRESHOLDS['z_degraded_min']} and (retention ≤ "
        f"{THRESHOLDS['retention_degraded_max']} or not a ratio scale); else `partial`.",
        "",
        f"{len(cards) + len(duplicates)} cards with a null control; "
        f"{report['n_batteries_counted']} batteries counted ({len(duplicates)} "
        "`.directions` duplicates of their base cards, listed below); "
        f"{sum(1 for c in cards if c['null_scores_source'] == 'sidecar')} with per-run "
        "null scores. Real and null cells are `mean ± sd (n)`.",
        "",
        "| card | family | specificity | real score | null score | d | retention | "
        "95% CI polarity·(real − null) | class |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for card in cards:
        leak = card["leak"]
        lines.append(
            f"| {card['card']} | {card['family']} | {card['specificity']['state']} "
            f"| {fmt_group(leak['real_mean'], leak['real_sd'], leak['n_real'])} "
            f"| {fmt_group(leak['null_mean'], leak['null_sd'], leak['n_null'])} "
            f"| {fmt(leak['d'], '.2f')} | {fmt(leak['retention'], '.2f')} "
            f"| {fmt_ci(leak['ci_difference'])} | {leak['leak_class']} |"
        )
    lines += [
        "",
        "## Crosstab: null family × specificity state → leak class",
        "",
        "| null family | specificity | " + " | ".join(CLASSES) + " | total |",
        "|---|---|" + "---|" * (len(CLASSES) + 1),
    ]
    for family in FAMILIES:
        for state in STATES:
            counts = table[family][STATE_LABEL[state]]
            lines.append(
                f"| {family} | {STATE_LABEL[state]} | "
                + " | ".join(str(counts[cls]) for cls in CLASSES)
                + f" | {sum(counts.values())} |"
            )
    lines += [
        "",
        "## Headline",
        "",
        "- structure-preserving null, specificity FAIL, null matches or exceeds the real "
        f"score (the null was too soft): **{head['structure_fail_null_matches_or_exceeds']}**",
        "- structure-preserving null, specificity FAIL, null degraded (the method is "
        "non-specific in structure while the score is task-specific): "
        f"**{head['structure_fail_null_degraded']}**",
        "",
        "## Duplicates of base cards",
        "",
        "Analysed identically to their base card (same runs, same null summary); "
        "excluded from the crosstab and the headline.",
        "",
    ]
    if duplicates:
        lines += ["| card | base | class |", "|---|---|---|"]
        lines += [
            f"| {d['card']} | {d['base']} | {d['record']['leak']['leak_class']} |"
            for d in duplicates
        ]
    else:
        lines.append("- none")
    lines += ["", "## Caveats", ""]
    lines += [f"- {text}" for text in caveats(report)] or ["- none"]
    lines += ["", "## Skipped", ""]
    lines += [f"- {s['card']}: {s['reason']}" for s in report["skipped"]] or ["- none"]
    lines += [
        "",
        "## Score polarity and scale, by card-name prefix",
        "",
        "| prefix | polarity | scale | evidence |",
        "|---|---|---|---|",
    ]
    for prefix in SCORE_POLARITY:
        lines.append(
            f"| {prefix} | {SCORE_POLARITY[prefix]:+d} | {SCORE_SCALE[prefix]} "
            f"| {SCORE_EVIDENCE[prefix]} |"
        )
    lines += [
        "",
        "Inputs digest (sha256 over the sorted (path, sha256) of every analysed card, "
        f"duplicates included): `{report['cards_digest']}`.",
        "",
        "*Generated by `references/null_score_leak.py` — do not edit by hand.*",
    ]
    return "\n".join(lines)


def close(a: float, b: float) -> bool:
    return abs(a - b) <= MEAN_TOLERANCE * max(1.0, abs(b))


def split_duplicates(
    records: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """The counted batteries and the ``.directions`` cards that re-analyse one.

    A ``<stem>.directions`` card re-expresses its base card's runs and null
    summary with direction structure. Its leak numbers must agree with the
    base's; it is then reported under the base instead of being counted again.
    """
    by_name = {record["card"]: record for record in records}
    counted, duplicates = [], []
    for record in records:
        name = record["card"]
        if not name.endswith(DIRECTIONS_SUFFIX):
            counted.append(record)
            continue
        base_name = name[: -len(DIRECTIONS_SUFFIX)]
        base = by_name.get(base_name)
        if base is None:
            raise SystemExit(
                f"{name}: base card {base_name} is not among the analysed cards; "
                "analyse the base stem instead"
            )
        for key in ("real_mean", "n_real", "null_mean", "n_null"):
            if not close(record["leak"][key], base["leak"][key]):
                raise SystemExit(
                    f"{name}: {key} {record['leak'][key]!r} differs from "
                    f"{base_name}'s {base['leak'][key]!r}; not a duplicate"
                )
        duplicates.append({"card": name, "base": base_name, "record": record})
    return counted, duplicates


def build_report(references_dir: str, only: Optional[str]) -> Dict[str, Any]:
    rows = [r for r in collect_rows([references_dir]) if r["kind"] == "stability card"]
    if only is not None:
        rows = [r for r in rows if os.path.basename(r["path"])[: -len(".json")] == only]
        if not rows:
            raise SystemExit(
                f"--only {only}: no stability card with that stem under {references_dir}"
            )
    records, skipped = [], []
    for row in rows:
        record = analyze_card(row)
        if record is None:
            skipped.append(
                {
                    "card": os.path.basename(row["path"])[: -len(".json")],
                    "reason": "no null control",
                }
            )
        else:
            records.append(record)
    cards, duplicates = split_duplicates(records)
    state_rank = {STATE_LABEL[s]: i for i, s in enumerate(STATES)}
    cards.sort(
        key=lambda c: (c["family"], state_rank[c["specificity"]["state"]], c["card"])
    )
    duplicates.sort(key=lambda d: d["card"])
    skipped.sort(key=lambda s: s["card"])
    table = crosstab(cards)
    return {
        "schema_version": SCHEMA_VERSION,
        "thresholds": THRESHOLDS,
        "score_polarity": SCORE_POLARITY,
        "score_scale": SCORE_SCALE,
        "n_batteries_counted": len(cards),
        "cards": cards,
        "duplicates_of_base_cards": duplicates,
        "skipped": skipped,
        "crosstab": table,
        "headline": headline(table),
        "cards_digest": cards_digest(cards + [d["record"] for d in duplicates]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--references", default=HERE)
    parser.add_argument(
        "--out", default=OUT_DIR, help="directory for the JSON and markdown"
    )
    parser.add_argument(
        "--json-only", action="store_true", help="write the JSON, skip the markdown"
    )
    parser.add_argument(
        "--only",
        metavar="CARD",
        help="analyze one card stem (e.g. swd_gpt2) and print it; writes no artifact",
    )
    args = parser.parse_args()
    report = build_report(args.references, args.only)
    if args.only is not None:
        print(markdown(report))
        return
    os.makedirs(args.out, exist_ok=True)
    json_path = os.path.join(args.out, JSON_NAME)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=1, ensure_ascii=False)
        handle.write("\n")
    if args.json_only:
        print(
            f"{report['n_batteries_counted']} batteries counted, "
            f"{len(report['duplicates_of_base_cards'])} duplicates; wrote {rel(json_path)}"
        )
        return
    text = markdown(report)
    md_path = os.path.join(args.out, MD_NAME)
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(text + "\n")
    print(text)
    print(f"\nwrote {rel(json_path)} and {rel(md_path)}")


if __name__ == "__main__":
    main()
