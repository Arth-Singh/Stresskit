"""Direction-native Stability Card for the refusal direction (arXiv:2406.11717).

Post-hoc regrade of an already-published battery. It runs no model and no GPU
work: ``run_refusal_direction_card.py`` saved every run's unit direction to
``cards/raw/refusal_<slug>/direction_<hash>.npy`` and wrote the run ledger to
``cards/refusal_direction_<slug>.runs.json``; this script reads those and
grades them with ``stresskit.from_findings``.

Why regrade at all. The published card had to invent a set proxy for a
direction — the top-32 vocabulary tokens the unit direction unembeds to — and
grade it with Jaccard, because StressKit had no direction-native structural
metric. That proxy has a ceiling: over 210 real run pairs on Llama-3.1-8B, run
pairs whose directions agree to cosine >= 0.98 share only 0.68 of their readout
tokens. The structural check was therefore measuring the readout, not the
direction. ``stresskit.direction`` grades the direction itself with mean
pairwise |cosine|; this card is the same 21 runs, same claims, same scores,
same null control, with that one substitution.

The published readout-proxy cards are not touched. This writes
``refusal_direction_<slug>.directions.{json,md}`` beside them.

    PYTHONPATH=src python references/refusal_direction_card_posthoc.py
"""

import argparse
import json
import os
import sys
from collections import Counter, OrderedDict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import stresskit as sk  # noqa: E402
from stresskit import metrics as M  # noqa: E402

CARDS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cards")

CLAIM_TEMPLATE = (
    "Refusal in {model} is mediated by a single residual-stream direction: "
    "ablating it removes refusal on held-out harmful instructions and adding "
    "it induces refusal on harmless ones"
)


def load_runs(cards_dir, slug):
    """The run ledger plus each run's saved unit direction, in file order."""
    ledger_path = os.path.join(cards_dir, f"refusal_direction_{slug}.runs.json")
    with open(ledger_path, encoding="utf-8") as f:
        ledger = json.load(f)
    raw_dir = os.path.join(cards_dir, ledger["raw_dir"])
    rows = []
    for run in ledger["runs"]:
        digest = run["meta"]["direction_sha256_16"]
        path = os.path.join(raw_dir, f"direction_{digest}.npy")
        if not os.path.exists(path):
            raise SystemExit(
                f"missing saved direction {path} for run "
                f"{run['group']}/{run['variant']} — regenerate the raw dir "
                "with references/run_refusal_direction_card.py before "
                "regrading it post hoc"
            )
        rows.append((run, np.load(path)))
    return ledger, rows


def to_finding(run, vector):
    return sk.direction(
        vector.tolist(),
        claim=run["claim"],
        score=run["score"],
        layer=run["meta"]["layer"],
        position=run["meta"]["position"],
        source_direction_sha256_16=run["meta"]["direction_sha256_16"],
    )


def layer_geometry(findings):
    """Mean and min pairwise |cos| within each selected layer, and overall."""
    by_layer = OrderedDict()
    for f in findings:
        by_layer.setdefault(f.meta["layer"], []).append(f.vector)
    out = []
    for layer, vectors in sorted(by_layer.items()):
        if len(vectors) < 2:
            continue
        pairs = M.pairwise_abs_cosine(vectors)
        out.append((layer, len(vectors), sum(pairs) / len(pairs), min(pairs)))
    return out


def same_layer_regrade(findings, null_findings, seed):
    """Grade each selected-layer group on its own, for the card to report
    beside the pooled verdict."""
    by_layer = OrderedDict()
    for f in findings:
        by_layer.setdefault(f.meta["layer"], []).append(f)
    lines = []
    for layer, group in sorted(by_layer.items(), key=lambda kv: -len(kv[1])):
        if len(group) < 4:  # below the bootstrap's minimum; no interval
            lines.append(f"L{layer} (n={len(group)}): too few runs to regrade")
            continue
        sub = sk.from_findings(group, null_findings=null_findings, seed=seed)
        ci = sub.pooled["mean_pairwise_abs_cosine_ci95"]
        lines.append(
            f"L{layer} (n={len(group)}): |cos| "
            f"{sub.pooled['mean_pairwise_abs_cosine']:.3f} "
            f"[{ci[0]:.3f}, {ci[1]:.3f}], beats_random "
            f"{sub.checks['beats_random']['value']:.1f}x, specificity "
            f"{sub.checks['specificity']['value']:.1f}x, grade {sub.grade}"
        )
    return lines


def published_readout_numbers(cards_dir, slug):
    """The graded structural check of the published readout-proxy card."""
    path = os.path.join(cards_dir, f"refusal_direction_{slug}.json")
    with open(path, encoding="utf-8") as f:
        card = json.load(f)
    check = card["verdict"]["checks"]["structural_stability"]
    return card["verdict"]["grade"], check["value"], check["ci"]


def published_model(cards_dir, slug):
    """The model the published card names, so a regrade of another slug can
    never inherit this file's default model in its claim statement."""
    path = os.path.join(cards_dir, f"refusal_direction_{slug}.json")
    with open(path, encoding="utf-8") as f:
        model = json.load(f)["claim"].get("model")
    if not model:
        raise SystemExit(f"{path}: card records no model to regrade against")
    return model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--slug", default="meta_llama_3p1_8b_instruct")
    parser.add_argument("--cards-dir", default=CARDS)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    model = published_model(args.cards_dir, args.slug)
    ledger, rows = load_runs(args.cards_dir, args.slug)
    real = [(run, vec) for run, vec in rows if run["group"] == "real"]
    nulls = [(run, vec) for run, vec in rows if run["group"] != "real"]
    if not real or not nulls:
        raise SystemExit("the run ledger must carry both real and null runs")

    real_findings = [to_finding(run, vec) for run, vec in real]
    null_findings = [to_finding(run, vec) for run, vec in nulls]
    axes = [run["axis"] for run, _ in real[1:]]

    result = sk.from_findings(
        real_findings,
        axes=axes,
        null_findings=null_findings,
        seed=args.seed,
        claim_statement=CLAIM_TEMPLATE.format(model=model),
        model=model,
        task="refusal direction (harmful vs harmless instructions, upstream splits)",
        method="difference-in-means direction, upstream selection rule, "
               "directional ablation / activation addition",
    )
    card = result.card
    dim = result.pooled["direction_dim"]
    graded = card.directions["order"]

    card.notes.append(
        "post-hoc regrade: the findings are the unit directions "
        "run_refusal_direction_card.py saved for the published card of the "
        f"same battery ({ledger['cache_key']}); no model was run and the "
        "published card, its runner and its raw outputs are unchanged. The "
        "only substitution is the structural metric: mean pairwise |cosine| "
        "between the directions themselves instead of Jaccard over the top-32 "
        "logit-lens readout tokens that stood in for them."
    )
    grade, readout_j, readout_ci = published_readout_numbers(
        args.cards_dir, args.slug)
    ac = result.pooled["mean_pairwise_abs_cosine"]
    ac_ci = result.pooled["mean_pairwise_abs_cosine_ci95"]
    card.notes.append(
        "readout proxy vs direction (the reason this card exists): the "
        f"published card grades top-32 readout Jaccard {readout_j:.3f} "
        f"[{readout_ci[0]:.3f}, {readout_ci[1]:.3f}] (grade {grade}) over "
        f"these same runs; the directions themselves agree to |cos| "
        f"{ac:.3f} [{ac_ci[0]:.3f}, {ac_ci[1]:.3f}]. Over 210 real run pairs "
        "the pairs whose directions agree to cosine >= 0.98 share only 0.68 "
        "of their readout tokens, so the proxy's structural check was bounded "
        "well below 1 for runs that recovered the same object. This card "
        "measures the object."
    )
    signed = [M.cosine_similarity(a.vector, b.vector)
              for i, a in enumerate(real_findings)
              for b in real_findings[i + 1:]]
    card.notes.append(
        "|cos| is graded, not signed cosine: a difference-in-means direction "
        "points from whichever class the extraction labelled positive, so its "
        "sign is a convention of the pipeline rather than a property of the "
        "model, and a run that flipped it would otherwise score as a total "
        f"structural failure. On this battery the signed cosines run from "
        f"{min(signed):.3f} to {max(signed):.3f} (mean "
        f"{sum(signed) / len(signed):.3f}), so no run flipped and the two "
        "metrics coincide here; the check does not depend on that holding."
    )

    layers = Counter(f.meta["layer"] for f in real_findings)
    within = layer_geometry(real_findings)
    within_txt = "; ".join(
        f"L{layer} (n={n}): mean {mean:.3f}, min {lo:.3f}"
        for layer, n, mean, lo in within
    )
    card.notes.append(
        "layer selection is inside the battery, so all "
        f"{len(real_findings)} real runs are graded together even though the "
        "upstream selection rule did not always choose the same layer "
        f"({', '.join(f'L{k}: {v}' for k, v in sorted(layers.items()))}). "
        "Two directions read off different layers are coordinates in the "
        "residual stream at different points of the forward pass; their "
        "cosine is defined but it mixes 'is the direction stable' with 'did "
        "the selection rule land in the same place'. Grading only same-layer "
        "runs would answer a conditional question and silently drop runs that "
        "moved, which is exactly the instability the battery exists to "
        f"surface. Within-layer values, not graded: {within_txt}."
    )
    card.notes.append(
        "same-layer regrade, not the verdict (each selected-layer group "
        "graded on its own against the same null control, first run of the "
        "group standing in as its base): "
        + "; ".join(same_layer_regrade(real_findings, null_findings, args.seed))
        + ". The pooled verdict above is the one this card reports; these say "
        "how much of the pooled spread is the selection rule moving layers."
    )
    card.notes.append(
        f"random null: the exact E[|cos|] between independent uniform unit "
        f"vectors in R^{dim} is "
        f"{result.pooled['expected_random_abs_cosine']:.5f} "
        "(metrics.expected_random_abs_cosine, the closed form the Monte-Carlo "
        "baselines.empirical_random_abs_cosine converges to). Beating that is "
        "a low bar in high dimension — near-orthogonality of random "
        "directions is concentration of measure, not evidence — so read "
        "beats_random as a floor and structural_stability as the check that "
        "carries the verdict."
    )
    card.notes.append(
        "graded order: directions.order indexes the pairwise |cos| matrix and "
        "follows the real runs of "
        f"refusal_direction_{args.slug}.runs.json in file order, whose "
        "meta.direction_sha256_16 names the saved vector under "
        f"{ledger['raw_dir']}/. Post-hoc grading relabels variants by axis "
        f"({graded[0]} .. {graded[-1]}), so the ledger is the mapping back to "
        "the original seed / resample / template / hyperparameter labels."
    )
    card.notes.append(
        "scope: unchanged from the published card — chat usage mode with the "
        "model's default template; refusal judged by the upstream substring "
        "list on the first 32 greedy tokens after folding typographic "
        "apostrophes to ASCII; compliance additionally requires coherence; "
        "held-out evaluation on 64 harmful and 64 harmless test instructions "
        "never seen by the finder; upstream splits at "
        "andyrdt/refusal_direction@9d852fa, SHA-256 verified."
    )
    card.notes.append(
        "null control: the labelled pool with harmful/harmless labels permuted "
        "once (seed 0x5EC); extraction AND selection both run on permuted "
        "labels, and the selected direction is still evaluated on the real "
        "held-out sets. Its directions scatter (|cos| "
        f"{result.null_summary['mean_pairwise_abs_cosine']:.3f} across "
        f"{result.null_summary['n_runs']} runs), so a comparably aligned real "
        "battery would have indicated the procedure, not the model, produces "
        "the structure."
    )

    base = os.path.join(args.cards_dir,
                        f"refusal_direction_{args.slug}.directions")
    card.save(base + ".json")
    with open(base + ".md", "w", encoding="utf-8") as f:
        f.write(card.to_markdown() + "\n")

    with open(base + ".json", encoding="utf-8") as f:
        report = sk.verify_card_dict(json.load(f))
    print(result)
    print(f"structural |cos| {ac:.4f} [{ac_ci[0]:.4f}, {ac_ci[1]:.4f}] "
          f"vs readout Jaccard {readout_j:.4f} "
          f"[{readout_ci[0]:.4f}, {readout_ci[1]:.4f}]")
    print(f"within layer: {within_txt}")
    print(f"wrote {base}.json / .md — verify ok={report['ok']} "
          f"grade={report['recomputed_grade']} {report['problems']}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
