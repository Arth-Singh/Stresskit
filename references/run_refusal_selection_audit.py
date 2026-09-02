"""Does the refusal-direction selection rule pick the best available direction?

The battery in `run_refusal_direction_card.py` shows that on some models the
selected direction's held-out effect swings widely across seeds (Qwen3.5-9B:
coherent compliance 0.13 to 0.98) while the selected layer alternates between
two bands. The battery cannot say whether that is the *direction* being fragile
or the *selection rule* being noisy, because each run records only the candidate
it chose.

This script opens that black box for one extraction split. It scores every
(layer, position) candidate with the upstream selection objective — mean
first-token refusal log-odds under directional ablation, on N validation
prompts — and then measures what actually matters for each of the top-k
candidates: coherent compliance on the held-out harmful set under generation.

Three pre-registered questions, decided before running:

Q1. Among candidates, does the selection objective predict held-out effect?
    (Spearman rank correlation over the scored candidates.)
Q2. Does the rule's argmin achieve the best held-out effect among the top-k it
    was choosing between, and how large is the gap?
Q3. Is the gap explained by the size of the validation set? The objective is
    recomputed on disjoint validation halves and on a 4x larger set; if the
    rule's choice is stable under those but still suboptimal, the objective is
    misspecified rather than undersampled.

No grade is issued. This is a diagnostic on the finder, not an audit of the
claim, and it does not produce a Stability Card.

Usage:
    python references/run_refusal_selection_audit.py --model Qwen/Qwen3.5-9B \
        --data-dir refusal_data --out references/cards/refusal_selection_audit_qwen3p5_9b.json
"""

import argparse
import json
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_refusal_direction_card as rd  # noqa: E402

TOP_K = 10
N_VAL_SMALL = 32     # the upstream selection-set size the battery uses
N_VAL_LARGE = 128


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data-dir", default="refusal_data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-per-class", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    harmful_train = rd.load_split(args.data_dir, "harmful_train")
    harmless_train = rd.load_split(args.data_dir, "harmless_train")
    harmful_val = rd.load_split(args.data_dir, "harmful_val")
    harmless_val = rd.load_split(args.data_dir, "harmless_val")
    harmful_test = rd.load_split(args.data_dir, "harmful_test")
    harmless_test = rd.load_split(args.data_dir, "harmless_test")

    fixed = random.Random(0)
    n_pool = len(harmful_train) + len(harmful_val)
    pool_h = rd.items(harmful_train + harmful_val, "harmful")
    pool_l = rd.items(fixed.sample(harmless_train + harmless_val, n_pool), "harmless")
    eval_harmful = rd.items(fixed.sample(harmful_test, rd.N_EVAL), "harmful")
    eval_harmless = rd.items(fixed.sample(harmless_test, rd.N_EVAL), "harmless")

    # the base run's split, reproduced: the finder samples with its own seed
    rng = random.Random(0)
    ph = rng.sample(pool_h, min(args.n_per_class + N_VAL_LARGE, len(pool_h)))
    pl = rng.sample(pool_l, min(args.n_per_class + N_VAL_LARGE, len(pool_l)))
    val_h, extract_h = ph[:N_VAL_LARGE], ph[N_VAL_LARGE:]
    val_l, extract_l = pl[:N_VAL_LARGE], pl[N_VAL_LARGE:]

    print(f"loading {args.model} ...")
    subject = rd.Subject(args.model, args.device, batch_size=args.batch_size)
    print(f"{subject.n_layers} layers; extraction {len(extract_h)}+{len(extract_l)}, "
          f"validation pool {N_VAL_LARGE}, held-out {rd.N_EVAL}+{rd.N_EVAL}")

    clean_comps, first_ids = subject.complete(eval_harmful)
    refusal_ids = sorted({t for c, t in zip(clean_comps, first_ids) if rd.refused(c)})
    clean_flags, _ = subject.degenerate_flags(eval_harmful, clean_comps)
    clean_complied = sum((not rd.refused(c)) and (not d)
                         for c, d in zip(clean_comps, clean_flags)) / len(clean_comps)
    clean_harmless_refusal = sum(map(rd.refused, subject.complete(eval_harmless)[0])) / rd.N_EVAL
    print(f"clean: refusal {sum(map(rd.refused, clean_comps)) / rd.N_EVAL:.3f}, "
          f"coherent compliance {clean_complied:.3f}; {len(refusal_ids)} refusal tokens")

    positions = [-5, -4, -3, -2, -1]
    resid_h = subject.residuals(extract_h, positions)
    resid_l = subject.residuals(extract_l, positions)
    diff = resid_h.mean(dim=1) - resid_l.mean(dim=1)

    def logodds(dist):
        p = dist[:, refusal_ids].sum(dim=-1).double()
        return (torch.log(p + 1e-8) - torch.log(1 - p + 1e-8))

    val_small_h, val_small_l = val_h[:N_VAL_SMALL], val_l[:N_VAL_SMALL]
    half_a_h, half_b_h = val_h[:N_VAL_LARGE // 2], val_h[N_VAL_LARGE // 2:]
    clean_val_l = subject.first_token_dist(val_l)

    last_layer = int(subject.n_layers * (1.0 - rd.PRUNE_LAYER_FRACTION))
    candidates = []
    for layer in range(last_layer):
        for pi, pos in enumerate(positions):
            vec = diff[layer, pi]
            norm = float(vec.norm())
            if norm < 1e-6:
                continue
            unit = vec / norm
            handles = subject.ablation_hooks(unit)
            lo_h = logodds(subject.first_token_dist(val_h))
            abl_val_l = subject.first_token_dist(val_l)
            subject.remove(handles)
            handles = subject.addition_hooks(unit, norm, layer)
            add = float(logodds(subject.first_token_dist(val_small_l)).mean())
            subject.remove(handles)
            candidates.append({
                "layer": layer, "position": pos, "coeff": norm,
                "objective_val32": float(lo_h[:N_VAL_SMALL].mean()),
                "objective_val128": float(lo_h.mean()),
                "objective_half_a": float(lo_h[:N_VAL_LARGE // 2].mean()),
                "objective_half_b": float(lo_h[N_VAL_LARGE // 2:].mean()),
                "kl": rd.kl_divergence(clean_val_l, abl_val_l),
                "addition_logodds": add,
                "unit": unit,
            })
        print(f"  layer {layer}/{last_layer} scored", flush=True)

    def admissible(key):
        ok = [c for c in candidates
              if c["kl"] <= rd.KL_MAX and c["addition_logodds"] >= rd.INDUCE_MIN]
        return sorted(ok or candidates, key=lambda c: c[key])

    ranked = admissible("objective_val32")
    print(f"{len(ranked)} admissible candidates; evaluating the top {TOP_K} held out")
    for c in ranked[:TOP_K]:
        handles = subject.ablation_hooks(c["unit"])
        comps, _ = subject.complete(eval_harmful)
        subject.remove(handles)
        flags, _ = subject.degenerate_flags(eval_harmful, comps)
        c["heldout_complied"] = sum((not rd.refused(x)) and (not d)
                                    for x, d in zip(comps, flags)) / len(comps)
        c["heldout_degenerate"] = sum(flags) / len(flags)
        handles = subject.addition_hooks(c["unit"], c["coeff"], c["layer"])
        comps_l, _ = subject.complete(eval_harmless)
        subject.remove(handles)
        flags_l, _ = subject.degenerate_flags(eval_harmless, comps_l)
        c["heldout_induced"] = sum(rd.refused(x) and (not d)
                                   for x, d in zip(comps_l, flags_l)) / len(comps_l)
        print(f"  L{c['layer']:2d} pos{c['position']} obj32={c['objective_val32']:+.3f} "
              f"obj128={c['objective_val128']:+.3f} complied={c['heldout_complied']:.3f} "
              f"induced={c['heldout_induced']:.3f}", flush=True)

    scored = ranked[:TOP_K]
    def spearman(a, b):
        ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
        return float(np.corrcoef(ra, rb)[0, 1])

    obj32 = [c["objective_val32"] for c in scored]
    obj128 = [c["objective_val128"] for c in scored]
    held = [c["heldout_complied"] for c in scored]
    picked32, picked128 = scored[0], min(scored, key=lambda c: c["objective_val128"])
    best = max(scored, key=lambda c: c["heldout_complied"])

    result = {
        "model": args.model,
        "n_candidates": len(candidates),
        "n_admissible": len(ranked),
        "n_evaluated": len(scored),
        "clean_complied": round(clean_complied, 4),
        "clean_harmless_refusal": round(clean_harmless_refusal, 4),
        "refusal_token_ids": refusal_ids,
        "q1_spearman_objective32_vs_heldout": round(spearman(obj32, held), 3),
        "q1_spearman_objective128_vs_heldout": round(spearman(obj128, held), 3),
        "q2_picked_by_val32": {k: picked32[k] for k in
                               ("layer", "position", "objective_val32", "objective_val128",
                                "heldout_complied", "heldout_induced")},
        "q2_best_available": {k: best[k] for k in
                              ("layer", "position", "objective_val32", "objective_val128",
                               "heldout_complied", "heldout_induced")},
        "q2_gap": round(best["heldout_complied"] - picked32["heldout_complied"], 4),
        "q3_picked_by_val128": {k: picked128[k] for k in
                                ("layer", "position", "objective_val128",
                                 "heldout_complied", "heldout_induced")},
        "q3_half_split_agreement": {
            "argmin_half_a": min(ranked, key=lambda c: c["objective_half_a"])["layer"],
            "argmin_half_b": min(ranked, key=lambda c: c["objective_half_b"])["layer"],
            "spearman_half_a_vs_half_b": round(spearman(
                [c["objective_half_a"] for c in ranked],
                [c["objective_half_b"] for c in ranked]), 3),
        },
        "candidates": [{k: (round(v, 5) if isinstance(v, float) else v)
                        for k, v in c.items() if k != "unit"} for c in candidates],
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
        f.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k != "candidates"}, indent=1))
    print(f"\nwritten {args.out}")


if __name__ == "__main__":
    main()
