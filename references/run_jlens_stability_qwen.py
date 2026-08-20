"""Reference Stability Card: Jacobian-lens readouts on Qwen3.5-4B.

Stresses the workspace-readout protocol of anthropics/jacobian-lens
("Verbalizable Representations Form a Global Workspace in Language
Models") on its own evaluation sets, using the released pre-fitted lens.

The finding under test, per the upstream hit criterion: the evoked
concept appears at lens rank <= k at some layer of the workspace band,
read at a single prompt position. The battery stresses the analytic
choices that criterion hides:

- k (pass@k cutoff), workspace band definition, readout position
  (hyperparams axis);
- item resampling (bootstrap) and subsampling (seeds axis);
- evaluation distribution: association vignettes vs multihop facts
  (templates axis);
- a derangement null: the same prompts scored against permuted targets
  (specificity check);
- junk contamination of the raw top-10 readout (the repo masks
  non-word-like tokens for display; the card measures them instead).

Phase 1 (GPU) caches ranked readouts per (item, position, layer); phase 2
(CPU) runs the battery from the cache. Re-runs skip phase 1.

Usage (GPU box with anthropics/jacobian-lens cloned + installed):
    python references/run_jlens_stability_qwen.py \
        [--jlens-repo ~/work/jacobian-lens] [--out-dir references/cards]
"""

import argparse
import gzip
import json
import os
import random

import stresskit as sk
from stresskit.adapters import jlens as skj

MODEL_NAME = "Qwen/Qwen3.5-4B"
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_REVISION = "qwen-n1000"
LENS_FILE = "qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt"

EVAL_SETS = ["lens-eval-association", "lens-eval-multihop"]
POSITIONS = [-1, -2]
TOP_N = 50   # ranks cached per (item, position, layer)


def load_items(jlens_repo, slug):
    path = os.path.join(jlens_repo, "data", "evaluations", f"{slug}.json")
    with open(path) as f:
        payload = json.load(f)
    items = payload["items"] if isinstance(payload, dict) else payload
    return [
        {"name": it.get("name", f"{slug}-{i}"),
         "prompt": it["prompt"].rstrip(),
         "intermediates": it["intermediates"]}
        for i, it in enumerate(items)
    ]


def precompute(jlens_repo, cache_path):
    import torch
    import transformers
    import jlens

    hf = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16
    ).cuda()
    tok = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    model = jlens.from_hf(hf, tok)
    lens = jlens.JacobianLens.from_pretrained(
        LENS_REPO, filename=LENS_FILE, revision=LENS_REVISION
    )
    n_layers = model.n_layers
    layers = list(lens.source_layers)   # the final layer(s) are not fitted

    cache = {"model": MODEL_NAME, "n_layers": n_layers, "sets": {}}
    for slug in EVAL_SETS:
        items = load_items(jlens_repo, slug)
        rows = []
        for it in items:
            lens_logits, _, _ = lens.apply(
                model, it["prompt"], layers=layers, positions=POSITIONS
            )
            readouts = {}
            for pi, pos in enumerate(POSITIONS):
                per_layer = {}
                for layer, logits in lens_logits.items():
                    top = logits[pi].topk(TOP_N).indices.tolist()
                    per_layer[str(layer)] = [tok.decode([t]) for t in top]
                readouts[str(pos)] = per_layer
            rows.append({**it, "readouts": readouts})
            print(f"[{slug}] {it['name']}: cached")
        cache["sets"][slug] = rows

    with gzip.open(cache_path, "wt") as f:
        json.dump(cache, f)
    print(f"cache -> {cache_path}")
    return cache


def make_finder(n_layers):
    def finder(data, seed, config):
        k = config.get("k", 5)
        band = config.get("band", "mid-third")
        pos = str(config.get("pos", -1))
        frac = config.get("subsample", 0.75)

        rng = random.Random(seed)
        sample = rng.sample(list(data), max(16, int(frac * len(data))))

        hits, best_layers, junk = [], [], []
        for it in sample:
            ranked_by_layer = {
                int(layer): ranked
                for layer, ranked in it["readouts"][pos].items()
            }
            # restrict the band to layers the lens was fitted on
            band_set = [L for L in skj.band_layers(n_layers, band)
                        if L in ranked_by_layer]
            for layer in band_set:
                junk.append(skj.junk_share(ranked_by_layer[layer][:10]))
            ranks = [
                skj.min_rank(ranked_by_layer, t, layers=band_set)
                for t in it["intermediates"]
            ]
            found = [r for r in ranks if r is not None and r <= k]
            if len(found) == len(it["intermediates"]):
                hits.append(it["name"])
                best = min(
                    (skj.min_rank({L: ranked_by_layer[L]}, t) or TOP_N + 1, L)
                    for t in it["intermediates"] for L in band_set
                )[1]
                best_layers.append(best)

        thirds = [0, 0, 0]
        for L in best_layers:
            thirds[min(2, 3 * L // n_layers)] += 1
        claim = ["early", "middle", "late"][thirds.index(max(thirds))] \
            if best_layers else "none"

        return sk.feature_set(
            hits,
            claim=claim,
            score=len(hits) / len(sample),
            universe_size=len(data),
            mean_junk_share_top10=round(sum(junk) / len(junk), 4),
        )

    return finder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jlens-repo", default=os.path.expanduser("~/work/jacobian-lens"))
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "cards"))
    ap.add_argument("--cache", default=None)
    ap.add_argument("--n-runs", type=int, default=6)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cache_path = args.cache or os.path.join(args.out_dir, "jlens_qwen_readouts.json.gz")

    if os.path.exists(cache_path):
        with gzip.open(cache_path, "rt") as f:
            cache = json.load(f)
        print(f"using cached readouts: {cache_path}")
    else:
        cache = precompute(args.jlens_repo, cache_path)

    n_layers = cache["n_layers"]
    association = cache["sets"]["lens-eval-association"]
    multihop = cache["sets"]["lens-eval-multihop"]

    # derangement null: same prompts, rotated targets (no fixed points)
    null_data = [
        {**it, "intermediates": association[(i + 1) % len(association)]["intermediates"]}
        for i, it in enumerate(association)
    ]

    result = sk.stress(
        make_finder(n_layers),
        association,
        battery=["seeds", "bootstrap", "templates", "hyperparams"],
        n_runs=args.n_runs,
        config={"k": 5, "band": "mid-third", "pos": -1},
        templates={"multihop": multihop},
        hyperparams={"k": [1, 10], "band": ["mid-half", "all"], "pos": [-2]},
        null_data=null_data,
        claim_statement=(
            "J-lens readouts surface the evoked concept at rank <= 5 "
            "within a mid-layer workspace band"
        ),
        model=MODEL_NAME,
        task="lens-eval-association (vs multihop)",
        method="Jacobian lens (pre-fitted, n=1000), upstream hit criterion",
        verbose=True,
    )

    print()
    print(result)
    print(result.to_markdown())

    base = os.path.join(args.out_dir, "jlens_qwen3p5_4b")
    result.card.save(base + ".json")
    with open(base + ".md", "w") as f:
        f.write(result.to_markdown() + "\n")
    with open(base + ".badge.json", "w") as f:
        json.dump(result.card.badge_dict(), f, indent=2)
        f.write("\n")

    junk_vals = [r.finding.meta.get("mean_junk_share_top10") for r in result.runs]
    junk_vals = [v for v in junk_vals if v is not None]
    print(f"\nmean junk share of top-10 readouts across runs: "
          f"{sum(junk_vals) / len(junk_vals):.3f}")
    print(f"artifacts -> {base}.*")


if __name__ == "__main__":
    main()
