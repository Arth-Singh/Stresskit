"""Reference battery: Jacobian lens vs logit lens vs tuned lens, Qwen3.5-4B.

Answers, under one pre-registered battery, the question the J-lens release
left open: how much better is the Jacobian transport than the two standard
linear transports on the same items, layers, positions and hit criterion?

Three transports of the layer-L residual h into the unembedding basis:

- jlens:  unembed(J_L h)         (released pre-fitted lens, qwen-n1000)
- logit:  unembed(final_norm(h)) (identity transport)
- tuned:  unembed(final_norm(h + A_L h + b_L))
          (affine translators trained by train_tuned_lens_qwen.py on the
           same corpus family the J-lens was fitted on)

Stage 1 (`precompute`, GPU) caches top-100 readouts per (item, position,
layer) for one lens into the same cache schema run_jlens_stability_qwen.py
uses. Stage 2 (`battery`, CPU) grades each lens under the identical battery
and derangement null, and writes a paired per-item comparison.

    python run_lens_baselines_qwen.py precompute --lens logit --out logit.json.gz
    python run_lens_baselines_qwen.py precompute --lens tuned \
        --tuned-ckpt tuned_lens_qwen3p5_4b.pt --out tuned.json.gz
    python run_lens_baselines_qwen.py battery \
        --cache jlens=jlens.json.gz --cache logit=logit.json.gz \
        --cache tuned=tuned.json.gz --out-dir references/cards
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
TOP_N = 100


def _text_cfg(cfg):
    return getattr(cfg, "text_config", None) or cfg


def load_items(jlens_repo, slug):
    path = os.path.join(jlens_repo, "data", "evaluations", f"{slug}.json")
    with open(path) as f:
        payload = json.load(f)
    items = payload["items"] if isinstance(payload, dict) else payload
    return [
        {"name": it.get("name", f"{slug}-{i}"),
         "prompt": it["prompt"].rstrip(),
         "intermediates": it["intermediates"],
         "set": slug}
        for i, it in enumerate(items)
    ]


# --------------------------------------------------------------- stage 1

def precompute(lens_name, jlens_repo, out_path, device, tuned_ckpt=None,
               model_name=MODEL_NAME, lens_file=LENS_FILE,
               lens_revision=LENS_REVISION):
    import torch
    import transformers

    hf = transformers.AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16
    ).to(device)
    hf.eval()
    tok = transformers.AutoTokenizer.from_pretrained(model_name)
    n_layers = _text_cfg(hf.config).num_hidden_layers

    if lens_name == "jlens":
        import jlens
        model = jlens.from_hf(hf, tok)
        lens = jlens.JacobianLens.from_pretrained(
            LENS_REPO, filename=lens_file, revision=lens_revision
        )
        layers = list(lens.source_layers)

        def readouts_for(prompt):
            lens_logits, _, _ = lens.apply(model, prompt, layers=layers,
                                           positions=POSITIONS)
            return {str(pos): {str(L): lens_logits[L][pi].topk(TOP_N).indices.tolist()
                               for L in lens_logits}
                    for pi, pos in enumerate(POSITIONS)}
    else:
        final_norm, lm_head = hf.model.norm, hf.lm_head
        translators = {}
        if lens_name == "tuned":
            blob = torch.load(tuned_ckpt, map_location=device, weights_only=True)
            for L, sd in blob["translators"].items():
                t = torch.nn.Linear(_text_cfg(hf.config).hidden_size, _text_cfg(hf.config).hidden_size,
                                    dtype=torch.float32, device=device)
                t.load_state_dict(sd)
                t.eval()
                translators[int(L)] = t
            layers = sorted(translators)
        else:
            layers = list(range(n_layers))

        @torch.no_grad()
        def readouts_for(prompt):
            ids = tok(prompt, return_tensors="pt").input_ids.to(device)
            hs = hf(ids, output_hidden_states=True).hidden_states
            out = {str(pos): {} for pos in POSITIONS}
            for L in layers:
                h = hs[L + 1][0]                      # [seq, d], post-block L
                if lens_name == "tuned":
                    hf32 = h.float()
                    h = (hf32 + translators[L](hf32)).to(torch.bfloat16)
                logits = lm_head(final_norm(h))
                for pos in POSITIONS:
                    top = logits[pos].topk(TOP_N).indices.tolist()
                    out[str(pos)][str(L)] = top
            return out

    cache = {"model": model_name, "lens": lens_name, "n_layers": n_layers,
             "sets": {}}
    for slug in EVAL_SETS:
        rows = []
        for it in load_items(jlens_repo, slug):
            token_ids = readouts_for(it["prompt"])
            readouts = {pos: {L: [tok.decode([t]) for t in ids]
                              for L, ids in per_layer.items()}
                        for pos, per_layer in token_ids.items()}
            rows.append({**it, "readouts": readouts})
            print(f"[{lens_name}/{slug}] {it['name']}: cached", flush=True)
        cache["sets"][slug] = rows

    with gzip.open(out_path, "wt") as f:
        json.dump(cache, f)
    print(f"cache -> {out_path}")


# --------------------------------------------------------------- stage 2

def make_finder(n_layers):
    def finder(data, seed, config):
        k = config.get("k", 5)
        band = config.get("band", "mid-third")
        pos = str(config.get("pos", -1))
        mask = config.get("mask", True)
        frac = config.get("subsample", 0.75)

        rng = random.Random(seed)
        sample = rng.sample(list(data), max(16, int(frac * len(data))))

        hits, best_layers, junk = [], [], []
        for it in sample:
            ranked_by_layer = {
                int(layer): ([t for t in ranked if skj.is_wordlike(t)]
                             if mask else ranked)
                for layer, ranked in it["readouts"][pos].items()
            }
            band_set = [L for L in skj.band_layers(n_layers, band)
                        if L in ranked_by_layer]
            for layer in band_set:
                junk.append(skj.junk_share(it["readouts"][pos][str(layer)][:10]))
            ranks = [skj.min_rank(ranked_by_layer, t, layers=band_set)
                     for t in it["intermediates"]]
            found = [r for r in ranks if r is not None and r <= k]
            if len(found) == len(it["intermediates"]):
                hits.append(it["name"])
                best = min(
                    (skj.min_rank({L: ranked_by_layer[L]}, t) or TOP_N + 1, L)
                    for t in it["intermediates"] for L in band_set
                )[1]
                best_layers.append(best)

        if best_layers:
            best_layers.sort()
            median_layer = best_layers[len(best_layers) // 2]
            claim = ["early", "middle", "late"][min(2, 3 * median_layer // n_layers)]
        else:
            claim = "none"

        return sk.feature_set(
            hits,
            claim=claim,
            score=len(hits) / len(sample),
            universe_size=len(data),
            universe=sample[0]["set"],
            mean_junk_share_top10=round(sum(junk) / len(junk), 4) if junk else None,
        )

    return finder


def item_min_rank(it, n_layers, pos="-1", band="all", mask=True):
    """Paired statistic: best rank of the item's intermediates, band-wide."""
    ranked_by_layer = {
        int(layer): ([t for t in ranked if skj.is_wordlike(t)] if mask else ranked)
        for layer, ranked in it["readouts"][pos].items()
    }
    band_set = [L for L in skj.band_layers(n_layers, band) if L in ranked_by_layer]
    ranks = [skj.min_rank(ranked_by_layer, t, layers=band_set)
             for t in it["intermediates"]]
    if any(r is None for r in ranks):
        return None
    return max(ranks)   # the item is only as read-out as its worst intermediate


def battery(caches, out_dir, n_runs):
    os.makedirs(out_dir, exist_ok=True)
    results, loaded = {}, {}
    model_name, model_slug = MODEL_NAME, "qwen3p5_4b"
    for lens_name, path in caches.items():
        with gzip.open(path, "rt") as f:
            cache = json.load(f)
        loaded[lens_name] = cache
        model_name = cache["model"]
        model_slug = model_name.split("/")[-1].lower().replace(".", "p").replace("-", "_")
        n_layers = cache["n_layers"]
        multihop = cache["sets"]["lens-eval-multihop"]
        association = cache["sets"]["lens-eval-association"]
        null_data = [
            {**it, "intermediates": multihop[(i + 1) % len(multihop)]["intermediates"]}
            for i, it in enumerate(multihop)
        ]
        result = sk.stress(
            make_finder(n_layers), multihop,
            battery=["seeds", "bootstrap", "templates", "hyperparams"],
            n_runs=n_runs,
            config={"k": 5, "band": "all", "pos": -1, "mask": True},
            templates={"association": association},
            hyperparams={"k": [1, 10], "band": ["mid-half", "mid-third"],
                         "pos": [-2], "mask": [False]},
            null_data=null_data,
            claim_statement=(f"{lens_name} readouts surface the latent intermediate "
                             "at rank <= 5, concentrated in a mid-to-late band"),
            model=model_name, task="lens-eval-multihop (vs association)",
            method=f"{lens_name} transport, upstream hit criterion",
            verbose=True,
        )
        results[lens_name] = result
        base = os.path.join(out_dir, f"lens_baseline_{lens_name}_{model_slug}")
        result.card.save(base + ".json")
        with open(base + ".md", "w") as f:
            f.write(result.to_markdown() + "\n")
        print(f"\n=== {lens_name} ===\n{result}")

    # paired per-item comparison, identical items and criterion
    lens_names = list(caches)
    comparison = {"model": model_name, "positions": POSITIONS, "sets": {}}
    for slug in EVAL_SETS:
        per_lens_ranks = {}
        for lens_name in lens_names:
            cache = loaded[lens_name]
            per_lens_ranks[lens_name] = {
                it["name"]: item_min_rank(it, cache["n_layers"])
                for it in cache["sets"][slug]
            }
        names = sorted(per_lens_ranks[lens_names[0]])
        rows = []
        for name in names:
            rows.append({"item": name, **{ln: per_lens_ranks[ln][name]
                                          for ln in lens_names}})
        summary = {}
        for ln in lens_names:
            ranks = [r for r in per_lens_ranks[ln].values()]
            hits5 = sum(1 for r in ranks if r is not None and r <= 5)
            found = [r for r in ranks if r is not None]
            summary[ln] = {
                "hit@5": round(hits5 / len(ranks), 4),
                "found@100": round(len(found) / len(ranks), 4),
                "median_rank_when_found": (sorted(found)[len(found) // 2]
                                           if found else None),
            }
        comparison["sets"][slug] = {"summary": summary, "items": rows}

    comp_path = os.path.join(out_dir, f"lens_baseline_comparison_{model_slug}.json")
    with open(comp_path, "w") as f:
        json.dump(comparison, f, indent=2)
        f.write("\n")

    lines = ["# Lens baseline comparison — " + model_name, ""]
    lines.append("Hit criterion: every intermediate of an item at rank <= 5 "
                 "(word-like mask, all fitted layers, position -1). "
                 "`found@100`: intermediate appears anywhere in the top-100.")
    for slug in EVAL_SETS:
        lines += ["", f"## {slug}", "",
                  "| lens | hit@5 | found@100 | median rank when found |",
                  "|---|---|---|---|"]
        for ln, s in comparison["sets"][slug]["summary"].items():
            lines.append(f"| {ln} | {s['hit@5']} | {s['found@100']} | "
                         f"{s['median_rank_when_found']} |")
    with open(os.path.join(out_dir, f"lens_baseline_comparison_{model_slug}.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="stage", required=True)
    p1 = sub.add_parser("precompute")
    p1.add_argument("--lens", required=True, choices=["jlens", "logit", "tuned"])
    p1.add_argument("--jlens-repo", default=os.path.expanduser("/root/work/jacobian-lens"))
    p1.add_argument("--out", required=True)
    p1.add_argument("--device", default="cuda:0")
    p1.add_argument("--tuned-ckpt", default=None)
    p1.add_argument("--model", default=MODEL_NAME)
    p1.add_argument("--lens-file", default=LENS_FILE)
    p1.add_argument("--lens-revision", default=LENS_REVISION)
    p2 = sub.add_parser("battery")
    p2.add_argument("--cache", action="append", required=True,
                    help="name=path.json.gz, repeatable")
    p2.add_argument("--out-dir", default=os.path.dirname(__file__) or ".")
    p2.add_argument("--n-runs", type=int, default=20)
    args = ap.parse_args()
    if args.stage == "precompute":
        if args.lens == "tuned" and not args.tuned_ckpt:
            raise SystemExit("tuned lens needs --tuned-ckpt")
        precompute(args.lens, args.jlens_repo, args.out, args.device, args.tuned_ckpt,
                   model_name=args.model, lens_file=args.lens_file,
                   lens_revision=args.lens_revision)
    else:
        caches = dict(kv.split("=", 1) for kv in args.cache)
        battery(caches, args.out_dir, args.n_runs)
