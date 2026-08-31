"""Fit-reproducibility battery for the Jacobian lens on Qwen3.5-4B.

Registry target ``jlens_fit_reproducibility_qwen35_4b``: fit lenses on
disjoint 250-prompt slices of the same corpus recipe as the released
n1000 lens (Salesforce wikitext, 128-token sequences), then measure how
much the *instrument itself* varies with its fitting sample:

- shard-vs-shard readout agreement (rank-biased overlap at matched
  item/layer/position over the repo's lens-eval items),
- merged(4 shards) vs the released lens (agreement + hit@5 under the
  upstream criterion).

    python refit_jlens_qwen.py fit --shard 0 --device cuda:0 --out shard0.pt
    python refit_jlens_qwen.py compare --shards 'shard*.pt' --out-dir out/
"""

import argparse
import glob
import json
import os

MODEL_NAME = "Qwen/Qwen3.5-4B"
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_REVISION = "qwen-n1000"
LENS_FILE = "qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt"
N_SHARDS = 4
PROMPTS_PER_SHARD = 250
POSITIONS = [-1, -2]
TOP_N = 100


def shard_prompts(tok, shard):
    from train_tuned_lens_qwen import corpus_chunks
    chunks = corpus_chunks(tok, n_seqs=N_SHARDS * PROMPTS_PER_SHARD)
    lo = shard * PROMPTS_PER_SHARD
    return [tok.decode(c) for c in chunks[lo:lo + PROMPTS_PER_SHARD]]


def load_model(device):
    import torch, transformers, jlens
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16
    ).to(device)
    hf.eval()
    tok = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    return jlens.from_hf(hf, tok), tok


def fit_shard(shard, device, out_path, dim_batch):
    import jlens
    model, tok = load_model(device)
    prompts = shard_prompts(tok, shard)
    released = jlens.JacobianLens.from_pretrained(
        LENS_REPO, filename=LENS_FILE, revision=LENS_REVISION
    )
    lens = jlens.fitting.fit(
        model, prompts,
        source_layers=list(released.source_layers),
        dim_batch=dim_batch, max_seq_len=128,
        checkpoint_path=out_path + ".ckpt", checkpoint_every=25,
    )
    lens.save(out_path)
    os.remove(out_path + ".ckpt")
    print(f"shard {shard}: {len(prompts)} prompts -> {out_path}")


def readouts(model, tok, lens, prompt):
    lens_logits, _, _ = lens.apply(
        model, prompt, layers=list(lens.source_layers), positions=POSITIONS
    )
    return {pos: {L: [tok.decode([t]) for t in lens_logits[L][pi].topk(TOP_N).indices]
                  for L in lens_logits}
            for pi, pos in enumerate(POSITIONS)}


def compare(shard_glob, device, out_dir, jlens_repo):
    import jlens
    import stresskit.metrics as M
    from stresskit.adapters import jlens as skj
    from run_lens_baselines_qwen import load_items, EVAL_SETS

    model, tok = load_model(device)
    shards = [jlens.JacobianLens.load(p) for p in sorted(glob.glob(shard_glob))]
    merged = jlens.JacobianLens.merge(shards)
    released = jlens.JacobianLens.from_pretrained(
        LENS_REPO, filename=LENS_FILE, revision=LENS_REVISION
    )
    lenses = {f"shard{i}": s for i, s in enumerate(shards)}
    lenses["merged"] = merged
    lenses["released_n1000"] = released

    report = {"model": MODEL_NAME, "prompts_per_shard": PROMPTS_PER_SHARD,
              "sets": {}}
    for slug in EVAL_SETS:
        items = load_items(jlens_repo, slug)
        per_lens = {name: [readouts(model, tok, lens, it["prompt"])
                           for it in items]
                    for name, lens in lenses.items()}
        shard_names = [n for n in per_lens if n.startswith("shard")]
        pair_rbo, agree_rows = [], []
        for i, it in enumerate(items):
            for pos in map(str, POSITIONS):
                for L in per_lens[shard_names[0]][i][pos]:
                    lists = [per_lens[n][i][pos][L] for n in shard_names]
                    r = M.pairwise_rbo(lists, p=0.9)
                    if r is not None:
                        pair_rbo.append(r)
                    mr = M.rbo(per_lens["merged"][i][pos][L],
                               per_lens["released_n1000"][i][pos][L], p=0.9)
                    agree_rows.append(mr)
        hit5 = {}
        for name in ("merged", "released_n1000"):
            hits = 0
            for i, it in enumerate(items):
                ranked_by_layer = {
                    int(L): [t for t in ranked if skj.is_wordlike(t)]
                    for L, ranked in per_lens[name][i]["-1"].items()
                }
                ranks = [skj.min_rank(ranked_by_layer, t)
                         for t in it["intermediates"]]
                if all(r is not None and r <= 5 for r in ranks):
                    hits += 1
            hit5[name] = round(hits / len(items), 4)
        report["sets"][slug] = {
            "shard_pairwise_rbo_mean": round(sum(pair_rbo) / len(pair_rbo), 4),
            "merged_vs_released_rbo_mean": round(sum(agree_rows) / len(agree_rows), 4),
            "hit@5": hit5,
            "n_items": len(items),
        }
        print(slug, report["sets"][slug])

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "jlens_refit_reproducibility.json"), "w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print("report ->", os.path.join(out_dir, "jlens_refit_reproducibility.json"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="stage", required=True)
    p1 = sub.add_parser("fit")
    p1.add_argument("--shard", type=int, required=True)
    p1.add_argument("--device", default="cuda:0")
    p1.add_argument("--dim-batch", type=int, default=64)
    p1.add_argument("--out", required=True)
    p2 = sub.add_parser("compare")
    p2.add_argument("--shards", required=True)
    p2.add_argument("--device", default="cuda:0")
    p2.add_argument("--jlens-repo", default="/root/work/jacobian-lens")
    p2.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    if args.stage == "fit":
        fit_shard(args.shard, args.device, args.out, args.dim_batch)
    else:
        compare(args.shards, args.device, args.out_dir, args.jlens_repo)
