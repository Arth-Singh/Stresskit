"""Reference Stability Card #1: GPT-2 small / IOI / attribution patching.

The first StressKit run on a real model. The finder is head-level
attribution patching (the EAP family, arXiv:2310.10348): rank the 144
attention heads of GPT-2 small by |(z_corr - z_clean) . dM/dz_clean| on the
Indirect Object Identification task, keep the top-k as "the circuit", and
score it by denoising faithfulness (fraction of the clean-vs-corrupted
logit-diff gap recovered when the circuit heads' clean activations are
patched into the corrupted run).

Battery: seeds (finder subsamples 75% of prompts per seed), bootstrap,
templates (ABBA-only vs BABA-only prompt orders — the axis arXiv:2606.16920
showed activates different circuits), hyperparams (top-k), plus a null
control where the answer tokens are random names unrelated to the prompt
(the "circuit" for a non-existent effect should not replicate).

Usage (on a GPU box):
    python references/run_ioi_gpt2_card.py [--n-prompts 96] [--n-runs 6]
Outputs: references/cards/ioi_gpt2_small.{json,md} + badge JSON.
"""

import argparse
import json
import os
import random

import torch
from transformer_lens import HookedTransformer

import stresskit as sk
from stresskit.adapters.transformer_lens import layer_band_claim

# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

NAMES = ["Mary", "John", "Tom", "James", "Dan", "Martin", "Amy", "Scott",
         "Sarah", "Kevin", "Anna", "Paul", "Laura", "Peter", "Emma", "Jason",
         "Karen", "Brian", "Susan", "Mark"]
PLACES = ["store", "park", "school", "office"]
OBJECTS = ["drink", "book", "ring", "snack"]

ABBA = "When {io} and {s} went to the {place}, {s} gave a {obj} to"
BABA = "When {s} and {io} went to the {place}, {s} gave a {obj} to"


def single_token_names(model):
    keep = []
    for name in NAMES:
        if model.to_tokens(" " + name, prepend_bos=False).shape[1] == 1:
            keep.append(name)
    assert len(keep) >= 10, f"only {len(keep)} single-token names"
    return keep


def make_dataset(model, n, seed=0, template="both", null=False):
    """Items: dict(clean, corr, io, s) with io/s as single-token strings."""
    rng = random.Random(seed)
    names = single_token_names(model)
    items = []
    while len(items) < n:
        io, s, c, x, y = rng.sample(names, 5)
        tpl = rng.choice([ABBA, BABA]) if template == "both" else \
            (ABBA if template == "ABBA" else BABA)
        place, obj = rng.choice(PLACES), rng.choice(OBJECTS)
        clean = tpl.format(io=io, s=s, place=place, obj=obj)
        # ABC corruption: second subject occurrence replaced by a third name
        corr = clean.replace(f", {s} gave", f", {c} gave")
        if null:
            io, s = x, y  # answer tokens unrelated to the prompt: no effect exists
        items.append({"clean": clean, "corr": corr, "io": " " + io, "s": " " + s})
    return items


# ---------------------------------------------------------------------------
# The finder: head-level attribution patching + denoising faithfulness
# ---------------------------------------------------------------------------

def make_finder(model, device):
    n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads
    z_names = [f"blocks.{l}.attn.hook_z" for l in range(n_layers)]

    def logit_diff(logits, io_ids, s_ids):
        last = logits[:, -1]
        return last.gather(1, io_ids[:, None]).squeeze(1) - \
            last.gather(1, s_ids[:, None]).squeeze(1)

    def finder(data, seed, config):
        top_k = config.get("top_k", 15)
        frac = config.get("subsample", 0.75)
        rng = random.Random(seed)
        torch.manual_seed(seed)

        sample = rng.sample(list(data), max(24, int(frac * len(data))))
        # keep only the modal token length so we can batch without padding
        lengths = [model.to_tokens(it["clean"]).shape[1] for it in sample]
        modal = max(set(lengths), key=lengths.count)
        sample = [it for it, ln in zip(sample, lengths) if ln == modal]

        clean_toks = model.to_tokens([it["clean"] for it in sample]).to(device)
        corr_toks = model.to_tokens([it["corr"] for it in sample]).to(device)
        io_ids = torch.tensor(
            [model.to_single_token(it["io"]) for it in sample], device=device)
        s_ids = torch.tensor(
            [model.to_single_token(it["s"]) for it in sample], device=device)

        # corrupted run (no grad), cache z
        with torch.no_grad():
            corr_logits, corr_cache = model.run_with_cache(
                corr_toks, names_filter=lambda n: n in z_names)
            m_corr = logit_diff(corr_logits, io_ids, s_ids).mean()

        # clean run with grad; capture z values and their grads
        clean_z, grads = {}, {}

        def save_z(z, hook):
            clean_z[hook.name] = z.detach()
            z.register_hook(lambda g, name=hook.name: grads.__setitem__(name, g.detach()))
            return z

        model.zero_grad(set_to_none=True)
        with model.hooks(fwd_hooks=[(n, save_z) for n in z_names]):
            clean_logits = model(clean_toks)
        m_clean_vec = logit_diff(clean_logits, io_ids, s_ids)
        m_clean = m_clean_vec.mean()
        m_clean.backward()

        # attribution per head: (z_corr - z_clean) . dM/dz_clean
        attrib = {}
        for l, name in enumerate(z_names):
            delta = corr_cache[name] - clean_z[name]          # [b, pos, head, d]
            per_head = (delta * grads[name]).sum(dim=(0, 1, 3))  # [head]
            for h in range(n_heads):
                attrib[f"L{l}H{h}"] = per_head[h].abs().item()
        model.zero_grad(set_to_none=True)

        heads = sorted(attrib, key=attrib.get, reverse=True)[:top_k]
        by_layer = {}
        for head in heads:
            l = int(head[1:head.index("H")])
            by_layer.setdefault(l, []).append(int(head[head.index("H") + 1:]))

        # denoising faithfulness: patch clean z of circuit heads into corrupted run
        def patch(z, hook):
            l = int(hook.name.split(".")[1])
            for h in by_layer.get(l, []):
                z[:, :, h] = clean_z[hook.name][:, :, h]
            return z

        with torch.no_grad(), model.hooks(
                fwd_hooks=[(z_names[l], patch) for l in by_layer]):
            patched_logits = model(corr_toks)
        m_patched = logit_diff(patched_logits, io_ids, s_ids).mean()

        denom = (m_clean - m_corr).item()
        faithfulness = ((m_patched - m_corr).item() / denom) if abs(denom) > 1e-6 else 0.0

        return sk.circuit(
            heads,
            claim=layer_band_claim(heads, lambda e: int(e[1:e.index("H")]), n_layers),
            score=faithfulness,
            universe_size=n_layers * n_heads,
            m_clean=round(m_clean.item(), 4),
            m_corr=round(m_corr.item(), 4),
            n_prompts=len(sample),
        )

    return finder


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2",
                    help="any GPT-2-family TransformerLens model name")
    ap.add_argument("--n-prompts", type=int, default=96)
    ap.add_argument("--n-runs", type=int, default=6)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "cards"))
    ap.add_argument("--cache-dir", default=None,
                    help="cache finder runs so a preempted battery resumes free")
    args = ap.parse_args()

    print(f"loading {args.model} on {args.device} ...")
    model = HookedTransformer.from_pretrained(args.model, device=args.device)
    model.eval()

    data = make_dataset(model, args.n_prompts, seed=0)
    templates = {
        "ABBA-only": make_dataset(model, args.n_prompts, seed=1, template="ABBA"),
        "BABA-only": make_dataset(model, args.n_prompts, seed=2, template="BABA"),
    }
    null_data = make_dataset(model, args.n_prompts, seed=3, null=True)

    finder = make_finder(model, args.device)
    result = sk.stress(
        finder,
        data,
        battery=["seeds", "bootstrap", "templates", "hyperparams"],
        n_runs=args.n_runs,
        config={"top_k": 15},
        templates=templates,
        hyperparams={"top_k": [8, 30]},
        null_data=null_data,
        claim_statement=(
            "IOI in GPT-2 small is implemented by ~15 attention heads "
            "concentrated in the late layers"
        ),
        model=args.model,
        task="IOI (ABC corruption)",
        method="head-level attribution patching, top-k by |attribution|",
        verbose=True,
        cache_dir=args.cache_dir,
        cache_key=(f"ioi-{args.model}-p{args.n_prompts}-r{args.n_runs}"
                   if args.cache_dir else None),
    )

    print()
    print(result)
    print()
    print(result.to_markdown())

    slug = {"gpt2": "gpt2_small"}.get(args.model, args.model.replace("-", "_"))
    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.join(args.out_dir, f"ioi_{slug}")
    result.card.save(base + ".json")
    with open(base + ".md", "w") as f:
        f.write(result.to_markdown() + "\n")
    with open(base + ".badge.json", "w") as f:
        json.dump(result.card.badge_dict(), f, indent=2)
        f.write("\n")

    print("\ncomputing verdict-stability trace ...")
    trace = result.verdict_trace(seed=0)
    with open(base + ".trace.json", "w") as f:
        json.dump(trace, f, indent=2)
        f.write("\n")
    with open(base + ".trace.md", "w") as f:
        f.write(sk.verdict_trace_markdown(trace) + "\n")
    print(sk.verdict_trace_markdown(trace))
    print(f"\nartifacts written to {base}.*")


if __name__ == "__main__":
    main()
