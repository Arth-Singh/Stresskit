"""Reference Stability Card: GPT-2 small / Greater-Than / attribution patching.

The Greater-Than task (Hanna et al. 2023, arXiv:2305.00586): prompts of the
form "The <noun> lasted from the year 17YY to the year 17" — the model
should place probability mass on years greater than YY. Finder: head-level
attribution patching over the 144 attention heads, scored by recovered
probability-mass difference. Corruption sets YY to 01 (removing the
threshold information), the task's standard corruption.

Battery: seeds (prompt subsampling), bootstrap, template phrasing swap,
top-k sweep, and a null control where the threshold used for scoring is a
random year unrelated to the prompt — the "circuit" for a nonexistent
comparison should not replicate.

Usage (GPU box):
    python references/run_greater_than_gpt2_card.py [--n-prompts 96] [--n-runs 6]
"""

import argparse
import json
import os
import random

import torch
from transformer_lens import HookedTransformer

import stresskit as sk
from stresskit.adapters.transformer_lens import layer_band_claim

NOUNS = ["war", "expedition", "dynasty", "trip", "voyage", "siege",
         "conflict", "famine", "drought", "rebellion", "plague", "career"]

LASTED = "The {noun} lasted from the year 17{yy} to the year 17"
WENT_ON = "The {noun} went on from the year 17{yy} to the year 17"


def two_digit_tokens(model):
    """Year suffixes 02..98 that GPT-2 encodes as a single token."""
    ok = {}
    for y in range(2, 99):
        s = f"{y:02d}"
        toks = model.to_tokens(s, prepend_bos=False)
        if toks.shape[1] == 1:
            ok[y] = toks[0, 0].item()
    assert len(ok) >= 80, f"only {len(ok)} single-token year suffixes"
    return ok


def make_dataset(n, year_pool, seed=0, template="both", null=False):
    rng = random.Random(seed)
    years = [y for y in year_pool if 10 <= y <= 90]
    items = []
    while len(items) < n:
        noun = rng.choice(NOUNS)
        yy = rng.choice(years)
        tpl = rng.choice([LASTED, WENT_ON]) if template == "both" else \
            (LASTED if template == "lasted" else WENT_ON)
        threshold = rng.choice(years) if null else yy
        items.append({
            "clean": tpl.format(noun=noun, yy=f"{yy:02d}"),
            "corr": tpl.format(noun=noun, yy="01"),
            "threshold": threshold,
        })
    return items


def make_finder(model, device, year_tokens):
    n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads
    z_names = [f"blocks.{l}.attn.hook_z" for l in range(n_layers)]
    yrs = sorted(year_tokens)
    year_ids = torch.tensor([year_tokens[y] for y in yrs], device=device)
    year_vals = torch.tensor(yrs, device=device)

    def prob_diff(logits, thresholds):
        # mass on years > threshold minus mass on years <= threshold
        probs = torch.softmax(logits[:, -1], dim=-1)[:, year_ids]  # [b, Y]
        gt = (year_vals[None, :] > thresholds[:, None]).float()
        return (probs * gt).sum(1) - (probs * (1 - gt)).sum(1)

    def finder(data, seed, config):
        top_k = config.get("top_k", 15)
        frac = config.get("subsample", 0.75)
        rng = random.Random(seed)
        torch.manual_seed(seed)

        sample = rng.sample(list(data), max(24, int(frac * len(data))))
        lengths = [model.to_tokens(it["clean"]).shape[1] for it in sample]
        modal = max(set(lengths), key=lengths.count)
        sample = [it for it, ln in zip(sample, lengths) if ln == modal]

        clean_toks = model.to_tokens([it["clean"] for it in sample]).to(device)
        corr_toks = model.to_tokens([it["corr"] for it in sample]).to(device)
        thresholds = torch.tensor([it["threshold"] for it in sample], device=device)

        with torch.no_grad():
            corr_logits, corr_cache = model.run_with_cache(
                corr_toks, names_filter=lambda n: n in z_names)
            m_corr = prob_diff(corr_logits, thresholds).mean()

        clean_z, grads = {}, {}

        def save_z(z, hook):
            clean_z[hook.name] = z.detach()
            z.register_hook(lambda g, name=hook.name: grads.__setitem__(name, g.detach()))
            return z

        model.zero_grad(set_to_none=True)
        with model.hooks(fwd_hooks=[(n, save_z) for n in z_names]):
            clean_logits = model(clean_toks)
        m_clean = prob_diff(clean_logits, thresholds).mean()
        m_clean.backward()

        attrib = {}
        for l, name in enumerate(z_names):
            delta = corr_cache[name] - clean_z[name]
            per_head = (delta * grads[name]).sum(dim=(0, 1, 3))
            for h in range(n_heads):
                attrib[f"L{l}H{h}"] = per_head[h].abs().item()
        model.zero_grad(set_to_none=True)

        heads = sorted(attrib, key=attrib.get, reverse=True)[:top_k]
        by_layer = {}
        for head in heads:
            l = int(head[1:head.index("H")])
            by_layer.setdefault(l, []).append(int(head[head.index("H") + 1:]))

        def patch(z, hook):
            l = int(hook.name.split(".")[1])
            for h in by_layer.get(l, []):
                z[:, :, h] = clean_z[hook.name][:, :, h]
            return z

        with torch.no_grad(), model.hooks(
                fwd_hooks=[(z_names[l], patch) for l in by_layer]):
            patched_logits = model(corr_toks)
        m_patched = prob_diff(patched_logits, thresholds).mean()

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-prompts", type=int, default=96)
    ap.add_argument("--n-runs", type=int, default=6)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "cards"))
    args = ap.parse_args()

    print(f"loading gpt2-small on {args.device} ...")
    model = HookedTransformer.from_pretrained("gpt2", device=args.device)
    model.eval()

    year_tokens = two_digit_tokens(model)
    pool = list(year_tokens)
    data = make_dataset(args.n_prompts, pool, seed=0)
    templates = {
        "lasted-only": make_dataset(args.n_prompts, pool, seed=1, template="lasted"),
        "went-on-only": make_dataset(args.n_prompts, pool, seed=2, template="went_on"),
    }
    null_data = make_dataset(args.n_prompts, pool, seed=3, null=True)

    finder = make_finder(model, args.device, year_tokens)
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
            "Greater-Than in GPT-2 small is implemented by ~15 attention "
            "heads concentrated in the mid-to-late layers"
        ),
        model="gpt2-small",
        task="Greater-Than (YY->01 corruption)",
        method="head-level attribution patching, top-k by |attribution|",
        verbose=True,
    )

    print()
    print(result)
    print(result.to_markdown())

    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.join(args.out_dir, "greater_than_gpt2_small")
    result.card.save(base + ".json")
    with open(base + ".md", "w") as f:
        f.write(result.to_markdown() + "\n")
    with open(base + ".badge.json", "w") as f:
        json.dump(result.card.badge_dict(), f, indent=2)
        f.write("\n")
    print(f"\nartifacts written to {base}.*")


if __name__ == "__main__":
    main()
