"""Reference Stability Card: the head-pair census of the communication map
(arXiv:2608.22007, richardzhewang/communication-map).

Claim under test (abstract, byte-exact): "The census of all candidate
channels, from 6.3x10^8 in GPT-2 to 1.3x10^11 in Pythia-6.9B, finds that
70-89% of head pairs are oriented far from chance, some coupled strongly and
others actively avoiding each other."

What the released code computes (experiments/map_build.py at the pinned
commit): for every causally eligible ordered head pair (writer in an earlier
layer than the reader) and each of the three composition channels K, Q, V,
the coupling coefficient C = ||RW||_F / (||R||_F ||W||_F) from the weights
alone, then a z-score of C^2 against the pair's theoretical rotation null
distribution (closed-form Weingarten moments, Appendix A.4). Table 2 of the
paper reports, per model and channel, the share of pairs with z >= 2
("super-coupled") and z <= -2 ("avoidant"); "far from chance" is |z| >= 2.
Seven models: GPT-2 small/medium/large, GPT-Neo-125M, Pythia-160m/2.8B/6.9B.

Arithmetic of the headline. The released census (results/map/*/
theory_census.json) gives, per (model, channel), far-from-chance shares
between 61% and 91%: four of the 21 entries sit outside 70-89% (GPT-2 V
90.5%, Pythia-2.8B K 68.5% and Q 63.8%, Pythia-6.9B Q 60.7%). Pooled over the
three channels of each model the shares run from 69.8% (Pythia-2.8B) to
89.0% (GPT-2), so the abstract's range is a per-model statement, not a
per-channel one. The card records both readings; the per-channel exceptions
are the finding's structure.

Finder = the released census as a pure function of (data, seed, config):

- data: chunk records {model, channel, chunk}. The pairs of each (model,
  channel) entry are split into N_CHUNKS chunks by pair index, and a record
  contributes its chunk's pairs, with multiplicity, so the bootstrap axis is
  a cluster bootstrap of head pairs inside every entry. Null records carry
  null="rotate-writers".
- seed: unused by the census, which has no randomness once the weights are
  fixed; the null control seeds its rotations with it.
- config: threshold (2, the paper; 3, also tabulated upstream), processing
  ("upstream": LayerNorm folded, writing weights and unembedding centred,
  Appendix B.1; "raw": the HF weights as stored), precision ("fp32", the
  paper; "fp16": weights rounded to half precision before the census).

Finding representation (fixed before any battery ran):

- components: the (model, channel) entries whose far-from-chance share,
  rounded to a whole percent, lies OUTSIDE 70-89: the exceptions to the
  abstract's range. Universe = 21 entries. (The complement is dense, 17 of
  21, and a dense set in a universe this small cannot beat a random draw;
  the exceptions carry the same information sparsely.)
- claim: "per-model pooled shares <inside|not inside> 70-89%; per-channel
  entries <all inside|not all inside>".
- score: the mean over the seven models of the pooled far-from-chance share
  (the quantity the abstract's range summarises; 0.805 in the released
  census).
- meta: every entry's above/below/far shares and pair count, the per-model
  pooled shares and their range, the config.

Battery: bootstrap (chunks of head pairs resampled with replacement),
hyperparams (threshold 3; raw weights; fp16 weights). Seeds axis not run:
the closed-form census has no randomness. Templates axis not run: the census
takes no text or prompt input; instead the card notes an extension census
over six further public models (Pythia-410m/1b/1.4b, GPT-Neo-1.3B,
OPT-125m/1.3b) under the paper's processing. Null control: every writer's
output factor O_w is replaced by R_w O_w with an independent Haar rotation
R_w per head, the exact hypothesis of the theoretical null (singular values
kept, orientation randomised), so the census should find about 5% of pairs
far from chance; band and thresholds unchanged.

Usage (GPU; about an hour on an H200, dominated by the Pythia-6.9B loads):
    python references/run_communication_map_card.py --upstream /path/to/communication-map \
        --out-dir references/cards --raw-dir references/cards/raw/communication_map
"""

import argparse
import gc
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

import stresskit as sk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from battery_shards import Shard  # noqa: E402

UPSTREAM_REPO = "richardzhewang/communication-map"
UPSTREAM_COMMIT = "c8ab3b02c11ab7541794682163cbbc03a5b5a4c1"
MODELS = ("gpt2", "gpt2-medium", "gpt2-large", "gpt-neo-125m",
          "pythia-160m", "pythia-2.8b", "pythia-6.9b")
EXTENSION_MODELS = ("pythia-410m", "pythia-1b", "pythia-1.4b", "gpt-neo-1.3B",
                    "opt-125m", "opt-1.3b")
CHANNELS = ("K", "Q", "V")
UPSTREAM_FILES = ("experiments/map_build.py",) + tuple(
    f"results/map/{m}/theory_census.json" for m in MODELS)
BAND = (70, 89)
N_CHUNKS = 40
BASE_SEED = 0
NULL_SEED = BASE_SEED ^ 0x5EC
BASE_CONFIG = {"threshold": 2.0, "processing": "upstream", "precision": "fp32"}
PLACEHOLDER = sk.Finding(claim="placeholder (other shard)", score=0.0, meta={"placeholder": True})
MIN_FREE_GB = 14.0


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def percent(share):
    return int(np.floor(100.0 * share + 0.5))


def inside_band(share):
    return BAND[0] <= percent(share) <= BAND[1]


def wait_for_gpu(min_free_gb, max_wait=3600):
    t0 = time.time()
    while True:
        free = torch.cuda.mem_get_info()[0] / 1e9
        if free >= min_free_gb or time.time() - t0 > max_wait:
            return
        print(f"  waiting for GPU memory: {free:.1f} GB free", flush=True)
        time.sleep(30)


def load_weights(mb, model, processing):
    """The upstream extraction (map_build.load_tl + extract) with the model
    resident on the CPU; the census itself runs on the GPU. 'raw' loads the
    same checkpoint through HookedTransformer without LayerNorm folding or
    centring and hands it to the same extract()."""
    import transformer_lens

    t0 = time.time()
    if processing == "upstream":
        saved, mb.DEV = mb.DEV, "cpu"
        try:
            tl = mb.load_tl(model)
        finally:
            mb.DEV = saved
    elif processing == "raw":
        kw = dict(fold_ln=False, center_writing_weights=False, center_unembed=False,
                  device="cpu", dtype=torch.float32)
        if model.startswith("pythia"):
            from transformers import AutoModelForCausalLM
            hf = AutoModelForCausalLM.from_pretrained(f"EleutherAI/{model}",
                                                      torch_dtype=torch.float32)
            if not hasattr(hf, "embed_out"):
                hf.embed_out = getattr(hf, "lm_head", None) or hf.get_output_embeddings()
            tl = transformer_lens.HookedTransformer.from_pretrained(model, hf_model=hf, **kw).eval()
            del hf
        else:
            tl = transformer_lens.HookedTransformer.from_pretrained(model, **kw).eval()
    else:
        raise ValueError(processing)
    W = mb.extract(tl)
    for p in tl.parameters():
        p.data = torch.empty(0)
    del tl
    gc.collect()
    W = {k: W[k] for k in ("L", "H", "d", "dh", "Q", "K", "V", "O")}
    print(f"  {model}/{processing}: extracted in {time.time() - t0:.0f}s "
          f"(L={W['L']} H={W['H']} d={W['d']} dh={W['dh']})", flush=True)
    return W


def to_precision(W, precision):
    if precision == "fp32":
        return W
    if precision == "fp16":
        return {k: (v.half().float() if torch.is_tensor(v) else v) for k, v in W.items()}
    raise ValueError(precision)


def rotate_writers(W, seed):
    """Null: an independent Haar rotation of every writer's output factor."""
    Nh, d, dh = W["L"] * W["H"], W["d"], W["dh"]
    writers = W["O"].reshape(Nh, d, dh)
    gen = torch.Generator(device="cuda").manual_seed(int(seed))
    out = torch.empty_like(writers)
    for w in range(Nh):
        A = torch.randn(d, d, generator=gen, device="cuda")
        Qm, R = torch.linalg.qr(A)
        Qm = Qm * torch.sign(torch.diagonal(R))[None, :]
        out[w] = (Qm @ writers[w].to("cuda")).cpu()
    W2 = dict(W)
    W2["O"] = out.reshape(W["O"].shape).contiguous()
    return W2


def pair_z(W, rows, labels):
    """Per-pair z of C^2 against the rotation null, the closed-form
    expression of map_build.theory_census written out per pair (the upstream
    function returns only the tail shares); checked against it in census()."""
    L, H, d = W["L"], W["H"], W["d"]
    NH = L * H

    def fold(X):
        return np.asarray(X, dtype=np.float64).reshape(NH, d, -1)

    def gram(X):
        return np.matmul(X.transpose(0, 2, 1), X)

    gQ, gK, gV, gO = (gram(fold(W[k])) for k in ("Q", "K", "V", "O"))

    def inv(a, b):
        ab = np.matmul(a, b)
        return np.einsum("hii->h", ab), np.einsum("hij,hji->h", ab, ab)

    trQK, trQK2 = inv(gQ, gK)
    trOV, trOV2 = inv(gO, gV)
    rinv = {"head_head_K": (trQK, trQK2), "head_head_Q": (trQK, trQK2),
            "head_head_V": (trOV, trOV2)}
    cH = trOV2 - trOV ** 2 / d
    index = {lab: i for i, lab in enumerate(labels)}
    out = {}
    for r in rows:
        trG, trG2 = rinv[r["cls"]]
        ri = np.array([index[x] for x in r["reader"]])
        wi = np.array([index[x] for x in r["writer"]])
        mu = trG[ri] * trOV[wi] / d
        var = 2.0 / ((d - 1) * (d + 2)) * (trG2 - trG ** 2 / d)[ri] * cH[wi]
        T = np.asarray(r["stat"], dtype=np.float64) ** 2 * trG[ri] * trOV[wi]
        out[r["cls"]] = (T - mu) / np.sqrt(var)
    return out


def tail_shares(z, thr):
    above, below = float((z >= thr).mean()), float((z <= -thr).mean())
    return {"above": above, "below": below, "far": above + below}


class Census:
    """Per (model, processing, precision, null) census, cached on disk as the
    per-pair z of every channel plus the upstream summary."""

    def __init__(self, mb, raw_dir):
        self.mb = mb
        self.dir = os.path.join(raw_dir, "census")
        os.makedirs(self.dir, exist_ok=True)
        self.weights = {}
        self.z = {}

    @staticmethod
    def key(model, processing, precision, null_seed):
        return f"{model}__{processing}__{precision}__" + (
            "none" if null_seed is None else f"rot{int(null_seed)}")

    def W(self, model, processing):
        if (model, processing) not in self.weights:
            self.weights[(model, processing)] = load_weights(self.mb, model, processing)
        return self.weights[(model, processing)]

    def release(self, model):
        for k in [k for k in self.weights if k[0] == model]:
            del self.weights[k]
        gc.collect()

    def summary(self, model, processing="upstream", precision="fp32", null_seed=None):
        with open(os.path.join(self.dir, self.key(model, processing, precision, null_seed) + ".json")) as f:
            return json.load(f)

    def get(self, model, processing, precision, null_seed=None):
        key = self.key(model, processing, precision, null_seed)
        if key in self.z:
            return self.z[key]
        path = os.path.join(self.dir, key + ".npz")
        if not os.path.exists(path):
            self.compute(model, processing, precision, null_seed, path)
        with np.load(path) as f:
            self.z[key] = {c: f[c] for c in f.files}
        return self.z[key]

    def compute(self, model, processing, precision, null_seed, path):
        t0 = time.time()
        W = to_precision(self.W(model, processing), precision)
        if null_seed is not None:
            W = rotate_writers(W, null_seed)
        wait_for_gpu(MIN_FREE_GB)
        rows = []
        hh = self.mb.head_head(W, rows)
        labels = list(hh["labels"])
        del hh
        torch.cuda.empty_cache()
        summary = self.mb.theory_census(W, rows)
        z = pair_z(W, rows, labels)
        for r in rows:
            up = summary["channels"][r["cls"]]
            mine = tail_shares(z[r["cls"]], 2.0), tail_shares(z[r["cls"]], 3.0)
            for (a, b), got in (("above2", "below2"), mine[0]), (("above3", "below3"), mine[1]):
                if abs(got["above"] - up[a]) > 1e-9 or abs(got["below"] - up[b]) > 1e-9:
                    raise RuntimeError(f"{model}/{r['cls']}: per-pair z disagrees with upstream "
                                       f"theory_census ({got} vs {up[a]}, {up[b]})")
        np.savez(path + ".tmp.npz", **z)
        os.replace(path + ".tmp.npz", path)
        summary["stresskit"] = {"model": model, "processing": processing, "precision": precision,
                                "null_seed": null_seed, "seconds": round(time.time() - t0, 1)}
        with open(path[:-4] + ".json", "w") as f:
            json.dump(summary, f, indent=1)
        print(f"  census {os.path.basename(path)[:-4]}: "
              + ", ".join(f"{c[-1]} far {tail_shares(z[c], 2.0)['far']:.3f}" for c in sorted(z))
              + f" ({time.time() - t0:.0f}s)", flush=True)


def make_records(models, null=None):
    return [{"model": m, "channel": ch, "chunk": j, **({"null": null} if null else {})}
            for m in models for ch in CHANNELS for j in range(N_CHUNKS)]


def make_finder(census, raw_dir):
    def compute(data, seed, cfg):
        t0 = time.time()
        null_seed = seed if data[0].get("null") else None
        entries = {}
        for d in data:
            entries.setdefault((d["model"], d["channel"]), []).append(d["chunk"])
        models = list(dict.fromkeys(m for m, _ in entries))
        per_entry = {}
        for (m, ch), chunks in entries.items():
            z = census.get(m, cfg["processing"], cfg["precision"], null_seed)[f"head_head_{ch}"]
            idx = np.arange(len(z))
            zs = z[np.concatenate([idx[idx % N_CHUNKS == j] for j in chunks])]
            shares = tail_shares(zs, cfg["threshold"])
            per_entry[f"{m}/{ch}"] = {**shares, "n": int(len(zs)),
                                      "n_far": int(round(shares["far"] * len(zs)))}
        pooled = {}
        for m in models:
            es = [per_entry[f"{m}/{ch}"] for ch in CHANNELS if f"{m}/{ch}" in per_entry]
            pooled[m] = sum(e["n_far"] for e in es) / sum(e["n"] for e in es)
        exceptions = {k for k, e in per_entry.items() if not inside_band(e["far"])}
        models_inside = all(inside_band(s) for s in pooled.values())
        claim = (f"per-model pooled shares {'inside' if models_inside else 'not inside'} 70-89%; "
                 f"per-channel entries {'all inside' if not exceptions else 'not all inside'}")
        score = float(np.mean(list(pooled.values())))
        meta = {"config": cfg, "null_seed": null_seed, "per_entry": per_entry,
                "pooled_per_model": pooled,
                "pooled_range": [min(pooled.values()), max(pooled.values())],
                "n_exceptions": len(exceptions), "wall_secs": round(time.time() - t0, 1)}
        print(f"  [thr={cfg['threshold']} {cfg['processing']}/{cfg['precision']} "
              f"{'null ' if null_seed is not None else ''}seed={seed}] pooled "
              f"{min(pooled.values()):.3f}-{max(pooled.values()):.3f}, "
              f"{len(exceptions)} exceptions {sorted(exceptions)} ({meta['wall_secs']}s)", flush=True)
        return sk.Finding(components=exceptions, universe_size=len(per_entry), claim=claim,
                          score=score, meta=meta)

    shard = Shard(os.path.join(raw_dir, "shard_cache"))

    def finder(data, seed, config):
        cfg = dict(BASE_CONFIG, **(config or {}))
        return shard.run(lambda: compute(data, seed, cfg), data, seed, cfg, PLACEHOLDER)

    finder.shard = shard
    return finder


def run_row(record, group):
    f = record.finding
    return {"group": group, "axis": record.axis, "variant": record.variant,
            "seed": record.seed, "config": record.config, "claim": f.claim,
            "score": f.score, "size": f.size,
            "components": sorted(str(c) for c in f.components), "meta": f.meta}


def released_shares(upstream, model):
    with open(os.path.join(upstream, "results", "map", model, "theory_census.json")) as f:
        d = json.load(f)["channels"]
    return {ch: d[f"head_head_{ch}"] for ch in CHANNELS}


def fmt_entry(shares):
    return f"{100 * shares['above']:.1f}/{100 * shares['below']:.1f} ({100 * shares['far']:.1f}%)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", required=True)
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "cards"))
    ap.add_argument("--raw-dir", default=None)
    ap.add_argument("--n-runs", type=int, default=20)
    ap.add_argument("--models", default=",".join(MODELS),
                    help="debugging switch; the card is the full seven-model census")
    ap.add_argument("--no-extension", action="store_true")
    args = ap.parse_args()
    models = tuple(args.models.split(","))

    raw_dir = args.raw_dir or os.path.join(args.out_dir, "raw", "communication_map")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    hashes = {p: sha256_file(os.path.join(args.upstream, p)) for p in UPSTREAM_FILES}

    sys.path.insert(0, os.path.join(args.upstream, "experiments"))
    import map_build as mb  # noqa: E402
    import transformer_lens  # noqa: E402
    import transformers  # noqa: E402

    census = Census(mb, raw_dir)
    for model in models:
        for processing, precision, null_seed in (("upstream", "fp32", None), ("upstream", "fp16", None),
                                                 ("upstream", "fp32", NULL_SEED), ("raw", "fp32", None)):
            census.get(model, processing, precision, null_seed)
        census.release(model)
    extension = () if args.no_extension else EXTENSION_MODELS
    for model in extension:
        census.get(model, "upstream", "fp32")
        census.release(model)

    finder = make_finder(census, raw_dir)
    result = sk.stress(
        finder, make_records(models),
        battery=["bootstrap", "hyperparams"],
        n_runs=args.n_runs, seed=BASE_SEED,
        config=dict(BASE_CONFIG),
        hyperparams={"threshold": [3.0], "processing": ["raw"], "precision": ["fp16"]},
        null_data=make_records(models, null="rotate-writers"),
        claim_statement=(
            "The census of all candidate channels, from 6.3x10^8 in GPT-2 to 1.3x10^11 in "
            "Pythia-6.9B, finds that 70-89% of head pairs are oriented far from chance, some "
            "coupled strongly and others actively avoiding each other"),
        model=", ".join(models),
        task="head-pair census: share of causally eligible head pairs whose coupling "
             "coefficient C^2 sits >= 2 SD from its rotation-null mean, per K/Q/V channel",
        method="upstream map_build.head_head + theory_census (closed-form Weingarten "
               "moments) at the pinned commit; weights extracted through the upstream loader",
        verbose=True,
    )

    if finder.shard.is_worker:
        print(f"shard {finder.shard.index}/{finder.shard.count} done; no artifacts written")
        return

    base = result.base.meta
    notes = result.card.notes
    notes.append(
        f"upstream: {UPSTREAM_REPO}@{UPSTREAM_COMMIT[:7]} (MIT); head_head, theory_census, load_tl "
        "and extract imported unmodified from experiments/map_build.py; file hashes "
        + ", ".join(f"{p.split('/')[-2] + '/' if 'map/' in p else ''}{os.path.basename(p)} {h[:12]}"
                    for p, h in hashes.items()))
    notes.append(
        f"environment: transformer-lens {transformer_lens.__version__ if hasattr(transformer_lens, '__version__') else '3.8.1'}, "
        f"transformers {transformers.__version__}, torch {torch.__version__} on an H200 (upstream lock: "
        "transformer-lens 3.7.0, transformers 5.14.1, torch 2.11+cu128 on an RTX 5090). Models are "
        "loaded on the CPU by switching map_build.DEV for the duration of load_tl (its GPU-resident "
        "fp32 copy of Pythia-2.8B/6.9B does not fit the GPU headroom here); the census runs on the GPU "
        "as upstream. Pythia-6.9B goes through load_tl rather than upstream's --stream-load path "
        "(validated bit-close upstream by verify_stream.py). Per-pair z is recomputed with the "
        "closed-form expression of theory_census, and its |z|>=2 and |z|>=3 shares are asserted equal "
        "to the upstream function's on every census")
    repro = []
    for model in models:
        rel = released_shares(args.upstream, model)
        mine = census.summary(model)["channels"]
        worst = max(abs(mine[f"head_head_{ch}"][k] - rel[ch][k])
                    for ch in CHANNELS for k in ("above2", "below2", "above3", "below3"))
        parts = []
        for ch in CHANNELS:
            r = rel[ch]
            m = mine[f"head_head_{ch}"]
            parts.append(f"{ch} {100 * (r['above2'] + r['below2']):.1f}->"
                         f"{100 * (m['above2'] + m['below2']):.1f}")
        repro.append(f"{model} {' '.join(parts)} (max |dshare| {worst:.2e})")
    notes.append("reproduction of the released census, far-from-chance share per channel "
                 "(released -> base run, %): " + "; ".join(repro))
    rel_entries = {f"{m}/{ch}": released_shares(args.upstream, m)[ch] for m in models for ch in CHANNELS}
    rel_far = {k: v["above2"] + v["below2"] for k, v in rel_entries.items()}
    rel_pooled = {m: (sum(rel_entries[f'{m}/{ch}']['n'] * rel_far[f'{m}/{ch}'] for ch in CHANNELS)
                      / sum(rel_entries[f'{m}/{ch}']['n'] for ch in CHANNELS)) for m in models}
    notes.append(
        "the abstract's 70-89% against the released Table 2 census: per (model, channel) the "
        f"far-from-chance shares span {100 * min(rel_far.values()):.1f}-{100 * max(rel_far.values()):.1f}% and "
        f"{sum(not inside_band(v) for v in rel_far.values())} of {len(rel_far)} entries fall outside the range "
        f"({', '.join(f'{k} {100 * v:.1f}%' for k, v in rel_far.items() if not inside_band(v))}); pooled over the "
        f"three channels of each model they span {100 * min(rel_pooled.values()):.1f}-{100 * max(rel_pooled.values()):.1f}% "
        f"({', '.join(f'{m} {100 * v:.1f}%' for m, v in rel_pooled.items())}), which is the reading under which the "
        "range holds. Base run: pooled "
        f"{100 * base['pooled_range'][0]:.1f}-{100 * base['pooled_range'][1]:.1f}%, exceptions "
        f"{sorted(result.base.components)}")
    for record in result.runs:
        if record.axis == "hyperparams":
            m = record.finding.meta
            notes.append(
                f"{record.variant}: pooled {100 * m['pooled_range'][0]:.1f}-{100 * m['pooled_range'][1]:.1f}%, "
                f"{m['n_exceptions']} exceptions {sorted(record.finding.components)}; per-channel far shares "
                + ", ".join(f"{k} {100 * e['far']:.1f}" for k, e in m["per_entry"].items()))
    if extension:
        ext = []
        for model in extension:
            s = census.summary(model)["channels"]
            far = {ch: s[f"head_head_{ch}"]["above2"] + s[f"head_head_{ch}"]["below2"] for ch in CHANNELS}
            n = {ch: s[f"head_head_{ch}"]["n"] for ch in CHANNELS}
            pooled = sum(far[ch] * n[ch] for ch in CHANNELS) / sum(n.values())
            ext.append(f"{model} K {100 * far['K']:.1f} Q {100 * far['Q']:.1f} V {100 * far['V']:.1f} "
                       f"pooled {100 * pooled:.1f}{'' if inside_band(pooled) else ' (outside)'}")
        notes.append("extension census (not part of the finding): six further models under the "
                     "upstream processing, far-from-chance shares at |z|>=2 (%): " + "; ".join(ext))
    null_meta = next((r.finding.meta for r in (result.null_runs or []) if r.axis == "base"), None)
    notes.append(
        "null control: each writer's output factor replaced by an independent Haar rotation of itself "
        f"(seed {NULL_SEED}), the census otherwise unchanged"
        + (f"; pooled far-from-chance shares {100 * null_meta['pooled_range'][0]:.1f}-"
           f"{100 * null_meta['pooled_range'][1]:.1f}% (chance is about 5%), every entry an exception"
           if null_meta else "")
        + ". The specificity ratio is uninformative for a range-membership finding: with no signal "
        "every entry is outside the band, and 'all outside' is as stable a set as 'these four outside'")
    notes.append("seeds axis not run: the closed-form census has no randomness once the weights are "
                 "fixed. Templates axis not run: the census takes no text or prompt input; the "
                 "extension census above is the nearest substitute and is reported, not graded")

    print()
    print(result)
    print(result.to_markdown())
    stem = os.path.join(args.out_dir, "communication_map")
    result.card.save(stem + ".json")
    with open(stem + ".md", "w") as f:
        f.write(result.to_markdown() + "\n")
    with open(stem + ".badge.json", "w") as f:
        json.dump(result.card.badge_dict(), f, indent=2)
        f.write("\n")
    trace = result.verdict_trace(seed=0)
    with open(stem + ".trace.json", "w") as f:
        json.dump(trace, f, indent=2)
        f.write("\n")
    with open(stem + ".trace.md", "w") as f:
        f.write(sk.verdict_trace_markdown(trace) + "\n")
    rows = [run_row(r, "real") for r in result.runs] + \
        [run_row(r, "null") for r in (result.null_runs or [])]
    with open(stem + ".runs.json", "w") as f:
        json.dump({"upstream_commit": UPSTREAM_COMMIT, "upstream_sha256": hashes,
                   "released_census": {m: released_shares(args.upstream, m) for m in models},
                   "raw_dir": os.path.relpath(raw_dir, args.out_dir), "runs": rows},
                  f, indent=1, default=str)
        f.write("\n")
    print(sk.verdict_trace_markdown(trace))
    print(f"\nartifacts written to {stem}.*")


if __name__ == "__main__":
    main()
