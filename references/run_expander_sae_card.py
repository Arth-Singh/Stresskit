"""Reference Stability Card: Expander-SAE CE-loss retention on Qwen2.5-3B
(arXiv:2607.01799, Qwen replication, layer 12).

Claim under test (abstract, byte-exact): "at the most compressed modern-LM
setting, Qwen2.5-3B with $d=7$ uses $293\\times$ fewer learned decoder values
than the full dense decoder while retaining $84$% of dense CE-loss recovered."
The shipped numbers behind it (results/qwen2_5_3b_replication.json at the
pinned commit, layer 12, k = 64, n = 16384, three seeds): expander d = 7
CE-recovered 0.833 / 0.827 / 0.825, dense warm-tied 0.983 / 0.983 / 0.982,
ratio 0.842; m / d = 2048 / 7 = 292.6.

Upstream pipeline (experiments/scaling_qwen2_5_3b.py, a Modal app): stream
monology/pile-uncopyrighted, tokenise each document to at most 128 tokens,
run Qwen2.5-3B in bf16 and collect the layer-12 residual stream until 205k
tokens; train TopK SAEs (5000 Adam steps, batch 256, cosine 3e-4 -> 1e-5,
dead-feature resampling every 1000 steps) on the first 200k; then, on the
first 100 documents of the same stream, replace the layer-12 residual with
each SAE's reconstruction and report CE-loss recovered =
(CE_zero - CE_recon) / (CE_zero - CE_clean), where CE_zero zero-ablates the
whole residual stream at that layer.

Finder = that pipeline with the SAE classes and training loop imported
unmodified from the pinned repository, as a pure function of (data, seed,
config):

- data: the list of training documents (text), in stream order. Activations
  are extracted from exactly these documents, so the bootstrap axis retrains
  on a resampled corpus.
- seed: the SAE seed (expander mask, decoder init, training sampler,
  dead-feature resampling), used for both the expander and the dense SAE as
  upstream does.
- config: d, k, n, steps, dense_arch, layer, denominator ("zero" |
  "mean"), eval_set ("upstream" | "held-out").

Finding representation (fixed before any battery ran):

- components: none. The finding is a scalar comparison between two trained
  dictionaries; feature indices are not comparable across seeds and there is
  no discrete structure to overlap. Structural checks are therefore absent
  from the card by construction, not skipped.
- claim: retention bucket of R = CE_recovered(expander d) /
  CE_recovered(dense): "retains >= 0.80" (the abstract's 84% sits here),
  "retains 0.60-0.80", "retains < 0.60". A second clause records whether the
  expander's absolute CE-recovered is at least 0.80.
- score: R.
- meta: CE_clean, CE_ablated, CE_recon for both SAEs, their absolute
  CE-recovered, relative reconstruction error, dead-feature fraction, token
  counts, the evaluation set and denominator, and wall-clock.

Battery: seeds (SAE seed), bootstrap (training documents resampled with
replacement), templates (a training corpus drawn from later in the same
stream; the evaluation documents are unchanged), hyperparams (k 32 and 128;
steps 15000; dense_tied comparator, whose learned values are the m x n
decoder only, exactly the quantity the 293x counts; layer 24, the paper's
other Qwen layer; denominator "mean": mean-ablation instead of
zero-ablation, the other convention for CE recovered; eval_set "held-out":
100 documents the SAEs never saw. Upstream evaluates on the first 100
documents of the stream, which are also the first documents of its training
pool).

No null control: with no component set there is no specificity ratio to
form. The comparison between conventions (zero vs mean ablation, seen vs
held-out text) is the substantive test here and is reported per run.

Usage (GPU, ~2 min per run at 5000 steps on an H200):
    python references/run_expander_sae_card.py \
        --upstream /path/to/expander-sae --data-dir /path/to/expander_data \
        --out-dir references/cards --raw-dir references/cards/raw/expander_sae_qwen2p5_3b
"""

import argparse
import hashlib
import json
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import stresskit as sk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from battery_shards import Shard  # noqa: E402

UPSTREAM_REPO = "rodrgo/expander-sae"
UPSTREAM_COMMIT = "598f59a6dd4cb6b4e0130763868253882ed87ebe"
UPSTREAM_FILES = ("experiments/scaling_qwen2_5_3b.py", "models/expander.py",
                  "models/dense.py", "models/training.py",
                  "results/qwen2_5_3b_replication.json")
MODEL = "Qwen/Qwen2.5-3B"
DATASET = "monology/pile-uncopyrighted"
DOC_CACHE = "pile_uncopyrighted_head.json"
N_DOCS_CACHED = 6000

M = 2048
N_TRAIN = 200_000
N_TEST = 5_000
SEQ_LEN = 128
CE_N_SEQUENCES = 100
BASE_DOCS = 2400         # the base corpus; upstream's 205k tokens end near document 1680
LATER_OFFSET = 2500      # first document of the "later slice" training corpus
HELD_OUT_OFFSET = 5000   # first document of the held-out evaluation set
SHIPPED = {"expander_d7": (0.833, 0.827, 0.825), "dense_warmtied": (0.983, 0.983, 0.982)}

BASE_CONFIG = {"d": 7, "k": 64, "n": 16384, "steps": 5000, "dense_arch": "dense_warmtied",
               "layer": 12, "denominator": "zero", "eval_set": "upstream"}
PLACEHOLDER = sk.Finding(claim="placeholder (other shard)", score=0.0, meta={"placeholder": True})


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_docs(data_dir):
    """The first N_DOCS_CACHED documents of the stream, fetched once and kept
    with the dataset revision they came from."""
    path = os.path.join(data_dir, DOC_CACHE)
    if not os.path.exists(path):
        from datasets import load_dataset
        from huggingface_hub import HfApi
        revision = HfApi().dataset_info(DATASET).sha
        ds = load_dataset(DATASET, split="train", streaming=True, revision=revision)
        docs = []
        for item in ds:
            docs.append(item.get("text") or "")
            if len(docs) >= N_DOCS_CACHED:
                break
        os.makedirs(data_dir, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"dataset": DATASET, "revision": revision, "docs": docs}, f)
    with open(path) as f:
        blob = json.load(f)
    return blob["docs"], blob["revision"], sha256_file(path)


class Subject:
    def __init__(self, model_name, device):
        self.device = device
        self.lm = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.bfloat16).to(device).eval()
        self.tok = AutoTokenizer.from_pretrained(model_name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.layers = self.lm.model.layers

    def encode(self, text):
        return self.tok(text, return_tensors="pt", truncation=True,
                        max_length=SEQ_LEN, padding=False)["input_ids"]

    @torch.no_grad()
    def activations(self, docs, layer, n_tokens):
        """Upstream stage 1: layer output on each document (<= 128 tokens),
        concatenated in document order and truncated to n_tokens."""
        captured = []

        def hook(_mod, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            captured.append(h.detach().float().reshape(-1, h.shape[-1]))

        handle = self.layers[layer].register_forward_hook(hook)
        chunks, total = [], 0
        try:
            for text in docs:
                text = (text or "").strip()
                if not text:
                    continue
                ids = self.encode(text).to(self.device)
                if ids.shape[1] < 4:
                    continue
                captured.clear()
                self.lm(ids)
                chunks.append(captured[0])
                total += captured[0].shape[0]
                if total >= n_tokens:
                    break
        finally:
            handle.remove()
        if total < n_tokens:
            raise RuntimeError(f"only {total} tokens available, need {n_tokens}")
        return torch.cat(chunks, dim=0)[:n_tokens]

    def eval_sequences(self, docs):
        """Upstream stage 3 selection: the first CE_N_SEQUENCES documents of
        at least 50 characters, each tokenised to at most 128 tokens."""
        out = []
        for text in docs:
            text = (text or "").strip()
            if not text or len(text) < 50:
                continue
            ids = self.encode(text)
            if ids.shape[1] < 4:
                continue
            out.append(ids)
            if len(out) >= CE_N_SEQUENCES:
                break
        if len(out) < CE_N_SEQUENCES:
            raise RuntimeError("not enough evaluation documents")
        return out

    @torch.no_grad()
    def cross_entropy(self, seqs, layer, hook_fn=None):
        """Token-weighted mean CE over the sequences, optionally with a hook
        rewriting the layer output (upstream _ce_with_hook)."""
        handle = self.layers[layer].register_forward_hook(hook_fn) if hook_fn else None
        tot, n_tok = 0.0, 0
        try:
            for ids in seqs:
                ids = ids.to(self.device)
                loss = float(self.lm(input_ids=ids, labels=ids).loss)
                tot += loss * ids.shape[1]
                n_tok += ids.shape[1]
        finally:
            if handle is not None:
                handle.remove()
        return tot / max(n_tok, 1)


def _unwrap(out):
    if isinstance(out, tuple):
        return out[0], lambda new_h: (new_h,) + out[1:]
    return out, lambda new_h: new_h


def zero_hook(_mod, _inp, out):
    h, rewrap = _unwrap(out)
    return rewrap(torch.zeros_like(h))


def mean_hook(mean_vec):
    def hook(_mod, _inp, out):
        h, rewrap = _unwrap(out)
        return rewrap(mean_vec.to(h.dtype).expand_as(h).clone())
    return hook


def recon_hook(sae):
    def hook(_mod, _inp, out):
        h, rewrap = _unwrap(out)
        B, S, H = h.shape
        with torch.no_grad():
            recon, _ = sae(h.float().reshape(-1, H))
        return rewrap(recon.reshape(B, S, H).to(h.dtype))
    return hook


def make_finder(subject, docs, raw_dir, upstream_models, device):
    build, train_sae = upstream_models.build, upstream_models.train_sae
    eval_docs = {
        "upstream": subject.eval_sequences(docs),
        "held-out": subject.eval_sequences(docs[HELD_OUT_OFFSET:]),
    }
    baseline_cache = {}

    def baselines(layer, denominator, eval_set, train_acts):
        key = (layer, denominator, eval_set)
        if denominator == "mean":
            # the mean depends on the run's own training pool; not cached
            mean_vec = train_acts.mean(dim=0)
            ce_clean = baseline_cache.setdefault(
                (layer, "clean", eval_set),
                subject.cross_entropy(eval_docs[eval_set], layer))
            return ce_clean, subject.cross_entropy(eval_docs[eval_set], layer,
                                                   mean_hook(mean_vec))
        if key not in baseline_cache:
            baseline_cache[(layer, "clean", eval_set)] = subject.cross_entropy(
                eval_docs[eval_set], layer)
            baseline_cache[key] = subject.cross_entropy(eval_docs[eval_set], layer, zero_hook)
        return baseline_cache[(layer, "clean", eval_set)], baseline_cache[key]

    def fit(arch, m, n, d, k, seed, steps, train_np):
        t0 = time.time()
        model = build(arch, m=m, n=n, d=d, k=k, seed=seed)
        model, _ = train_sae(model, train_np, steps=steps, batch_size=256,
                             lr_max=3e-4, lr_min=1e-5, grad_clip=1.0,
                             resample_interval=1000, device=device)
        return model.to(device).eval(), time.time() - t0

    def compute(data, seed, cfg):
        layer, m, n, d, k = cfg["layer"], M, cfg["n"], cfg["d"], cfg["k"]
        t0 = time.time()
        acts = subject.activations(data, layer, N_TRAIN + N_TEST)
        train_t, test_t = acts[:N_TRAIN], acts[N_TRAIN:N_TRAIN + N_TEST]
        train_np = train_t.cpu().numpy()
        ce_clean, ce_ablated = baselines(layer, cfg["denominator"], cfg["eval_set"], train_t)
        denom = ce_ablated - ce_clean

        out = {}
        for tag, arch, dd in (("expander", "expander_tied", d),
                              ("dense", cfg["dense_arch"], m)):
            sae, secs = fit(arch, m, n, dd, k, seed, cfg["steps"], train_np)
            with torch.no_grad():
                y_hat, h = sae(test_t)
                rel_err = float((torch.norm(test_t - y_hat, dim=-1) /
                                 torch.norm(test_t, dim=-1).clamp(min=1e-12)).mean())
                dead = float((h.abs().sum(dim=0) == 0).float().mean())
            ce_recon = subject.cross_entropy(eval_docs[cfg["eval_set"]], layer, recon_hook(sae))
            out[tag] = {"arch": arch, "d": dd, "ce_reconstructed": ce_recon,
                        "ce_recovered": (ce_ablated - ce_recon) / denom,
                        "rel_err": rel_err, "dead_frac": dead, "train_secs": round(secs, 1),
                        "learned_decoder_values": dd * n}
            del sae
            torch.cuda.empty_cache()
        del acts, train_t, test_t
        torch.cuda.empty_cache()

        ratio = out["expander"]["ce_recovered"] / out["dense"]["ce_recovered"]
        if ratio >= 0.80:
            bucket = "retains >= 0.80"
        elif ratio >= 0.60:
            bucket = "retains 0.60-0.80"
        else:
            bucket = "retains < 0.60"
        absolute = ("expander CE-recovered >= 0.80"
                    if out["expander"]["ce_recovered"] >= 0.80
                    else "expander CE-recovered < 0.80")
        meta = {"config": cfg, "ce_clean": ce_clean, "ce_ablated": ce_ablated,
                "denominator_nats": denom, "n_train_docs": len(data),
                "n_unique_train_docs": len(set(data)),
                "decoder_value_ratio": (m * n) / (d * n),
                "expander": out["expander"], "dense": out["dense"],
                "wall_secs": round(time.time() - t0, 1)}
        digest = hashlib.sha256(
            json.dumps(meta, sort_keys=True, default=str).encode()).hexdigest()[:12]
        with open(os.path.join(raw_dir, f"run_{digest}.json"), "w") as f:
            json.dump(meta, f, indent=1)
        print(f"  [{cfg['eval_set']}/{cfg['denominator']} L{layer} d={d} k={k} steps={cfg['steps']} "
              f"seed={seed}] expander {out['expander']['ce_recovered']:.3f} "
              f"dense {out['dense']['ce_recovered']:.3f} ratio {ratio:.3f} "
              f"({meta['wall_secs']}s)", flush=True)
        return sk.Finding(claim=f"{bucket}; {absolute}", score=ratio, meta=meta)

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
            "score": f.score, "meta": f.meta}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "cards"))
    ap.add_argument("--raw-dir", default=None)
    ap.add_argument("--n-runs", type=int, default=12)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    raw_dir = args.raw_dir or os.path.join(args.out_dir, "raw", "expander_sae_qwen2p5_3b")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    hashes = {p: sha256_file(os.path.join(args.upstream, p)) for p in UPSTREAM_FILES}
    sys.path.insert(0, args.upstream)
    import models as upstream_models  # noqa: E402

    docs, revision, docs_sha = load_docs(args.data_dir)
    print(f"{len(docs)} documents cached from {DATASET}@{revision[:12]} (sha256 {docs_sha[:12]})")
    subject = Subject(MODEL, args.device)
    print(f"{MODEL}: {len(subject.layers)} layers, d_model {subject.lm.config.hidden_size}")

    # the base training corpus: the first BASE_DOCS documents in stream order.
    # Extraction stops at 205k tokens, so the base run trains on exactly the
    # tokens upstream used; the extra documents are slack for the bootstrap
    # axis, whose resampled corpora (with duplicates) must still reach 205k.
    base_docs = docs[:BASE_DOCS]
    total = sum(subject.encode(t.strip()).shape[1] for t in base_docs
                if t and t.strip() and subject.encode(t.strip()).shape[1] >= 4)
    if total < 1.3 * (N_TRAIN + N_TEST):
        raise RuntimeError(f"base corpus has only {total} tokens; need slack above 205k")
    later_docs = docs[LATER_OFFSET:HELD_OUT_OFFSET]
    print(f"base corpus: first {BASE_DOCS} documents ({total} tokens, truncated to "
          f"{N_TRAIN + N_TEST} at extraction); later slice starts at document "
          f"{LATER_OFFSET}; held-out evaluation from document {HELD_OUT_OFFSET}")

    finder = make_finder(subject, docs, raw_dir, upstream_models, args.device)
    result = sk.stress(
        finder, base_docs,
        battery=["seeds", "bootstrap", "templates", "hyperparams"],
        n_runs=args.n_runs,
        config=dict(BASE_CONFIG),
        templates={"pile-later-slice": later_docs},
        hyperparams={"k": [32, 128], "steps": [15000], "dense_arch": ["dense_tied"],
                     "layer": [24], "denominator": ["mean"], "eval_set": ["held-out"]},
        claim_statement=(
            "Qwen2.5-3B with d=7 uses 293x fewer learned decoder values than the "
            "full dense decoder while retaining 84% of dense CE-loss recovered"),
        model=MODEL,
        task="layer-12 residual-stream TopK SAEs (k=64, n=16384) trained on 200k "
             "pile-uncopyrighted tokens; CE-loss recovered on 100 sequences",
        method="Expander SAE (left-7-regular tied support) vs dense warm-tied SAE, "
               "upstream classes and training loop at the pinned commit",
        verbose=True,
    )

    if finder.shard.is_worker:
        print(f"shard {finder.shard.index}/{finder.shard.count} done; no artifacts written")
        return

    base = result.base.meta
    result.card.notes.append(
        f"upstream: {UPSTREAM_REPO}@{UPSTREAM_COMMIT[:7]} (MIT); SAE classes and training "
        "loop imported unmodified; the Modal orchestration is replaced by a local driver "
        "that follows experiments/scaling_qwen2_5_3b.py stage by stage; file hashes "
        + ", ".join(f"{os.path.basename(p)} {h[:12]}" for p, h in hashes.items()))
    result.card.notes.append(
        f"data: {DATASET} at revision {revision[:12]}, first {N_DOCS_CACHED} documents "
        f"cached (sha256 {docs_sha[:12]}); base corpus = the first {BASE_DOCS} documents, "
        f"extracted in order and truncated to {N_TRAIN} training + {N_TEST} test tokens "
        f"(upstream's 205k tokens end near document 1680; the rest is slack for the "
        f"bootstrap axis); the 'later slice' corpus "
        f"starts at document {LATER_OFFSET}; held-out evaluation uses documents from "
        f"{HELD_OUT_OFFSET} on, which no training corpus in this battery touches")
    result.card.notes.append(
        f"reproduction: shipped layer-12 CE-recovered {SHIPPED['expander_d7']} (expander "
        f"d=7) and {SHIPPED['dense_warmtied']} (dense warm-tied), ratio 0.842; base run "
        f"here {base['expander']['ce_recovered']:.3f} and {base['dense']['ce_recovered']:.3f}, "
        f"ratio {result.base.score:.3f} (bf16 on a different GPU and torch; exact byte "
        "reproduction is not expected)")
    result.card.notes.append(
        f"denominators in the base run: CE_clean {base['ce_clean']:.3f}, CE_zero-ablated "
        f"{base['ce_ablated']:.3f} nats/token, so 'CE recovered' is measured against a "
        f"{base['denominator_nats']:.1f}-nat collapse; the mean-ablation and held-out runs "
        "in the hyperparams axis show what the same dictionaries retain under the other "
        "convention and on text they were not trained on")
    result.card.notes.append(
        "evaluation overlap in the upstream protocol: the 100 CE-evaluation sequences are "
        "the first 100 documents of the stream, and the training pool is the first ~205k "
        "tokens of the same stream, so upstream's CE recovered is measured on training "
        "text; the 'eval_set=held-out' run is the same protocol on unseen documents")
    result.card.notes.append(
        "no null control: the finding has no component set, so no specificity ratio can "
        "be formed; the substantive comparison is between measurement conventions, "
        "reported per run in the manifest")

    print()
    print(result)
    print(result.to_markdown())
    stem = os.path.join(args.out_dir, "expander_sae_qwen2p5_3b")
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
    with open(stem + ".runs.json", "w") as f:
        json.dump({"upstream_commit": UPSTREAM_COMMIT, "upstream_sha256": hashes,
                   "dataset_revision": revision, "docs_sha256": docs_sha,
                   "raw_dir": os.path.relpath(raw_dir, args.out_dir),
                   "runs": [run_row(r, "real") for r in result.runs]}, f, indent=1)
        f.write("\n")
    print(sk.verdict_trace_markdown(trace))
    print(f"\nartifacts written to {stem}.*")


if __name__ == "__main__":
    main()
