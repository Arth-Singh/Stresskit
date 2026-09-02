"""Reference Stability Card: factual vs opinion sycophancy probes on
Llama-3.1-8B-Instruct (arXiv:2607.07003, antbaez/dissociating-sycophancy).

Claim under test (abstract, byte-exact): "We find that different LLMs
represent these subtypes differently, with either more aligned or more
distinct representations". For Llama-3.1-8B-Instruct the paper quantifies
"distinct" as a transfer gap: probes trained on one subtype reach 0.91
(factual) / 0.92 (opinion) ROC-AUC in domain and 0.70 (factual -> opinion)
/ 0.61 (opinion -> factual) across subtypes, a drop of about 0.30 (Tables
1-2, final layer, mean of five seeds). The Gemma-3-12B half of the paper
does not fit next to the other tenants of the GPU and is not audited.

Upstream pipeline (probes/ at the pinned commit): GPT-5-mini-generated
prompts, model responses committed with their GPT-5 sycophancy labels and
gpt-5-mini truncation, length balancing (process_lengths.py: the first 600
sycophantic and 600 non-sycophantic conversations per subtype, trimmed to
500 each by equalising mean token length), activations from a forward hook
on every decoder layer (generate_responses.ActivationExtractor,
extract_prediction_activations_batch), per-layer nn.Linear probes
(linear_probes.LinearProbeTrainer.train_probe: stratified 80/10/10 split,
Adam lr 1e-3, 100 epochs, batch 100, best-validation-loss checkpoint) and
transfer evaluation of each subtype's probe on the other subtype's held-out
split. The released activation cache (HF dataset antbaez/sycophancy-mech)
is private, so the activations are regenerated here with the upstream
extractor from the committed conversations.

Finder = that pipeline with the upstream extractor, sample class, length
balancing and probe trainer imported unmodified, as a pure function of
(data, seed, config):

- data: the candidate pool, 2 x 1200 records (subtype, conversation id,
  label); the finder applies upstream's length balancing to it. Bootstrap
  resamples the pool with replacement; duplicates are collapsed before
  training (a duplicate across the train and test splits would leak), so a
  bootstrap run trains on the ~63% distinct conversations of its resample
  and balances down to 5/6 of each class, upstream's 500 of 600.
- seed: the probe seed (torch.manual_seed before training, as upstream's
  trial loop does).
- config: epochs, batch_size, lr, weight_decay (train_probe arguments);
  length_balance ("upstream" | "off": the first 5/6 of each class without
  equalising lengths); layer ("final-index": upstream's last probe index,
  the paper's "final layer" | "true-final": decoder layer 31 |
  "best-in-domain": the index with the highest mean in-domain AUC).

The upstream extractor stacks the hooked layers in sorted() order of their
module names, i.e. lexicographically ("model.layers.0", "model.layers.1",
"model.layers.10", ...), so probe index 31 is decoder layer 9 and decoder
layer 31 is index 25; the map is recorded at runtime and every layer is
named by its decoder index in the finding. The hook reads position -2 of
the left-padded full conversation, the last token of the (truncated)
assistant response before its end-of-turn token; the eot_positions the
extractor computes are not used by the hook (the tokens actually read are
recorded in a note).

Finding representation (fixed before any battery ran):

- components: the eight decoder layers with the largest transfer drop,
  drop_i = mean over both directions of (in-domain AUC - transfer AUC) at
  probe index i, named "L{decoder layer}". Universe = 32 layers. A
  threshold set (drop >= 0.15) is empty under the permuted-label null and
  J(empty, empty) = 1 makes the specificity ratio degenerate, so the
  component set is a fixed-size top-k; the threshold set is recorded in
  meta.
- claim: "Llama: distinct (transfer drop >= 0.15) | shared (< 0.15);
  in-domain <bucket>" with the drop and the in-domain AUC read at the
  config's layer and the bucket in {">=0.85", "0.70-0.85", "<0.70"}. The
  paper's numbers give "distinct; in-domain >=0.85".
- score: the transfer drop at the config's layer (paper: about 0.30).
- meta: the four AUC curves over probe indices (factual->factual,
  factual->opinion, opinion->opinion, opinion->factual), layer-averaged
  AUCs (upstream's avg_auc), per-index drop, the cosine between the factual
  and opinion probe weights per index (probes/cosine_sim.py's quantity),
  best epochs, sample counts, the index -> decoder-layer map.

Battery: seeds, bootstrap, hyperparams (epochs 30; lr 1e-4; weight decay
0.01; batch 20; no length balancing; the two other layer choices), plus a
null control: the same pool through upstream's shuffle_labels=True path
(training and validation labels permuted, test labels intact), so the
probes are fitted to noise. The combined (factual+opinion) probe of the
paper is not trained: the claim compares the two subtypes. There is no
templates axis: the conversations are fixed upstream artifacts produced and
labelled with closed models.

Usage (GPU, ~45 s per run after a one-off activation pass of a few minutes):
    python references/run_sycophancy_probe_card.py --upstream /path/to/dissociating-sycophancy \
        --out-dir references/cards --raw-dir references/cards/raw/sycophancy_llama3p1_8b
"""

import argparse
import gc
import hashlib
import json
import os
import re
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

import stresskit as sk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from battery_shards import Shard  # noqa: E402

UPSTREAM_REPO = "antbaez/dissociating-sycophancy"
UPSTREAM_COMMIT = "47e02ef106896fdf45f7a13e86e306f118f109ac"
UPSTREAM_FILES = ("probes/generate_responses.py", "probes/linear_probes.py",
                  "probes/process_lengths.py",
                  "probes/response_datasets/Llama-3.1-8B-Instruct/factual_prompts_with_responses.json",
                  "probes/response_datasets/Llama-3.1-8B-Instruct/opinion_prompts_with_responses.json")
MODEL_NAME = "Llama-3.1-8B-Instruct"   # upstream key; load_model resolves it to meta-llama/
SUBTYPES = ("factual", "opinion")
POOL_PER_CLASS = 600      # process_lengths.N
KEEP_PER_CLASS = 500      # process_lengths.TARGET_PER_CLASS
BASE_SEED = 42
UPSTREAM_SEEDS = (42, 43, 44, 45, 46)
TOP_K = 8
DROP_THRESHOLD = 0.15
SHIPPED = {"ff": 0.91, "oo": 0.92, "fo": 0.70, "of": 0.61}
DIRECTIONS = {"ff": ("factual", "factual"), "fo": ("factual", "opinion"),
              "oo": ("opinion", "opinion"), "of": ("opinion", "factual")}
BASE_CONFIG = {"epochs": 100, "batch_size": 100, "lr": 1e-3, "weight_decay": 0.0,
               "length_balance": "upstream", "layer": "final-index"}
PLACEHOLDER = sk.Finding(claim="placeholder (other shard)", score=0.0, meta={"placeholder": True})


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def import_upstream(upstream):
    """generate_responses.py and process_lengths.py call huggingface_hub.login
    at import time (with the HF_TOKEN environment variable, absent here);
    the call is made a no-op before the import, the modules are otherwise
    untouched."""
    import huggingface_hub

    huggingface_hub.login = lambda *a, **k: None
    sys.path.insert(0, os.path.join(upstream, "probes"))
    import generate_responses as GR  # noqa: E402
    import linear_probes as LP  # noqa: E402
    import process_lengths as PL  # noqa: E402
    return GR, LP, PL


def load_pool(upstream):
    """process_lengths.py's candidate pool: the first 600 sycophantic and
    600 non-sycophantic conversations per subtype, in file order."""
    pool, texts = [], {}
    for sub in SUBTYPES:
        path = os.path.join(upstream, "probes", "response_datasets", MODEL_NAME,
                            f"{sub}_prompts_with_responses.json")
        with open(path) as f:
            items = json.load(f)
        syc = [it for it in items if it["is_sycophantic"] is True][:POOL_PER_CLASS]
        non = [it for it in items if it["is_sycophantic"] is False][:POOL_PER_CLASS]
        for it in syc + non:
            pool.append({"subtype": sub, "id": int(it["id"]), "label": bool(it["is_sycophantic"])})
            texts[(sub, int(it["id"]))] = it["conversation"]
    return pool, texts


class Activations:
    """Upstream extractor output for every pool conversation, cached in the
    raw dir; the model is loaded once, for the first pass only."""

    def __init__(self, raw_dir, GR, texts, pool, batch_size):
        self.raw_dir, self.GR, self.texts, self.batch_size = raw_dir, GR, texts, batch_size
        self.ids = {sub: [d["id"] for d in pool if d["subtype"] == sub] for sub in SUBTYPES}
        self.acts = {}
        self.meta_path = os.path.join(raw_dir, "extraction_meta.json")
        self.lengths_path = os.path.join(raw_dir, "token_lengths.json")

    def _path(self, sub):
        return os.path.join(self.raw_dir, f"acts_{sub}.pt")

    def ensure(self):
        missing = [sub for sub in SUBTYPES if not os.path.exists(self._path(sub))]
        if missing or not os.path.exists(self.meta_path) or not os.path.exists(self.lengths_path):
            self._extract(missing or list(SUBTYPES))
        with open(self.meta_path) as f:
            self.meta = json.load(f)
        with open(self.lengths_path) as f:
            self.lengths = {tuple(k.split("/")): v for k, v in json.load(f).items()}
        self.lengths = {(sub, int(i)): n for (sub, i), n in self.lengths.items()}
        for sub in SUBTYPES:
            blob = torch.load(self._path(sub), map_location="cpu")
            if blob["ids"] != self.ids[sub]:
                raise RuntimeError(f"cached activations for {sub} do not match the pool")
            self.acts[sub] = {i: blob["acts"][k] for k, i in enumerate(blob["ids"])}

    def _extract(self, subs):
        GR = self.GR
        t0 = time.time()
        model, tokenizer = GR.load_model(MODEL_NAME)
        order = sorted(n for n, _ in model.named_modules()
                       if n.split(".")[-1].isdigit()
                       and ("model.layers." in n or "language_model.layers." in n))
        extractor = GR.ActivationExtractor(model, tokenizer)
        read_tokens, date_string = [], None
        for sub in subs:
            convs = [GR.parse_conversation(self.texts[(sub, i)]) for i in self.ids[sub]]
            for conv in convs[:3]:
                formatted = tokenizer.apply_chat_template(conv, tokenize=False,
                                                          add_generation_prompt=False)
                ids = tokenizer(formatted)["input_ids"]
                read_tokens.append({"subtype": sub, "position_-2": tokenizer.decode([ids[-2]]),
                                    "position_-1": tokenizer.decode([ids[-1]]),
                                    "n_tokens": len(ids)})
                m = re.search(r"Today Date: ([^\n]+)", formatted)
                date_string = m.group(1) if m else date_string
            acts = extractor.extract_prediction_activations_batch(convs, batch_size=self.batch_size)
            torch.save({"ids": self.ids[sub], "acts": torch.stack(acts)}, self._path(sub) + ".tmp")
            os.replace(self._path(sub) + ".tmp", self._path(sub))
            print(f"  activations {sub}: {len(acts)} x {tuple(acts[0].shape)} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        lengths = {f"{sub}/{i}": len(tokenizer.encode(self.texts[(sub, i)]))
                   for sub in SUBTYPES for i in self.ids[sub]}
        with open(self.lengths_path, "w") as f:
            json.dump(lengths, f)
        with open(self.meta_path, "w") as f:
            json.dump({"hook_order": order,
                       "true_layers": [int(n.split(".")[-1]) for n in order],
                       "read_tokens": read_tokens, "chat_template_date": date_string,
                       "extract_batch_size": self.batch_size,
                       "model_revision": getattr(model.config, "_commit_hash", None),
                       "dtype": str(next(model.parameters()).dtype)}, f, indent=1)
        extractor.clear_hooks()
        del extractor, model
        gc.collect()
        torch.cuda.empty_cache()


def make_finder(acts, GR, LP, PL, texts, raw_dir):
    true_layers = acts.meta["true_layers"]
    n_idx = len(true_layers)

    def length_of(sample):
        return acts.lengths[(sample.subtype, sample.sample_id)]

    def compute(data, seed, cfg):
        t0 = time.time()
        shuffle = bool(data[0].get("shuffle_labels", False))
        seen, records = set(), []
        for d in data:
            key = (d["subtype"], d["id"])
            if key not in seen:
                seen.add(key)
                records.append(d)
        samples, counts = {}, {}
        for sub in SUBTYPES:
            recs = [d for d in records if d["subtype"] == sub]
            by_label = {lab: [] for lab in (True, False)}
            for d in recs:
                s = GR.ConversationSample(sample_id=d["id"], conversation=texts[(sub, d["id"])],
                                          is_sycophantic=d["label"], activations=acts.acts[sub][d["id"]])
                s.subtype = sub
                by_label[d["label"]].append(s)
            target = int(round(min(len(by_label[True]), len(by_label[False]))
                               * KEEP_PER_CLASS / POOL_PER_CLASS))
            if cfg["length_balance"] == "upstream":
                syc_eq, non_eq = PL.equalize_mean_lengths(by_label[True], by_label[False],
                                                          target_per_class=target, len_fn=length_of)
            elif cfg["length_balance"] == "off":
                syc_eq, non_eq = by_label[True][:target], by_label[False][:target]
            else:
                raise ValueError(cfg["length_balance"])
            samples[sub] = syc_eq + non_eq
            counts[sub] = {"pool": len(recs), "syc": len(syc_eq), "non_syc": len(non_eq),
                           "mean_len_syc": float(np.mean([length_of(s) for s in syc_eq])),
                           "mean_len_non_syc": float(np.mean([length_of(s) for s in non_eq]))}

        torch.manual_seed(seed)
        trainers, layer_data, best_epoch = {}, {}, {}
        auc = {k: [float("nan")] * n_idx for k in DIRECTIONS}
        for sub in SUBTYPES:
            tr = LP.LinearProbeTrainer()
            ld = tr.prepare_data(samples[sub])
            best_epoch[sub] = []
            for i, (X, y) in sorted(ld.items()):
                m = tr.train_probe(X, y, probe_name=f"layer_{i}", epochs=cfg["epochs"],
                                   batch_size=cfg["batch_size"], lr=cfg["lr"],
                                   shuffle_labels=shuffle, weight_decay=cfg["weight_decay"])
                auc[sub[0] + sub[0]][i] = float(m["auc"])
                best_epoch[sub].append(m["epoch"])
            trainers[sub], layer_data[sub] = tr, ld
        dev = trainers["factual"].device
        for key, (src, dst) in DIRECTIONS.items():
            if src == dst:
                continue
            for i in range(n_idx):
                probe = trainers[src].probes[f"layer_{i}"]
                X, y = layer_data[dst][i]
                idx = trainers[dst].test_indices[f"layer_{i}"]
                with torch.no_grad():
                    p = torch.sigmoid(probe(X[idx].to(dev)).squeeze()).cpu().numpy()
                auc[key][i] = float(roc_auc_score(y[idx].numpy(), p))
        cosine = [float(F.cosine_similarity(
            trainers["factual"].probes[f"layer_{i}"].weight.data.squeeze(0).float().unsqueeze(0),
            trainers["opinion"].probes[f"layer_{i}"].weight.data.squeeze(0).float().unsqueeze(0)))
            for i in range(n_idx)]

        drop = [0.5 * ((auc["ff"][i] - auc["fo"][i]) + (auc["oo"][i] - auc["of"][i]))
                for i in range(n_idx)]
        in_domain = [0.5 * (auc["ff"][i] + auc["oo"][i]) for i in range(n_idx)]
        if cfg["layer"] == "final-index":
            at = n_idx - 1
        elif cfg["layer"] == "true-final":
            at = true_layers.index(max(true_layers))
        elif cfg["layer"] == "best-in-domain":
            at = int(np.argmax(in_domain))
        else:
            raise ValueError(cfg["layer"])
        order = np.argsort(-np.asarray(drop))
        top = [int(i) for i in order[:TOP_K]]
        bucket = (">=0.85" if in_domain[at] >= 0.85 else
                  "0.70-0.85" if in_domain[at] >= 0.70 else "<0.70")
        distinct = drop[at] >= DROP_THRESHOLD
        claim = (f"Llama: {'distinct' if distinct else 'shared'} (transfer drop "
                 f"{'>=' if distinct else '<'} {DROP_THRESHOLD}); in-domain {bucket}")
        meta = {"config": cfg, "shuffle_labels": shuffle, "n_pool_unique": len(records),
                "samples": counts, "true_layers": true_layers, "score_index": at,
                "score_layer": true_layers[at], "auc": auc, "drop": drop, "in_domain": in_domain,
                "layer_avg_auc": {k: float(np.mean(v)) for k, v in auc.items()},
                "at_index": {k: auc[k][at] for k in DIRECTIONS},
                "cosine_factual_opinion": cosine, "best_epoch": best_epoch,
                "top8": [f"L{true_layers[i]}" for i in top],
                "layers_drop_ge_threshold": sorted(true_layers[i] for i in range(n_idx)
                                                   if drop[i] >= DROP_THRESHOLD),
                "wall_secs": round(time.time() - t0, 1)}
        print(f"  [{cfg['layer']} ep={cfg['epochs']} bs={cfg['batch_size']} lr={cfg['lr']} "
              f"wd={cfg['weight_decay']} {cfg['length_balance']} seed={seed}"
              f"{' NULL' if shuffle else ''}] at index {at} (L{true_layers[at]}): "
              f"ff {auc['ff'][at]:.3f} oo {auc['oo'][at]:.3f} fo {auc['fo'][at]:.3f} "
              f"of {auc['of'][at]:.3f} drop {drop[at]:.3f}; layer-avg drop "
              f"{np.mean(drop):.3f} ({meta['wall_secs']}s)", flush=True)
        digest = hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()[:12]
        with open(os.path.join(raw_dir, f"run_{digest}.json"), "w") as f:
            json.dump(meta, f, indent=1)
        return sk.Finding(components={f"L{true_layers[i]}" for i in top}, universe_size=n_idx,
                          claim=claim, score=float(drop[at]), meta=meta)

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", required=True)
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "cards"))
    ap.add_argument("--raw-dir", default=None)
    ap.add_argument("--n-runs", type=int, default=20)
    ap.add_argument("--extract-batch", type=int, default=25)
    args = ap.parse_args()

    raw_dir = args.raw_dir or os.path.join(args.out_dir, "raw", "sycophancy_llama3p1_8b")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    hashes = {p: sha256_file(os.path.join(args.upstream, p)) for p in UPSTREAM_FILES}

    GR, LP, PL = import_upstream(args.upstream)
    pool, texts = load_pool(args.upstream)
    n_pool = {sub: sum(d["subtype"] == sub for d in pool) for sub in SUBTYPES}
    if any(n != 2 * POOL_PER_CLASS for n in n_pool.values()):
        raise RuntimeError(f"pool sizes {n_pool}, expected {2 * POOL_PER_CLASS} per subtype")
    print(f"pool: {n_pool}; {MODEL_NAME}")

    acts = Activations(raw_dir, GR, texts, pool, args.extract_batch)
    acts.ensure()
    print(f"hook order maps probe index -> decoder layer: {acts.meta['true_layers']}")

    null_data = [dict(d, shuffle_labels=True) for d in pool]
    finder = make_finder(acts, GR, LP, PL, texts, raw_dir)
    result = sk.stress(
        finder, pool,
        battery=["seeds", "bootstrap", "hyperparams"],
        n_runs=args.n_runs, seed=BASE_SEED,
        config=dict(BASE_CONFIG),
        hyperparams={"epochs": [30], "lr": [1e-4], "weight_decay": [0.01], "batch_size": [20],
                     "length_balance": ["off"], "layer": ["true-final", "best-in-domain"]},
        null_data=null_data,
        claim_statement=(
            "We find that different LLMs represent these subtypes differently, with either "
            "more aligned or more distinct representations"),
        model="meta-llama/Llama-3.1-8B-Instruct",
        task="factual vs opinion sycophancy: per-layer linear probes on the residual stream at "
             "the end of the assistant's (truncated) response, 500+500 length-balanced "
             "conversations per subtype, in-domain vs cross-subtype ROC-AUC",
        method="upstream extractor, length balancing and probe trainer at the pinned commit "
               "(nn.Linear, Adam, 100 epochs, best-validation checkpoint, 80/10/10 split); "
               "transfer = each subtype's probe on the other subtype's held-out split",
        verbose=True,
    )

    if finder.shard.is_worker:
        print(f"shard {finder.shard.index}/{finder.shard.count} done; no artifacts written")
        return

    base = result.base.meta
    em = acts.meta
    result.card.notes.append(
        f"upstream: {UPSTREAM_REPO}@{UPSTREAM_COMMIT[:7]} (MIT); ActivationExtractor, "
        "ConversationSample, parse_conversation, load_model, equalize_mean_lengths and "
        "LinearProbeTrainer imported unmodified; huggingface_hub.login, which two upstream "
        "modules call at import time, is a no-op here (the model is loaded from the local "
        "cache); file hashes " + ", ".join(
            f"{os.path.basename(p)} {h[:12]}" for p, h in hashes.items()))
    result.card.notes.append(
        "activations: regenerated with the upstream extractor (the released cache "
        "antbaez/sycophancy-mech is private); it hooks every decoder layer and stacks them in "
        "sorted() order of the module names, i.e. lexicographically, so probe index i is "
        f"decoder layer {em['true_layers']}[i]: index {len(em['true_layers']) - 1} (the paper's "
        f"'final layer') is decoder layer {em['true_layers'][-1]} and decoder layer "
        f"{max(em['true_layers'])} is index {em['true_layers'].index(max(em['true_layers']))}. "
        "The hook reads position -2 of the left-padded full conversation (the eot_positions "
        "the extractor computes are unused); tokens read for the first conversations: "
        + "; ".join(f"{r['subtype']} -2={r['position_-2']!r} -1={r['position_-1']!r}"
                    for r in em["read_tokens"]) +
        f". Chat template date string: {em['chat_template_date']!r}; extraction batch "
        f"{em['extract_batch_size']} (upstream 100), {em['dtype']}, model revision "
        f"{em['model_revision']}")
    repro = [r for r in result.runs if r.axis in ("base", "seeds") and r.seed in UPSTREAM_SEEDS]
    n_idx = len(em["true_layers"])
    final = {k: float(np.mean([r.finding.meta["auc"][k][n_idx - 1] for r in repro]))
             for k in DIRECTIONS}
    avg = {k: float(np.mean([r.finding.meta["layer_avg_auc"][k] for r in repro]))
           for k in DIRECTIONS}
    tf = em["true_layers"].index(max(em["true_layers"]))
    true_final = {k: float(np.mean([r.finding.meta["auc"][k][tf] for r in repro]))
                  for k in DIRECTIONS}
    result.card.notes.append(
        f"reproduction (Tables 1-2, final layer, mean of seeds 42-46 -> seeds "
        f"{sorted(r.seed for r in repro)} here): factual->factual {SHIPPED['ff']} -> "
        f"{final['ff']:.3f}, opinion->opinion {SHIPPED['oo']} -> {final['oo']:.3f}, "
        f"factual->opinion {SHIPPED['fo']} -> {final['fo']:.3f}, opinion->factual "
        f"{SHIPPED['of']} -> {final['of']:.3f} at probe index {n_idx - 1} (decoder layer "
        f"{em['true_layers'][-1]}); layer-averaged (upstream avg_auc): "
        + ", ".join(f"{k} {v:.3f}" for k, v in avg.items()) +
        f"; at decoder layer {max(em['true_layers'])} (index {tf}): "
        + ", ".join(f"{k} {v:.3f}" for k, v in true_final.items()) +
        f". Base run (seed {BASE_SEED}): in-domain {base['in_domain'][base['score_index']]:.3f}, "
        f"drop {result.base.score:.3f}, layers with drop >= {DROP_THRESHOLD}: "
        f"{base['layers_drop_ge_threshold']}; samples {base['samples']}")
    for record in result.runs:
        if record.axis == "hyperparams":
            m = record.finding.meta
            result.card.notes.append(
                f"{record.variant}: at index {m['score_index']} (L{m['score_layer']}) "
                + ", ".join(f"{k} {v:.3f}" for k, v in m["at_index"].items()) +
                f", drop {record.finding.score:.3f}; layer-averaged drop "
                f"{np.mean(m['drop']):.3f}; top-8 {m['top8']}")
    result.card.notes.append(
        "DEVIATIONS: the combined factual+opinion probe is not trained (the claim compares the "
        "two subtypes); the transfer AUC is computed by applying the trained probe to the other "
        "subtype's held-out split exactly as upstream evaluate_probes_on_all_datasets does, but "
        "inline, because that function reads probes and pickles from fixed relative paths; "
        "activations are extracted for the 1200-conversation pool per subtype rather than all "
        "3000, in batches of 25 rather than 100 (bf16 padding differs), and the chat template "
        "stamps the extraction date into the system header as it did upstream")
    result.card.notes.append(
        "bootstrap: the pool is resampled with replacement and duplicates are collapsed before "
        "training, so each bootstrap run uses the ~63% distinct conversations of its resample "
        "and balances to 5/6 of each class; the templates axis is not run (the conversations "
        "are fixed artifacts generated, labelled and truncated with closed models)")
    result.card.notes.append(
        "null control: the same pool through upstream's shuffle_labels=True path, which "
        "permutes the training and validation labels and leaves the test labels intact, so "
        "every probe is fitted to noise and both in-domain and transfer AUC sit near chance")

    print()
    print(result)
    print(result.to_markdown())
    stem = os.path.join(args.out_dir, "sycophancy_llama3p1_8b")
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
                   "extraction": em, "shipped": SHIPPED,
                   "raw_dir": os.path.relpath(raw_dir, args.out_dir), "runs": rows},
                  f, indent=1, default=str)
        f.write("\n")
    print(sk.verdict_trace_markdown(trace))
    print(f"\nartifacts written to {stem}.*")


if __name__ == "__main__":
    main()
