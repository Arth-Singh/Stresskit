"""Reference Stability Card: "represented but not decoded" on Llama-3.1-8B-Instruct
(arXiv:2608.02486, AragonerUA/folkmotif).

Claim under test (abstract, byte-exact): "The residual stream cleanly
distinguishes cultures, well above a name-string baseline, yet the decoder
collapses culturally-specific tokens onto dominant-tradition ones." The
abstract carries no number; the quantified form is Table 2 and the released
per-model results: for Llama-3.1-8B-Instruct, culture-probe peak accuracy
0.881 at layer 8 against a character n-gram name baseline of 0.604, output
accuracy 0.248, and a 2x2 decomposition of the 270 (motif, culture) cells
into Preserved 32 / DecodingSuppressed 206 / SurfaceLuck 6 /
RepresentationallyFlat 26, i.e. a DecodingSuppressed share of 76.3%.

Upstream pipeline (scripts/run_pilot.py at the pinned commit, prompt mode
v3_e6_english, fp16): for each of the 270 cells, the entity name is placed
in a culture-free sentence ("{name} embodies the role of {description}.")
and the residual stream is mean-pooled over the entity tokens at every
layer; a 10-way ridge probe (StandardScaler + RidgeClassifier alpha = 1,
5-fold stratified CV, seed 0) is fitted per layer and the layer with the
highest accuracy is the peak; the model is then asked, in five English
paraphrases with its chat template and greedy decoding, to name the entity
of that motif in that culture, and a generation counts as correct if it
matches the gold name exactly, by substring in either direction, or at
normalised Levenshtein similarity >= 0.8 (majority over the five
paraphrases); each cell is labelled Preserved / DecodingSuppressed /
SurfaceLuck / RepresentationallyFlat from probe-correctness at the peak
layer (cross-validated prediction) x output-correctness.

Finder = that pipeline with the upstream extraction, generation, scoring,
probe and labelling functions imported unmodified, as a pure function of
(data, seed, config):

- data: the list of cells (motif, culture, gold name); each carries the
  prompt mode its generations use. Hidden states and generations are
  deterministic given the model and are cached per (dtype, prompt mode).
- seed: the CV fold seed.
- config: dtype ("fp16" | "bf16"), alpha, n_splits, agg ("majority" | "any"
  | "all"), scoring ("lenient" | "exact"), peak ("argmax" | "frac0.5").

Finding representation (fixed before any battery ran):

- components: the Preserved cells (probe correct AND output correct),
  "{motif}__{culture}". Universe = 270. The DecodingSuppressed set is the
  headline structure but covers three quarters of the universe, where the
  size-matched random null is saturated; Preserved is the sparse side of
  the same output-correctness axis and identifies the cells the model both
  represents and expresses. The DecodingSuppressed share is the score.
- claim: "DS <plurality|not plurality>; probe <well above|not well above>
  name n-gram" where plurality means DecodingSuppressed is the largest of
  the four buckets and well above means the peak probe accuracy exceeds
  the character-n-gram name baseline (same CV) by at least 0.10.
- score: the DecodingSuppressed share of the 270 cells.
- meta: peak layer and accuracy, n-gram baseline, output accuracy, the four
  bucket counts, the config.

Battery: seeds (CV seed), templates (the upstream v2_chat and v3_h6_native
prompt modes for the generation side; extraction unchanged), hyperparams
(alpha 0.1 and 10; 10 folds; any-of-five and all-of-five aggregation
instead of majority; exact-match scoring instead of the lenient rule; the
decomposition layer fixed at half depth instead of the CV-selected peak;
bfloat16 weights), plus a null control: culture labels permuted across
cells once (seed 0x5EC), run through the same finder, so the probe is
fitted to scrambled labels while the generations are unchanged.

The bootstrap axis is not run: stratified CV over a resampled cell list
puts copies of the same cell in training and held-out folds, which inflates
probe accuracy by construction; the card says so.

Usage (GPU, ~16 GB; generations ~10 min per prompt mode on an H200):
    FOLKPAPER_GROUND_TRUTH=<upstream>/dataset/ground_truth_staging.json \
    python references/run_folkmotif_card.py --upstream /path/to/folkmotif \
        --out-dir references/cards --raw-dir references/cards/raw/folkmotif_llama3p1_8b
"""

import argparse
import gc
import hashlib
import json
import os
import random
import sys
import time

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

import stresskit as sk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from battery_shards import Shard  # noqa: E402

UPSTREAM_REPO = "AragonerUA/folkmotif"
UPSTREAM_COMMIT = "cb0ae7cb9411cad485c1bfb91d1ecf759a8f7ce3"
UPSTREAM_FILES = ("pipeline/extract.py", "pipeline/probe.py", "pipeline/output_extract.py",
                  "pipeline/scoring.py", "pipeline/decomposition.py", "pipeline/prompts.py",
                  "pipeline/prompts_v3.py", "dataset/ground_truth_staging.json")
MODEL = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_KEY = "llama-3.1-8b"
MAX_NEW_TOKENS = 256
N_VARIANTS = {"v2_chat": 1, "v3_e6_english": 5, "v3_h6_native": 5}
BASE_PROMPT_MODE = "v3_e6_english"
# results/llama-3.1-8b_fp16_v3e6/{summary.json,decomposition.csv} at the pinned
# commit (the base prompt mode). The paper's decomposition table
# (analysis/v3_decomposition_shares.csv) lists the v3h6 run for this model, and
# its 0.248 output accuracy is the rescored majority: analysis/rescore_v3.py
# scores the raw generation instead of the trimmed one.
SHIPPED = {"probe_peak_acc": 0.881, "probe_peak_layer": 8, "ngram_acc": 0.604,
           "output_acc": 0.185, "output_acc_rescored_majority": 0.248,
           "buckets": {"Preserved": 45, "DecodingSuppressed": 193, "SurfaceLuck": 5,
                       "RepresentationallyFlat": 27},
           "buckets_v3h6": {"Preserved": 32, "DecodingSuppressed": 206, "SurfaceLuck": 6,
                            "RepresentationallyFlat": 26}}
BASE_CONFIG = {"dtype": "fp16", "alpha": 1.0, "n_splits": 5, "agg": "majority",
               "scoring": "lenient", "peak": "argmax"}
PLACEHOLDER = sk.Finding(claim="placeholder (other shard)", score=0.0, meta={"placeholder": True})
BUCKETS = ("Preserved", "DecodingSuppressed", "SurfaceLuck", "RepresentationallyFlat")


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


class Subject:
    """Model by dtype, one resident at a time (two copies do not fit next to
    the other tenants of the GPU)."""

    def __init__(self):
        self.dtype = None
        self.model = self.tok = None

    def get(self, dtype):
        if self.dtype != dtype:
            self.model = self.tok = None
            gc.collect()
            torch.cuda.empty_cache()
            tok = AutoTokenizer.from_pretrained(MODEL)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                MODEL, dtype={"fp16": torch.float16, "bf16": torch.bfloat16}[dtype],
                device_map="cuda").eval()
            self.model, self.tok, self.dtype = model, tok, dtype
        return self.model, self.tok


class Caches:
    """Hidden states per dtype and generations per (dtype, prompt mode),
    computed with the upstream functions and kept on disk."""

    def __init__(self, raw_dir, subject, rows, up):
        self.raw_dir, self.subject, self.rows, self.up = raw_dir, subject, rows, up
        self.hidden, self.outputs = {}, {}

    def hidden_states(self, dtype):
        if dtype in self.hidden:
            return self.hidden[dtype]
        path = os.path.join(self.raw_dir, f"hidden_{dtype}.npy")
        if not os.path.exists(path):
            model, tok = self.subject.get(dtype)
            t0 = time.time()
            H = np.stack([self.up.extract_entity_hiddens(model, tok, row) for row in self.rows])
            np.save(path + ".tmp.npy", H)
            os.replace(path + ".tmp.npy", path)
            print(f"  hidden states {dtype}: {H.shape} ({time.time() - t0:.0f}s)", flush=True)
        self.hidden[dtype] = np.load(path)
        return self.hidden[dtype]

    def generations(self, dtype, prompt_mode):
        key = (dtype, prompt_mode)
        if key in self.outputs:
            return self.outputs[key]
        path = os.path.join(self.raw_dir, f"outputs_{dtype}_{prompt_mode}.json")
        if not os.path.exists(path):
            model, tok = self.subject.get(dtype)
            t0 = time.time()
            records = []
            for i, row in enumerate(self.rows):
                for v in range(N_VARIANTS[prompt_mode]):
                    gen, raw, info = self.up.generate_entity(
                        model, tok, row, max_new_tokens=MAX_NEW_TOKENS, hf_id=MODEL,
                        prompt_mode=prompt_mode, variant_idx=v)
                    score = self.up.score_generation(gen, row.name, row.native)
                    records.append({"cell_id": row.cell_id, "variant": v, "generated": gen,
                                    "raw": raw, "lang_match": info.get("lang_match"), **score})
                if (i + 1) % 50 == 0:
                    print(f"  generations {dtype}/{prompt_mode}: {i + 1}/{len(self.rows)} "
                          f"({time.time() - t0:.0f}s)", flush=True)
            with open(path + ".tmp", "w") as f:
                json.dump(records, f, ensure_ascii=False, indent=0)
            os.replace(path + ".tmp", path)
        with open(path) as f:
            self.outputs[key] = json.load(f)
        return self.outputs[key]


def cv_predictions(X, y, n_splits, alpha, seed):
    """Upstream's decomposition-layer probe: per-fold StandardScaler +
    RidgeClassifier, cross-validated predictions for every row."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    preds = np.empty_like(y)
    for tr, te in skf.split(X, y):
        sc = StandardScaler()
        clf = RidgeClassifier(alpha=alpha)
        clf.fit(sc.fit_transform(X[tr]), y[tr])
        preds[te] = clf.predict(sc.transform(X[te]))
    return preds


def ngram_baseline(names, y, n_splits, seed):
    """Character n-gram probe on the entity name string, same CV: the
    'name-string baseline' the abstract compares against."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    names = np.asarray(names)
    accs = []
    for tr, te in skf.split(names, y):
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
        clf = RidgeClassifier(alpha=1.0)
        clf.fit(vec.fit_transform(names[tr]), y[tr])
        accs.append(float((clf.predict(vec.transform(names[te])) == y[te]).mean()))
    return float(np.mean(accs))


def make_finder(caches, up, raw_dir):
    rows_by_id = {r.cell_id: r for r in caches.rows}

    def compute(data, seed, cfg):
        t0 = time.time()
        prompt_mode = data[0]["prompt_mode"]
        H = caches.hidden_states(cfg["dtype"])
        gens = caches.generations(cfg["dtype"], prompt_mode)
        idx = np.array([d["idx"] for d in data])
        cultures = np.array([d["culture"] for d in data])
        classes, y = np.unique(cultures, return_inverse=True)
        n_layers = H.shape[1]
        layer_acc = [up._layer_probe(H[idx, L, :], y, n_splits=cfg["n_splits"],
                                     alpha=cfg["alpha"], seed=seed)[0] for L in range(n_layers)]
        peak = (int(np.argmax(layer_acc)) if cfg["peak"] == "argmax"
                else int(round((n_layers - 1) * 0.5)))
        preds = cv_predictions(H[idx, peak, :], y, cfg["n_splits"], cfg["alpha"], seed)
        probe_correct = preds == y
        ngram = ngram_baseline([d["name"] for d in data], y, cfg["n_splits"], seed)

        by_cell = {}
        for r in gens:
            if cfg["scoring"] == "exact":
                ok = r["exact"]
            elif cfg["scoring"] == "raw":
                row = rows_by_id[r["cell_id"]]
                ok = up.score_generation(r["raw"], row.name, row.native)["correct"]
            else:
                ok = r["correct"]
            by_cell.setdefault(r["cell_id"], []).append(bool(ok))
        agg = {"majority": lambda v: sum(v) >= (len(v) + 1) // 2,
               "any": any, "all": all}[cfg["agg"]]
        output_correct = np.array([agg(by_cell[d["cell_id"]]) for d in data])

        buckets = [up.label_cell(bool(p), bool(o)) for p, o in zip(probe_correct, output_correct)]
        counts = {b: buckets.count(b) for b in BUCKETS}
        preserved = {d["cell_id"] for d, b in zip(data, buckets) if b == "Preserved"}
        ds_share = counts["DecodingSuppressed"] / len(data)
        plurality = max(counts, key=counts.get) == "DecodingSuppressed"
        margin = layer_acc[peak] - ngram
        claim = (f"DS {'plurality' if plurality else 'not plurality'}; probe "
                 f"{'well above' if margin >= 0.10 else 'not well above'} name n-gram")
        meta = {"config": cfg, "prompt_mode": prompt_mode, "n_cells": len(data),
                "peak_layer": peak, "peak_acc": float(layer_acc[peak]),
                "probe_acc_by_layer": [float(a) for a in layer_acc],
                "ngram_acc": ngram, "probe_minus_ngram": float(margin),
                "output_acc": float(output_correct.mean()), "buckets": counts,
                "wall_secs": round(time.time() - t0, 1)}
        print(f"  [{cfg['dtype']} {prompt_mode} a={cfg['alpha']} k={cfg['n_splits']} "
              f"{cfg['agg']}/{cfg['scoring']}/{cfg['peak']} seed={seed}] peak L{peak} "
              f"{layer_acc[peak]:.3f} ngram {ngram:.3f} out {output_correct.mean():.3f} "
              f"DS {ds_share:.3f} {counts} ({meta['wall_secs']}s)", flush=True)
        return sk.Finding(components=preserved, universe_size=len(data), claim=claim,
                          score=ds_share, meta=meta)

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
    ap.add_argument("--n-runs", type=int, default=10)
    args = ap.parse_args()

    raw_dir = args.raw_dir or os.path.join(args.out_dir, "raw", "folkmotif_llama3p1_8b")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    hashes = {p: sha256_file(os.path.join(args.upstream, p)) for p in UPSTREAM_FILES}
    os.environ["FOLKPAPER_GROUND_TRUTH"] = os.path.join(
        args.upstream, "dataset", "ground_truth_staging.json")
    os.environ.setdefault("FOLKPAPER_RESULTS", os.path.join(raw_dir, "upstream_results"))
    sys.path.insert(0, args.upstream)
    from pipeline import decomposition, extract, output_extract, probe, scoring  # noqa: E402
    from pipeline.data import load_ground_truth  # noqa: E402

    class Up:
        extract_entity_hiddens = staticmethod(extract.extract_entity_hiddens)
        generate_entity = staticmethod(output_extract.generate_entity)
        score_generation = staticmethod(scoring.score_generation)
        label_cell = staticmethod(decomposition.label_cell)
        _layer_probe = staticmethod(probe._layer_probe)

    rows = load_ground_truth()
    if len(rows) != 270:
        raise RuntimeError(f"expected 270 cells, got {len(rows)}")
    print(f"{len(rows)} cells; {MODEL}")

    def cells(prompt_mode, cultures=None):
        return [{"idx": i, "cell_id": r.cell_id, "motif_id": r.motif_id,
                 "culture": r.culture if cultures is None else cultures[i],
                 "name": r.name, "prompt_mode": prompt_mode} for i, r in enumerate(rows)]

    data = cells(BASE_PROMPT_MODE)
    permuted = [r.culture for r in rows]
    random.Random(0x5EC).shuffle(permuted)
    null_data = cells(BASE_PROMPT_MODE, cultures=permuted)

    caches = Caches(raw_dir, Subject(), rows, Up)
    finder = make_finder(caches, Up, raw_dir)
    result = sk.stress(
        finder, data,
        battery=["seeds", "templates", "hyperparams"],
        n_runs=args.n_runs,
        config=dict(BASE_CONFIG),
        templates={"v2_chat": cells("v2_chat"), "v3_h6_native": cells("v3_h6_native")},
        hyperparams={"alpha": [0.1, 10.0], "n_splits": [10], "agg": ["any", "all"],
                     "scoring": ["exact", "raw"], "peak": ["frac0.5"], "dtype": ["bf16"]},
        null_data=null_data,
        claim_statement=(
            "The residual stream cleanly distinguishes cultures, well above a name-string "
            "baseline, yet the decoder collapses culturally-specific tokens onto "
            "dominant-tradition ones"),
        model=MODEL,
        task="FolkMotif: 270 (motif, culture) cells; 10-way culture probe on entity-token "
             "residuals vs. named-entity generation; 2x2 decomposition",
        method="upstream pipeline (ridge probe with stratified CV, greedy generation with "
               "lenient string scoring, Preserved/DecodingSuppressed/SurfaceLuck/"
               "RepresentationallyFlat labelling) at the pinned commit",
        verbose=True,
    )
    if finder.shard.is_worker:
        print(f"shard {finder.shard.index}/{finder.shard.count} done; no artifacts written")
        return

    base = result.base.meta
    result.card.notes.append(
        f"upstream: {UPSTREAM_REPO}@{UPSTREAM_COMMIT[:7]} (MIT code, CC-BY-4.0 data); "
        "extraction, generation, scoring, probe and labelling functions imported unmodified; "
        "file hashes " + ", ".join(f"{os.path.basename(p)} {h[:12]}" for p, h in hashes.items()))
    result.card.notes.append(
        f"reproduction (released llama-3.1-8b fp16 v3e6 results -> base run): probe peak "
        f"{SHIPPED['probe_peak_acc']} at layer {SHIPPED['probe_peak_layer']} -> "
        f"{base['peak_acc']:.3f} at layer {base['peak_layer']}; name n-gram baseline "
        f"{SHIPPED['ngram_acc']} -> {base['ngram_acc']:.3f}; output accuracy (majority) "
        f"{SHIPPED['output_acc']} -> {base['output_acc']:.3f}; buckets "
        f"{SHIPPED['buckets']} -> {base['buckets']}. The paper's "
        f"{SHIPPED['output_acc_rescored_majority']} output accuracy for this model is the "
        "released rescored majority (analysis/rescore_v3.py scores the raw generation "
        "instead of the trimmed one; the scoring=raw run below), and its decomposition "
        f"table lists the v3h6 run, {SHIPPED['buckets_v3h6']} (the template=v3_h6_native "
        "run below). The n-gram baseline here is a character 2-4-gram ridge probe on the "
        "name string under the same folds; upstream's analysis script uses its own n-gram "
        "classifier, not shipped with the pipeline")
    for record in result.runs:
        if record.axis in ("templates", "hyperparams"):
            m = record.finding.meta
            result.card.notes.append(
                f"{record.variant}: peak L{m['peak_layer']} acc {m['peak_acc']:.3f}, n-gram "
                f"{m['ngram_acc']:.3f}, output acc {m['output_acc']:.3f}, buckets {m['buckets']}, "
                f"DS share {record.finding.score:.3f}")
    result.card.notes.append(
        "the bootstrap axis is not run: stratified CV over a resampled cell list puts "
        "copies of one cell in training and held-out folds and inflates probe accuracy by "
        "construction")
    result.card.notes.append(
        "null control: culture labels permuted across the 270 cells once (seed 0x5EC); "
        "generations and their correctness are unchanged, only the probe sees scrambled labels")

    print()
    print(result)
    print(result.to_markdown())
    stem = os.path.join(args.out_dir, "folkmotif_llama3p1_8b")
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
    rows_out = [run_row(r, "real") for r in result.runs] + \
        [run_row(r, "null") for r in (result.null_runs or [])]
    with open(stem + ".runs.json", "w") as f:
        json.dump({"upstream_commit": UPSTREAM_COMMIT, "upstream_sha256": hashes,
                   "raw_dir": os.path.relpath(raw_dir, args.out_dir), "runs": rows_out},
                  f, indent=1, default=str)
        f.write("\n")
    print(sk.verdict_trace_markdown(trace))
    print(f"\nartifacts written to {stem}.*")


if __name__ == "__main__":
    main()
