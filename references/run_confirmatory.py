"""Confirmatory-profile certificates for reference cards whose finder is cheap
once its features are cached: HARC (Llama-3.1-8B-Instruct), REINS-Gate
(Qwen3.5-2B-Base), the faithfulness-steering cross-cue convergence
(Gemma-3-4B-it) and the sycophancy probes (Llama-3.1-8B-Instruct).

The diagnostic cards vary one axis at a time around the paper's own setting
and grade with bootstrap intervals; they say where a finding is fragile. The
confirmatory profile (docs/CALIBRATION_REPORT_v0.1.md, stresskit.confirmatory)
asks a narrower question with finite-sample guarantees: over IID draws from a
frozen product distribution of defensible specifications, does the registered
claim clear every registered gate? Runs are independent specification draws,
checks use disjoint-pair Hoeffding intervals with a Bonferroni familywise
budget at 95%, at least 200 real and 200 null runs are required, and the
verdict is pass, fail or inconclusive; a check whose interval crosses its bar
is inconclusive, never a pass.

Everything below was frozen before the first confirmatory run and is not
changed after seeing outcomes.

Specification spaces. Each target's space is the product of the axes its
diagnostic battery varied, drawn jointly. Seed axes (the extraction split,
the calibration split, the task subsample, the probe seed) are uniform over
20 values, the first being the paper's own seed. The "resample" axis draws
the records with replacement under a fixed seed (values 1-9) or leaves them
as the paper's rows (value 0); the paper's rows carry half the mass. Every
other axis is a defensible alternative to the paper's own setting: the
paper's value carries probability 1/2 and the alternatives share the other
1/2 equally. Axes that change the size of the component set (HARC top_k,
REINS topk) are not varied: two runs with different set sizes cannot reach
a Jaccard of 1 and the confirmatory profile has no size-comparability
exclusion, so the registered representation fixes them at the paper's value
(top_k 8 on HARC, topk 256 on REINS). Faithfulness steering varies the
paper's completion wording, pooling and prompt cueing but not the extra
completion sets of the diagnostic battery, so that twelve activation passes
cover the whole space.

- harc_llama: split_seed {0..19} x resample x features {cb_ultrachat/chat
  (paper), cb_ultrachat/raw, advbench_alpaca/chat, advbench_alpaca/raw} x
  estimator {diffmeans, probe} x harm_position {t_inst, mean_content} x
  response_window {32, 8} x n_extract {300, 100} x drop_truncated {False,
  True}; null: the same draws with the harmful/harmless labels permuted
  inside the extraction split.
- reins_gate: split_seed {12 (paper), 100..118} x resample x template
  {answer_en_v1 (paper), plain, answer_en_v2} x target_negative_fpr {0.10,
  0.05, 0.20} x folds {5, 3} x layers {all, last, late} x negatives
  {matched_safe, all_safe}; null: labels permuted within each category's
  calibration set.
- faithfulness: subsample_seed {42 (paper), 100..118} x resample x phrasing
  {upstream, paraphrase_a, paraphrase_b} x pooling {completion_mean,
  last_token} x n_per_cue {None, 20, 50} x subsample {0.8, 1.0} x ref_layer
  {17, 11} x prompt {cued, uncued}; null: polarity-balanced random halves.
- sycophancy: probe_seed {42 (paper), 100..118} x resample x epochs {100,
  30} x lr {1e-3, 1e-4} x weight_decay {0.0, 0.01} x batch_size {100, 20} x
  length_balance {upstream, off} x layer {final-index, true-final,
  best-in-domain}; null: upstream shuffle_labels.

Claims. Each target's claim classes are the finite set of strings its
finder can return (the cross product of its pre-registered buckets), listed
in CLAIM_CLASSES below; a run whose claim is outside the set aborts the
audit.

Thresholds, identical for every target: structural_stability >= 0.80 (mean
Jaccard of disjoint IID run pairs; the same overlap bar the diagnostic
profile uses), beats_random >= 0.20 (Jaccard minus the exact pair-size-
matched uniform-set expectation; the README's frozen default), claim
stability >= 0.80 (population modal share over the registered classes),
specificity >= 0.20 (real minus null mean Jaccard; the README's frozen
default). Seeds: real manifest 20260901, null manifest 20260902, pairing
master seed 20260903. 200 real and 200 null draws per target, minimum 200.

The finders, nulls and caches are the diagnostic runners' own
(run_harc_card.py, run_reins_gate_card.py, run_faithfulness_steering_card.py,
run_sycophancy_probe_card.py), imported and called unchanged; this driver
only draws the manifests, maps each configuration onto (data, seed, config)
and writes the confirmatory card (the card carries every run's component set,
digest-checked by `stresskit verify`; the `.runs.json` manifest carries each
draw's configuration, seed, config, claim, score, size and meta). Runs shard
across processes with
references/battery_shards.Shard (STRESSKIT_SHARD=i/n) and are cached, so the
assembler re-derives nothing.

Usage (on the box; --phase extract needs a GPU, --phase run is a CPU pass
except for the sycophancy probes):
    python references/run_confirmatory.py harc_llama --upstream upstream/HARC --data-dir harc_data \\
        --raw-dir out/cards/raw/harc_llama3p1_8b --out-dir out/cards/confirmatory --phase extract
    STRESSKIT_SHARD=0/16 python references/run_confirmatory.py harc_llama ... --phase run
    python references/run_confirmatory.py harc_llama ... --phase run      # assembler
"""

import argparse
import hashlib
import importlib.util
import itertools
import json
import os
import random
import sys
import time
import types

import stresskit as sk
from stresskit.confirmatory import confirmatory_from_findings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from battery_shards import Shard  # noqa: E402

MANIFEST_SEED = 20260901
NULL_MANIFEST_SEED = 20260902
MASTER_SEED = 20260903
N_RUNS = 200
MINIMUM_RUNS = 200
CONFIDENCE = 0.95
THRESHOLDS = {"structural_stability": 0.80, "beats_random": 0.20,
              "claim_stability": 0.80, "specificity": 0.20}
JUSTIFICATIONS = {
    "structural_stability": "mean Jaccard of disjoint IID run pairs at least 0.80: the same "
                            "overlap bar the diagnostic profile registers, applied to independent "
                            "specification draws instead of one-axis-at-a-time variants",
    "beats_random": "Jaccard at least 0.20 above the exact pair-size-matched uniform-set "
                    "expectation: the README's frozen minimum excess over chance",
    "claim_stability": "population modal claim share at least 0.80 over the registered classes: "
                       "the diagnostic profile's filability bar",
    "specificity": "real minus null-control mean Jaccard at least 0.20: the README's frozen "
                   "minimum real-minus-null difference",
}
RESAMPLE_VALUES = list(range(10))
RESAMPLE_WEIGHTS = [0.5] + [0.5 / 9] * 9
PLACEHOLDER = sk.Finding(claim="placeholder (other shard)", score=0.0, meta={"placeholder": True})


def default_first(values):
    """The paper's value (first) carries half the mass; alternatives share the rest."""
    n = len(values)
    return [0.5] + [0.5 / (n - 1)] * (n - 1)


def load_runner(module_name):
    spec = importlib.util.spec_from_file_location(module_name, os.path.join(HERE, module_name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def resample(records, k):
    if k == 0:
        return [dict(r) for r in records]
    rng = random.Random(10_000 + k)
    return [dict(records[rng.randrange(len(records))]) for _ in records]


def seed_axis(paper_seed):
    return [paper_seed] + [s for s in range(100, 119) if s != paper_seed]


# ------------------------------------------------------------------ targets

class Target:
    name = ""
    stem = ""
    runner_module = ""
    claim_statement = ""
    model = task = method = ""
    space = None
    claim_classes = ()

    def __init__(self, args):
        self.args = args
        self.runner = load_runner(self.runner_module)
        self.conf_raw = os.path.join(args.raw_dir, "confirmatory")
        os.makedirs(self.conf_raw, exist_ok=True)

    def hashes(self):
        return {p: self.runner.sha256_file(os.path.join(self.args.upstream, p))
                for p in self.runner.UPSTREAM_FILES}

    def extract(self):
        raise NotImplementedError

    def setup(self):
        raise NotImplementedError

    def realise(self, configuration, null):
        raise NotImplementedError


class Harc(Target):
    name = "harc_llama"
    stem = "harc_llama3p1_8b"
    runner_module = "run_harc_card"
    claim_statement = ("aligned LLMs encode harmfulness and refusal as separable directions in the "
                       "residual stream at prompt-side token positions; HARC pairs the two directions "
                       "across both prompt and response positions")
    model = "meta-llama/Llama-3.1-8B-Instruct with the released HARC LoRA adapter (microsoft/HARC/adapters/harc_llama3.1_8b)"
    task = ("per-layer cosine between the harmfulness direction (t_inst) and the refusal direction "
            "(t_post), base model vs HARC adapter, prompt and response side; the eight cells with "
            "the largest coupling gain")
    method = "upstream difference-of-means or logistic-probe extraction on cached residuals, both models"
    space = sk.SpecificationSpace(
        axes={"split_seed": list(range(20)), "resample": RESAMPLE_VALUES,
              "features": ["cb_ultrachat/chat", "cb_ultrachat/raw", "advbench_alpaca/chat",
                           "advbench_alpaca/raw"],
              "estimator": ["diffmeans", "probe"], "harm_position": ["t_inst", "mean_content"],
              "response_window": [32, 8], "n_extract": [300, 100], "drop_truncated": [False, True]},
        weights={"resample": RESAMPLE_WEIGHTS, "features": default_first([0] * 4),
                 "estimator": default_first([0] * 2), "harm_position": default_first([0] * 2),
                 "response_window": default_first([0] * 2), "n_extract": default_first([0] * 2),
                 "drop_truncated": default_first([0] * 2)})
    claim_classes = tuple(
        f"{profile}; HARC couples {coupled}; prompt gain peaks {loc}"
        for profile in ("base: late-decoupled", "base: no late decoupling")
        for coupled in ("prompt+response", "prompt only", "response only", "neither")
        for loc in ("in/after band", "upstream of band"))

    def _load(self):
        h = self.runner
        self.spec = h.MODELS["llama"]
        self.up_data, self.up_dirs, self.up_paper = h.import_upstream(self.args.upstream)
        self.pools, self.pairs, self.cb_rows = h.load_pools(self.args.data_dir, self.up_data, h.POOL_N)
        self.manifest_path = os.path.join(self.args.raw_dir, "features", "manifest.json")

    def _features_present(self):
        h = self.runner
        need = [h.feature_path(self.args.raw_dir, k, v) for k in h.FEATURE_VARIANTS for v in h.VARIANTS] + \
               [h.response_path(self.args.raw_dir, t, v) for t in h.PAIR_TYPES for v in h.VARIANTS]
        return all(os.path.exists(p) for p in need) and os.path.exists(self.manifest_path)

    def extract(self):
        h = self.runner
        self._load()
        if not self._features_present():
            ns = types.SimpleNamespace(data_dir=self.args.data_dir)
            manifest = h.prepare(ns, self.spec, self.up_data, self.up_dirs, self.up_paper, self.pools,
                                 self.pairs, self.cb_rows, self.args.raw_dir, False)
        else:
            manifest = json.load(open(self.manifest_path))
        self._truncation(manifest)
        print("features ready:", {k: v for k, v in manifest.items() if k in ("n_layers", "hidden_size")})

    def _truncation(self, manifest):
        h = self.runner
        if any("truncated_rows" not in manifest[k] for k in h.FEATURE_VARIANTS):
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(self.spec["base"])
            for key in h.FEATURE_VARIANTS:
                pool, template = key.split("/")
                prompts = self.pools[pool]["harmful"] + self.pools[pool]["harmless"]
                texts, _P = h.render(tok, self.up_dirs, self.up_paper, self.spec["base"], prompts, template)
                lengths = [len(tok(t, add_special_tokens=False).input_ids) for t in texts]
                manifest[key]["truncated_rows"] = [i for i, ln in enumerate(lengths) if ln > h.MAX_LENGTH]
                manifest[key]["truncated"] = len(manifest[key]["truncated_rows"])
            json.dump(manifest, open(self.manifest_path, "w"), indent=1)

    def setup(self):
        h = self.runner
        self._load()
        if not self._features_present():
            raise RuntimeError("HARC features missing: run --phase extract on a GPU first")
        manifest = json.load(open(self.manifest_path))
        self._truncation(manifest)
        caches = h.Caches(self.args.raw_dir, manifest)
        self.finder = h.make_finder(self.spec, caches, self.conf_raw)
        self.provenance = {"upstream": h.UPSTREAM_REPO, "commit": h.UPSTREAM_COMMIT, "files": self.hashes(),
                           "adapter": f"{h.ADAPTER_REPO}/{self.spec['adapter']}",
                           "collector_checks": {k: v for k, v in manifest.items() if k.startswith("collector_check")}}

    def realise(self, c, null):
        h = self.runner
        data = resample(h.make_records(c["features"], h.POOL_N), c["resample"])
        if null:
            for r in data:
                r["null"] = True
        cfg = dict(h.BASE_CONFIG, estimator=c["estimator"], harm_position=c["harm_position"],
                   response_window=c["response_window"], n_extract=c["n_extract"],
                   drop_truncated=c["drop_truncated"], top_k=h.TOP_K)
        return data, c["split_seed"], cfg


class Reins(Target):
    name = "reins_gate"
    stem = "reins_gate_qwen3p5_2b_base"
    runner_module = "run_reins_gate_card"
    claim_statement = ("The frozen Qwen3.5-2B-Base gate opened on 98.7% of harmful evaluation prompts "
                       "and 4.7% of negative evaluation prompts. This preserves high harmful coverage "
                       "and keeps negative openings rare.")
    model = "Qwen/Qwen3.5-2B-Base with the released REINS-SAE bundle"
    task = ("REINS-Gate on GUISE: per-category sparse cosine router over prompt-side SAE feature means, "
            "harmful vs matched-safe prompts; the five category gates' coordinates")
    method = "upstream split_samples, feature_means and fit_prompt_gate at the pinned commit (topk 256)"
    space = sk.SpecificationSpace(
        axes={"split_seed": seed_axis(12), "resample": RESAMPLE_VALUES,
              "template": ["answer_en_v1", "plain", "answer_en_v2"],
              "target_negative_fpr": [0.10, 0.05, 0.20], "folds": [5, 3],
              "layers": ["all", "last", "late"], "negatives": ["matched_safe", "all_safe"]},
        weights={"resample": RESAMPLE_WEIGHTS, "template": default_first([0] * 3),
                 "target_negative_fpr": default_first([0] * 3), "folds": default_first([0] * 2),
                 "layers": default_first([0] * 3), "negatives": default_first([0] * 2)})
    claim_classes = tuple(f"harmful open {h}; matched-safe open {s}"
                          for h in (">=0.9", "0.7-0.9", "<0.7") for s in ("<=0.10", "0.10-0.20", ">0.20"))

    def _load(self):
        r = self.runner
        self.D, self.G, self.GEN, self.H, self.INT, self.R = r.import_upstream(self.args.upstream)
        self.samples, self.by_id = r.load_guise(self.args.upstream, self.D)
        self.released, _rows = r.released_gates(self.args.upstream, self.G)
        self.store = r.FeatureStore(self.args.raw_dir)

    def extract(self):
        r = self.runner
        self._load()
        ns = types.SimpleNamespace(raw_dir=self.args.raw_dir, model_path=self.args.model_path,
                                   sae_root=self.args.sae_root, device=self.args.device,
                                   templates=self.args.templates, smoke=False)
        r.extract_features(ns, self.D, self.GEN, self.R, self.samples)

    def setup(self):
        r = self.runner
        self._load()
        for t in self.space.axes["template"]:
            if not self.store.has(t):
                raise RuntimeError(f"REINS features for {t} missing: run --phase extract first")
        inner = r.make_finder(self.store, self.D, self.G, self.by_id, self.released, self.conf_raw)
        shard = Shard(os.path.join(self.conf_raw, "shard_cache"))

        def finder(data, seed, cfg):
            return shard.run(lambda: inner(data, seed, cfg), data, seed, cfg, PLACEHOLDER)

        finder.shard = shard
        self.finder = finder
        self.provenance = {"upstream": r.UPSTREAM_REPO, "commit": r.UPSTREAM_COMMIT, "files": self.hashes(),
                           "model": r.MODEL_ID, "sae": r.SAE_ID}
        self._base = [{"id": s.sample_id, "category": s.major_category}
                      for s in sorted(self.samples, key=lambda s: s.sample_id)]

    def realise(self, c, null):
        r = self.runner
        data = resample([dict(d, template=c["template"]) for d in self._base], c["resample"])
        if null:
            for d in data:
                d["null"] = True
        cfg = dict(r.BASE_CONFIG, target_negative_fpr=c["target_negative_fpr"], folds=c["folds"],
                   layers=c["layers"], negatives=c["negatives"])
        return data, c["split_seed"], cfg


class Faithfulness(Target):
    name = "faithfulness"
    stem = "faithfulness_steering_gemma3_4b"
    runner_module = "run_faithfulness_steering_card"
    claim_statement = ("when steering is effective, its effect generalizes broadly across cue types and "
                       "datasets--in cross-cue and cross-dataset analyses, effect size is determined "
                       "primarily by the evaluation setting, rather than the vector's train setting. How "
                       "the vector is built also matters little--four construction methods, including one "
                       "whose optimization target mentions no specific cue, yield similar effect sizes.")
    model = "google/gemma-3-4b-it"
    task = ("cross-cue convergence of synthetic difference-of-means cue-acknowledgment vectors rebuilt "
            "at every decoder layer; the convergence band of layers")
    method = "upstream synthetic row builder and activation collector, difference of means per cue"
    space = sk.SpecificationSpace(
        axes={"subsample_seed": seed_axis(42), "resample": RESAMPLE_VALUES,
              "phrasing": ["upstream", "paraphrase_a", "paraphrase_b"],
              "pooling": ["completion_mean", "last_token"], "n_per_cue": [None, 20, 50],
              "subsample": [0.8, 1.0], "ref_layer": [17, 11], "prompt": ["cued", "uncued"]},
        weights={"resample": RESAMPLE_WEIGHTS, "phrasing": default_first([0] * 3),
                 "pooling": default_first([0] * 2), "n_per_cue": default_first([0] * 3),
                 "subsample": default_first([0] * 2), "ref_layer": default_first([0] * 2),
                 "prompt": default_first([0] * 2)})
    claim_classes = tuple(f"cross-cue convergence at L{ref} {bucket}; absolute band {size}"
                          for ref in (17, 11) for bucket in (">=0.8", "0.5-0.8", "<0.5")
                          for size in (">=20 layers", "10-19 layers", "1-9 layers", "no layer"))

    def _load(self):
        f = self.runner
        from transformers import AutoTokenizer
        self.BOE, self.BSE, self.BSV, self.CM, self.template, self.cues = f.import_upstream(self.args.upstream)
        tokenizer = AutoTokenizer.from_pretrained(f.MODEL_ID, trust_remote_code=True)
        self.rows, self.rows_uncued, self.ids = f.build_rows(self.args.upstream, self.BSE, self.BOE, self.CM,
                                                             self.template, self.cues, tokenizer)
        self.cache = f.ActivationCache(self.args.raw_dir, self.rows, self.rows_uncued, self.ids, self.BSV,
                                       self.args.device, self.args.batch_size)

    def _variants(self):
        f = self.runner
        base = dict(f.BASE_CONFIG)
        out = []
        for phrasing, pooling, prompt in itertools.product(self.space.axes["phrasing"], self.space.axes["pooling"],
                                                           self.space.axes["prompt"]):
            pos, neg = f.completion_texts(self.BSE, self.BOE, phrasing, base)
            out.append((pos, neg, pooling, prompt))
        return out

    def extract(self):
        self._load()
        for pos, neg, pooling, prompt in self._variants():
            self.cache.get(pos, neg, pooling, prompt)
        self.cache.release()
        print(f"{len(self._variants())} activation variants cached under {self.cache.dir}")

    def setup(self):
        f = self.runner
        self._load()
        for pos, neg, pooling, prompt in self._variants():
            key = self.cache.key(pos, neg, pooling, prompt)
            if not os.path.exists(os.path.join(self.cache.dir, f"{key}.pt")):
                raise RuntimeError("faithfulness activation variant missing: run --phase extract first")
        shipped = f.load_shipped_synthetic(self.args.upstream)
        self.finder = f.make_finder(self.cache, self.BSE, self.BOE, shipped, self.conf_raw)
        self.provenance = {"upstream": f.UPSTREAM_REPO, "commit": f.UPSTREAM_COMMIT,
                           "monitorability_commit": f.MONITORABILITY_COMMIT, "files": self.hashes(),
                           "model": f.MODEL_ID}
        self._base = [{"cue": c, "task_id": t} for c in f.CUES for t in self.ids[c]]

    def realise(self, c, null):
        f = self.runner
        data = resample([dict(d, phrasing=c["phrasing"]) for d in self._base], c["resample"])
        if null:
            for d in data:
                d["null"] = True
        cfg = dict(f.BASE_CONFIG, pooling=c["pooling"], n_per_cue=c["n_per_cue"], subsample=c["subsample"],
                   ref_layer=c["ref_layer"], prompt=c["prompt"])
        return data, c["subsample_seed"], cfg


class Sycophancy(Target):
    name = "sycophancy"
    stem = "sycophancy_llama3p1_8b"
    runner_module = "run_sycophancy_probe_card"
    claim_statement = ("We find that different LLMs represent these subtypes differently, with either "
                       "more aligned or more distinct representations")
    model = "meta-llama/Llama-3.1-8B-Instruct"
    task = ("factual vs opinion sycophancy: per-layer linear probes on the residual stream, in-domain vs "
            "cross-subtype ROC-AUC; the eight decoder layers with the largest transfer drop")
    method = "upstream extractor, length balancing and probe trainer at the pinned commit"
    space = sk.SpecificationSpace(
        axes={"probe_seed": seed_axis(42), "resample": RESAMPLE_VALUES, "epochs": [100, 30],
              "lr": [1e-3, 1e-4], "weight_decay": [0.0, 0.01], "batch_size": [100, 20],
              "length_balance": ["upstream", "off"], "layer": ["final-index", "true-final", "best-in-domain"]},
        weights={"resample": RESAMPLE_WEIGHTS, "epochs": default_first([0] * 2), "lr": default_first([0] * 2),
                 "weight_decay": default_first([0] * 2), "batch_size": default_first([0] * 2),
                 "length_balance": default_first([0] * 2), "layer": default_first([0] * 3)})
    claim_classes = tuple(
        f"Llama: {'distinct' if d else 'shared'} (transfer drop {'>=' if d else '<'} 0.15); in-domain {bucket}"
        for d in (True, False) for bucket in (">=0.85", "0.70-0.85", "<0.70"))

    def hashes(self):
        s = self.runner
        files = tuple(s.UPSTREAM_CODE) + tuple(
            f"probes/response_datasets/{s.MODEL['upstream_key']}/{sub}_prompts_with_responses.json"
            for sub in s.SUBTYPES)
        return {p: s.sha256_file(os.path.join(self.args.upstream, p)) for p in files}

    def _load(self):
        s = self.runner
        self.GR, self.LP, self.PL = s.import_upstream(self.args.upstream)
        self.pool, self.texts, class_counts = s.load_pool(self.args.upstream)
        self.pool_per_class = {sub: min(c.values()) for sub, c in class_counts.items()}
        self.acts = s.Activations(self.args.raw_dir, self.GR, self.texts, self.pool, self.args.extract_batch)

    def extract(self):
        self._load()
        self.acts.ensure()
        print("activations ready:", self.acts.meta["true_layers"])

    def setup(self):
        s = self.runner
        self._load()
        for sub in s.SUBTYPES:
            if not os.path.exists(os.path.join(self.args.raw_dir, f"acts_{sub}.pt")):
                raise RuntimeError("sycophancy activations missing: run --phase extract on a GPU first")
        self.acts.ensure()
        self.finder = s.make_finder(self.acts, self.GR, self.LP, self.PL, self.texts, self.conf_raw,
                                    self.pool_per_class)
        self.provenance = {"upstream": s.UPSTREAM_REPO, "commit": s.UPSTREAM_COMMIT, "files": self.hashes(),
                           "model": s.MODEL["hf_id"], "extraction": self.acts.meta}

    def realise(self, c, null):
        s = self.runner
        data = resample(self.pool, c["resample"])
        if null:
            for d in data:
                d["shuffle_labels"] = True
        cfg = dict(s.BASE_CONFIG, epochs=c["epochs"], lr=c["lr"], weight_decay=c["weight_decay"],
                   batch_size=c["batch_size"], length_balance=c["length_balance"], layer=c["layer"])
        return data, c["probe_seed"], cfg


TARGETS = {t.name: t for t in (Harc, Reins, Faithfulness, Sycophancy)}


# --------------------------------------------------------------------- run

def space_dict(space):
    return {"axes": {k: list(v) for k, v in space.axes.items()},
            "weights": {k: space.axis_probabilities(k) for k in space.axis_names},
            "size": space.size}


def run_manifest(target, manifest, null, label):
    findings, rows = [], []
    for row in manifest:
        data, seed, cfg = target.realise(row["configuration"], null)
        t0 = time.time()
        finding = target.finder(data, seed, cfg)
        placeholder = bool(finding.meta.get("placeholder"))
        if not placeholder:
            if finding.claim not in target.claim_classes:
                raise RuntimeError(f"claim {finding.claim!r} is outside the registered classes")
            print(f"[{label} {row['draw_index']:3d}] {finding.claim} | score {finding.score:.4f} | "
                  f"size {finding.size} | {time.time() - t0:.0f}s", flush=True)
        findings.append(finding)
        rows.append({"group": label, "draw_index": row["draw_index"], "configuration": row["configuration"],
                     "seed": seed, "config": cfg, "claim": None if placeholder else finding.claim,
                     "score": None if placeholder else finding.score,
                     "size": None if placeholder else finding.size,
                     "meta": None if placeholder else finding.meta})
    return findings, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", choices=sorted(TARGETS))
    ap.add_argument("--upstream", required=True)
    ap.add_argument("--raw-dir", required=True, help="the diagnostic card's raw dir (features live there)")
    ap.add_argument("--out-dir", default=os.path.join(HERE, "cards", "confirmatory"))
    ap.add_argument("--phase", choices=["extract", "run", "dry-run"], default="run")
    ap.add_argument("--n-runs", type=int, default=N_RUNS)
    ap.add_argument("--smoke", action="store_true", help="4 real + 4 null draws, nothing written")
    ap.add_argument("--data-dir", default=None, help="HARC: circuit_breakers/, advbench/, xstest/, adapters/")
    ap.add_argument("--model-path", default=None, help="REINS: base model snapshot")
    ap.add_argument("--sae-root", default=None, help="REINS: SAE bundle directory")
    ap.add_argument("--templates", default=None, help="REINS extract: comma-separated renderings")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=8, help="faithfulness extraction batch")
    ap.add_argument("--extract-batch", type=int, default=25, help="sycophancy extraction batch")
    args = ap.parse_args()

    cls = TARGETS[args.target]
    n_runs = 4 if args.smoke else args.n_runs
    manifest = cls.space.sample_manifest(n_runs=n_runs, seed=MANIFEST_SEED)
    null_manifest = cls.space.sample_manifest(n_runs=n_runs, seed=NULL_MANIFEST_SEED)
    if args.phase == "dry-run":
        from collections import Counter
        print(json.dumps({"target": cls.name, "space": space_dict(cls.space),
                          "claim_classes": list(cls.claim_classes)}, indent=1, default=str)[:3000])
        for axis in cls.space.axis_names:
            print(axis, dict(Counter(str(r["configuration"][axis]) for r in manifest)))
        return

    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    target = cls(args)
    if args.phase == "extract":
        target.extract()
        return

    target.setup()
    real, real_rows = run_manifest(target, manifest, False, "real")
    null, null_rows = run_manifest(target, null_manifest, True, "null")
    shard = getattr(target.finder, "shard", None)
    if shard is not None and shard.is_worker:
        done = sum(1 for f in real + null if not f.meta.get("placeholder"))
        print(f"shard {shard.index}/{shard.count} done: {done} of {len(real) + len(null)} runs computed here")
        return

    result = confirmatory_from_findings(
        real, manifest, null_findings=null, null_manifest=null_manifest,
        claim_statement=target.claim_statement, thresholds=dict(THRESHOLDS),
        threshold_justifications=dict(JUSTIFICATIONS), claim_classes=list(target.claim_classes),
        confidence_level=CONFIDENCE, minimum_runs=MINIMUM_RUNS, seed=MASTER_SEED,
        model=target.model, task=target.task, method=target.method,
        claim_id=f"{target.stem}_confirmatory_v1",
        provenance={"driver": "references/run_confirmatory.py", "specification_space": space_dict(target.space),
                    "manifest_seed": MANIFEST_SEED, "null_manifest_seed": NULL_MANIFEST_SEED,
                    "resample_rule": "value 0 keeps the paper's rows; value k>0 draws the records with "
                                     "replacement under random.Random(10000 + k)",
                    **target.provenance})
    print()
    print(result.to_markdown())
    print(json.dumps({k: v for k, v in result.metrics.items()}, indent=1, default=str))
    if args.smoke:
        print("smoke complete; nothing written")
        return
    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.join(args.out_dir, target.stem + ".confirmatory")
    result.card.save(stem + ".json")
    with open(stem + ".md", "w") as f:
        f.write(result.to_markdown() + "\n\n")
        f.write("Frozen specification space (product distribution, drawn IID):\n\n")
        for axis in target.space.axis_names:
            probs = target.space.axis_probabilities(axis)
            f.write(f"- {axis}: " + ", ".join(f"{v} ({p:.3f})" for v, p in zip(target.space.axes[axis], probs)) + "\n")
        f.write(f"\nSeeds: real manifest {MANIFEST_SEED}, null manifest {NULL_MANIFEST_SEED}, "
                f"pairing master {MASTER_SEED}. Runs: {len(real)} real, {len(null)} null.\n")
    with open(stem + ".runs.json", "w") as f:
        json.dump({"target": target.name, "stem": target.stem, "space": space_dict(target.space),
                   "manifest_seed": MANIFEST_SEED, "null_manifest_seed": NULL_MANIFEST_SEED,
                   "master_seed": MASTER_SEED, "thresholds": THRESHOLDS,
                   "provenance": target.provenance, "runs": real_rows + null_rows},
                  f, indent=1, default=str, ensure_ascii=False)
        f.write("\n")
    digest = hashlib.sha256(open(stem + ".json", "rb").read()).hexdigest()[:12]
    print(f"\nconfirmatory card written to {stem}.json (sha256 {digest}); verdict {result.state}")


if __name__ == "__main__":
    main()
