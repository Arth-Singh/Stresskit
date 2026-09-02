"""Reference Stability Card: Sparse Weight Decomposition on GPT-2 small,
layer-8 mlp.c_proj (arXiv:2608.03913, veri-safe/SWD).

Claims under test (abstract, byte-exact): "Across single-matrix replacements,
SWD matches the held-out fidelity achieved by Transcoder and other strong
baselines while using less than 1% of the data that those baselines use to
train their replacements. For matched replacement fidelity, SWD reaches the
same circuit sufficiency and necessity targets with fewer active read/write
edges and selected units across tasks on GPT-2, Qwen2.5, and Qwen3.5-27B."
Only the GPT-2 single-matrix surface is audited (the paper's Section 3.3,
Figures 2-3, configs/gpt2_single_projection.yaml).

Upstream pipeline at the pinned commit: 16 FineWeb-Edu blocks of 1,024 tokens
(16,384 calibration tokens) give the input Gram of mlp.c_proj; the vendored
Double Sparse Factorization solver (40 outer iterations, 20 fixed-support
finalisation steps) writes W ~ A B with A [3072 x 768] and B [768 x 768] at
50% joint sparsity, so the 768 bottleneck coordinates are the circuit units.
Replacement fidelity is CE delta (replacement minus dense CE) on 2,048
held-out blocks, with forward KL and the projection's output cosine. Circuit
extraction per task family: units are ranked by positive first-order
task-margin attribution on circuit_train, the smallest top-k prefix whose
validation necessity beats the 95th percentile of 50 size-matched random
subsets is admissible, and held-out sufficiency and necessity are reported
for every prefix from that k on, with cost counted as selected units and as
active read/write edges. The released comparator is the Transcoder (12,288
features) checkpoint at 2,048,000 optimizer-replay tokens, whose replacement
CE delta (0.000979) is inside the paper's 0.001 matching band of SWD's
0.000889 at 16,384 tokens (0.8% of the data).

Finder = the three upstream stages (factorize_experiment,
evaluate_experiment, run_circuit_experiment) driven by a generated YAML
config, as a pure function of (data, seed, config):

- data: 16 calibration-slot records {"slot": i, "pool": name}. A slot names
  a position in a seeded draw from a pool of 4,096 pre-tokenised blocks; the
  base seed's draw is the pool order, so the base run uses the pool's first
  16 blocks as upstream does. Bootstrap resamples slots (a duplicated block
  counts twice in the Gram). The templates axis swaps the pool (Wikipedia;
  FineWeb without the educational filter).
- seed: the draw of calibration blocks and the circuit stage's random
  size-matched control subsets (upstream's config seed).
- config: sparsity, blocks (calibration blocks used; above 16 the draw is
  read past the records), outer_iterations, final_iterations, eval_set
  ("held-out" | "in-sample", the calibration blocks themselves),
  random_seeds (number of random control subsets), circuit_batch_size.

The battery's circuit stage runs IOI, docstring and gendered-pronoun. The
greater-than family (9,160 validation prompts x 50 random controls x 20
prefixes, more than an hour per run through the upstream loop) is run once,
on the base calibration blocks, as a reproduction check against the released
SWD and Transcoder frontiers; it is not part of the battery. This scoping was
fixed before any battery run completed.

Finding representation (fixed before any battery ran):

- components: the frontier cells "family:metric@target:axis" (3 families x
  {sufficiency, necessity_drop} x {0.8, 0.9, 0.95} x {units, edges};
  universe 36) on which SWD reaches the target at a strictly lower minimum
  cost than the released Transcoder-12k frontier (a target the Transcoder
  never reaches counts as infinite cost).
- claim: "CE delta <bucket>, <matched|not matched> to TC-12k; fewer than
  TC-12k on <all|most|some|none> contested cells", matching under the
  paper's own 0.001 band, buckets <=0.002 / 0.002-0.01 / >0.01 nats, and
  contested = cells reached by at least one method where the two costs
  differ (all: every contested cell; most: at least two thirds).
- score: the replacement CE delta in nats at the run's calibration budget.
- meta: dense and replacement CE, KL, output cosine and relative MSE, token
  exposure, per-family k_min, minimum cost at every target for this run and
  the released Transcoder, the run's config.

Battery: seeds (calibration draw and control subsets), bootstrap
(calibration blocks resampled), templates (Wikipedia and FineWeb pools),
hyperparams (sparsity 0.75; 1 and 64 calibration blocks; 8 outer
iterations; no finalisation; in-sample evaluation; 10 random controls), plus
a null control: calibration blocks of uniformly random token ids through the
same pipeline, so the Gram carries nothing about the model's input
distribution.

Usage (GPU, ~9 min per run on a shared H200 for GPT-2 small):
    python references/run_swd_card.py --upstream /path/to/SWD --data-dir /path/to/swd_data \
        --calibration-parquet 000_00000.parquet --evaluation-parquet 013_00000.parquet \
        --wiki-parquet wiki.parquet --fineweb-parquet fineweb.parquet \
        --out-dir references/cards --raw-dir references/cards/raw/swd_gpt2 \
        [--prepare-only | --reproduce-greater-than]
"""

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
import types

import numpy as np
import torch
import yaml

import stresskit as sk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from battery_shards import Shard  # noqa: E402

UPSTREAM_REPO = "veri-safe/SWD"
UPSTREAM_COMMIT = "4c44b7281bc7c78f80e431dac3aa75f397dd3043"
UPSTREAM_FILES = ("src/swd/pipeline.py", "src/swd/factorization.py", "src/swd/evaluation.py",
                  "src/swd/circuits.py", "src/swd/modules.py", "src/swd/data.py",
                  "src/swd/_vendor/doublesparse.py", "configs/gpt2_single_projection.yaml",
                  "paper_data/exp1_gpt2_single/replacement_quality.csv",
                  "paper_data/exp1_gpt2_single/reconstruction_quality.csv",
                  "paper_data/exp1_gpt2_single/circuit_frontiers.csv")
MODEL = "gpt2"
MODULE = "transformer.h.8.mlp.c_proj"
SEQ_LEN = 1024
N_BLOCKS = 16
POOL_BLOCKS = 4096
EVAL_BLOCKS = 2048
BASE_SEED = 20260606   # configs/gpt2_single_projection.yaml
FAMILIES = ("ioi", "docstring", "gendered_pronoun")
REPRODUCTION_FAMILY = "greater_than"
TOP_K = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 640, 768]
TARGETS = (0.8, 0.9, 0.95)
METRICS = {"sufficiency": "method_relative_sufficiency", "necessity_drop": "necessity_drop"}
AXES = {"bottleneck_units": "units", "effective_active_edges": "edges"}
RELEASED_SWD = "swd_s0p5"
COMPARATOR = "transcoder_12k"
COMPARATOR_TOKENS = 2048000
MATCH_BAND = 0.001
BASE_CONFIG = {"sparsity": 0.5, "blocks": N_BLOCKS, "outer_iterations": 40, "final_iterations": 20,
               "eval_set": "held-out", "random_seeds": 50, "circuit_batch_size": 64}
PLACEHOLDER = sk.Finding(claim="placeholder (other shard)", score=0.0, meta={"placeholder": True})


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def cell(family, metric, target, axis):
    return f"{family}:{metric}@{target}:{AXES[axis]}"


def all_cells(families):
    return [cell(f, m, t, a) for f in families for m in METRICS for t in TARGETS for a in AXES]


def released_costs(frontier_rows, method, families):
    """Minimum released cost per cell: the frontier CSV lists, for every
    reached threshold, the cheapest prefix reaching it."""
    out = {}
    for family in families:
        for metric, metric_name in METRICS.items():
            for axis in AXES:
                rows = [r for r in frontier_rows if r["method"] == method and r["family"] == family
                        and r["metric"] == metric_name and r["cost_axis"] == axis]
                for t in TARGETS:
                    reached = [float(r["min_cost"]) for r in rows if float(r["threshold"]) >= t]
                    out[cell(family, metric, t, axis)] = min(reached) if reached else math.inf
    return out


def run_costs(test_rows, families):
    out = {}
    for family in families:
        rows = [r for r in test_rows if r["task_family"] == family]
        for metric in METRICS:
            for axis in AXES:
                for t in TARGETS:
                    reached = [float(r[axis]) for r in rows
                               if not math.isnan(float(r[metric])) and float(r[metric]) >= t]
                    out[cell(family, metric, t, axis)] = min(reached) if reached else math.inf
    return out


def compare(costs, comparator, families):
    won, contested, ties = set(), [], []
    for c in all_cells(families):
        a, b = costs[c], comparator[c]
        if math.isinf(a) and math.isinf(b):
            continue
        if a == b:
            ties.append(c)
            continue
        contested.append(c)
        if a < b:
            won.add(c)
    return won, contested, ties


def claim_text(ce_delta, comparator_ce_delta, won, contested):
    bucket = "<=0.002" if ce_delta <= 0.002 else "0.002-0.01" if ce_delta <= 0.01 else ">0.01"
    matched = "matched" if abs(ce_delta - comparator_ce_delta) <= MATCH_BAND else "not matched"
    frac = len(won) / len(contested) if contested else 0.0
    level = ("all" if contested and frac == 1 else "most" if frac >= 2 / 3
             else "some" if frac > 0 else "none")
    return f"CE delta {bucket}, {matched} to TC-12k; fewer than TC-12k on {level} contested cells"


def finite(costs):
    return {c: (None if math.isinf(v) else v) for c, v in costs.items()}


def fmt_cost(value):
    return "none" if value is None else f"{value:.0f}"


def build_config(*, name, root, calib, n_blocks, sparsity, eval_path, eval_sequences, task_registry,
                 families, outer_iterations, final_iterations, random_seeds, batch_size, seed):
    return {
        "experiment": {"id": name, "paper_section": "Single Matrix Replacement"},
        "model": {"id": MODEL, "dtype": "float32", "modules": [MODULE], "local_files_only": True},
        "data": {"calibration_tokens": calib, "calibration_sequences": n_blocks,
                 "evaluation_tokens": eval_path, "task_registry": task_registry},
        "factorization": {"objective": "activation_gram", "sparsities": [sparsity],
                          "primary_sparsity": sparsity,
                          "calibration_sequences_by_sparsity": {str(sparsity): n_blocks},
                          "outer_iterations": int(outer_iterations),
                          "final_iterations": int(final_iterations)},
        "evaluation": {"sequences": eval_sequences, "reconstruction_modules": [MODULE]},
        "circuit": {"enabled": True, "families": list(families),
                    "split_roles": {"score": "circuit_train", "select": "circuit_val",
                                    "report": "circuit_test"},
                    "top_k": list(TOP_K), "random_seeds": int(random_seeds),
                    "batch_size": int(batch_size)},
        "runtime": {"device": "auto", "factor_batch_size": 4, "eval_batch_size": 2},
        "output": {"root": root},
        "seed": int(seed),
    }


def write_blocks(path, ids):
    torch.save({"input_ids": ids, "attention_mask": torch.ones_like(ids)}, path)


def prepare_data(up, data_dir, args):
    pools = {}
    for name, parquet in (("edu", args.calibration_parquet), ("wiki", args.wiki_parquet),
                          ("fineweb", args.fineweb_parquet)):
        path = os.path.join(data_dir, f"pool_{name}.pt")
        if not os.path.exists(path):
            print(f"preparing pool {name}: {POOL_BLOCKS} blocks from {parquet}", flush=True)
            up.prepare_lm_data(model_id=MODEL, input_pattern=parquet, output=path,
                               seq_len=SEQ_LEN, num_sequences=POOL_BLOCKS, seed=0)
        pools[name] = path
    eval_path = os.path.join(data_dir, "evaluation.pt")
    if not os.path.exists(eval_path):
        print(f"preparing evaluation blocks from {args.evaluation_parquet}", flush=True)
        up.prepare_lm_data(model_id=MODEL, input_pattern=args.evaluation_parquet, output=eval_path,
                           seq_len=SEQ_LEN, num_sequences=EVAL_BLOCKS, seed=1)
    return pools, eval_path


def reproduce_greater_than(up, pools, eval_path, task_registry, run_root, raw_dir, frontier):
    """Upstream circuit stage on greater-than for the base factorisation
    (first 16 pool blocks, base seed); one run, outside the battery."""
    t0 = time.time()
    root = os.path.join(run_root, "reproduction_greater_than")
    os.makedirs(root, exist_ok=True)
    ids = torch.load(pools["edu"], map_location="cpu", weights_only=False)["input_ids"][:N_BLOCKS]
    calib = os.path.join(root, "calibration.pt")
    write_blocks(calib, ids)
    config = build_config(name="swd_reproduction_greater_than", root=root, calib=calib,
                          n_blocks=N_BLOCKS, sparsity=BASE_CONFIG["sparsity"], eval_path=eval_path,
                          eval_sequences=EVAL_BLOCKS, task_registry=task_registry,
                          families=(REPRODUCTION_FAMILY,),
                          outer_iterations=BASE_CONFIG["outer_iterations"],
                          final_iterations=BASE_CONFIG["final_iterations"],
                          random_seeds=BASE_CONFIG["random_seeds"],
                          batch_size=BASE_CONFIG["circuit_batch_size"], seed=BASE_SEED)
    cfg_path = os.path.join(root, "config.yaml")
    with open(cfg_path, "w") as f:
        yaml.safe_dump(config, f)
    summary = up.factorize_experiment(cfg_path)
    artifact = summary["outputs"][0]["artifact"]
    circuit = up.run_circuit_experiment(cfg_path, artifact_path=artifact)
    rows = up.read_csv(os.path.join(root, "circuits", circuit["artifact_label"], "test_circuit_rows.csv"))
    families = (REPRODUCTION_FAMILY,)
    costs = run_costs(rows, families)
    payload = {"family": REPRODUCTION_FAMILY, "seed": BASE_SEED, "n_blocks": N_BLOCKS,
               "k_min": next((int(r["k_min"]) for r in rows), None), "costs": finite(costs),
               "released_swd": finite(released_costs(frontier, RELEASED_SWD, families)),
               "comparator": finite(released_costs(frontier, COMPARATOR, families)),
               "test_rows": rows, "wall_secs": round(time.time() - t0, 1)}
    with open(os.path.join(raw_dir, "reproduction_greater_than.json"), "w") as f:
        json.dump(payload, f, indent=1, default=str)
    print(f"greater-than reproduction written ({payload['wall_secs']}s): "
          f"{ {c: v for c, v in payload['costs'].items() if 'units' in c} }", flush=True)


def make_finder(up, pools, eval_path, task_registry, run_root, raw_dir, comparator_costs,
                comparator_ce_delta, vocab_size):
    shard = Shard(os.path.join(raw_dir, "shard_cache"))
    loaded = {}

    def pool_ids(name):
        if name not in loaded:
            loaded[name] = torch.load(pools[name], map_location="cpu", weights_only=False)["input_ids"]
        return loaded[name]

    def blocks_for(data, seed, cfg):
        pool = data[0]["pool"]
        slots = [int(d["slot"]) for d in data]
        n = int(cfg["blocks"])
        if n < len(slots):
            slots = slots[:n]
        elif n > len(slots):
            slots = slots + list(range(len(slots), n))
        if pool == "random":
            rows = [torch.randint(vocab_size, (SEQ_LEN,),
                                  generator=torch.Generator().manual_seed((int(seed) * 1_000_003 + s) % 2**31))
                    for s in slots]
            return torch.stack(rows), slots
        ids = pool_ids(pool)
        draw = (np.arange(len(ids)) if int(seed) == BASE_SEED
                else np.random.default_rng(int(seed)).permutation(len(ids)))
        return ids[[int(draw[s]) for s in slots]], slots

    def compute(data, seed, cfg):
        t0 = time.time()
        key = shard.key(data, seed, cfg)
        root = os.path.join(run_root, key)
        os.makedirs(root, exist_ok=True)
        ids, slots = blocks_for(data, seed, cfg)
        calib = os.path.join(root, "calibration.pt")
        write_blocks(calib, ids)
        in_sample = cfg["eval_set"] == "in-sample"
        config = build_config(name=f"swd_{key}", root=root, calib=calib, n_blocks=len(slots),
                              sparsity=float(cfg["sparsity"]),
                              eval_path=calib if in_sample else eval_path,
                              eval_sequences=len(slots) if in_sample else EVAL_BLOCKS,
                              task_registry=task_registry, families=FAMILIES,
                              outer_iterations=cfg["outer_iterations"],
                              final_iterations=cfg["final_iterations"],
                              random_seeds=cfg["random_seeds"],
                              batch_size=cfg["circuit_batch_size"], seed=seed)
        cfg_path = os.path.join(root, "config.yaml")
        with open(cfg_path, "w") as f:
            yaml.safe_dump(config, f)
        summary = up.factorize_experiment(cfg_path)
        artifact = summary["outputs"][0]["artifact"]
        metrics = up.evaluate_experiment(cfg_path, artifact_path=artifact)
        t1 = time.time()
        circuit = up.run_circuit_experiment(cfg_path, artifact_path=artifact)
        label = circuit["artifact_label"]
        test_rows = up.read_csv(os.path.join(root, "circuits", label, "test_circuit_rows.csv"))
        selection = up.read_csv(os.path.join(root, "circuits", label, "validation_selection.csv"))
        torch.cuda.empty_cache()

        costs = run_costs(test_rows, FAMILIES)
        won, contested, ties = compare(costs, comparator_costs, FAMILIES)
        ce_delta = float(metrics["ce_delta"])
        recon = metrics["activation_reconstruction"][0]
        k_min = {family: next((int(r["k_min"]) for r in test_rows if r["task_family"] == family), None)
                 for family in FAMILIES}
        claim = claim_text(ce_delta, comparator_ce_delta, won, contested)
        meta = {"config": cfg, "pool": data[0]["pool"], "slots": slots,
                "n_unique_blocks": len(set(slots)), "token_exposure": int(metrics["token_exposure"]),
                "dense_ce": float(metrics["dense_ce"]), "replacement_ce": float(metrics["replacement_ce"]),
                "ce_delta": ce_delta, "kl": float(metrics["kl_dense_to_replacement"]),
                "top_token_agreement": float(metrics["top_token_agreement"]),
                "cosine_mean": float(recon["cosine_mean"]), "relative_mse": float(recon["relative_mse"]),
                "k_min": k_min, "costs": finite(costs), "won": sorted(won), "contested": sorted(contested),
                "ties": sorted(ties), "factorize_eval_secs": round(t1 - t0, 1),
                "wall_secs": round(time.time() - t0, 1)}
        gp = cell("gendered_pronoun", "sufficiency", 0.95, "bottleneck_units")
        print(f"  [{data[0]['pool']} s={cfg['sparsity']} blocks={len(slots)} it={cfg['outer_iterations']}/"
              f"{cfg['final_iterations']} {cfg['eval_set']} seed={seed}] CE delta {ce_delta:.5f} "
              f"KL {meta['kl']:.5f} cos {meta['cosine_mean']:.4f} won {len(won)}/{len(contested)} "
              f"gp units suff@0.95 {fmt_cost(meta['costs'][gp])} ({meta['wall_secs']}s)", flush=True)
        with open(os.path.join(raw_dir, f"run_{key}.json"), "w") as f:
            json.dump({**meta, "test_rows": test_rows, "validation_selection": selection,
                       "factorization": summary}, f, indent=1, default=str)
        return sk.Finding(components=won, universe_size=len(costs), claim=claim, score=ce_delta,
                          meta=meta)

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
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--calibration-parquet", required=True)
    ap.add_argument("--evaluation-parquet", required=True)
    ap.add_argument("--wiki-parquet", required=True)
    ap.add_argument("--fineweb-parquet", required=True)
    ap.add_argument("--task-registry", default=None)
    ap.add_argument("--run-root", default=None)
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "cards"))
    ap.add_argument("--raw-dir", default=None)
    ap.add_argument("--n-runs", type=int, default=10)
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--reproduce-greater-than", action="store_true")
    args = ap.parse_args()

    # upstream joins relative data paths onto the config's directory, so every path is absolute
    args.upstream, args.data_dir, args.out_dir = (os.path.abspath(p) for p in
                                                  (args.upstream, args.data_dir, args.out_dir))
    raw_dir = os.path.abspath(args.raw_dir or os.path.join(args.out_dir, "raw", "swd_gpt2"))
    run_root = os.path.abspath(args.run_root or os.path.join(args.data_dir, "runs"))
    task_registry = os.path.abspath(
        args.task_registry or os.path.join(args.data_dir, "tasks", "gpt2", "task_examples.jsonl"))
    for d in (raw_dir, run_root, args.out_dir, args.data_dir):
        os.makedirs(d, exist_ok=True)
    hashes = {p: sha256_file(os.path.join(args.upstream, p)) for p in UPSTREAM_FILES}

    sys.path.insert(0, os.path.join(args.upstream, "src"))
    from swd.circuits import run_circuit_experiment  # noqa: E402
    from swd.data import prepare_lm_data  # noqa: E402
    from swd.evaluation import evaluate_experiment  # noqa: E402
    from swd.io import read_csv  # noqa: E402
    from swd.pipeline import factorize_experiment  # noqa: E402
    from transformers import AutoTokenizer  # noqa: E402

    up = types.SimpleNamespace(
        factorize_experiment=factorize_experiment, evaluate_experiment=evaluate_experiment,
        run_circuit_experiment=run_circuit_experiment, prepare_lm_data=prepare_lm_data,
        read_csv=read_csv)

    pools, eval_path = prepare_data(up, args.data_dir, args)
    if not os.path.exists(task_registry):
        raise FileNotFoundError(f"task registry missing: {task_registry} (scripts/prepare_task_data.py)")
    if args.prepare_only:
        print("data prepared")
        return

    base = os.path.join(args.upstream, "paper_data", "exp1_gpt2_single")
    with open(os.path.join(base, "replacement_quality.csv")) as f:
        replacement = list(csv.DictReader(f))
    with open(os.path.join(base, "reconstruction_quality.csv")) as f:
        reconstruction = list(csv.DictReader(f))
    with open(os.path.join(base, "circuit_frontiers.csv")) as f:
        frontier = list(csv.DictReader(f))
    if args.reproduce_greater_than:
        reproduce_greater_than(up, pools, eval_path, task_registry, run_root, raw_dir, frontier)
        return

    vocab_size = len(AutoTokenizer.from_pretrained(MODEL))
    comparator_row = next(r for r in replacement
                          if r["method"] == COMPARATOR and int(r["tokens"]) == COMPARATOR_TOKENS)
    comparator_ce_delta = float(comparator_row["ce_delta"])
    released_row = next(r for r in replacement
                        if r["method"] == RELEASED_SWD and int(r["tokens"]) == N_BLOCKS * SEQ_LEN)
    released_cosine = next(float(r["value"]) for r in reconstruction
                           if r["method"] == RELEASED_SWD and float(r["tokens"]) == N_BLOCKS * SEQ_LEN
                           and r["metric"] == "cosine_mean")
    comparator_costs = released_costs(frontier, COMPARATOR, FAMILIES)
    released_swd_costs = released_costs(frontier, RELEASED_SWD, FAMILIES)
    released_won, released_contested, _ = compare(released_swd_costs, comparator_costs, FAMILIES)

    def records(pool):
        return [{"slot": i, "pool": pool} for i in range(N_BLOCKS)]

    finder = make_finder(up, pools, eval_path, task_registry, run_root, raw_dir, comparator_costs,
                         comparator_ce_delta, vocab_size)
    result = sk.stress(
        finder, records("edu"),
        battery=["seeds", "bootstrap", "templates", "hyperparams"],
        n_runs=args.n_runs, seed=BASE_SEED,
        config=dict(BASE_CONFIG),
        templates={"wikipedia": records("wiki"), "fineweb": records("fineweb")},
        hyperparams={"sparsity": [0.75], "blocks": [1, 64], "outer_iterations": [8],
                     "final_iterations": [0], "eval_set": ["in-sample"], "random_seeds": [10]},
        null_data=records("random"),
        claim_statement=(
            "Across single-matrix replacements, SWD matches the held-out fidelity achieved by "
            "Transcoder and other strong baselines while using less than 1% of the data that those "
            "baselines use to train their replacements. For matched replacement fidelity, SWD "
            "reaches the same circuit sufficiency and necessity targets with fewer active read/write "
            "edges and selected units across tasks on GPT-2, Qwen2.5, and Qwen3.5-27B"),
        model=MODEL,
        task="GPT-2 small layer-8 mlp.c_proj replaced by a 50%-sparse two-factor decomposition from "
             "16,384 FineWeb-Edu calibration tokens; held-out CE delta on 2,048 blocks and "
             "sufficiency/necessity circuit frontiers on IOI, docstring and gendered-pronoun against "
             "the released Transcoder-12k frontier (greater-than reproduced once, outside the battery)",
        method="upstream factorize/evaluate/circuit stages (vendored Double Sparse Factorization, "
               "task-margin attribution ranking, validation-selected prefix, mean ablation) at the "
               "pinned commit",
        verbose=True,
    )
    if finder.shard.is_worker:
        print(f"shard {finder.shard.index}/{finder.shard.count} done; no artifacts written")
        return

    b = result.base.meta
    base_won = set(result.base.components)
    result.card.notes.append(
        f"upstream: {UPSTREAM_REPO}@{UPSTREAM_COMMIT[:7]} (Apache-2.0); the factorize, evaluate and "
        "circuit stages, the vendored solver and the released tables imported unmodified; file hashes "
        + ", ".join(f"{os.path.basename(p)} {h[:12]}" for p, h in hashes.items()))
    same_eval = abs(float(released_row["dense_ce"]) - b["dense_ce"]) < 1e-4
    result.card.notes.append(
        f"reproduction (released SWD s=0.5 at {N_BLOCKS * SEQ_LEN} calibration tokens -> base run): "
        f"CE delta {float(released_row['ce_delta']):.6f} -> {b['ce_delta']:.6f}; KL "
        f"{float(released_row['kl']):.6f} -> {b['kl']:.6f}; output cosine {released_cosine:.4f} -> "
        f"{b['cosine_mean']:.4f}; dense CE {float(released_row['dense_ce']):.6f} -> {b['dense_ce']:.6f} "
        f"({'the same 2,048 evaluation blocks' if same_eval else 'different evaluation blocks'}). "
        "The calibration blocks differ: upstream drew them from FineWeb-Edu sample-10BT files 000-012 "
        "under a seeded streaming shuffle, this card from file 000 alone (pool of 4,096 blocks, "
        "seed 0), so the base run is the same protocol on a different 16-block sample, not the "
        "released checkpoint")
    union = base_won | released_won
    result.card.notes.append(
        f"comparator: released Transcoder-12k checkpoint at {COMPARATOR_TOKENS:,} optimizer-replay "
        f"tokens, CE delta {comparator_ce_delta:.6f} (the paper's CE-matched partner of SWD s=0.5 at "
        f"{N_BLOCKS * SEQ_LEN:,} tokens, {100 * N_BLOCKS * SEQ_LEN / COMPARATOR_TOKENS:.1f}% of its "
        "data); its frontier is read from paper_data/exp1_gpt2_single/circuit_frontiers.csv and is "
        "not retrained. On the three battery families the released SWD run beats it on "
        f"{len(released_won)}/{len(released_contested)} contested cells; the base run on "
        f"{len(base_won)}/{len(b['contested'])}, Jaccard to the released won set "
        f"{(len(base_won & released_won) / len(union)) if union else 1.0:.2f}")
    released_swd_finite = finite(released_swd_costs)
    comparator_finite = finite(comparator_costs)
    for family in FAMILIES:
        parts = []
        for metric in METRICS:
            for t in TARGETS:
                c = cell(family, metric, t, "bottleneck_units")
                e = cell(family, metric, t, "effective_active_edges")
                parts.append(f"{metric}@{t}: units {fmt_cost(b['costs'][c])} (released SWD "
                             f"{fmt_cost(released_swd_finite[c])}, TC {fmt_cost(comparator_finite[c])}), "
                             f"edges {fmt_cost(b['costs'][e])} (released SWD "
                             f"{fmt_cost(released_swd_finite[e])}, TC {fmt_cost(comparator_finite[e])})")
        result.card.notes.append(f"base run {family} (k_min {b['k_min'][family]}): " + "; ".join(parts))
    rep_path = os.path.join(raw_dir, "reproduction_greater_than.json")
    if os.path.exists(rep_path):
        with open(rep_path) as f:
            rep = json.load(f)
        parts = []
        for metric in METRICS:
            for t in TARGETS:
                c = cell(REPRODUCTION_FAMILY, metric, t, "bottleneck_units")
                e = cell(REPRODUCTION_FAMILY, metric, t, "effective_active_edges")
                parts.append(f"{metric}@{t}: units {fmt_cost(rep['costs'][c])} (released SWD "
                             f"{fmt_cost(rep['released_swd'][c])}, TC {fmt_cost(rep['comparator'][c])}), "
                             f"edges {fmt_cost(rep['costs'][e])} (released SWD "
                             f"{fmt_cost(rep['released_swd'][e])}, TC {fmt_cost(rep['comparator'][e])})")
        result.card.notes.append(
            f"greater-than reproduction (base calibration blocks, upstream protocol with 50 random "
            f"controls, one run outside the battery, k_min {rep['k_min']}, {rep['wall_secs']:.0f}s): "
            + "; ".join(parts))
    else:
        result.card.notes.append("greater-than reproduction not available: --reproduce-greater-than "
                                 "did not finish before assembly")
    for record in result.runs:
        if record.axis in ("templates", "hyperparams"):
            m = record.finding.meta
            g = cell("gendered_pronoun", "sufficiency", 0.95, "bottleneck_units")
            ge = cell("gendered_pronoun", "sufficiency", 0.95, "effective_active_edges")
            result.card.notes.append(
                f"{record.variant}: CE delta {m['ce_delta']:.5f}, KL {m['kl']:.5f}, cosine "
                f"{m['cosine_mean']:.4f}, token exposure {m['token_exposure']:,}; won "
                f"{len(m['won'])}/{len(m['contested'])} contested cells; gendered-pronoun "
                f"sufficiency@0.95 units {fmt_cost(m['costs'][g])} edges {fmt_cost(m['costs'][ge])}; "
                f"k_min {m['k_min']}")
    result.card.notes.append(
        "the battery's circuit stage runs IOI, docstring and gendered-pronoun; greater-than (9,160 "
        "validation prompts x 50 random controls x 20 prefixes through the upstream per-prompt loop, "
        "over an hour per run) is reproduced once on the base blocks and kept out of the battery. "
        "Circuit batch size 64 instead of upstream's 16 and evaluation batch size 2 instead of 4 "
        "(runtime settings for a shared GPU; the margin is read at each prompt's last real token and "
        "CE/KL are token sums, so batching changes nothing but float summation order)")
    result.card.notes.append(
        "null control: calibration blocks of uniformly random GPT-2 token ids (seeded per slot); "
        "the factorisation, evaluation and circuit stages are unchanged, only the Gram no longer "
        "reflects the model's input distribution")

    print()
    print(result)
    print(result.to_markdown())
    stem = os.path.join(args.out_dir, "swd_gpt2")
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
                   "comparator": {"method": COMPARATOR, "tokens": COMPARATOR_TOKENS,
                                  "ce_delta": comparator_ce_delta, "costs": comparator_finite},
                   "released_swd": {"row": released_row, "cosine": released_cosine,
                                    "costs": released_swd_finite},
                   "raw_dir": os.path.relpath(raw_dir, args.out_dir), "runs": rows},
                  f, indent=1, default=str)
        f.write("\n")
    print(sk.verdict_trace_markdown(trace))
    print(f"\nartifacts written to {stem}.*")


if __name__ == "__main__":
    main()
