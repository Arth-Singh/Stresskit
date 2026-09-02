"""Reference Stability Card: CoAx backup-head recovery on the GPT-2-small IOI
circuit (arXiv:2607.01940, GongZhiren/Conditional-Co-Ablation).

Claim under test (abstract, byte-exact): "On the GPT-2-small IOI circuit,
CoAx raises backup-head recovery from 0.33 to 0.91 ROC-AUC, outperforming
all baselines, including self-repair-aware gradient scores (best 0.82)".
The released repository's own results/reference_metrics.json reports the
same experiment at 0.941 +/- 0.004 with the single-ablation baseline at
0.603, so the "from 0.33" is version-dependent; both are recorded.

Upstream pipeline (experiments/paper/backup_recovery_full.py at the pinned
commit): 96 IOI prompts from one template ("When {IO} and {S} went to the
{place}, {S} gave a {obj} to", 16 names, 8 places, 8 objects, numpy seed);
CoAblation measures, for every head u, the Fisher energy of ablating u on
the intact model (single) and after the three documented name-mover heads
are ablated (conditional); the compensation score is their difference.
Candidates are the 141 heads that are not name movers; positives are the 8
documented backup name movers of Wang et al. (2022); the metric is ROC-AUC
of the compensation score against those labels. AtP, EAP-IG, AtP* GradDrop
and a conditional attribution baseline are scored on the same prompts.

Finder = that pipeline with CoAblation, the baselines and the circuit
labels imported unmodified, as a pure function of (data, seed, config):

- data: prompt records (template string, third-name flag, index into the
  seeded draw of 96 prompts). Bootstrap resamples records; the templates
  axis swaps the template string.
- seed: the prompt-generation seed of the upstream generator. The seeds
  axis redraws the 96 prompts from the same template; the scoring is
  deterministic given the prompts.
- config: position_mode ("last" | "all"), top_r (0 = full vocabulary |
  192, the arXiv setting), ablation_mode ("zero" | "mean"), feature_mode
  ("fisher_centered" | "l2_uncentered"), freeze_ln ("none" | "full"),
  primaries ("documented" | "atp-top3" | "energy-top3").

Finding representation (fixed before any battery ran):

- components: the eight highest-scoring candidate heads, "L{layer}H{head}".
  Universe = 141 candidates. Eight is the size of the documented backup set
  and the cut-off upstream reports precision at.
- claim: "CoAx AUC <bucket>; <beats|does not beat> AtP*" with bucket in
  {">=0.90", "0.80-0.90", "<0.80"}; the abstract's 0.91 and 0.82 sit in the
  first bucket with "beats".
- score: the CoAx ROC-AUC.
- meta: AUC of single-ablation, AtP, EAP-IG, AtP*, conditional attribution;
  average precision and precision@8 of CoAx; the primary set actually used;
  the top-8 list; the config.

Battery: seeds (the 96 prompts redrawn), bootstrap (prompts resampled), templates
(three other IOI templates with the same names, places, objects and seed;
upstream hard-codes one), hyperparams (energy over all positions; mean
ablation; uncentred L2 feature; frozen LayerNorm; top-192 logits, the
setting the arXiv version reports; and two label-free primary sets: the
three heads with the largest AtP attribution, and the three heads with the
largest single-ablation energy, since the abstract calls the score
label-free but the headline conditions on the documented name movers), plus
a null control: prompts in which a third name is the giver, so no name is
repeated and the indirect-object task is not posed; labels unchanged.

Usage (GPU, ~90 s per run on an H200 for GPT-2 small):
    python references/run_coax_backup_card.py --upstream /path/to/Conditional-Co-Ablation \
        --out-dir references/cards --raw-dir references/cards/raw/coax_backup_gpt2
"""

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

import stresskit as sk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from battery_shards import Shard  # noqa: E402

UPSTREAM_REPO = "GongZhiren/Conditional-Co-Ablation"
UPSTREAM_COMMIT = "1c04682d705926cf11d23729ee3bf25c19071dd9"
UPSTREAM_FILES = ("src/curvgraph/coablation.py", "src/curvgraph/circuits.py",
                  "src/curvgraph/baselines.py", "experiments/paper/backup_recovery_full.py")
MODEL_KEY = "gpt2-small"
N_PROMPTS = 96
BASE_SEED = 1          # first of the Makefile's seeds {1, 15, 22, 8}
TOP_K = 8
ARXIV = {"coax": 0.91, "single": 0.33, "atp_star": 0.82}

TEMPLATES = {
    "upstream": "When {IO} and {S} went to the {place}, {S} gave a {obj} to",
    "afterwards": "Afterwards, {IO} and {S} went to the {place}. {S} gave a {obj} to",
    "friends": "Friends {IO} and {S} found a {obj} at the {place}. {S} gave it to",
    "argument": "Then, {IO} and {S} had a long argument, and afterwards {S} said to",
}
NULL_TEMPLATE = "When {IO} and {S} went to the {place}, {X} gave a {obj} to"
BASE_CONFIG = {"position_mode": "last", "top_r": 0, "ablation_mode": "zero",
               "feature_mode": "fisher_centered", "freeze_ln": "none",
               "primaries": "documented"}
PLACEHOLDER = sk.Finding(claim="placeholder (other shard)", score=0.0, meta={"placeholder": True})


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def make_prompts(C, template, n, seed, third_name=False):
    """Upstream ioi_prompts with the template as a parameter; the generator
    is consumed in the same order, so the upstream template yields the
    upstream prompts (checked at startup). third_name=True draws a third
    name for the {X} slot after the upstream draws."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        io, s = rng.choice(C._IOI_NAMES, size=2, replace=False)
        place = rng.choice(C._IOI_PLACES)
        obj = rng.choice(C._IOI_OBJECTS)
        x = None
        if third_name:
            x = rng.choice([nm for nm in C._IOI_NAMES if nm not in (io, s)])
        out.append({"prompt": template.format(IO=io, S=s, place=place, obj=obj, X=x),
                    "io": " " + io, "s": " " + s})
    return out


def make_records(template, n, third_name=False):
    return [{"template": template, "third_name": third_name, "index": i} for i in range(n)]


def realise(C, data, seed):
    """Prompt records -> upstream prompt dicts: draw the template's prompts
    with this seed and pick the indices the records name."""
    drawn = make_prompts(C, data[0]["template"], N_PROMPTS, seed, data[0]["third_name"])
    return [drawn[d["index"]] for d in data]


def make_finder(bundle, C, B, CoAblation, raw_dir):
    import torch

    dev = next(bundle.model.parameters()).device
    nH, nU = bundle.num_heads, bundle.num_layers * bundle.num_heads
    documented = C.IOI_CIRCUIT["name_mover"]
    backup = {C.head_index(layer, h, nH) for (layer, h) in C.IOI_CIRCUIT["backup_name_mover"]}
    vocab = int(bundle.tokenizer.vocab_size)

    def name(u):
        layer, head = C.head_layer_head(u, nH)
        return f"L{layer}H{head}"

    def compute(data, seed, cfg):
        t0 = time.time()
        prompts = realise(C, data, seed)
        seqs = [bundle.tokenizer(e["prompt"], return_tensors="pt").to(dev)["input_ids"]
                for e in prompts]
        seqs = [s for s in seqs if s.shape[1] >= 4]
        co = CoAblation(bundle, seqs, top_r=cfg["top_r"] or vocab,
                        ablation_mode=cfg["ablation_mode"], feature_mode=cfg["feature_mode"],
                        freeze_ln=cfg["freeze_ln"], position_mode=cfg["position_mode"])
        atp = np.nan_to_num(np.asarray(B.head_attribution_patching(bundle, prompts), dtype=float))
        if cfg["primaries"] == "documented":
            prim_h = list(documented)
        elif cfg["primaries"] == "atp-top3":
            prim_h = [C.head_layer_head(int(u), nH) for u in np.argsort(-np.abs(atp))[:3]]
        elif cfg["primaries"] == "energy-top3":
            energy = np.nan_to_num(np.asarray(co.single_energy(), dtype=float))
            prim_h = [C.head_layer_head(int(u), nH) for u in np.argsort(-energy)[:3]]
        else:
            raise ValueError(cfg["primaries"])
        prim = {C.head_index(layer, h, nH) for (layer, h) in prim_h}
        cand = [u for u in range(nU) if u not in prim]
        y = np.array([1 if u in backup else 0 for u in cand])

        r = co.conditional_compensation(prim_h, head_set=list(range(nU)))
        eap = np.asarray(B.integrated_gradient_attribution(bundle, prompts), dtype=float)
        aps = np.asarray(B.head_attribution_graddrop(bundle, prompts), dtype=float)
        gim = np.asarray(B.conditional_attribution_patching(bundle, prompts, prim_h), dtype=float)
        vectors = {"coax": r["compensation"], "single": r["single"],
                   "conditional": r["conditional"], "atp": atp, "eap_ig": eap,
                   "atp_star": aps, "conditional_grad": gim}

        def auc(vec):
            v = np.nan_to_num(np.asarray([vec[u] for u in cand], dtype=float), nan=0.0)
            return float(roc_auc_score(y, v)) if 0 < y.sum() < len(y) else float("nan")

        aucs = {k: auc(v) for k, v in vectors.items()}
        comp = np.nan_to_num(np.asarray([r["compensation"][u] for u in cand], dtype=float))
        order = np.argsort(-comp)
        top = [cand[i] for i in order[:TOP_K]]
        bucket = (">=0.90" if aucs["coax"] >= 0.90 else
                  "0.80-0.90" if aucs["coax"] >= 0.80 else "<0.80")
        beats = "beats AtP*" if aucs["coax"] > aucs["atp_star"] else "does not beat AtP*"
        torch.cuda.empty_cache()
        meta = {"config": cfg, "n_prompts": len(seqs), "prompt_seed": seed,
                "n_unique_prompts": len({e["prompt"] for e in prompts}),
                "template": data[0]["template"], "first_prompt": prompts[0]["prompt"],
                "primaries": [list(map(int, p)) for p in prim_h],
                "primaries_documented": [list(p) in [list(d) for d in documented] for p in prim_h],
                "auc": aucs,
                "average_precision": float(average_precision_score(y, comp)),
                "precision_at_8": float(y[order[:TOP_K]].mean()),
                "top8": [name(u) for u in top], "wall_secs": round(time.time() - t0, 1)}
        print(f"  [{cfg['primaries']} {cfg['position_mode']} top_r={cfg['top_r']} "
              f"{cfg['ablation_mode']}/{cfg['feature_mode']}/{cfg['freeze_ln']} seed={seed}] "
              f"CoAx {aucs['coax']:.3f} single {aucs['single']:.3f} AtP* {aucs['atp_star']:.3f} "
              f"EAP-IG {aucs['eap_ig']:.3f} p@8 {meta['precision_at_8']:.2f} "
              f"({meta['wall_secs']}s)", flush=True)
        digest = hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()[:12]
        with open(os.path.join(raw_dir, f"run_{digest}.json"), "w") as f:
            json.dump({**meta, "scores": {k: [float(x) for x in np.nan_to_num(v)]
                                          for k, v in vectors.items()}}, f, indent=1)
        return sk.Finding(components={name(u) for u in top}, universe_size=len(cand),
                          claim=f"CoAx AUC {bucket}; {beats}", score=aucs["coax"], meta=meta)

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
    ap.add_argument("--n-runs", type=int, default=12)
    args = ap.parse_args()

    raw_dir = args.raw_dir or os.path.join(args.out_dir, "raw", "coax_backup_gpt2")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    hashes = {p: sha256_file(os.path.join(args.upstream, p)) for p in UPSTREAM_FILES}
    reference_path = os.path.join(args.upstream, "results", "reference_metrics.json")
    reference = json.load(open(reference_path)) if os.path.exists(reference_path) else None

    sys.path.insert(0, os.path.join(args.upstream, "src"))
    import curvgraph  # noqa: E402,F401
    from curvgraph import baselines as B  # noqa: E402
    from curvgraph import circuits as C  # noqa: E402
    from curvgraph._core.config import load_config  # noqa: E402
    from curvgraph._core.model import load_model_bundle  # noqa: E402
    from curvgraph.coablation import CoAblation  # noqa: E402

    cfg = load_config(os.path.join(args.upstream, "configs", "default.yaml"))
    bundle = load_model_bundle(cfg["model"]["models"][MODEL_KEY], cfg["model"].get("tokenizer", {}))
    print(f"{MODEL_KEY}: {bundle.num_layers} layers x {bundle.num_heads} heads")

    data = make_records(TEMPLATES["upstream"], N_PROMPTS)
    if realise(C, data, BASE_SEED) != C.ioi_prompts(N_PROMPTS, seed=BASE_SEED):
        raise RuntimeError("prompt generator does not reproduce the upstream prompts")
    alt = {label: make_records(tpl, N_PROMPTS)
           for label, tpl in TEMPLATES.items() if label != "upstream"}
    null_data = make_records(NULL_TEMPLATE, N_PROMPTS, third_name=True)

    finder = make_finder(bundle, C, B, CoAblation, raw_dir)
    result = sk.stress(
        finder, data,
        battery=["seeds", "bootstrap", "templates", "hyperparams"],
        n_runs=args.n_runs, seed=BASE_SEED,
        config=dict(BASE_CONFIG),
        templates=alt,
        hyperparams={"position_mode": ["all"], "ablation_mode": ["mean"],
                     "feature_mode": ["l2_uncentered"], "freeze_ln": ["full"],
                     "top_r": [192], "primaries": ["atp-top3", "energy-top3"]},
        null_data=null_data,
        claim_statement=(
            "On the GPT-2-small IOI circuit, CoAx raises backup-head recovery from 0.33 to "
            "0.91 ROC-AUC, outperforming all baselines, including self-repair-aware gradient "
            "scores (best 0.82)"),
        model="gpt2",
        task="IOI backup-name-mover recovery: rank 141 candidate heads against the 8 "
             "documented backups after ablating the 3 documented name movers",
        method="conditional co-ablation compensation score (Fisher energy of ablating a head "
               "given the primaries ablated, minus unconditional), upstream code at the "
               "pinned commit",
        verbose=True,
    )

    if finder.shard.is_worker:
        print(f"shard {finder.shard.index}/{finder.shard.count} done; no artifacts written")
        return

    base = result.base.meta
    result.card.notes.append(
        f"upstream: {UPSTREAM_REPO}@{UPSTREAM_COMMIT[:7]} (MIT); CoAblation, baselines and "
        "circuit labels imported unmodified; file hashes " + ", ".join(
            f"{os.path.basename(p)} {h[:12]}" for p, h in hashes.items()))
    result.card.notes.append(
        f"reproduction: base run (seed {BASE_SEED}, {N_PROMPTS} prompts, last position, full "
        f"vocabulary) CoAx {base['auc']['coax']:.3f}, single-ablation {base['auc']['single']:.3f}, "
        f"AtP* {base['auc']['atp_star']:.3f}, EAP-IG {base['auc']['eap_ig']:.3f}, AtP "
        f"{base['auc']['atp']:.3f}, conditional attribution {base['auc']['conditional_grad']:.3f}; "
        f"precision@8 {base['precision_at_8']:.2f}, average precision "
        f"{base['average_precision']:.3f}. arXiv v1 abstract: CoAx {ARXIV['coax']}, single "
        f"{ARXIV['single']}, best baseline {ARXIV['atp_star']}"
        + (f"; released results/reference_metrics.json: "
           f"{json.dumps(reference)[:300]}" if reference else ""))
    result.card.notes.append(
        "templates: upstream hard-codes one IOI template; the three alternatives use the "
        "same 16 names, 8 places, 8 objects and the same seed, and are labelled as such")
    for record in result.runs:
        if record.axis == "hyperparams" and record.config.get("primaries", "documented") != "documented":
            m = record.finding.meta
            result.card.notes.append(
                f"{record.variant}: primaries selected label-free = "
                f"{[f'L{layer}H{h}' for layer, h in m['primaries']]} (documented name movers: "
                f"{m['primaries_documented']}); CoAx AUC {record.finding.score:.3f}, "
                f"AtP* {m['auc']['atp_star']:.3f}")
    result.card.notes.append(
        "null control: the same prompt seed with a third name as the giver ('... {X} gave a "
        "{obj} to'), so no name is repeated and the indirect-object task is not posed; "
        "labels unchanged; the finder still ranks candidates against the documented backups")

    print()
    print(result)
    print(result.to_markdown())
    stem = os.path.join(args.out_dir, "coax_backup_gpt2")
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
                   "reference_metrics": reference,
                   "raw_dir": os.path.relpath(raw_dir, args.out_dir), "runs": rows},
                  f, indent=1, default=str)
        f.write("\n")
    print(sk.verdict_trace_markdown(trace))
    print(f"\nartifacts written to {stem}.*")


if __name__ == "__main__":
    main()
