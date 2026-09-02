"""Reference Stability Card: OMP recovery of the 32-coordinate finite-effect
map in Mechanistic Tomography (arXiv:2608.19338, Section 5.1 / Figure 2).

Claim under test (registry statement, byte-exact): "Orthogonal matching
pursuit recovers the 32-coordinate finite-effect map with Pearson r = 0.989
and held-out R-squared = 0.935."

Upstream pipeline (kwisatzh/mechanistic-tomography @ 5c097d2, Apache-2.0).
``nt_mi_correspondence.py`` perturbs the released 4-layer HMM observer along
a per-layer belief direction at 32 coordinates (layer x time bin, step
epsilon = 0.6) and records, on one fixed evaluation batch, the 32 single-
coordinate effects on the implied belief z1 (the coordinate-patching
reference map) plus 256 aggregate effects of random signed masks (density
0.30). ``sparse_tomography_posthoc.py`` permutes the 256 aggregate
measurements (seed 7), holds out 64, keeps 64 for validation, fits OMP on
the first n_train = 12 of the rest for every support size in a grid, keeps
the support size with the best validation R^2, and reports held-out R^2
and the Pearson r between the recovered map and the reference.

Finder = that post-hoc reducer, imported unmodified from the pinned upstream
file, as a pure function of (data, seed, config):

- data: the list of aggregate measurements (mask, response, reference map),
  so the bootstrap axis resamples measurements;
- seed: the split permutation (which measurements are train / validation /
  held-out), exactly upstream's ``--seed``;
- config: n_train, n_val, holdout_frac, the OMP support-size grid, and the
  OMP refit ridge.

Finding representation (pre-registered; thresholds fixed before any run):

- components: the OMP-selected support, coordinates named ``L{layer}B{bin}``.
  Universe = 32 coordinates.
- score: held-out aggregate R^2 (the claim's second number).
- claim: ``"<recovery>; <sparsity>"``. recovery = "recovered" if held-out
  R^2 >= 0.90 and Pearson r vs the reference >= 0.95 (the two threshold
  crossings upstream itself declares in sparse_recovery_summary.json);
  "predictive-only" if R^2 >= 0.90 but r < 0.95 (upstream's own
  confounded-basis diagnostic); otherwise "not recovered". sparsity =
  "sparse support (k<=8)" when at most a quarter of the coordinates are
  selected, else "dense support (k>8)".
- meta: Pearson/Spearman vs the reference, validation/train R^2, selected k,
  split sizes, pool label, and a digest of the raw fit.

Battery: seeds (split permutation), bootstrap (measurement resampling),
templates (measurement pools re-measured on CPU from the released checkpoint
with the upstream generator: the released design re-measured, two fresh
signed designs, and one Bernoulli-mask design -- the second measurement
family upstream exposes), hyperparams (n_train 8 and 16; validation budget
matched to the training budget, n_val = 12; support size fixed at the
paper's k = 4 instead of validation-selected), plus a null control where the
measurement-to-response pairing is permuted once -- the registry's declared
null (random designs at matched budget) realized as a re-pairing: every
response meets a design independent of the one that produced it.

Data: the released measurement pool experiments/hmm/frozen/nt_mi_set1_v2 and
the released checkpoint, SHA-256 verified against the frozen intake
inventory. Base seed 7 reproduces the released row exactly.

Usage (CPU, minutes; a venv with the upstream requirements -- torch, numpy,
pandas, matplotlib, tqdm -- on Python 3.10/3.11):
    PYTHONPATH=src python references/run_mechtomo_omp_card.py \
        --upstream-dir /path/to/mechanistic-tomography  # checkout of 5c097d2
"""

import argparse
import hashlib
import importlib
import json
import math
import os
import random
import statistics
import sys
import urllib.request

os.environ.setdefault("MPLBACKEND", "Agg")  # upstream imports pyplot at module load

import numpy as np

import stresskit as sk

UPSTREAM_REPO = "kwisatzh/mechanistic-tomography"
UPSTREAM_COMMIT = "5c097d2175c632a4a15e359cb4d94ec923168472"
HMM_DIR = "experiments/hmm"
POOL_DIR = f"{HMM_DIR}/frozen/nt_mi_set1_v2"
RELEASED_ROWS = f"{HMM_DIR}/frozen/nt_mi_sparse_v1/sparse_recovery_sample_efficiency.csv"
# benchmark/intake/mechtomo_finite_effect_map_recovery/artifact-inventory.json
UPSTREAM_SHA256 = {
    f"{HMM_DIR}/sparse_tomography_posthoc.py":
        "db809578b161a8c43036f541c1f0a59e7408203d0650081d70629958e28d76c3",
    f"{HMM_DIR}/nt_mi_correspondence.py":
        "e53aa11056e971022201d4f48342514261a60a057995d75c6dd3e798707a7db0",
    f"{HMM_DIR}/hmm_observer_control.py":
        "3f69529a4833de87813b4b562ddc00c558e3ce2283de21bb77f544496ca03b3b",
    f"{HMM_DIR}/frozen/model.pt":
        "4d8689f8615cd2e78972f46dc022ae6b11d02eb2a4430bbae6cc013b0f299983",
    f"{POOL_DIR}/measurement_matrix_A.npy":
        "a59abe984e70a904185d8c5a3e295bcdf89d70f2da9a1fde1959b1061e0dc8bd",
    f"{POOL_DIR}/tomography_measurements.csv":
        "fe9b58ac7738b3a78367df35ab9ca816e5d44695f9ba57ec1dee88571b4393c2",
    f"{POOL_DIR}/direct_mi_z1.npy":
        "15096074eb36d6a9fba855fa366abe0b28b7646d1899cb1d46a4ac23bb05d4bc",
    RELEASED_ROWS:
        "212e1876e7f79e123d5867191de6bcd78b3cf364a1c2ad96a0d4c21972d75cdc",
}

CLAIM = ("Orthogonal matching pursuit recovers the 32-coordinate finite-effect "
         "map with Pearson r = 0.989 and held-out R-squared = 0.935.")
N_LAYERS, N_BINS = 4, 8
N_COORDS = N_LAYERS * N_BINS
# upstream nt_mi_correspondence.py defaults, as recorded in the released
# mt_summary.json (seed 7, 8 bins, epsilon 0.6, batch 384, 20 direction
# batches of 120000 samples, 256 signed masks at density 0.30)
INSTRUMENT_SEED = 7
N_MEASUREMENTS = 256
MASK_DENSITY = 0.30
EPSILON = 0.6
EVAL_BATCH = 384
DIRECTION_BATCHES = 20
DIRECTION_SAMPLES = 120000
# upstream sparse_tomography_posthoc.py defaults (seed 7, 25% held out, 25%
# validation, k grid, refit ridge); n_val = int(0.25 * 256)
BASE_SEED = 7
BASE_CONFIG = {"n_train": 12, "n_val": 64, "holdout_frac": 0.25,
               "k_grid": "1,2,3,4,5,6,8,10,12,16", "refit_ridge": 1e-10}
# claim-label thresholds: upstream's own threshold_crossings bars
R2_BAR = 0.90
PEARSON_BAR = 0.95
SPARSE_MAX_K = 8
NULL_PERMUTATION_SEED = 0x5EC


# ---------------------------------------------------------------------------
# Pinned upstream files and modules
# ---------------------------------------------------------------------------

def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def upstream_file(root, relpath):
    """Path to a pinned upstream file: downloaded on demand from the pinned
    commit, always SHA-256 verified against the frozen intake inventory."""
    path = os.path.join(root, relpath)
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        url = (f"https://raw.githubusercontent.com/{UPSTREAM_REPO}/"
               f"{UPSTREAM_COMMIT}/{relpath}")
        urllib.request.urlretrieve(url, path)
    digest = sha256_of(path)
    if digest != UPSTREAM_SHA256[relpath]:
        raise RuntimeError(
            f"{path}: sha256 {digest} does not match the pinned upstream file "
            f"{UPSTREAM_SHA256[relpath]} (commit {UPSTREAM_COMMIT[:7]})")
    return path


def import_upstream(root):
    """(posthoc, generator, harness): the three upstream modules, unmodified."""
    for name in ("sparse_tomography_posthoc", "nt_mi_correspondence",
                 "hmm_observer_control"):
        upstream_file(root, f"{HMM_DIR}/{name}.py")
    sys.path.insert(0, os.path.join(root, HMM_DIR))
    posthoc = importlib.import_module("sparse_tomography_posthoc")
    generator = importlib.import_module("nt_mi_correspondence")
    harness = importlib.import_module("hmm_observer_control")
    return posthoc, generator, harness


def released_arrays(root):
    """(A, y, mi) of the released pool: 256 signed masks, their aggregate z1
    responses, and the 32 single-coordinate reference effects."""
    A = np.load(upstream_file(root, f"{POOL_DIR}/measurement_matrix_A.npy"))
    meas = np.loadtxt(upstream_file(root, f"{POOL_DIR}/tomography_measurements.csv"),
                      delimiter=",", skiprows=1)  # measurement, y_z1, y_z2
    mi = np.load(upstream_file(root, f"{POOL_DIR}/direct_mi_z1.npy"))
    if A.shape != (N_MEASUREMENTS, N_COORDS) or meas.shape[0] != N_MEASUREMENTS \
            or mi.shape != (N_COORDS,):
        raise RuntimeError(f"released pool has unexpected shapes: A {A.shape}, "
                           f"measurements {meas.shape}, mi {mi.shape}")
    return A, meas[:, 1], mi


def released_row(root):
    """The released OMP n_train=12 row of sparse_recovery_sample_efficiency.csv."""
    path = upstream_file(root, RELEASED_ROWS)
    with open(path, encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        for line in f:
            row = dict(zip(header, line.strip().split(",")))
            if row["method"] == "omp" and int(row["n_train"]) == BASE_CONFIG["n_train"]:
                return {"selected_k": int(float(row["selected_k"])),
                        "pearson_vs_mi": float(row["pearson_vs_mi"]),
                        "holdout_r2": float(row["holdout_r2"])}
    raise RuntimeError(f"{path}: no omp row at n_train={BASE_CONFIG['n_train']}")


# ---------------------------------------------------------------------------
# Measurement pools
# ---------------------------------------------------------------------------

def coord_name(index):
    return f"L{index // N_BINS}B{index % N_BINS}"


def make_pool(A, y, mi, label):
    """One record per aggregate measurement. Every record carries the pool's
    reference map so the finder stays a pure function of its data argument
    (bootstrap resamples records; templates swap pools)."""
    reference = tuple(float(v) for v in mi)
    return [{"mask": tuple(float(v) for v in row), "y": float(resp),
             "mi": reference, "pool": label} for row, resp in zip(A, y)]


def permuted_pairing(pool, seed):
    """Null control: the same masks and the same responses, paired at random."""
    responses = [rec["y"] for rec in pool]
    random.Random(seed).shuffle(responses)
    return [dict(rec, y=resp, pool=f"{rec['pool']} (responses permuted)")
            for rec, resp in zip(pool, responses)]


def load_released_model(root, harness, device):
    import torch
    ckpt = torch.load(upstream_file(root, f"{HMM_DIR}/frozen/model.pt"),
                      map_location=device, weights_only=True)
    cfg = harness.ExperimentConfig(**ckpt["cfg"])
    params = harness.HMMParams(**ckpt["params"])
    model = harness.TinyCausalTransformer(
        vocab_size=4, seq_len=cfg.seq_len - 1, d_model=cfg.d_model,
        n_layers=cfg.n_layers, n_heads=cfg.n_heads, d_mlp=cfg.d_mlp,
        dropout=cfg.dropout).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg, params


class Instrument:
    """The upstream measurement instrument on the released checkpoint: per-layer
    belief directions, one fixed evaluation batch, the 32 single-coordinate
    reference effects, and the aggregate response to any mask. Mirrors
    nt_mi_correspondence.main() step for step (same seed, same defaults), so
    with design seed 7 the masks are the released masks; only the device
    differs (CPU here, MPS upstream)."""

    def __init__(self, root, generator, harness, device):
        import torch
        self.torch = torch
        self.generator = generator
        self.device = torch.device(device)
        self.model, self.cfg, self.params = load_released_model(root, harness, self.device)
        harness.set_seed(INSTRUMENT_SEED)
        self.T = self.cfg.seq_len - 1
        self.bins = generator.build_time_bins(self.T, N_BINS)
        self.comps = generator.component_index(self.model.n_layers, N_BINS)
        if len(self.comps) != N_COORDS:
            raise RuntimeError(f"released model yields {len(self.comps)} coordinates, expected {N_COORDS}")
        self.layer_dirs = generator.compute_layer_directions(
            self.model, self.cfg, self.params, self.device,
            batches=DIRECTION_BATCHES, batch_size=self.cfg.batch_size,
            seq_len=self.cfg.seq_len, samples=DIRECTION_SAMPLES)
        batch = harness.generate_batch(EVAL_BATCH, self.cfg.seq_len, self.params, self.device)
        self.idx = batch["tokens"][:, :-1]
        with torch.no_grad():
            base_logits, _ = self.model(self.idx)
            self.base_z1 = generator.implied_z_from_logits(base_logits, self.params, "z1")[:, -1]
            self.base_z2 = generator.implied_z_from_logits(base_logits, self.params, "z2")[:, -1]
        self.mi = self.measure(np.eye(N_COORDS))

    def respond(self, mask):
        strengths = self.generator.strengths_from_mask(
            mask, self.comps, self.bins, self.model.n_layers, self.T,
            EVAL_BATCH, EPSILON, self.device)
        with self.torch.no_grad():
            dz1, _ = self.generator.measure_effect(
                self.model, self.idx, self.params, strengths, self.layer_dirs,
                self.base_z1, self.base_z2)
        return dz1

    def measure(self, A):
        return np.array([self.respond(row) for row in A], dtype=np.float64)

    def design(self, design_seed, mask_kind):
        rng = np.random.default_rng(design_seed)
        return self.generator.make_random_masks(
            N_MEASUREMENTS, N_COORDS, mask_kind, MASK_DENSITY, rng)


# ---------------------------------------------------------------------------
# Finder
# ---------------------------------------------------------------------------

def claim_label(holdout_r2, pearson, k):
    if not math.isfinite(holdout_r2) or holdout_r2 < R2_BAR:
        recovery = "not recovered"
    elif math.isfinite(pearson) and pearson >= PEARSON_BAR:
        recovery = "recovered"
    else:
        recovery = "predictive-only"
    sparsity = (f"sparse support (k<={SPARSE_MAX_K})" if k <= SPARSE_MAX_K
                else f"dense support (k>{SPARSE_MAX_K})")
    return f"{recovery}; {sparsity}"


def finite_or_none(x):
    x = float(x)
    return x if math.isfinite(x) else None


def make_finder(posthoc, raw_path):
    """The upstream post-hoc reducer as a pure function of (data, seed, config).
    Split, OMP fit, support-size selection and metrics are upstream code
    (evaluate_fit); this wrapper builds the arrays, names the coordinates,
    labels the claim, and logs the raw fit."""

    def finder(data, seed, config):
        A = np.array([rec["mask"] for rec in data], dtype=np.float64)
        y = np.array([rec["y"] for rec in data], dtype=np.float64)
        mi = np.array(data[0]["mi"], dtype=np.float64)
        n = len(data)
        n_train, n_val = int(config["n_train"]), int(config["n_val"])
        n_hold = max(1, int(float(config["holdout_frac"]) * n))
        if n_hold + n_val + n_train > n:
            raise ValueError(f"split needs {n_hold} + {n_val} + {n_train} measurements, pool has {n}")
        perm = np.random.default_rng(seed).permutation(n)
        hold_idx = perm[:n_hold]
        val_idx = perm[n_hold:n_hold + n_val]
        fit_idx = perm[n_hold + n_val:][:n_train]

        upstream_args = argparse.Namespace(
            omp_k_grid=posthoc.parse_int_list(config["k_grid"]),
            omp_refit_ridge=float(config["refit_ridge"]))
        beta, intercept, info = posthoc.evaluate_fit(
            "omp", A[fit_idx], y[fit_idx], A[val_idx], y[val_idx],
            A[hold_idx], y[hold_idx], mi, upstream_args)
        support = [int(i) for i in np.flatnonzero(beta)]
        if len(support) != int(info["support_size"]):
            raise RuntimeError(f"support {support} disagrees with upstream support_size {info['support_size']}")
        holdout_r2 = float(info["holdout_r2"])
        if not math.isfinite(holdout_r2):
            raise RuntimeError("held-out R^2 is not finite; the held-out split is degenerate")
        pearson = float(info.get("pearson_vs_mi", float("nan")))
        claim = claim_label(holdout_r2, pearson, len(support))

        record = {
            "pool": data[0]["pool"], "seed": int(seed), "config": dict(config),
            "support": support, "beta": [float(b) for b in beta],
            "intercept": float(intercept),
            "hold_idx": [int(i) for i in hold_idx], "val_idx": [int(i) for i in val_idx],
            "fit_idx": [int(i) for i in fit_idx],
            "holdout_r2": holdout_r2, "claim": claim,
            "info": {k: finite_or_none(v) for k, v in info.items()},
        }
        digest = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()[:16]
        record["fit_sha256_16"] = digest
        with open(raw_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        return sk.feature_set(
            [coord_name(i) for i in support],
            claim=claim,
            score=holdout_r2,
            universe_size=N_COORDS,
            selected_k=len(support),
            pearson_vs_mi=finite_or_none(pearson),
            spearman_vs_mi=finite_or_none(info.get("spearman_vs_mi", float("nan"))),
            top5_overlap=finite_or_none(info.get("top5_overlap", float("nan"))),
            val_r2=finite_or_none(info.get("val_r2", float("nan"))),
            train_r2=finite_or_none(info.get("train_r2", float("nan"))),
            n_train=len(fit_idx), n_val=len(val_idx), n_holdout=len(hold_idx),
            pool=data[0]["pool"],
            fit_sha256_16=digest,
        )

    return finder


# ---------------------------------------------------------------------------
# Post-hoc notes and random samples
# ---------------------------------------------------------------------------

def released_row_note(result, released):
    base = result.base
    same = (base.meta["selected_k"] == released["selected_k"]
            and base.meta["pearson_vs_mi"] is not None
            and abs(base.meta["pearson_vs_mi"] - released["pearson_vs_mi"]) < 1e-9
            and abs(base.score - released["holdout_r2"]) < 1e-9)
    support = ", ".join(sorted(base.components))
    pearson = base.meta["pearson_vs_mi"]
    return (f"upstream row: the base run (seed {BASE_SEED}, released pool, n_train="
            f"{BASE_CONFIG['n_train']}) selects k={base.meta['selected_k']} "
            f"{{{support}}} with Pearson r = {'nan' if pearson is None else f'{pearson:.6f}'} "
            f"and held-out R^2 = {base.score:.6f}; the released "
            f"sparse_recovery_sample_efficiency.csv row has k={released['selected_k']}, "
            f"r = {released['pearson_vs_mi']:.6f}, R^2 = {released['holdout_r2']:.6f} "
            f"-- {'reproduced to 1e-9' if same else 'NOT reproduced'}")


def support_notes(result):
    base = result.base.components
    real = [r for r in result.runs if r.axis != "base"]
    notes = []
    by_axis = {}
    for r in real:
        hit, total = by_axis.get(r.axis, (0, 0))
        by_axis[r.axis] = (hit + (r.finding.components == base), total + 1)
    per_axis = ", ".join(f"{axis} {hit}/{total}" for axis, (hit, total) in sorted(by_axis.items()))
    n_hit = sum(h for h, _ in by_axis.values())
    counts = {}
    for r in real:
        for c in r.finding.components:
            counts[c] = counts.get(c, 0) + 1
    frequent = sorted((c for c, n in counts.items() if n >= len(real) / 2),
                      key=lambda c: -counts[c])
    notes.append(
        f"support identity (not graded): {n_hit}/{len(real)} perturbed real runs select "
        f"exactly the base support {{{', '.join(sorted(base))}}} ({per_axis}); coordinates "
        f"selected in at least half of the perturbed real runs: "
        f"{', '.join(f'{c} ({counts[c]}/{len(real)})' for c in frequent) or 'none'}")
    if result.null_runs:
        null_counts, claim_counts = {}, {}
        for r in result.null_runs:
            key = tuple(sorted(r.finding.components))
            null_counts[key] = null_counts.get(key, 0) + 1
            claim_counts[r.finding.claim] = claim_counts.get(r.finding.claim, 0) + 1
        modal, n_modal = max(null_counts.items(), key=lambda kv: kv[1])
        sizes = [r.finding.size for r in result.null_runs]
        scores = [r.finding.score for r in result.null_runs]
        notes.append(
            f"null-control supports: {len(null_counts)} distinct supports across "
            f"{len(result.null_runs)} null runs (most common {{{', '.join(modal) or 'empty'}}} "
            f"x{n_modal}); median size {sorted(sizes)[len(sizes) // 2]}; held-out R^2 "
            f"mean {np.mean(scores):.3f}, max {np.max(scores):.3f}; claims: "
            + ", ".join(f"`{k}` x{v}" for k, v in
                        sorted(claim_counts.items(), key=lambda kv: -kv[1])))
    return notes


def recovery_note(result):
    rows = []
    for axis in ("seeds", "bootstrap", "templates", "hyperparams"):
        group = [r for r in result.runs if r.axis == axis]
        if not group:
            continue
        n_recovered = sum(r.finding.claim.startswith("recovered") for r in group)
        median_r2 = statistics.median(r.finding.score for r in group)
        rows.append(f"{axis} {n_recovered}/{len(group)} recovered "
                    f"(median held-out R^2 {median_r2:.2f})")
    return (f"recovery by axis (label 'recovered' = held-out R^2 >= {R2_BAR} and r >= "
            f"{PEARSON_BAR}): " + "; ".join(rows) + ". Bootstrap resamples measurements "
            "with replacement, so a training measurement's duplicate can sit in the held-out "
            "set; that biases held-out R^2 upward, not downward")


def null_geometry_note(result):
    """How much null data the graded specificity ratio actually rests on."""
    null_base = next(r.finding for r in result.null_runs if r.axis == "base")
    sets = [r.finding.components for r in result.null_runs]
    sized = [s for s in sets if null_base.size / 2 <= len(s) <= null_base.size * 2]
    graded = sized if sized else sets
    j_all = sk.metrics.mean_pairwise_jaccard(sets)
    j_real = result.pooled["mean_pairwise_jaccard"]
    ratio_all = f"{j_real / j_all:.1f}x" if j_all and j_all > 1e-9 else "unbounded"
    return (f"specificity basis: the null base run selected k={null_base.size}, so StressKit's "
            f"2x size guard grades null Jaccard on {len(graded)}/{len(sets)} null runs (sizes "
            f"{sorted(len(s) for s in graded)}) = "
            f"{result.null_summary['mean_pairwise_jaccard']:.3f}, too few for a CI (state "
            f"inconclusive); over all {len(sets)} null runs the null Jaccard is {j_all:.3f} and "
            f"the real/null ratio would be {ratio_all}, so the graded "
            f"{result.checks['specificity']['value']:.2f}x is the conservative (harsher) "
            "reading")


def fmt_ci(ci):
    return "no CI" if ci is None else f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def posthoc_regrade_note(result):
    """The verdict trace regrades in post-hoc mode (from_findings, bootstrap
    seed 0, null runs pooled without the size guard); say where that disagrees
    with the card so a reader is not surprised by two confidence labels."""
    full = sk.from_findings(
        [r.finding for r in result.runs],
        null_findings=[r.finding for r in result.null_runs] if result.null_runs else None,
        seed=0)
    diffs = []
    for name, check in full.checks.items():
        card_check = result.checks.get(name)
        if card_check is None:
            continue
        if check["state"] != card_check["state"] or check["passed"] != card_check["passed"]:
            diffs.append(f"{name}: card {card_check['state']} (CI {fmt_ci(card_check['ci'])}, "
                         f"value {card_check['value']:.3f}) vs post-hoc {check['state']} "
                         f"(CI {fmt_ci(check['ci'])}, value {check['value']:.3f})")
    return (f"post-hoc regrade (verdict-trace mode, from_findings at bootstrap seed 0): grade "
            f"{full.grade}, {full.pooled['confidence']} confidence vs the card's "
            f"{result.pooled['confidence']}; "
            + (("checks that differ -- " + "; ".join(diffs)) if diffs
               else "every check has the same state as on the card")
            + " -- a check whose CI end sits at its bar is decided by the bootstrap seed, "
              "not by the data")


def instrument_note(instrument, released_A, released_y, released_mi):
    import torch
    A_cpu = instrument.design(INSTRUMENT_SEED, "signed")
    same_masks = np.array_equal(A_cpu, released_A)
    y_cpu = instrument.measure(released_A)
    r_y = float(np.corrcoef(y_cpu, released_y)[0, 1])
    r_mi = float(np.corrcoef(instrument.mi, released_mi)[0, 1])
    top4_cpu = set(np.argsort(-np.abs(instrument.mi))[:4].tolist())
    top4_rel = set(np.argsort(-np.abs(released_mi))[:4].tolist())
    return (
        f"instrument transfer (not graded): the upstream generator re-run on CPU (torch "
        f"{torch.__version__}) from the released checkpoint regenerates the released masks "
        f"{'bit-identically' if same_masks else 'DIFFERENTLY'} from numpy seed {INSTRUMENT_SEED}; "
        f"re-measuring the released design gives responses with Pearson r = {r_y:.4f} to the "
        f"released (MPS) responses, mean |diff| {np.mean(np.abs(y_cpu - released_y)):.3f} "
        f"(response sd {np.std(released_y):.3f}), and a reference map with r = {r_mi:.4f} "
        f"({'same' if top4_cpu == top4_rel else 'different'} top-4 coordinates); the "
        f"template pools inherit this CPU evaluation batch"), y_cpu


def mask_string(mask):
    chars = "".join("+" if v > 0 else "-" if v < 0 else "." for v in mask)
    return " ".join(chars[i:i + N_BINS] for i in range(0, len(chars), N_BINS))


def write_samples(path, pool, result, raw_by_digest, seed=0):
    rng = random.Random(seed)
    base_raw = raw_by_digest[result.base.meta["fit_sha256_16"]]
    beta = np.array(base_raw["beta"])
    mi = np.array(pool[0]["mi"])
    role = {}
    for name, idx in (("held-out", base_raw["hold_idx"]), ("validation", base_raw["val_idx"]),
                      ("train", base_raw["fit_idx"])):
        for i in idx:
            role[i] = name
    lines = [
        "# Randomly selected raw records (released pool and battery runs)", "",
        "Selected with `random.Random(0)`, not cherry-picked. Masks are shown per "
        "layer (L0..L3), one character per time bin: `+` = +epsilon, `-` = -epsilon, "
        "`.` = untouched.", "",
        f"## Base run (seed {BASE_SEED}, released pool): recovered coefficients vs "
        "the coordinate-patching reference", "",
        "Selected coordinates first (deterministic, not sampled), then the four "
        "largest unselected reference effects.", "",
        "| coordinate | OMP coefficient | reference effect | selected |", "|---|---|---|---|",
    ]
    selected = sorted(base_raw["support"], key=lambda i: -abs(mi[i]))
    unselected = [i for i in np.argsort(-np.abs(mi)) if i not in base_raw["support"]][:4]
    for i in selected + [int(i) for i in unselected]:
        lines.append(f"| {coord_name(i)} | {beta[i]: .4f} | {mi[i]: .4f} | "
                     f"{'yes' if i in base_raw['support'] else 'no'} |")
    lines += ["", "## Six randomly selected released measurements (base-run split role, "
              "base-fit prediction)", "",
              "| # | mask (L0 L1 L2 L3) | response y | base fit | role in base run |",
              "|---|---|---|---|---|"]
    for i in sorted(rng.sample(range(len(pool)), 6)):
        rec = pool[i]
        pred = float(np.dot(beta, rec["mask"]) + base_raw["intercept"])
        lines.append(f"| {i} | `{mask_string(rec['mask'])}` | {rec['y']: .4f} | {pred: .4f} | "
                     f"{role.get(i, 'unused (pool beyond the first 12)')} |")
    lines += ["", "## Five randomly selected perturbed runs", "",
              "| axis | variant | support | held-out R^2 | Pearson r | claim |",
              "|---|---|---|---|---|---|"]
    perturbed = [r for r in result.runs if r.axis != "base"]
    for r in rng.sample(perturbed, 5):
        f = r.finding
        pr = f.meta["pearson_vs_mi"]
        lines.append(f"| {r.axis} | {r.variant} | {', '.join(sorted(f.components)) or '(empty)'} | "
                     f"{f.score: .4f} | {'nan' if pr is None else f'{pr: .4f}'} | {f.claim} |")
    if result.null_runs:
        lines += ["", "## Three randomly selected null-control runs (responses permuted)", "",
                  "| axis | variant | support | held-out R^2 | Pearson r | claim |",
                  "|---|---|---|---|---|---|"]
        for r in rng.sample(result.null_runs, 3):
            f = r.finding
            pr = f.meta["pearson_vs_mi"]
            lines.append(f"| {r.axis} | {r.variant} | {', '.join(sorted(f.components)) or '(empty)'} | "
                         f"{f.score: .4f} | {'nan' if pr is None else f'{pr: .4f}'} | {f.claim} |")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream-dir", default="mechtomo_upstream",
                    help="checkout of the pinned upstream commit, or a directory the "
                         "needed files are downloaded into (SHA-256 verified either way)")
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "cards"))
    ap.add_argument("--raw-dir", default=None,
                    help="where per-run fits and regenerated pools are saved "
                         "(default: <out-dir>/raw/mechtomo_omp)")
    ap.add_argument("--n-runs", type=int, default=24)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    raw_dir = args.raw_dir or os.path.join(args.out_dir, "raw", "mechtomo_omp")
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)
    raw_path = os.path.join(raw_dir, "fits.jsonl")
    open(raw_path, "w").close()

    posthoc, generator, harness = import_upstream(args.upstream_dir)
    released_A, released_y, released_mi = released_arrays(args.upstream_dir)
    released = released_row(args.upstream_dir)
    pool = make_pool(released_A, released_y, released_mi, "released nt_mi_set1_v2")
    null_pool = permuted_pairing(pool, NULL_PERMUTATION_SEED)

    print(f"loading the released checkpoint on {args.device} and re-running the "
          f"upstream instrument (seed {INSTRUMENT_SEED}) ...")
    instrument = Instrument(args.upstream_dir, generator, harness, args.device)
    transfer_note, y_cpu = instrument_note(instrument, released_A, released_y, released_mi)
    print(transfer_note)
    templates = {}
    saved = {"cpu_reference_mi": instrument.mi}

    def add_template(label, A, y=None):
        if y is None:
            print(f"measuring template pool {label} ...")
            y = instrument.measure(A)
        templates[label] = make_pool(A, y, instrument.mi, label)
        saved[f"{label}.A"], saved[f"{label}.y"] = A.astype(np.int8), y

    add_template("released-design-cpu-remeasured", released_A, y_cpu)
    add_template("fresh-signed-design-11", instrument.design(11, "signed"))
    add_template("fresh-signed-design-12", instrument.design(12, "signed"))
    add_template("bernoulli-design-11", instrument.design(11, "bernoulli"))
    np.savez(os.path.join(raw_dir, "pools.npz"), **saved)

    finder = make_finder(posthoc, raw_path)
    result = sk.stress(
        finder,
        pool,
        battery=["seeds", "bootstrap", "templates", "hyperparams"],
        n_runs=args.n_runs,
        seed=BASE_SEED,
        config=BASE_CONFIG,
        templates=templates,
        hyperparams={"n_train": [8, 16], "n_val": [12], "k_grid": ["4"]},
        null_data=null_pool,
        claim_statement=CLAIM,
        model=("released HMM observer checkpoint experiments/hmm/frozen/model.pt "
               "(4-layer, d_model 96, seed 7; kwisatzh/mechanistic-tomography@"
               f"{UPSTREAM_COMMIT[:7]})"),
        task=("recover the 32-coordinate (layer x time-bin) finite-effect map on the "
              "implied belief z1 from 12 aggregate signed-mask interventions "
              "(epsilon 0.6, density 0.30)"),
        method=("orthogonal matching pursuit with validation-selected support size, "
                "upstream sparse_tomography_posthoc.py"),
        verbose=True,
    )
    result.card.notes.append(
        "scope: graded artifact = the released measurement pool nt_mi_set1_v2 (256 "
        "signed-mask aggregate measurements on the released seed-7 HMM observer, "
        "epsilon 0.6, density 0.30) reduced by the upstream OMP post-hoc script at "
        f"n_train={BASE_CONFIG['n_train']} with validation-selected support size; usage "
        "mode = forward-only aggregate recovery on the fixed 4-layer x 8-bin coordinate "
        "basis, scored against upstream's own coordinate-patching reference on the same "
        "evaluation batch. The battery does NOT test the Section 5.2 attribution-patching "
        "calibration, the Qwen-2.5-7B experiment, designed (non-random) measurement "
        "optimality, other coordinate bases or intervention sizes, or any pretrained model")
    result.card.notes.append(
        f"data: {UPSTREAM_REPO}@{UPSTREAM_COMMIT[:7]} (Apache-2.0 code; the checkpoint and "
        "frozen numeric artifacts carry no explicit file-level license, per the intake "
        "inventory), every file SHA-256 verified against the frozen intake inventory; "
        "the checkpoint is loaded with weights_only=True after verification")
    result.card.notes.append(
        f"claim label thresholds pre-registered from upstream's own summary: held-out R^2 "
        f">= {R2_BAR} and Pearson r >= {PEARSON_BAR} (sparse_recovery_summary.json "
        f"threshold_crossings); sparse = k <= {SPARSE_MAX_K}; score = held-out R^2")
    result.card.notes.append(released_row_note(result, released))
    result.card.notes.append(
        f"null control: the released pool with responses permuted once (seed "
        f"{NULL_PERMUTATION_SEED:#x}), so each response is paired with a design independent "
        "of the one that produced it -- the registry's declared null (random measurement "
        "designs at matched budget) realized as a re-pairing, run through the same finder on "
        "the same seeds/bootstrap axes. Direction: on permuted data OMP still returns a "
        "support of validation-selected size, so null Jaccard sits at the size-matched random "
        "level rather than zero; a real-data Jaccard near that level would mean the split, "
        "not the measurements, picks the coordinates")
    result.card.notes.append(
        "templates: pools re-measured on CPU from the released checkpoint with the upstream "
        f"generator at instrument seed {INSTRUMENT_SEED} (same directions and evaluation batch "
        "for all four): the released design re-measured, two fresh signed designs (design "
        "seeds 11, 12), and one Bernoulli-mask design (seed 11; upstream's second mask "
        "family). Their reference map is the CPU re-measured coordinate-patching map")
    result.card.notes.append(
        "budget: the reported 12-measurement model additionally uses 64 validation "
        "measurements to choose its support size (and 64 held-out to score it); the n_val=12 "
        "hyperparameter run tests selection at a budget matched to the training count, and "
        "n_train=8 lies below the budget the claim is made at (upstream's own curve fails "
        "there too), so read the hyperparams row of the per-axis table before the pooled one")
    result.card.notes.append(transfer_note)
    result.card.notes.extend(support_notes(result))
    result.card.notes.append(recovery_note(result))
    if result.null_runs:
        result.card.notes.append(null_geometry_note(result))
    result.card.notes.append(posthoc_regrade_note(result))

    print()
    print(result)
    print(result.to_markdown())

    base = os.path.join(args.out_dir, "mechtomo_omp_recovery")
    result.card.save(base + ".json")
    with open(base + ".md", "w", encoding="utf-8") as f:
        f.write(result.to_markdown() + "\n")
    with open(base + ".badge.json", "w", encoding="utf-8") as f:
        json.dump(result.card.badge_dict(), f, indent=2)
        f.write("\n")
    print("\ncomputing verdict-stability trace ...")
    trace = result.verdict_trace(seed=0)
    with open(base + ".trace.json", "w", encoding="utf-8") as f:
        json.dump(trace, f, indent=2)
        f.write("\n")
    with open(base + ".trace.md", "w", encoding="utf-8") as f:
        f.write(sk.verdict_trace_markdown(trace) + "\n")
    raw_by_digest = {}
    with open(raw_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            raw_by_digest[rec["fit_sha256_16"]] = rec
    write_samples(base + ".samples.md", pool, result, raw_by_digest)
    print(sk.verdict_trace_markdown(trace))
    print(f"\nartifacts written to {base}.*")


if __name__ == "__main__":
    main()
