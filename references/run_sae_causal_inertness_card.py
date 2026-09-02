"""Reference Stability Card: the causally-inert census of cosine-recovered SAE
features (arXiv:2607.12166, abstract and Table 5).

Claim under test (registry statement, byte-exact from the arXiv abstract):
"Our central result is causal: subjecting every recovered feature to ablation
and steering, we find up to 77% of features passing a recovery bar (cosine >=
0.90) in a degraded SAE -- and 9% in a well-trained one -- are causally inert:
the matched atom never fires when the feature is present, including matches at
cosine ~1.000."

Upstream pipeline (mohamed-bal/sae-causal-audit @ 3915d95, MIT). ``toy.py``
trains the Elhage et al. (2022) bottleneck model (32 features into 8 dims,
sparsity 0.95, seed 0) and two TopK SAEs on its hidden activations (d_sae 128,
k=4 "good" and k=13 "bad"). ``matching.py`` matches every ground-truth direction
to its best decoder atom by signed cosine; ``metrics.py`` measures, per matched
pair, the fraction of feature-ON samples on which that atom's code is nonzero
(``fired_frac``), and flags the pair ``causally_inert`` when that fraction is
exactly zero. ``audit.py`` assembles the per-pair results and the census over
pairs whose unsigned cosine clears the recovery bar.

Finder = that pipeline, imported unmodified from the pinned upstream files, as a
pure function of (data, seed, config):

- data: the audit cohort -- one record per (SAE, ground-truth feature) pair, so
  the bootstrap axis resamples cohort members and the templates axis swaps which
  features are eligible for the census. Each record also carries the probe
  permutation the run is to use, which is what the null control varies.
- seed: the SAE initialisation/training seed, the probe sampling seed, and the
  audit seed together -- upstream's own ``seed`` arguments, moved as one.
- config: SAE training steps, toy-model training steps, dictionary size, the two
  TopK sparsities, samples per pair, and the cosine recovery bar.

The toy model is held at upstream's ``ToyConfig(seed=0)`` in every run. It is the
instrument, not the object under test: holding it fixed keeps feature indices
0..31 a single comparable universe across runs, the same way the mechanistic-
tomography card holds the released checkpoint fixed and varies the split.

Finding representation (pre-registered; fixed before any run, see card notes):

- components: ``"{sae}:f{feature_idx}"`` for every cohort pair that is both
  correlationally recovered (unsigned cosine >= the run's bar) and causally
  inert (``fired_frac == 0``). Universe = 2 SAEs x 32 toy features = 64.
- claim: ``"<presence>; <ordering>"``. presence is read off which SAEs have a
  non-zero inert rate ("inert in both" / "inert in degraded only" / "inert in
  well-trained only" / "no inert recovered features"); ordering is whether the
  degraded SAE's inert rate is at least the well-trained one's. Both halves come
  straight off the claim sentence, which asserts inertness in both SAEs and more
  of it in the degraded one. No free threshold, so nothing to tune post hoc.
- score: the pooled inert rate among recovered pairs, both SAEs together. The
  sentence quotes a rate per SAE; the pooled rate is the one scalar that moves
  when either does, and the per-SAE rates ride in meta.
- meta: per-SAE recovered/inert counts and rates, the cosine bar, the cohort
  label and size, the smallest cosine among inert pairs (the sentence's "matches
  at cosine ~1.000" is a claim about that number), the well-represented count,
  and a digest of the raw census so every run is recomputable.

Battery: seeds (SAE/probe/audit seed), bootstrap (cohort resampling), templates
(alternative probe datasets: the feature presented inside a sparse background
instead of in isolation, that same in-context probe over a denser background,
and both cohorts widened from the 22 well-represented features to all 32),
hyperparams (cosine bar 0.85 and 0.95 against upstream's 0.90, 200 samples per
side instead of 500, 2000 SAE training steps instead of 4000), plus a null
control that permutes the feature-to-probe pairing once, so every matched atom
is causally tested against a ground-truth feature it was not matched to. The
null runs through the SAME finder on the same axes.

Data: no external dataset. Every number is regenerated on CPU from the pinned
upstream source files, each SHA-256 verified against this script before import.

Usage (CPU, ~10 minutes):
    PYTHONPATH=src python references/run_sae_causal_inertness_card.py \
        --upstream-dir /path/to/sae-causal-audit  # checkout of 3915d95
"""

import argparse
import hashlib
import importlib
import json
import os
import random
import statistics
import sys
import urllib.request

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MKL_CBWR", "COMPATIBLE")
os.environ.setdefault("PYTHONHASHSEED", "0")

import numpy as np

import stresskit as sk

UPSTREAM_REPO = "mohamed-bal/sae-causal-audit"
UPSTREAM_COMMIT = "3915d95549c1ad8d0f5a84fdbed6c7a97ef483ee"
PACKAGE = "src/sae_causal_audit"
EXPECTED = "expected_results.json"
UPSTREAM_SHA256 = {
    f"{PACKAGE}/__init__.py":
        "0b14a33f78cdc2a0658b2634b82ff72299736b7fc9aa13df8d475787df12be2c",
    f"{PACKAGE}/audit.py":
        "d8a94761f7a4bb61f3824fe2a0b72f23885e378b5489dae9bf2322260c437af3",
    f"{PACKAGE}/interfaces.py":
        "8f8b6d8140a45c0e8defa8a4c37bdae11c261e4394c108e525095c4debdf5b61",
    f"{PACKAGE}/matching.py":
        "045dc9e7311656343ebde601812635d258f0342a5ea1e5a15d947d7e6c39afb1",
    f"{PACKAGE}/metrics.py":
        "8fd71e8481b3d2b74866cbe5759506ade299ec5a12482f7a723cb329da4fa9a4",
    f"{PACKAGE}/report.py":
        "1c813d9105300977cbfa22e865ccc5118fcd905c3c05de0f5597b52e6e6e98ce",
    f"{PACKAGE}/stats.py":
        "e88d2d0533800fa153de9ba6e84fbe0cc0e27fe66736f76f72253b6fa53b89ec",
    f"{PACKAGE}/toy.py":
        "4cf40fb0e86d73bf1efc33452cbdd2732eff56f3cc3d2b7d2cde07c76e59be1d",
    "scripts/reproduce_toy_results.py":
        "8fa13198ee0afad69582e12e64152fae76a10c8f67b04c7926b46359d5a78ef2",
    EXPECTED:
        "f91a535aecbee1b90b40cda32ab62d660e8b84aa77c0edefc988a3043d424ad0",
    "LICENSE":
        "b67dcf8bd115f63a71b1ab10ef1622471b695a52ec81c1e3b637f26141eb5832",
}

CLAIM = ("Our central result is causal: subjecting every recovered feature to "
         "ablation and steering, we find up to 77% of features passing a recovery "
         "bar (cosine >= 0.90) in a degraded SAE -- and 9% in a well-trained one "
         "-- are causally inert: the matched atom never fires when the feature is "
         "present, including matches at cosine ~1.000.")

# upstream scripts/reproduce_toy_results.py defaults
N_FEATURES = 32
SAE_LABELS = {"good_k4": "k_good", "bad_k13": "k_bad"}
BASE_SEED = 0
BASE_CONFIG = {
    "toy_steps": 4000, "sae_steps": 4000, "d_sae": 128,
    "k_good": 4, "k_bad": 13, "n_samples": 500, "cosine_bar": 0.90,
    "well_represented_norm": 0.1,
}
NULL_PERMUTATION_SEED = 0x5EC
UNIVERSE = len(SAE_LABELS) * N_FEATURES


# ---------------------------------------------------------------------------
# Pinned upstream files and modules
# ---------------------------------------------------------------------------

def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def upstream_file(root, relpath):
    """Path to a pinned upstream file: downloaded on demand from the pinned
    commit, always SHA-256 verified before it is imported or read."""
    path = os.path.join(root, relpath)
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
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
    """(audit, toy): the two upstream modules, unmodified."""
    for relpath in UPSTREAM_SHA256:
        upstream_file(root, relpath)
    sys.path.insert(0, os.path.join(root, "src"))
    return (importlib.import_module("sae_causal_audit"),
            importlib.import_module("sae_causal_audit.toy"))


def expected_rows(root):
    """The released headline numbers from expected_results.json."""
    with open(upstream_file(root, EXPECTED), encoding="utf-8") as handle:
        return json.load(handle)


# ---------------------------------------------------------------------------
# Cohorts and the null pairing
# ---------------------------------------------------------------------------

def make_cohort(feature_indices, label, permutation=None, regime="isolated",
                background_sparsity=None):
    """One record per (SAE, ground-truth feature). Every record also carries the
    probe settings the run is to use, so the finder stays a pure function of its
    data argument: bootstrap resamples records, templates swap the probe
    dataset or the eligible features, and the null swaps the pairing."""
    perm = None if permutation is None else tuple(int(v) for v in permutation)
    sparsity = None if background_sparsity is None else float(background_sparsity)
    return [{"sae": sae, "feature_idx": int(idx), "cohort": label,
             "probe_permutation": perm, "probe_regime": regime,
             "background_sparsity": sparsity}
            for sae in sorted(SAE_LABELS)
            for idx in sorted(int(i) for i in feature_indices)]


def derangement(n, seed):
    """A permutation with no fixed point, so no feature is probed with itself."""
    rng = random.Random(seed)
    order = list(range(n))
    while True:
        rng.shuffle(order)
        if all(a != b for a, b in enumerate(order)):
            return tuple(order)


def context_probe(torch, model, seed, background_sparsity=None, magnitude=1.0):
    """Feature-ON samples with the feature present *inside* a sparse background.

    Upstream's ToyProbe presents one feature at a time with every other input
    coordinate at exactly zero, and Table 5's census is computed in that regime.
    It is not the only defensible reading of "the feature is present": upstream's
    own real-model regime defines feature-ON as "the concept is present in the
    text", where the rest of the input is present too. This probe is that reading
    in the toy setting -- upstream's own background draw with the target feature
    forced on -- and it leaves the feature-OFF side exactly as upstream draws it.
    """
    generator = torch.Generator().manual_seed(int(seed) + 3)
    sparsity = (model.cfg.sparsity if background_sparsity is None
                else float(background_sparsity))
    n_features = model.cfg.n_features

    def background(n_samples):
        values = torch.rand(n_samples, n_features, generator=generator)
        mask = torch.rand(n_samples, n_features, generator=generator) >= sparsity
        return values * mask

    class ContextProbe:
        def activations_with_feature(self, feature_idx, n_samples):
            x = background(n_samples)
            x[:, feature_idx] = magnitude * torch.rand(
                n_samples, generator=generator).clamp_min(0.5)
            with torch.no_grad():
                return model.hidden(x)

        def activations_without_feature(self, feature_idx, n_samples):
            x = background(n_samples)
            x[:, feature_idx] = 0.0
            with torch.no_grad():
                return model.hidden(x)

    return ContextProbe()


def permuted_probe(probe, permutation):
    """The upstream probe with its feature index remapped. run_audit asks the
    probe for feature i's activations; under the null it is handed feature
    permutation[i]'s instead, while matching, the readout dimension and the
    recovery bar are untouched."""

    class PermutedProbe:
        def activations_with_feature(self, feature_idx, n_samples):
            return probe.activations_with_feature(
                permutation[feature_idx], n_samples)

        def activations_without_feature(self, feature_idx, n_samples):
            return probe.activations_without_feature(
                permutation[feature_idx], n_samples)

    return PermutedProbe()


# ---------------------------------------------------------------------------
# Finder
# ---------------------------------------------------------------------------

def claim_label(rate_good, rate_bad):
    if rate_good > 0 and rate_bad > 0:
        presence = "inert in both"
    elif rate_good > 0:
        presence = "inert in well-trained only"
    elif rate_bad > 0:
        presence = "inert in degraded only"
    else:
        presence = "no inert recovered features"
    ordering = ("degraded >= well-trained" if rate_bad >= rate_good
                else "well-trained > degraded")
    return f"{presence}; {ordering}"


def make_finder(audit_module, toy, raw_path):
    """The upstream audit as a pure function of (data, seed, config).

    Training, matching, firing and specificity are upstream code; this wrapper
    fixes the toy model, routes the run's seed into upstream's own seed
    arguments, applies the recovery bar and the cohort, names the components and
    logs the raw census. Trained models and audit reports are memoised on their
    full (seed, config, permutation) key -- a cache, never a change of value."""
    import torch

    models, saes, audits = {}, {}, {}

    def toy_model(config):
        key = (int(config["toy_steps"]),)
        if key not in models:
            model = toy.train_toy_model(toy.ToyConfig(seed=0),
                                        steps=int(config["toy_steps"]))
            mask = toy.well_represented_mask(model)
            models[key] = (model, mask)
        return models[key]

    def sae_for(name, seed, config):
        model, _ = toy_model(config)
        k = int(config[SAE_LABELS[name]])
        key = (name, int(seed), k, int(config["d_sae"]), int(config["sae_steps"]),
               int(config["toy_steps"]))
        if key not in saes:
            saes[key] = toy.train_topk_sae(
                model, d_sae=int(config["d_sae"]), k=k,
                steps=int(config["sae_steps"]), seed=int(seed))
        return saes[key]

    def audit_for(name, seed, config, permutation, regime, sparsity):
        key = (name, int(seed), int(config[SAE_LABELS[name]]),
               int(config["d_sae"]), int(config["sae_steps"]),
               int(config["toy_steps"]), int(config["n_samples"]), permutation,
               regime, sparsity)
        if key not in audits:
            model, _ = toy_model(config)
            probe = (toy.ToyProbe(model, seed=int(seed)) if regime == "isolated"
                     else context_probe(torch, model, int(seed),
                                        background_sparsity=sparsity))
            if permutation is not None:
                probe = permuted_probe(probe, permutation)
            report = audit_module.run_audit(
                sae=sae_for(name, seed, config),
                downstream=model.output,
                probe=probe,
                true_directions=toy.true_directions(model),
                config=audit_module.AuditConfig(
                    n_samples_on=int(config["n_samples"]),
                    n_samples_off=int(config["n_samples"]),
                    cosine_threshold=float(config["cosine_bar"]),
                    seed=int(seed)),
                metadata={"sae": name, "regime": "toy"})
            audits[key] = {r.feature_idx: r for r in report.results}
        return audits[key]

    def finder(data, seed, config):
        if not data:
            raise ValueError("empty audit cohort")
        permutation = data[0]["probe_permutation"]
        regime = data[0]["probe_regime"]
        sparsity = data[0]["background_sparsity"]
        bar = float(config["cosine_bar"])
        per_sae = {name: {"recovered": 0, "inert": 0} for name in SAE_LABELS}
        inert_components, cosines = set(), []
        rows = []
        for record in data:  # a multiset: bootstrap can repeat a cohort member
            result = audit_for(record["sae"], seed, config, permutation,
                               regime, sparsity)[record["feature_idx"]]
            if result.cosine < bar:
                continue
            per_sae[record["sae"]]["recovered"] += 1
            if result.causally_inert:
                per_sae[record["sae"]]["inert"] += 1
                inert_components.add(f"{record['sae']}:f{record['feature_idx']}")
                cosines.append(float(result.cosine))
            rows.append([record["sae"], int(record["feature_idx"]),
                         round(float(result.cosine), 6),
                         round(float(result.fired_frac), 6),
                         bool(result.causally_inert)])

        def rate(name):
            counts = per_sae[name]
            return counts["inert"] / counts["recovered"] if counts["recovered"] else 0.0

        recovered = sum(c["recovered"] for c in per_sae.values())
        inert = sum(c["inert"] for c in per_sae.values())
        pooled = inert / recovered if recovered else 0.0
        claim = claim_label(rate("good_k4"), rate("bad_k13"))

        record = {
            "cohort": data[0]["cohort"], "seed": int(seed), "config": dict(config),
            "permuted": permutation is not None, "probe_regime": regime,
            "background_sparsity": sparsity,
            "components": sorted(inert_components), "rows": rows,
            "n_recovered": recovered, "n_inert": inert, "claim": claim,
        }
        digest = hashlib.sha256(
            json.dumps(record, sort_keys=True).encode()).hexdigest()[:16]
        record["census_sha256_16"] = digest
        with open(raw_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

        return sk.feature_set(
            sorted(inert_components),
            claim=claim,
            score=round(pooled, 6),
            universe_size=UNIVERSE,
            cohort=data[0]["cohort"],
            cohort_size=len(data),
            cosine_bar=bar,
            probe_regime=regime,
            background_sparsity=sparsity,
            permuted_probe=permutation is not None,
            n_recovered=recovered,
            n_inert=inert,
            good_recovered=per_sae["good_k4"]["recovered"],
            good_inert=per_sae["good_k4"]["inert"],
            good_inert_rate=round(rate("good_k4"), 6),
            bad_recovered=per_sae["bad_k13"]["recovered"],
            bad_inert=per_sae["bad_k13"]["inert"],
            bad_inert_rate=round(rate("bad_k13"), 6),
            min_inert_cosine=round(min(cosines), 6) if cosines else None,
            max_inert_cosine=round(max(cosines), 6) if cosines else None,
            census_sha256_16=digest,
        )

    finder.toy_model = toy_model
    finder.torch_version = torch.__version__
    return finder


# ---------------------------------------------------------------------------
# Post-hoc notes and random samples
# ---------------------------------------------------------------------------

def released_row_note(result, expected):
    """Did the base run reproduce the released census? Stated per SAE, against
    expected_results.json's own values and tolerances."""
    meta = result.base.meta
    lines = []
    for name, key_recovered, key_inert in (
            ("good_k4", "good_recovered", "good_inert"),
            ("bad_k13", "bad_recovered", "bad_inert")):
        row = expected[name]

        def value_and_tolerance(field):
            entry = row[field]
            if isinstance(entry, dict):
                return float(entry["value"]), float(entry.get("atol", 0.0))
            return float(entry), 0.0

        exp_rec, tol_rec = value_and_tolerance("recovered")
        exp_inert, tol_inert = value_and_tolerance("recovered_inert")
        got_rec, got_inert = meta[key_recovered], meta[key_inert]
        ok_rec = abs(got_rec - exp_rec) <= tol_rec
        ok_inert = abs(got_inert - exp_inert) <= tol_inert
        lines.append(
            f"{name}: recovered {got_rec} vs released {exp_rec:g}"
            f"{f' +/-{tol_rec:g}' if tol_rec else ''} "
            f"({'reproduced' if ok_rec else 'NOT reproduced'}); "
            f"causally inert {got_inert} vs released {exp_inert:g}"
            f"{f' +/-{tol_inert:g}' if tol_inert else ''} "
            f"({'reproduced' if ok_inert else 'NOT reproduced'})")
    return ("released census: the base run (seed 0, upstream defaults, "
            "upstream's 22 well-represented features) against "
            f"expected_results.json at {UPSTREAM_COMMIT[:7]} -- " + "; ".join(lines))


def cohort_note(result):
    by_cohort = {}
    for run in result.runs:
        by_cohort.setdefault(run.finding.meta["cohort"], []).append(run.finding)
    parts = []
    for cohort, findings in sorted(by_cohort.items()):
        parts.append(
            f"{cohort} (n={len(findings)}): pooled inert rate "
            f"{statistics.median(f.score for f in findings):.3f} median, "
            f"good {statistics.median(f.meta['good_inert_rate'] for f in findings):.3f}, "
            f"degraded {statistics.median(f.meta['bad_inert_rate'] for f in findings):.3f}")
    return "inert rate by eligibility cohort (not graded): " + "; ".join(parts)


def axis_note(result):
    parts = []
    for axis in ("seeds", "bootstrap", "templates", "hyperparams"):
        group = [run.finding for run in result.runs if run.axis == axis]
        if not group:
            continue
        good = [f.meta["good_inert_rate"] for f in group]
        bad = [f.meta["bad_inert_rate"] for f in group]
        parts.append(
            f"{axis} (n={len(group)}): good {np.mean(good):.3f}+/-{np.std(good):.3f}, "
            f"degraded {np.mean(bad):.3f}+/-{np.std(bad):.3f}, "
            f"census size {min(f.size for f in group)}-{max(f.size for f in group)}")
    return ("per-SAE inert rates by axis (not graded; the claim sentence quotes "
            "9% for the well-trained SAE and up to 77% for the degraded one): "
            + "; ".join(parts))


def presentation_note(result):
    """The claim's verb is "never fires when the feature is present". Presenting
    the feature alone, with every other input coordinate at exactly zero, and
    presenting it inside a normal sparse background are both readings of
    "present"; upstream's census uses the first. This separates them."""
    isolated, in_context = [], []
    for run in result.runs:
        finding = run.finding
        target = in_context if finding.meta["probe_regime"] == "in-context" else isolated
        target.append(finding)
    if not in_context:
        return None

    def summarise(findings):
        return (f"n={len(findings)}, census {np.mean([f.size for f in findings]):.1f} "
                f"pairs on average, pooled inert rate "
                f"{np.mean([f.score for f in findings]):.3f}")

    return ("presentation regime (not graded, and the largest single effect in this "
            f"battery): with the feature presented in isolation as upstream does "
            f"({summarise(isolated)}); with the same feature presented inside a "
            f"sparse background, which is how upstream's own real-model regime "
            f"defines feature-ON ({summarise(in_context)}). Most of the census is "
            "specific to the isolation regime: an atom that never wins the TopK "
            "competition for a feature presented alone can still win it when the "
            "feature arrives in company. This is a statement about the scope of the "
            "measurement, not about whether the isolated-regime census is correct -- "
            "it is upstream's declared regime and it reproduces there")


def cosine_note(result):
    """"including matches at cosine ~1.000" is an existential claim: at least one
    inert pair sits at effectively perfect cosine. Grade it as one -- report the
    largest inert cosine per run, and the spread the census covers."""
    highs = [run.finding.meta["max_inert_cosine"] for run in result.runs
             if run.finding.meta["max_inert_cosine"] is not None]
    lows = [run.finding.meta["min_inert_cosine"] for run in result.runs
            if run.finding.meta["min_inert_cosine"] is not None]
    empty = sum(1 for run in result.runs if run.finding.size == 0)
    if not highs:
        return ("inert-pair geometry (not graded): no run produced a non-empty "
                "census, so the sentence's cosine claim is untested here")
    near_one = sum(1 for value in highs if value >= 0.999)
    return (f"inert-pair geometry (not graded): across {len(highs)} real runs with a "
            f"non-empty census, {near_one}/{len(highs)} contain at least one inert "
            f"pair at cosine >= 0.999 (largest inert cosine {np.mean(highs):.4f} on "
            f"average, min over runs {np.min(highs):.4f}), which is the existential "
            f"reading the claim sentence makes; the census is not confined to those "
            f"pairs, and its lowest inert cosine is {np.mean(lows):.4f} on average "
            f"(min {np.min(lows):.4f}); {empty} run(s) produced an empty census")


def null_note(result):
    if not result.null_runs:
        return None
    sizes = [run.finding.size for run in result.null_runs]
    scores = [run.finding.score for run in result.null_runs]
    claims = {}
    for run in result.null_runs:
        claims[run.finding.claim] = claims.get(run.finding.claim, 0) + 1
    real_base = result.base.components
    overlaps = [len(set(run.finding.components) & set(real_base)) for run in result.null_runs]
    return (f"null-control census: over {len(result.null_runs)} null runs the census "
            f"holds {np.mean(sizes):.1f} pairs on average (range {min(sizes)}-{max(sizes)}, "
            f"real base {len(real_base)}), pooled inert rate "
            f"{np.mean(scores):.3f}+/-{np.std(scores):.3f}, and shares "
            f"{np.mean(overlaps):.1f} pairs with the real base census; claims: "
            + ", ".join(f"`{k}` x{v}" for k, v in sorted(claims.items(), key=lambda kv: -kv[1])))


def specificity_basis_note(result):
    """How much null data the graded specificity ratio actually rests on."""
    null_base = next(run.finding for run in result.null_runs if run.axis == "base")
    sets = [run.finding.components for run in result.null_runs]
    sized = [s for s in sets if null_base.size / 2 <= len(s) <= null_base.size * 2]
    graded = sized if sized else sets
    jaccard_all = sk.metrics.mean_pairwise_jaccard(sets)
    jaccard_real = result.pooled["mean_pairwise_jaccard"]
    ratio_all = (f"{jaccard_real / jaccard_all:.2f}x"
                 if jaccard_all and jaccard_all > 1e-9 else "unbounded")
    return (f"specificity basis: the null base run selected {null_base.size} pairs, so "
            f"StressKit's 2x size guard grades null Jaccard on {len(graded)}/{len(sets)} "
            f"null runs = {result.null_summary['mean_pairwise_jaccard']:.3f}; over all "
            f"{len(sets)} null runs the null Jaccard is {jaccard_all:.3f} and the "
            f"real/null ratio would be {ratio_all}")


def fmt_ci(ci):
    return "no CI" if ci is None else f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def posthoc_regrade_note(result):
    """The verdict trace regrades in post-hoc mode; say where that disagrees
    with the card so a reader is not surprised by two confidence labels."""
    full = sk.from_findings(
        [run.finding for run in result.runs],
        null_findings=[run.finding for run in result.null_runs]
        if result.null_runs else None,
        seed=0)
    diffs = []
    for name, check in full.checks.items():
        card_check = result.checks.get(name)
        if card_check is None:
            continue
        if check["state"] != card_check["state"] or check["passed"] != card_check["passed"]:
            diffs.append(f"{name}: card {card_check['state']} (CI "
                         f"{fmt_ci(card_check['ci'])}, value {card_check['value']:.3f}) "
                         f"vs post-hoc {check['state']} (CI {fmt_ci(check['ci'])}, "
                         f"value {check['value']:.3f})")
    return (f"post-hoc regrade (verdict-trace mode, from_findings at bootstrap seed 0): "
            f"grade {full.grade}, {full.pooled['confidence']} confidence vs the card's "
            f"{result.pooled['confidence']}; "
            + (("checks that differ -- " + "; ".join(diffs)) if diffs
               else "every check has the same state as on the card")
            + " -- a check whose CI end sits at its bar is decided by the bootstrap "
              "seed, not by the data")


def run_row(record, group):
    finding = record.finding
    return {
        "group": group, "axis": record.axis, "variant": record.variant,
        "seed": record.seed, "config": record.config, "claim": finding.claim,
        "score": finding.score, "size": finding.size,
        "components": sorted(str(c) for c in finding.components),
        "meta": finding.meta,
    }


def write_samples(path, result, raw_by_digest, seed=0):
    rng = random.Random(seed)
    base_raw = raw_by_digest[result.base.meta["census_sha256_16"]]
    lines = [
        "# Randomly selected raw records (base run, battery runs, null runs)", "",
        "Selected with `random.Random(0)`, not cherry-picked. A pair is "
        "*recovered* when its unsigned cosine to the best decoder atom clears the "
        "bar, and *causally inert* when the matched atom's code is nonzero on "
        "exactly none of the feature-ON samples (`fired_frac = 0`).", "",
        f"## Base run (seed {BASE_SEED}, upstream defaults, cohort "
        f"`{result.base.meta['cohort']}`)", "",
        "Every inert pair first (deterministic, not sampled), then six randomly "
        "selected recovered-but-firing pairs.", "",
        "| SAE | feature | cosine | fired_frac | causally inert |",
        "|---|---|---|---|---|",
    ]
    inert = [row for row in base_raw["rows"] if row[4]]
    firing = [row for row in base_raw["rows"] if not row[4]]
    for row in sorted(inert) + sorted(rng.sample(firing, min(6, len(firing)))):
        lines.append(f"| {row[0]} | f{row[1]} | {row[2]:.4f} | {row[3]:.4f} | "
                     f"{'yes' if row[4] else 'no'} |")
    lines += ["", "## Five randomly selected perturbed runs", "",
              "| axis | variant | census | pooled inert rate | good | degraded | claim |",
              "|---|---|---|---|---|---|---|"]
    perturbed = [run for run in result.runs if run.axis != "base"]
    for run in rng.sample(perturbed, min(5, len(perturbed))):
        finding = run.finding
        lines.append(
            f"| {run.axis} | {run.variant} | "
            f"{', '.join(sorted(finding.components)) or '(empty)'} | {finding.score:.4f} | "
            f"{finding.meta['good_inert']}/{finding.meta['good_recovered']} | "
            f"{finding.meta['bad_inert']}/{finding.meta['bad_recovered']} | {finding.claim} |")
    if result.null_runs:
        lines += ["", "## Three randomly selected null-control runs "
                  "(feature-to-probe pairing permuted)", "",
                  "| axis | variant | census size | pooled inert rate | good | degraded | claim |",
                  "|---|---|---|---|---|---|---|"]
        for run in rng.sample(result.null_runs, min(3, len(result.null_runs))):
            finding = run.finding
            lines.append(
                f"| {run.axis} | {run.variant} | {finding.size} | {finding.score:.4f} | "
                f"{finding.meta['good_inert']}/{finding.meta['good_recovered']} | "
                f"{finding.meta['bad_inert']}/{finding.meta['bad_recovered']} | "
                f"{finding.claim} |")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-dir", default="sae_causal_audit_upstream",
                        help="checkout of the pinned upstream commit, or a directory "
                             "the needed files are downloaded into (SHA-256 verified "
                             "either way)")
    parser.add_argument("--out-dir",
                        default=os.path.join(os.path.dirname(__file__), "cards"))
    parser.add_argument("--raw-dir", default=None,
                        help="where per-run censuses are saved "
                             "(default: <out-dir>/raw/sae_causal_inertness)")
    parser.add_argument("--n-runs", type=int, default=8)
    args = parser.parse_args()

    raw_dir = args.raw_dir or os.path.join(args.out_dir, "raw", "sae_causal_inertness")
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)
    raw_path = os.path.join(raw_dir, "censuses.jsonl")
    open(raw_path, "w").close()

    audit_module, toy = import_upstream(args.upstream_dir)
    expected = expected_rows(args.upstream_dir)

    finder = make_finder(audit_module, toy, raw_path)
    mask = finder.toy_model(BASE_CONFIG)[1]
    well_represented = [i for i, flag in enumerate(mask.tolist()) if flag]
    print(f"toy model: {len(well_represented)}/{N_FEATURES} well-represented features")

    pool = make_cohort(well_represented, "released-22-well-represented")
    permutation = derangement(N_FEATURES, NULL_PERMUTATION_SEED)
    null_pool = make_cohort(well_represented,
                            "released-22-well-represented (probe permuted)",
                            permutation=permutation)

    # Templates swap the probe dataset: how the feature is presented when the
    # audit asks whether its atom ever fires, and which pairs are eligible.
    # "The matched atom never fires when the feature is present" is a claim
    # about a presentation regime as much as about an atom, and upstream fixes
    # the regime (one feature at a time, everything else exactly zero) without
    # revisiting it. The feature-index universe is unchanged throughout.
    #
    # Two weaker template axes were tried first and rejected before grading,
    # for the same reason in both cases -- they were copies of the base run:
    # upstream's well_represented_mask threshold (this toy model's feature norms
    # are bimodal, so 0.05, 0.10 and 0.25 all select the same 22 features), and
    # the probe's feature-ON magnitude (TopK selection is near scale-invariant,
    # so magnitudes 0.5, 2.0 and 4.0 all returned the base census exactly).
    templates = {
        "in-context-presentation": make_cohort(
            well_represented, "in-context-presentation", regime="in-context"),
        "in-context-dense-background": make_cohort(
            well_represented, "in-context-dense-background", regime="in-context",
            background_sparsity=0.80),
        "all-32-features": make_cohort(range(N_FEATURES), "all-32-features"),
        "all-32-in-context": make_cohort(
            range(N_FEATURES), "all-32-in-context", regime="in-context"),
    }
    for label, cohort in templates.items():
        print(f"template {label}: {len(cohort)} cohort records")

    result = sk.stress(
        finder,
        pool,
        battery=["seeds", "bootstrap", "templates", "hyperparams"],
        n_runs=args.n_runs,
        seed=BASE_SEED,
        config=BASE_CONFIG,
        templates=templates,
        hyperparams={"cosine_bar": [0.85, 0.95], "n_samples": [200],
                     "sae_steps": [2000]},
        null_data=null_pool,
        claim_statement=CLAIM,
        model=("Elhage et al. (2022) toy bottleneck model trained by the released "
               "harness (32 features, 8 hidden dims, sparsity 0.95, seed 0) with two "
               "TopK SAEs on its hidden activations (d_sae 128, k=4 well-trained and "
               f"k=13 degraded); {UPSTREAM_REPO}@{UPSTREAM_COMMIT[:7]}"),
        task=("census of causally inert pairs among cosine-recovered "
              "(ground-truth direction, decoder atom) matches"),
        method=("signed cosine matching against W_dec, then per-pair fired_frac / "
                "ablation / steering measurement through encode+decode, upstream "
                "sae_causal_audit.run_audit"),
        verbose=True,
    )

    result.card.notes.append(
        "scope: graded artifact = the released toy-regime pipeline (Elhage-style "
        "bottleneck model at ToyConfig(seed=0), two TopK SAEs at k=4 and k=13, "
        "d_sae 128) audited by upstream's own run_audit; usage mode = the "
        "known-ground-truth toy regime that the paper's Table 5 census is computed "
        "in. The battery does NOT test the real-model regime (GPT-2-small, 83 "
        "concepts, gpt2-small-res-jb), the superposition phase-diagram or "
        "TopK-versus-L1 reproductions, the ablation/steering specificity medians, "
        "or the read-inert versus write-inert taxonomy; and the 77% in the claim "
        "sentence is the originating write-up's figure, which the released "
        "instrument does not recompute -- Table 5's own degraded-SAE census is 17%")
    result.card.notes.append(
        f"data: no external dataset; every number is regenerated on CPU from "
        f"{UPSTREAM_REPO}@{UPSTREAM_COMMIT[:7]} (MIT), each imported source file "
        f"SHA-256 verified against this runner before import. torch "
        f"{finder.torch_version}; upstream guarantees byte-exact results only "
        "inside its pinned CI environment (ubuntu-24.04, torch==2.13.0+cpu) and "
        "semantic agreement within rtol=1e-4 elsewhere")
    result.card.notes.append(
        "finding representation pre-registered before any run: components = "
        "recovered-and-inert (SAE, feature) pairs over a universe of "
        f"{UNIVERSE} pairs; score = pooled inert rate among recovered pairs; claim "
        "= presence of inertness in each SAE plus which SAE has the higher rate, "
        "both read directly off the claim sentence with no free threshold")
    result.card.notes.append(released_row_note(result, expected))
    result.card.notes.append(
        f"null control: the feature-to-probe pairing permuted once by a derangement "
        f"(seed {NULL_PERMUTATION_SEED:#x}), so every matched atom is asked whether it "
        "fires for a ground-truth feature it was not matched to, while matching, the "
        "readout dimension and the recovery bar stay untouched. Direction: the null is "
        "strict rather than conservative. Inertness is the absence of an effect, so "
        "breaking the pairing pushes the census toward saturation -- a large, "
        "near-identical set every run, which inflates null Jaccard and therefore "
        "depresses the specificity ratio. Read a specificity failure here as 'the "
        "identity of the inert pairs is not more stable than a saturated census', not "
        "as evidence that the real census is random; StressKit's separate beats-random "
        "check already carries the size-matched random comparison")
    result.card.notes.append(cohort_note(result))
    presentation = presentation_note(result)
    if presentation:
        result.card.notes.append(presentation)
    result.card.notes.append(axis_note(result))
    result.card.notes.append(cosine_note(result))
    null_summary = null_note(result)
    if null_summary:
        result.card.notes.append(null_summary)
        result.card.notes.append(specificity_basis_note(result))
    result.card.notes.append(posthoc_regrade_note(result))

    print()
    print(result)
    print(result.to_markdown())

    base = os.path.join(args.out_dir, "sae_causal_inertness")
    result.card.save(base + ".json")
    with open(base + ".md", "w", encoding="utf-8") as handle:
        handle.write(result.to_markdown() + "\n")
    with open(base + ".badge.json", "w", encoding="utf-8") as handle:
        json.dump(result.card.badge_dict(), handle, indent=2)
        handle.write("\n")
    print("\ncomputing verdict-stability trace ...")
    trace = result.verdict_trace(seed=0)
    with open(base + ".trace.json", "w", encoding="utf-8") as handle:
        json.dump(trace, handle, indent=2)
        handle.write("\n")
    with open(base + ".trace.md", "w", encoding="utf-8") as handle:
        handle.write(sk.verdict_trace_markdown(trace) + "\n")
    raw_by_digest = {}
    with open(raw_path, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            raw_by_digest[row["census_sha256_16"]] = row
    write_samples(base + ".samples.md", result, raw_by_digest)
    rows = [run_row(r, "real") for r in result.runs] + \
        [run_row(r, "null") for r in (result.null_runs or [])]
    with open(base + ".runs.json", "w", encoding="utf-8") as handle:
        json.dump({"upstream": f"{UPSTREAM_REPO}@{UPSTREAM_COMMIT}",
                   "raw_dir": os.path.relpath(raw_dir, args.out_dir),
                   "runs": rows}, handle, indent=1)
        handle.write("\n")
    print(sk.verdict_trace_markdown(trace))
    print(f"\nartifacts written to {base}.*")


if __name__ == "__main__":
    main()
