"""Reference Stability Card: the homonym/polyseme reconvergence profile
(arXiv:2608.01816, Scott, Pat & Liesaputra 2026).

Claim under test (registry statement, verbatim): "Homonym and polyseme
representations become maximally distinct in middle layers and partially
reconverge in late layers, while the KL divergence between their next-token
predictions peaks in the final layers."

Finder = the upstream Experiment-1 logit-lens profile
(``Experiment 1 - Logit lens/Unified_logit_lens_experiment_v5.py`` at commit
dd08338) re-implemented on plain Hugging Face models as a pure function of
(data, seed, config). ``data`` is a list of word-pair stimuli, one item per
unordered pair of validated contexts of one word (so bootstrap resamples
pairs); the seed draws a 75% subsample of the items. Per item and layer the
finder computes the upstream quantities at the target-word token of the
post-block residual stream (TransformerLens ``hook_resid_post``): cosine
distance between the two residuals ("activation distance"), cosine distance
between their unembedded projections ("logit distance"; the upstream raw
lens = residual x final-norm gain x W_U with no normalisation, logits centred
over the vocabulary as TransformerLens ``center_unembed`` does, residuals
mean-centred for LayerNorm models as ``center_writing_weights`` does) and the
symmetrised KL divergence between the softmaxes of those projections
(upstream ``safe_kl_div`` numerics). Items are aggregated as upstream does:
mean over a word's pairs, then mean (or median) across words, giving one
curve per metric over the L layers.

Finding representation (fixed before any run):

- components: the "maximal distinctness band" = the top-k layers of the
  activation-distance curve, k = max(3, round(0.2 L)); universe = L layers.
- score: reconvergence ratio r = mean activation distance over the final
  band (last max(2, round(0.1 L)) layers) / peak activation distance.
- claim: "<zone> peak; <late reconvergence | no late reconvergence>;
  KL peak <final | zone>". zone = majority manuscript layer group of the
  band (paper section 3.6: GPT-2 0-3/4-8/9-11, Llama-3.2-3B 0-6/7-16/17-27,
  Qwen2.5-32B 0-14/15-48/49-63; other depths use Llama's fractions); late
  reconvergence iff r <= 0.9 (the smallest late-vs-middle decline the paper
  reports is 13%); KL peak is "final" iff the KL curve's argmax lies in the
  final band. The paper's claim is the label
  "middle peak; late reconvergence; KL peak final".

Battery: seeds (75% item subsample), bootstrap (pair resampling), templates
(the upstream concept-drift polyseme set; the homonym set read one token
after the target word), hyperparams (band metric = logit distance; lens =
normed, i.e. the final norm applied before unembedding as the standard logit
lens does; aggregation = median across words; band size k = round(0.15 L)
and round(0.25 L)).

Null control (graded): the homonym items with the pairing permuted across
words (sentence A of word w paired with sentence B of a different word w',
each read at its own validated target token), which removes the
shared-surface-form condition the claim is about while every sentence,
position and computation stays unchanged. Alternative null (reported in the
notes): the upstream sequence-order control set (matching tokens in
reordered sentence pairs, Control_analysis/
Sequence_order_and_transition_analysis.py) through the same finder.

Data: upstream stimulus files at commit dd08338 (MIT), fetched from the
pinned commit and SHA-256 verified; parsed as literals, no upstream code is
executed.

Usage (GPU):
    python references/run_homonym_reconvergence_card.py \
        --model unsloth/Llama-3.2-3B --data-dir homonym_data \
        --out-dir references/cards --cache-dir .cache
"""

import argparse
import ast
import hashlib
import json
import math
import os
import random
import urllib.parse
import urllib.request
import warnings

import numpy as np
import torch
import transformers

import stresskit as sk

UPSTREAM_REPO = "scoki211/Divergent_LLM_Predictions_Convergent_Reps_amb_words"
UPSTREAM_COMMIT = "dd083383752da7ec7725d290dae13925bd8b852b"
UPSTREAM_FILES = {
    "homonyms": (
        "Experiment 1 - Logit lens/validated_homonym_dict.py", "ALL_HOMONYMS",
        "f4082968f539de2e866bad329162eb2034bd5404271aeb542f5a4f4581649663"),
    "polysemes": (
        "Experiment 1 - Logit lens/validated_concept_drift_dict.py",
        "CONCEPT_DRIFT_POLYSEMES",
        "ce8b13dbce593e4eca41279715d8249b31bfc9471757ef926fe3572804216c87"),
    "sequence_order": (
        "Control_analysis/Sequence_order_and_transition_analysis.py",
        "sequence_pairs",
        "567ced4f3c09f475941d2a3bfea1c08409b0efcd5db621882b21bfb092cd09e0"),
}
CLAIM = ("Homonym and polyseme representations become maximally distinct in "
         "middle layers and partially reconverge in late layers, while the KL "
         "divergence between their next-token predictions peaks in the final "
         "layers.")
PAPER_CLAIM_LABEL = "middle peak; late reconvergence; KL peak final"
# paper Table 1: validated logit-lens stimuli (homonyms, polysemes)
PAPER_VALIDATED = {"gpt2": (167, 93), "llama-3.2-3b": (166, 94), "qwen2.5-32b": (156, 86)}
K_FRAC, K_MIN, K_ALT_FRACS = 0.2, 3, (0.15, 0.25)
FINAL_FRAC, FINAL_MIN = 0.1, 2
RECONVERGENCE_MAX_RATIO = 0.9
SUBSAMPLE_FRAC = 0.75
# paper section 3.6 layer groups: (last early layer, last middle layer) by depth
MANUSCRIPT_ZONES = {12: (3, 8), 28: (6, 16), 64: (14, 48)}
ZONE_FRACS = (7 / 28, 17 / 28)  # Llama-3.2-3B's boundaries, for other depths
KL_EPS, KL_RATIO_MAX, KL_MASS_MIN = 1e-10, 1e8, 1e-6  # upstream safe_kl_div
SEQ_EXCLUDE = {".", ","}  # upstream extract_pair_activations
NULL_SEED = 0x5EC


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def fetch_upstream(data_dir, key):
    """Fetch one upstream source file from the pinned commit, verify its
    SHA-256 and return the named top-level literal it assigns."""
    rel, name, digest = UPSTREAM_FILES[key]
    path = os.path.join(data_dir, os.path.basename(rel))
    if not os.path.exists(path):
        os.makedirs(data_dir, exist_ok=True)
        url = (f"https://raw.githubusercontent.com/{UPSTREAM_REPO}/"
               f"{UPSTREAM_COMMIT}/{urllib.parse.quote(rel)}")
        urllib.request.urlretrieve(url, path)
    with open(path, "rb") as f:
        raw = f.read()
    got = hashlib.sha256(raw).hexdigest()
    if got != digest:
        raise RuntimeError(
            f"{path}: sha256 {got} does not match the pinned upstream file "
            f"{digest} (commit {UPSTREAM_COMMIT[:7]})")
    tree = ast.parse(raw.decode("utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise RuntimeError(f"{path}: no top-level assignment to {name}")


# ---------------------------------------------------------------------------
# Model access
# ---------------------------------------------------------------------------

class Subject:
    """A causal LM with the upstream logit-lens quantities exposed."""

    def __init__(self, name, device, dtype):
        self.name = name
        self.device = device
        self.tok = transformers.AutoTokenizer.from_pretrained(name)
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            name, dtype=dtype, device_map=device).eval()
        self.dtype = str(dtype).replace("torch.", "")
        self.revision = getattr(self.model.config, "_commit_hash", None)
        lists = [(n, m) for n, m in self.model.named_modules()
                 if isinstance(m, torch.nn.ModuleList) and len(m) >= 6]
        if len(lists) != 1:
            raise RuntimeError(f"{name}: expected one decoder layer list, found "
                               f"{[n for n, _ in lists]}")
        owner = self.model.get_submodule(lists[0][0].rsplit(".", 1)[0])
        self.layers = lists[0][1]
        self.n_layers = len(self.layers)
        self.final_norm = None
        for attr in ("norm", "ln_f"):
            if hasattr(owner, attr):
                self.final_norm = getattr(owner, attr)
        if self.final_norm is None:
            raise RuntimeError(f"{name}: could not locate the final norm")
        if type(self.final_norm).__name__.lower().startswith("gemma"):
            raise RuntimeError("Gemma-style (1 + w) norms are not handled")
        self.is_layernorm = isinstance(self.final_norm, torch.nn.LayerNorm)
        self.norm_eps = getattr(self.final_norm, "eps", None)
        if self.norm_eps is None:
            self.norm_eps = getattr(self.final_norm, "variance_epsilon", None)
        if self.norm_eps is None:
            raise RuntimeError(f"{name}: final norm {type(self.final_norm).__name__} "
                               "exposes no epsilon")
        lm_head = self.model.get_output_embeddings()
        if getattr(lm_head, "bias", None) is not None:
            raise RuntimeError(f"{name}: unembedding bias is not handled")
        self.W_U = lm_head.weight.detach().float()             # [V, d]
        self.gamma = self.final_norm.weight.detach().float()   # [d]
        self.beta = (self.final_norm.bias.detach().float()
                     if self.is_layernorm and self.final_norm.bias is not None else None)
        self.vocab_size, self.d_model = self.W_U.shape
        self.bos_id = self.tok.bos_token_id
        if self.bos_id is None:
            self.bos_id = self.tok.eos_token_id
        self.special_ids = set(self.tok.all_special_ids)
        self._tok_cache = {}
        self._resid_cache = {}

    def tokenize(self, sentence):
        """BOS-prefixed ids and per-token strings (TransformerLens
        to_tokens / to_str_tokens conventions)."""
        if sentence not in self._tok_cache:
            ids = [self.bos_id] + self.tok(sentence, add_special_tokens=False).input_ids
            strs = [self.tok.decode([i]) for i in ids]
            self._tok_cache[sentence] = (ids, strs)
        return self._tok_cache[sentence]

    @torch.no_grad()
    def residuals(self, sentence):
        """[L, T, d] float32 post-block residual stream (hook_resid_post) on
        the CPU; mean-centred over d_model for LayerNorm models, as
        TransformerLens' center_writing_weights leaves it."""
        if sentence not in self._resid_cache:
            ids, _ = self.tokenize(sentence)
            captured = [None] * self.n_layers

            def make_hook(i):
                def hook(module, args, out):
                    captured[i] = (out[0] if isinstance(out, tuple) else out).detach()
                return hook

            handles = [layer.register_forward_hook(make_hook(i))
                       for i, layer in enumerate(self.layers)]
            try:
                self.model(torch.tensor([ids], device=self.device))
            finally:
                for h in handles:
                    h.remove()
            if any(c is None for c in captured):
                raise RuntimeError("a decoder layer produced no output")
            resid = torch.stack([c[0] for c in captured]).float()  # [L, T, d]
            if self.is_layernorm:
                resid = resid - resid.mean(dim=-1, keepdim=True)
            self._resid_cache[sentence] = resid.cpu()
        return self._resid_cache[sentence]

    def lens_logits(self, h, lens):
        """[n, d] residuals -> [n, V] logits centred over the vocabulary.
        'raw' is the upstream projection (gain-folded W_U, no normalisation);
        'normed' applies the model's final norm first."""
        if lens == "raw":
            x = h * self.gamma
        elif lens == "normed":
            if self.is_layernorm:
                mu = h.mean(dim=-1, keepdim=True)
                var = h.var(dim=-1, keepdim=True, unbiased=False)
                x = (h - mu) / torch.sqrt(var + self.norm_eps) * self.gamma
                if self.beta is not None:
                    x = x + self.beta
            else:
                x = h * torch.rsqrt(h.pow(2).mean(dim=-1, keepdim=True) + self.norm_eps)
                x = x * self.gamma
        else:
            raise ValueError(f"unknown lens {lens!r}")
        logits = x @ self.W_U.T
        return logits - logits.mean(dim=-1, keepdim=True)

    @torch.no_grad()
    def item_curves(self, item, lens):
        """Per-layer activation distance, logit distance and symmetrised KL
        for one word-pair item: three float64 arrays of length L."""
        ra = self.residuals(item["a"])[:, item["pos_a"] + item["offset"]].to(self.device)
        rb = self.residuals(item["b"])[:, item["pos_b"] + item["offset"]].to(self.device)
        act = 1.0 - torch.nn.functional.cosine_similarity(ra, rb, dim=-1)
        la, lb = self.lens_logits(ra, lens), self.lens_logits(rb, lens)
        logit = 1.0 - torch.nn.functional.cosine_similarity(la, lb, dim=-1)
        pa = torch.softmax(la.double(), dim=-1)
        pb = torch.softmax(lb.double(), dim=-1)
        kl = 0.5 * (safe_kl(pa, pb) + safe_kl(pb, pa))
        return (act.double().cpu().numpy(), logit.double().cpu().numpy(),
                kl.cpu().numpy())


def safe_kl(p, q):
    """Upstream safe_kl_div, row-wise on [L, V] float64 distributions."""
    p = p / p.sum(dim=-1, keepdim=True)
    q = q / q.sum(dim=-1, keepdim=True)
    p = p.clamp_min(KL_EPS)
    q = q.clamp_min(KL_EPS)
    p = p / p.sum(dim=-1, keepdim=True)
    q = q / q.sum(dim=-1, keepdim=True)
    ratio = (p / q).clamp(KL_EPS, KL_RATIO_MAX)
    kl = (p * ratio.log() * (p > KL_MASS_MIN)).sum(dim=-1)
    return torch.where(torch.isfinite(kl), kl, torch.zeros_like(kl))


# ---------------------------------------------------------------------------
# Stimuli -> items
# ---------------------------------------------------------------------------

def validate_word(subject, word, contexts):
    """Upstream validate_word_contexts: substring token match, first-position
    rule, consistent token string, >= 2 contexts. Returns (found, reason)."""
    found, first_position = [], False
    for sentence in contexts:
        _, strs = subject.tokenize(sentence)
        positions = [i for i, t in enumerate(strs) if word.lower() in t.strip().lower()]
        if not positions:
            continue
        pos = positions[0]
        if pos <= 1:
            first_position = True
            continue
        found.append((sentence, pos, strs[pos]))
    if not found:
        return None, "not_found"
    if len(found) < 2:
        return None, "first_position" if first_position else "insufficient_contexts"
    if len({t for _, _, t in found}) > 1:
        return None, "inconsistent_tokenization"
    return found, None


def pair_items(subject, word_dict, cond, offset=0):
    """One item per unordered pair of validated contexts of each word."""
    items, failures = [], {}
    for word, contexts in word_dict.items():
        found, why = validate_word(subject, word, contexts)
        if found is None:
            failures[why] = failures.get(why, 0) + 1
            continue
        if offset:
            found = [(s, p, t) for s, p, t in found
                     if p + offset < len(subject.tokenize(s)[0])]
            if len(found) < 2:
                failures["offset_out_of_range"] = failures.get("offset_out_of_range", 0) + 1
                continue
        for i in range(len(found)):
            for j in range(i + 1, len(found)):
                (sa, pa, tok), (sb, pb, _) = found[i], found[j]
                items.append({"word": word, "cond": cond, "token": tok,
                              "a": sa, "pos_a": pa, "b": sb, "pos_b": pb,
                              "offset": offset})
    return items, failures


def sequence_order_items(subject, pairs):
    """Upstream extract_pair_activations(target_word=None): every token
    string common to both orderings (last occurrence in sentence 1, first in
    sentence 2), excluding special tokens, '.', ',' and 1-character tokens."""
    items = []
    for s1, s2 in pairs:
        ids1, strs1 = subject.tokenize(s1)
        _, strs2 = subject.tokenize(s2)
        common = {}
        for i, t in enumerate(strs1):
            if t in strs2:
                common[t] = (i, strs2.index(t))
        for t, (i, j) in common.items():
            if ids1[i] in subject.special_ids or not t.strip() or t in SEQ_EXCLUDE \
                    or len(t.strip()) <= 1:
                continue
            items.append({"word": t, "cond": "sequence-order", "token": t,
                          "a": s1, "pos_a": i, "b": s2, "pos_b": j, "offset": 0})
    return items


def permuted_items(items, seed=NULL_SEED):
    """Pair sentence A of every item with sentence B of an item of a
    different word (seeded derangement over words)."""
    rng = random.Random(seed)
    order = list(range(len(items)))
    rng.shuffle(order)
    for _ in range(10 * len(items)):
        bad = [i for i, j in enumerate(order) if items[i]["word"] == items[j]["word"]]
        if not bad:
            break
        for i in bad:
            n = (i + 1) % len(order)
            order[i], order[n] = order[n], order[i]
    else:
        raise RuntimeError("could not derange the word pairing")
    out = []
    for i, j in enumerate(order):
        a, b = items[i], items[j]
        out.append({"word": a["word"], "cond": "permuted",
                    "token": f"{a['token']}|{b['token']}",
                    "a": a["a"], "pos_a": a["pos_a"], "b": b["b"], "pos_b": b["pos_b"],
                    "offset": 0})
    return out


# ---------------------------------------------------------------------------
# Finder
# ---------------------------------------------------------------------------

def zone_bounds(n_layers):
    if n_layers in MANUSCRIPT_ZONES:
        return MANUSCRIPT_ZONES[n_layers]
    return (round(ZONE_FRACS[0] * n_layers) - 1, round(ZONE_FRACS[1] * n_layers) - 1)


def zone(layer, n_layers):
    early_last, middle_last = zone_bounds(n_layers)
    if layer <= early_last:
        return "early"
    if layer <= middle_last:
        return "middle"
    return "late"


def final_band(n_layers):
    n_final = max(FINAL_MIN, round(FINAL_FRAC * n_layers))
    return list(range(n_layers - n_final, n_layers))


def base_k(n_layers):
    return max(K_MIN, round(K_FRAC * n_layers))


def alt_ks(n_layers):
    return sorted({max(2, round(f * n_layers)) for f in K_ALT_FRACS} - {base_k(n_layers)})


def aggregate(items, per_item, agg):
    """Upstream two-stage aggregation: mean over a word's pairs, then mean or
    median across words. KL values of exactly 0.0 are excluded, as upstream's
    aggregate_finite_only does."""
    by_word = {}
    for it, curves in zip(items, per_item):
        by_word.setdefault(it["word"], []).append(curves)
    word_curves = {"activation": [], "logit": [], "kl": []}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for rows in by_word.values():
            word_curves["activation"].append(np.mean([r[0] for r in rows], axis=0))
            word_curves["logit"].append(np.mean([r[1] for r in rows], axis=0))
            kls = np.stack([r[2] for r in rows])
            word_curves["kl"].append(np.nanmean(np.where(kls > 0.0, kls, np.nan), axis=0))
        reduce = np.nanmean if agg == "mean" else np.nanmedian
        return {m: reduce(np.stack(v), axis=0) for m, v in word_curves.items()}


def profile_labels(curves, config, n_layers):
    """Band, score and claim label from one set of aggregate curves."""
    metric_curve = curves[config["metric"]]
    if not np.isfinite(metric_curve).all():
        raise RuntimeError("non-finite values in the band metric curve")
    order = sorted(range(n_layers), key=lambda layer: (-metric_curve[layer], layer))
    k = int(config["k"])
    band, peak = order[:k], order[0]
    fb = final_band(n_layers)
    r = float(np.mean(metric_curve[fb]) / metric_curve[peak])
    zones = [zone(layer, n_layers) for layer in band]
    counts = {z: zones.count(z) for z in ("early", "middle", "late")}
    majority = [z for z, c in counts.items() if c == max(counts.values())]
    band_zone = majority[0] if len(majority) == 1 else zone(peak, n_layers)
    kl_curve = curves["kl"]
    kl_peak = int(np.nanargmax(kl_curve)) if np.isfinite(kl_curve).any() else None
    if kl_peak is None:
        kl_label = "undefined"
    elif kl_peak in fb:
        kl_label = "final"
    else:
        kl_label = zone(kl_peak, n_layers)
    reconv = ("late reconvergence" if r <= RECONVERGENCE_MAX_RATIO
              else "no late reconvergence")
    return {
        "band": band, "peak_layer": peak, "score": r, "kl_peak_layer": kl_peak,
        "claim": f"{band_zone} peak; {reconv}; KL peak {kl_label}",
    }


def curve_list(x):
    return [None if not np.isfinite(v) else round(float(v), 6) for v in x]


def make_finder(subject):
    n_layers = subject.n_layers

    def finder(data, seed, config):
        rng = random.Random(seed)
        n_sub = math.ceil(SUBSAMPLE_FRAC * len(data))
        sample = rng.sample(list(data), n_sub)
        per_item = [subject.item_curves(it, config["lens"]) for it in sample]
        curves = aggregate(sample, per_item, config["agg"])
        lab = profile_labels(curves, config, n_layers)
        return sk.feature_set(
            lab["band"],
            claim=lab["claim"],
            score=round(lab["score"], 4),
            universe_size=n_layers,
            peak_layer=lab["peak_layer"],
            kl_peak_layer=lab["kl_peak_layer"],
            k=int(config["k"]),
            metric=config["metric"],
            lens=config["lens"],
            agg=config["agg"],
            offset=sample[0]["offset"],
            condition=sample[0]["cond"],
            n_items=len(sample),
            n_words=len({it["word"] for it in sample}),
            activation_curve=curve_list(curves["activation"]),
            logit_curve=curve_list(curves["logit"]),
            kl_curve=curve_list(curves["kl"]),
        )

    return finder


# ---------------------------------------------------------------------------
# Post-hoc notes and raw records
# ---------------------------------------------------------------------------

def full_profile(subject, items, config):
    """Deterministic profile over ALL items (no subsample): per-item curves,
    per-word curves and aggregate curves, for the notes and samples."""
    per_item = [subject.item_curves(it, config["lens"]) for it in items]
    curves = aggregate(items, per_item, config["agg"])
    by_word = {}
    for it, c in zip(items, per_item):
        by_word.setdefault(it["word"], []).append(c[0])
    word_act = {w: np.mean(rows, axis=0) for w, rows in by_word.items()}
    return per_item, word_act, curves


def zone_stats(word_act, n_layers):
    """Mean and median of per-word activation distance over (word, layer)
    observations in each manuscript layer group (the paper's group numbers)."""
    out = {}
    for z in ("early", "middle", "late"):
        layers = [layer for layer in range(n_layers) if zone(layer, n_layers) == z]
        vals = np.concatenate([c[layers] for c in word_act.values()])
        out[z] = (float(np.mean(vals)), float(np.median(vals)))
    return out


def fmt_zones(stats):
    return "; ".join(f"{z} mean {m:.3f} / median {md:.3f}" for z, (m, md) in stats.items())


def profile_note(label, finding, n_layers):
    m = finding.meta
    act = m["activation_curve"]
    return (f"{label}: activation-distance peak at layer {m['peak_layer']} "
            f"({act[m['peak_layer']]:.3f}), band {sorted(finding.components)}, "
            f"final-band mean {np.mean([act[layer] for layer in final_band(n_layers)]):.3f}, "
            f"r = {finding.score:.3f}, KL argmax layer {m['kl_peak_layer']}, "
            f"label '{finding.claim}' ({m['n_items']} items, {m['n_words']} words)")


def claim_counts(runs):
    counts = {}
    for r in runs:
        counts[r.finding.claim] = counts.get(r.finding.claim, 0) + 1
    return ", ".join(f"'{c}' x{n}" for c, n in sorted(counts.items(), key=lambda kv: -kv[1]))


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a | b) else 1.0


def shared_label_parts(claim_a, claim_b):
    """The '; '-separated label components two claim labels agree on."""
    parts = [x for x, y in zip(claim_a.split("; "), claim_b.split("; ")) if x == y]
    return ", ".join(f"'{p}'" for p in parts) if parts else "none"


def run_records(runs):
    return [{"axis": r.axis, "variant": r.variant, "seed": r.seed, "config": r.config,
             "components": sorted(r.finding.components), "claim": r.finding.claim,
             "score": r.finding.score, "meta": r.finding.meta} for r in runs]


def write_samples(path, subject, datasets, base_config, seed=0):
    """Randomly selected raw items with their per-layer curves (base
    configuration, full item sets), plus the aggregate base curves."""
    rng = random.Random(seed)
    lines = ["# Randomly selected raw examples (base configuration)", "",
             "Selected with `random.Random(0)`, not cherry-picked. Activation distance "
             "= cosine distance between the two residuals at the read position; KL = "
             "symmetrised KL between the raw-lens next-token distributions. Layers 0..L-1 "
             "are post-block residuals.", ""]
    for name, (items, per_item, curves, n_show) in datasets.items():
        lines += [f"## {name} ({len(items)} items)", ""]
        lines.append("Aggregate curves (mean over words of per-word means):")
        lines.append("")
        lines.append("| layer | activation | logit | KL |")
        lines.append("|---|---|---|---|")
        for layer in range(subject.n_layers):
            lines.append(f"| {layer} | {curves['activation'][layer]:.4f} | "
                         f"{curves['logit'][layer]:.4f} | {curves['kl'][layer]:.3f} |")
        lines.append("")
        for idx in sorted(rng.sample(range(len(items)), min(n_show, len(items)))):
            it, (act, _, kl) = items[idx], per_item[idx]
            lines += [f"**{it['word']}** (token `{it['token']!r}`, read offset {it['offset']})",
                      f"- A: {it['a']} (position {it['pos_a']})",
                      f"- B: {it['b']} (position {it['pos_b']})",
                      f"- activation distance by layer: {[round(float(x), 3) for x in act]}",
                      f"- KL by layer: {[round(float(x), 2) for x in kl]}",
                      f"- item peak layer {int(np.argmax(act))}, KL argmax {int(np.argmax(kl))}",
                      ""]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dtype", default="auto",
                    choices=["auto", "float16", "bfloat16", "float32"],
                    help="auto = upstream's choice: float32 for GPT-2, float16 otherwise")
    ap.add_argument("--data-dir", default="homonym_data")
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "cards"))
    ap.add_argument("--raw-dir", default=None,
                    help="per-run curves and per-item base-run curves "
                         "(default: <out-dir>/raw/homonym_<slug>)")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--n-runs", type=int, default=12)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    slug = args.model.split("/")[-1].lower().replace("-", "_").replace(".", "p")
    raw_dir = args.raw_dir or os.path.join(args.out_dir, "raw", f"homonym_{slug}")
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)
    if args.dtype == "auto":
        dtype = torch.float32 if "gpt2" in args.model.lower() else torch.float16
    else:
        dtype = getattr(torch, args.dtype)

    homonym_dict = fetch_upstream(args.data_dir, "homonyms")
    polyseme_dict = fetch_upstream(args.data_dir, "polysemes")
    sequence_pairs = fetch_upstream(args.data_dir, "sequence_order")

    print(f"loading {args.model} ({dtype}) on {args.device} ...")
    subject = Subject(args.model, args.device, dtype)
    L = subject.n_layers
    print(f"{L} layers, d_model {subject.d_model}, vocab {subject.vocab_size}, "
          f"{'LayerNorm' if subject.is_layernorm else 'RMSNorm'} final norm")

    homonyms, hom_fail = pair_items(subject, homonym_dict, "homonym")
    polysemes, poly_fail = pair_items(subject, polyseme_dict, "polyseme")
    shifted, shift_fail = pair_items(subject, homonym_dict, "homonym", offset=1)
    seq_items = sequence_order_items(subject, sequence_pairs)
    null_items = permuted_items(homonyms)
    n_hom_words = len({it["word"] for it in homonyms})
    n_poly_words = len({it["word"] for it in polysemes})
    print(f"homonyms: {n_hom_words}/{len(homonym_dict)} words validated "
          f"({len(homonyms)} pairs; failures {hom_fail})")
    print(f"polysemes: {n_poly_words}/{len(polyseme_dict)} words validated "
          f"({len(polysemes)} pairs; failures {poly_fail})")
    print(f"sequence-order control: {len(seq_items)} matched-token items from "
          f"{len(sequence_pairs)} pairs; permuted null: {len(null_items)} items")

    base_config = {"metric": "activation", "lens": "raw", "agg": "mean", "k": base_k(L)}
    hyperparams = {"metric": ["logit"], "lens": ["normed"], "agg": ["median"]}
    if alt_ks(L):
        hyperparams["k"] = alt_ks(L)
    templates = {"polysemes": polysemes, "next-token-position": shifted}
    finder = make_finder(subject)
    cache_key = f"homonym-{slug}-r{args.n_runs}-v1" if args.cache_dir else None
    common = dict(
        battery=["seeds", "bootstrap", "templates", "hyperparams"],
        n_runs=args.n_runs, config=base_config, templates=templates,
        hyperparams=hyperparams, claim_statement=CLAIM, model=args.model,
        task="layer-wise logit-lens profile of homonym pairs (upstream Experiment 1 "
             "stimuli; activation distance, logit distance, KL divergence)",
        method="upstream per-layer distance/KL profile; top-k activation-distance "
               "band, reconvergence ratio, fixed-threshold profile label",
        verbose=True, cache_dir=args.cache_dir,
    )
    result = sk.stress(finder, homonyms, null_data=null_items, cache_key=cache_key,
                       **common)
    print("\n[alternative null] upstream sequence-order control through the same finder")
    alt = sk.stress(finder, homonyms, null_data=seq_items,
                    cache_key=f"{cache_key}-seqnull" if cache_key else None, **common)

    # ---- post-hoc: full-data profiles for the notes and samples --------------
    hom_items_curves, hom_word_act, hom_curves = full_profile(subject, homonyms, base_config)
    poly_items_curves, _, poly_curves = full_profile(subject, polysemes, base_config)
    null_items_curves, _, null_curves = full_profile(subject, null_items, base_config)
    seq_items_curves, seq_word_act, seq_curves = full_profile(subject, seq_items, base_config)
    hom_zones = zone_stats(hom_word_act, L)
    seq_zones = zone_stats(seq_word_act, L)
    paper_counts = PAPER_VALIDATED.get(args.model.split("/")[-1].lower())

    notes = result.card.notes
    notes.append(
        f"scope: {args.model} (HF revision {subject.revision or 'unrecorded'}) in "
        f"{subject.dtype}, {L} post-block residual layers read at the target-word "
        f"token, one sentence per forward pass, no chat template; the paper's models "
        f"are gpt2, meta-llama/Llama-3.2-3B and Qwen/Qwen2.5-32B via TransformerLens "
        f"2.1.6 — this card grades the profile on the model named here only, with the "
        f"TransformerLens conventions the upstream inherits (fold_ln, center_unembed, "
        f"center_writing_weights for LayerNorm models) reproduced on plain Hugging Face "
        f"weights")
    notes.append(
        f"data: upstream stimulus files at {UPSTREAM_REPO}@{UPSTREAM_COMMIT[:7]} (MIT), "
        f"SHA-256 verified; upstream tokenisation validation reproduced: homonyms "
        f"{n_hom_words}/{len(homonym_dict)} words -> {len(homonyms)} context pairs "
        f"(failures {hom_fail}); polysemes {n_poly_words}/{len(polyseme_dict)} words -> "
        f"{len(polysemes)} pairs (failures {poly_fail})"
        + (f"; paper Table 1 reports {paper_counts[0]} homonyms and {paper_counts[1]} "
           f"polysemes for this model" if paper_counts else "")
        + "; the paper's KL > 0.5 stimulus screen is not part of the released "
          "Experiment-1 entrypoint and is not applied")
    notes.append(
        f"representation (fixed before running): band = top-{base_config['k']} layers by "
        f"activation distance (k = max(3, round(0.2 L))); score r = mean activation "
        f"distance over the final band {final_band(L)} / peak; label = majority "
        f"manuscript layer group of the band (early <= {zone_bounds(L)[0]}, middle <= "
        f"{zone_bounds(L)[1]}, late above) + 'late reconvergence' iff r <= "
        f"{RECONVERGENCE_MAX_RATIO} + KL argmax in the final band -> 'final'; the paper's "
        f"claim is the label '{PAPER_CLAIM_LABEL}'; each finder call subsamples "
        f"{int(SUBSAMPLE_FRAC * 100)}% of the items with its seed")
    notes.append(profile_note("base run (subsampled)", result.base, L))
    base_peak = result.base.meta["peak_layer"]
    base_zone = result.base.claim.split(" peak")[0]
    if zone(base_peak, L) != base_zone:
        notes.append(
            f"the base activation-distance peak (layer {base_peak}) falls in the "
            f"manuscript's '{zone(base_peak, L)}' group while the band majority is "
            f"'{base_zone}': the peak sits on a group boundary, so the zone label "
            f"rests on the band's majority — see the single-run axes for the variants "
            f"that relabel it")
    notes.append(
        f"full homonym set, base configuration: activation-distance peak layer "
        f"{int(np.argmax(hom_curves['activation']))} ({np.max(hom_curves['activation']):.3f}), "
        f"final-band mean {np.mean(hom_curves['activation'][final_band(L)]):.3f}, KL argmax "
        f"layer {int(np.nanargmax(hom_curves['kl']))}; per-(word, layer) activation distance "
        f"by manuscript group: {fmt_zones(hom_zones)} (paper: GPT-2 means "
        f"0.069/0.215/0.167, Llama-3.2-3B medians 0.48/0.55/0.48, Qwen2.5-32B medians "
        f"0.38/0.63/0.42 for early/middle/late)")
    notes.append(
        "null control (graded): the homonym items with sentence B re-paired to a "
        f"different word (seeded derangement, seed {NULL_SEED:#x}, {len(null_items)} items, "
        "same sentences, same read positions, same finder) — the effect under test "
        "(one surface form, two senses) is absent while everything else is held fixed. "
        "Fair because the finder's output is a stable top-k band on any smooth depth "
        "profile, so the specificity ratio asks whether band stability is diagnostic of "
        "the ambiguity effect; it does NOT test whether the null recovers the same band "
        "or label — see the next notes. Conservative direction: different-word pairs are "
        "far apart at every depth, so their profile is shaped by the global geometry "
        "(anisotropy), not by sense resolution")
    null_base = result.null_runs[0].finding
    notes.append(profile_note("null base run (permuted pairs)", null_base, L))
    notes.append(
        f"null claim distribution: {claim_counts(result.null_runs)}; Jaccard between the "
        f"real and null base bands {jaccard(result.base.components, null_base.components):.3f}"
        f"; full permuted set: peak layer {int(np.argmax(null_curves['activation']))}, "
        f"final-band mean / peak {np.mean(null_curves['activation'][final_band(L)]) / np.max(null_curves['activation']):.3f}, "
        f"KL argmax layer {int(np.nanargmax(null_curves['kl']))}")
    alt_spec = alt.checks.get("specificity", {})
    alt_null_base = alt.null_runs[0].finding
    alt_ci = alt_spec.get("ci")
    alt_ci_text = (f"[{alt_ci[0]:.3f}, {alt_ci[1]:.3f}]" if alt_ci else "unavailable")
    notes.append(
        "alternative null (reported, not graded): the upstream sequence-order control "
        f"set ({len(seq_items)} matched tokens from {len(sequence_pairs)} reordered "
        f"sentence pairs) through the same finder — null Jaccard "
        f"{alt.null_summary['mean_pairwise_jaccard']:.3f}, specificity ratio "
        f"{alt_spec.get('value', float('nan')):.3f} (95% CI {alt_ci_text}), "
        f"{'pass' if alt_spec.get('passed') else 'fail'} at >= 1.5; null claim "
        f"distribution: {claim_counts(alt.null_runs)}; Jaccard between the real and "
        f"sequence-order base bands {jaccard(result.base.components, alt_null_base.components):.3f}")
    notes.append(profile_note("sequence-order null base run", alt_null_base, L))
    notes.append(
        "label components of the real base run that the null base runs also produce "
        f"— permuted pairs: {shared_label_parts(result.base.claim, null_base.claim)}; "
        f"sequence-order control: {shared_label_parts(result.base.claim, alt_null_base.claim)}"
        ". A component the finder also returns without the ambiguity effect is not "
        "evidence of that effect on this model")
    notes.append(
        "magnitude comparison the paper does make for this control (its Figure 4, "
        "Llama-3.2-3B, medians early/middle/late homonym 0.483/0.546/0.482 vs "
        f"sequence-order 0.113/0.223/0.190): here homonym {fmt_zones(hom_zones)}; "
        f"sequence-order {fmt_zones(seq_zones)}. The graded finder is magnitude-blind: "
        "a control with the same profile shape at lower magnitude reproduces the label")
    single = [r for r in result.runs if r.axis in ("templates", "hyperparams")]
    notes.append(
        "single-run axes: " + "; ".join(
            f"{r.variant}: '{r.finding.claim}', band {sorted(r.finding.components)}, "
            f"r = {r.finding.score:.3f}, KL argmax {r.finding.meta['kl_peak_layer']}"
            for r in single))
    notes.append(
        f"full polyseme set, base configuration: peak layer "
        f"{int(np.argmax(poly_curves['activation']))}, final-band mean / peak "
        f"{np.mean(poly_curves['activation'][final_band(L)]) / np.max(poly_curves['activation']):.3f}, "
        f"KL argmax layer {int(np.nanargmax(poly_curves['kl']))}")
    notes.append(
        "the 'lens=normed' hyperparameter applies the model's final norm before "
        "unembedding (the standard logit lens); the upstream raw projection lets the "
        "growing residual norm sharpen late-layer softmaxes, so the final-layer KL peak "
        "under 'raw' partly measures norm growth — compare the two KL argmax values above")

    print()
    print(result)
    print(result.to_markdown())

    base = os.path.join(args.out_dir, f"homonym_reconvergence_{slug}")
    result.card.save(base + ".json")
    with open(base + ".md", "w") as f:
        f.write(result.to_markdown() + "\n")
    with open(base + ".badge.json", "w") as f:
        json.dump(result.card.badge_dict(), f, indent=2)
        f.write("\n")
    trace = result.verdict_trace(seed=0)
    with open(base + ".trace.json", "w") as f:
        json.dump(trace, f, indent=2)
        f.write("\n")
    with open(base + ".trace.md", "w") as f:
        f.write(sk.verdict_trace_markdown(trace) + "\n")
    with open(os.path.join(raw_dir, f"runs_{slug}.json"), "w") as f:
        json.dump({
            "model": args.model, "revision": subject.revision, "dtype": subject.dtype,
            "n_layers": L, "base_config": base_config,
            "real_runs": run_records(result.runs),
            "null_runs_permuted": run_records(result.null_runs),
            "null_runs_sequence_order": run_records(alt.null_runs),
            "alt_null_summary": alt.null_summary, "alt_specificity": alt_spec,
            "homonym_failures": hom_fail, "polyseme_failures": poly_fail,
            "shifted_failures": shift_fail,
        }, f, indent=1, default=str)
    with open(os.path.join(raw_dir, f"items_{slug}.json"), "w") as f:
        json.dump({
            name: [dict(it, activation=curve_list(c[0]), logit=curve_list(c[1]),
                        kl=curve_list(c[2])) for it, c in zip(items, per_item)]
            for name, (items, per_item) in {
                "homonyms": (homonyms, hom_items_curves),
                "polysemes": (polysemes, poly_items_curves),
                "permuted_null": (null_items, null_items_curves),
                "sequence_order": (seq_items, seq_items_curves),
            }.items()
        }, f)
    write_samples(base + ".samples.md", subject, {
        "Homonyms": (homonyms, hom_items_curves, hom_curves, 5),
        "Polysemes (template)": (polysemes, poly_items_curves, poly_curves, 3),
        "Permuted pairs (graded null)": (null_items, null_items_curves, null_curves, 3),
        "Sequence-order control (alternative null)": (seq_items, seq_items_curves, seq_curves, 3),
    }, base_config)
    print(sk.verdict_trace_markdown(trace))
    print(f"\nartifacts written to {base}.*")


if __name__ == "__main__":
    main()
