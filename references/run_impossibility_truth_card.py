"""Reference Stability Card: truth vs impossibility directions (arXiv:2608.12852).

Claim under test (registry entry impossibility_truth_double_dissociation_gemma3_4b):
"The truth direction and the impossibility direction show a double
dissociation across held-out families and are close to orthogonal, while the
model's verbal labels conflate contingent falsehood with contradiction."

Finder = the upstream probe pipeline (analyze_axes.py at the pinned commit) as
a pure function of (data, seed, config):

- data: the 60 modality-set statements in the true / false / improbable /
  impossible conditions (15 topic families x 4 conditions), each carrying its
  family, condition label and prompt template. Bootstrap resamples statements.
- seed: the assignment of the 15 families to the 5 held-out folds (seeded
  permutation, round-robin). The probe itself is an L2 logistic regression
  solved by lbfgs -- convex, deterministic given its training set -- so the
  family split is the only place a seed can enter. There is no separate
  "probe init" randomness to sweep, and the card says so.
- config: ``depth`` (hidden-state index; 16 = transformer layer 15, the
  upstream impossibility peak) and ``C`` (inverse L2 strength; upstream 0.1).

Per call: for each fold, fit an impossibility probe (impossible vs true +
false + improbable) and a truth probe (false vs true) on the training
families -- StandardScaler + LogisticRegression(C, class_weight="balanced"),
upstream's make_probe() verbatim -- and score every statement of the held-out
families with both probes. Then fit both probes on all training data for the
direction cosine (upstream axis_cosines()).

Finding representation (fixed before any battery ran):

- components: tagged held-out selections {"impossibility:<id>" : the
  impossibility probe fires on statement id} + {"truth:<id>" : the truth
  probe fires on id}, each probe restricted to its own contrast (60 and 30
  statements). Universe = 90. Selections are made on the fixed evaluation
  universe (every statement, scored while its family is held out), so
  bootstrap changes what the probes are trained on, not which statements are
  scored. Rejected alternatives: top-k residual coordinates of a probe (a
  basis artifact) and the set of depths where the dissociation holds (a
  contiguous band covering half the depths cannot beat the size-matched
  random null by 3x, so the representation rather than the finding would fix
  the grade).
- claim: "<decoding>; <dissociation>; <geometry>" from fixed thresholds.
  decoding: both in-axis held-out AUCs >= 0.8. dissociation: impossibility
  probe within +/-0.15 of chance on false-vs-true AND truth probe <= 0.65 on
  impossible-vs-false. The second criterion is one-sided on purpose: it is
  the paper's Table 2 reading ("the truth direction cannot see the
  difference between the impossible and the false"), whose upstream value is
  0.20 -- a *reversed* ordering (impossible statements score less false than
  contingent falsehoods) that the paper counts as "at or below chance". The
  stricter two-sided variant is recorded per run in meta and summarised in
  the notes. geometry: |cos| < 0.2 between the two full-data probe
  directions, in standardised feature space as upstream computes it (the
  raw-residual-space cosine is in meta).
- score: double-dissociation index in [0, 1] = mean in-axis held-out AUC of
  the two probes minus the mean off-axis excess over chance
  (|AUC_imp(false vs true) - 0.5| and max(0, AUC_truth(impossible vs false)
  - 0.5)). 1.0 = both directions perfect on their own contrast and blind on
  the other's. |cos| is in meta: as a score its CV would only measure the
  noise floor of a near-zero quantity.

Battery: seeds (family split), bootstrap (statement resampling), templates
(two alternative prompt framings; the upstream repository ships a single
template, so both alternatives are constructed here and labelled as such),
hyperparams (depth +/- 4, C 0.01 / 1.0), plus a null control: condition
labels permuted once within each family (the upstream permutation-test
scheme), run through the same finder.

Data: data/questions_modality.json at commit f1ead9a of
sixticket/representing-the-impossible (MIT), SHA-256 verified; its 75 items
are byte-identical to the modality items of questions_combined.json, the
file the upstream run used. Model: google/gemma-3-4b-it at revision 093f9f3,
the snapshot recorded in the upstream run_config.json.

Usage (GPU):
    python references/run_impossibility_truth_card.py --out-dir references/cards \
        --raw-dir /path/to/raw/impossibility_gemma_3_4b_it
"""

import argparse
import hashlib
import json
import os
import random
import re
import urllib.request
from collections import Counter

import numpy as np
import torch
import transformers
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import stresskit as sk

UPSTREAM_REPO = "sixticket/representing-the-impossible"
UPSTREAM_COMMIT = "f1ead9af6cb7104c170325906a0bd78baa989473"
DATA_FILE = "questions_modality.json"
DATA_SHA256 = "7d4ca5b1a96025298464e915059ebabc7d374978288976cfc756048d3800cdef"
UPSTREAM_COMBINED_SHA256 = "20fbcc5c0710a0cdd7114675d6403d683a164098256545f0280ca6242a99f99e"
MODEL = "google/gemma-3-4b-it"
MODEL_REVISION = "093f9f388b31de276ce2de164bdc2081324b9767"

TEMPLATES = {
    "upstream": (
        "Statement: {statement}\n\nClassify the statement using exactly one label: "
        "coherent, contradiction, paradox, or underdetermined. Then explain the "
        "classification in one sentence."),
    "paraphrased-instruction": (
        "Consider this statement: {statement}\n\nWhich single label fits it best: "
        "coherent, contradiction, paradox, or underdetermined? Give the label, then "
        "justify it in one sentence."),
    "bare-statement": "{statement}",
}
BASE_TEMPLATE = "upstream"
LABELS = ("coherent", "contradiction", "paradox", "underdetermined")
MODALITIES = ("true", "false", "improbable", "anomalous", "impossible")
POSSIBLE = ("true", "false", "improbable")
PROBES = {"impossibility": ("impossible", POSSIBLE), "truth": ("false", ("true",))}
CONTRASTS = {
    "false_vs_true": ("false", ("true",)),
    "impossible_vs_true": ("impossible", ("true",)),
    "impossible_vs_false": ("impossible", ("false",)),
    "impossible_vs_possible": ("impossible", POSSIBLE),
}
IN_AXIS = {"impossibility": "impossible_vs_possible", "truth": "false_vs_true"}
N_FOLDS = 5
UPSTREAM_DEPTH = 16
BASE_C = 0.1
MAX_NEW_TOKENS = 48
DECODE_AUC_MIN = 0.8
CHANCE_BAND = 0.15
ORTHO_COS_MAX = 0.2
DEPTH_OFFSET = 4
C_ALTERNATIVES = [0.01, 1.0]

# upstream results/reference_run/axes/axes_results.json at UPSTREAM_COMMIT, depth 16
UPSTREAM_REFERENCE = {
    "ba_impossibility": 0.967, "ba_truth": 0.900,
    "auc_impossibility_false_vs_true": 0.511, "auc_truth_impossible_vs_false": 0.200,
    "auc_truth_impossible_vs_true": 0.933, "auc_truth_impossible_vs_possible": 0.578,
    "cos_standardized": -0.010,
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_items(data_dir):
    path = os.path.join(data_dir, DATA_FILE)
    if not os.path.exists(path):
        os.makedirs(data_dir, exist_ok=True)
        url = (f"https://raw.githubusercontent.com/{UPSTREAM_REPO}/"
               f"{UPSTREAM_COMMIT}/data/{DATA_FILE}")
        urllib.request.urlretrieve(url, path)
    with open(path, "rb") as f:
        raw = f.read()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != DATA_SHA256:
        raise RuntimeError(
            f"{path}: sha256 {digest} does not match the pinned upstream file "
            f"{DATA_SHA256} (commit {UPSTREAM_COMMIT[:7]})")
    dataset = json.loads(raw)
    if dataset["prompt_template"] != TEMPLATES[BASE_TEMPLATE]:
        raise RuntimeError("the pinned data file's prompt_template differs from "
                           "TEMPLATES['upstream']; refusing to run with a stale template")
    items = []
    for it in dataset["items"]:
        prefix, family, modality = it["id"].split("_")
        if prefix != "mod" or modality not in MODALITIES \
                or it["category"] != f"modality_{modality}":
            raise RuntimeError(f"unexpected item id/category: {it['id']} / {it['category']}")
        items.append({"id": it["id"], "family": family, "modality": modality,
                      "statement": it["statement_en"],
                      "expected_label": it["expected_label"]})
    counts = Counter(it["modality"] for it in items)
    if set(counts) != set(MODALITIES) or len(set(counts.values())) != 1:
        raise RuntimeError(f"modality set is not balanced: {dict(counts)}")
    return items


def with_template(items, template):
    return [dict(it, template=template) for it in items]


def permute_within_family(items, seed):
    """Null control: shuffle the condition labels among each family's statements."""
    rng = random.Random(seed)
    out = []
    for family in sorted({it["family"] for it in items}):
        members = [it for it in items if it["family"] == family]
        labels = [it["modality"] for it in members]
        rng.shuffle(labels)
        out.extend(dict(it, modality=lab) for it, lab in zip(members, labels))
    return out


def parse_predicted_label(response):
    match = re.search(r"\b(coherent|contradiction|paradox|underdetermined)\b",
                      response.lower())
    return match.group(1) if match else None


# ---------------------------------------------------------------------------
# Model access and activation cache
# ---------------------------------------------------------------------------

class Subject:
    """The chat model, read at the final prompt token (upstream extraction)."""

    def __init__(self, name, revision, device):
        self.device = device
        self.tok = transformers.AutoTokenizer.from_pretrained(name, revision=revision)
        config = transformers.AutoConfig.from_pretrained(name, revision=revision)
        arch = (getattr(config, "architectures", None) or [""])[0]
        cls = getattr(transformers, arch, None) or transformers.AutoModelForCausalLM
        self.model = cls.from_pretrained(
            name, revision=revision, dtype=torch.bfloat16, device_map=device).eval()
        self.model_class = type(self.model).__name__

    def encode(self, prompt):
        rendered = self.tok.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True)
        enc = self.tok(rendered, return_tensors="pt", add_special_tokens=False)
        return {k: v.to(self.device) for k, v in enc.items()}

    @torch.no_grad()
    def hidden_last(self, prompt):
        """[n_depths, d] float32 residual stream at the final prompt token.
        Upstream stored float16; Gemma 3's final-token residual exceeds the
        float16 range at some depths under the bare-statement template, so the
        cache keeps float32 and the card says so."""
        enc = self.encode(prompt)
        out = self.model(**enc, output_hidden_states=True, return_dict=True,
                         use_cache=False)
        if out.hidden_states is None:
            raise RuntimeError(f"{self.model_class} returned no hidden states")
        states = torch.stack([h[0, -1, :].float().cpu() for h in out.hidden_states])
        return states.numpy().astype(np.float32), int(enc["input_ids"].shape[1])

    @torch.no_grad()
    def complete(self, prompt):
        enc = self.encode(prompt)
        gen = self.model.generate(
            **enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, top_p=None,
            top_k=None, pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(gen[0, enc["input_ids"].shape[1]:],
                               skip_special_tokens=True).strip()


class ActivationStore:
    """Per-template residual caches: extracted once, then read from disk."""

    def __init__(self, items, raw_dir, model_name, revision, device):
        self.items = items
        self.raw_dir = raw_dir
        self.model_name = model_name
        self.revision = revision
        self.device = device
        self._acts = {}
        self._subject = None

    def _subject_or_load(self):
        if self._subject is None:
            print(f"loading {self.model_name}@{self.revision[:7]} on {self.device} ...")
            self._subject = Subject(self.model_name, self.revision, self.device)
            os.makedirs(self.raw_dir, exist_ok=True)
            with open(self.meta_path(), "w") as f:
                json.dump({"model": self.model_name, "revision": self.revision,
                           "model_class": self._subject.model_class,
                           "transformers": transformers.__version__,
                           "torch": torch.__version__}, f, indent=1)
        return self._subject

    def meta_path(self):
        return os.path.join(self.raw_dir, "extraction_meta.json")

    def extraction_meta(self):
        with open(self.meta_path()) as f:
            meta = json.load(f)
        if meta["model"] != self.model_name or meta["revision"] != self.revision:
            raise RuntimeError(
                f"{self.meta_path()}: cached activations were extracted from "
                f"{meta['model']}@{meta['revision'][:7]}, not "
                f"{self.model_name}@{self.revision[:7]}; use a different --raw-dir")
        return meta

    def release(self):
        if self._subject is not None:
            del self._subject
            self._subject = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def path(self, template):
        return os.path.join(self.raw_dir, f"activations_{template}.npz")

    def responses_path(self):
        return os.path.join(self.raw_dir, f"responses_{BASE_TEMPLATE}.json")

    def ensure(self, template):
        if os.path.exists(self.path(template)):
            return
        subject = self._subject_or_load()
        os.makedirs(self.raw_dir, exist_ok=True)
        rows, counts = [], []
        for it in self.items:
            states, n_tok = subject.hidden_last(TEMPLATES[template].format(statement=it["statement"]))
            rows.append(states)
            counts.append(n_tok)
        np.savez_compressed(
            self.path(template) + ".tmp.npz",
            item_ids=np.array([it["id"] for it in self.items]),
            activations=np.stack(rows), token_counts=np.array(counts, dtype=np.int32))
        os.replace(self.path(template) + ".tmp.npz", self.path(template))
        print(f"extracted {template}: {np.stack(rows).shape}")

    def ensure_responses(self):
        if os.path.exists(self.responses_path()):
            return
        subject = self._subject_or_load()
        out = []
        for it in self.items:
            response = subject.complete(TEMPLATES[BASE_TEMPLATE].format(statement=it["statement"]))
            out.append({"id": it["id"], "response": response,
                        "predicted_label": parse_predicted_label(response)})
        with open(self.responses_path(), "w") as f:
            json.dump(out, f, indent=1, ensure_ascii=False)

    def activations(self, template):
        """[n_items, n_depths, d] float32, rows aligned with self.items."""
        if template not in self._acts:
            self.ensure(template)
            with np.load(self.path(template)) as z:
                ids = [str(x) for x in z["item_ids"]]
                if ids != [it["id"] for it in self.items]:
                    raise RuntimeError(f"{self.path(template)}: item order does not "
                                       "match the pinned data file")
                acts = z["activations"].astype(np.float32)
            if not np.isfinite(acts).all():
                raise RuntimeError(f"{self.path(template)}: non-finite activations; "
                                   "re-extract (delete the cache) before running")
            self._acts[template] = acts
        return self._acts[template]

    def token_counts(self, template):
        self.ensure(template)
        with np.load(self.path(template)) as z:
            return z["token_counts"].astype(float)

    def responses(self):
        self.ensure_responses()
        with open(self.responses_path()) as f:
            return {r["id"]: r for r in json.load(f)}

    @property
    def n_depths(self):
        return self.activations(BASE_TEMPLATE).shape[1]


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------

def make_probe(C, seed):
    """upstream analyze_axes.make_probe with C and random_state exposed."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, class_weight="balanced", max_iter=2000,
                           random_state=seed))


def family_folds(families, seed):
    fams = list(families)
    random.Random(seed).shuffle(fams)
    return {f: i % N_FOLDS for i, f in enumerate(fams)}


def in_contrast(item, pos, neg):
    return item["modality"] == pos or item["modality"] in neg


def contrast_auc(items, scores, pos, neg):
    y = [int(it["modality"] == pos) for it in items if in_contrast(it, pos, neg)]
    s = [sc for it, sc in zip(items, scores) if in_contrast(it, pos, neg)]
    if len(set(y)) < 2:
        return None
    return float(roc_auc_score(y, s))


def cosine(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def mean(xs):
    return float(np.mean(xs)) if xs else float("nan")


def direction_hash(vec):
    return hashlib.sha256(np.asarray(vec, dtype=np.float32).tobytes()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Finder
# ---------------------------------------------------------------------------

def make_finder(store, items, raw_dir):
    eval_items = [it for it in items if it["modality"] != "anomalous"]
    families = sorted({it["family"] for it in eval_items})
    index = {it["id"]: i for i, it in enumerate(items)}

    def rows(X, subset):
        return X[[index[it["id"]] for it in subset]]

    def fit(X, subset, pos, neg, C, seed, what):
        y = [int(it["modality"] == pos) for it in subset]
        if len(set(y)) < 2:
            raise RuntimeError(
                f"{what}: the training set for the {pos}-vs-{'/'.join(neg)} probe "
                f"contains a single class (n={len(subset)}); this resample cannot "
                "support the probe")
        return make_probe(C, seed).fit(rows(X, subset), y)

    def held_out(data, X, C, seed):
        """Per-item held-out decisions on the fixed evaluation universe, plus
        fold-averaged AUCs for every contrast and in-axis balanced accuracy."""
        fold_of = family_folds(families, seed)
        decisions = {name: {} for name in PROBES}
        aucs = {name: {c: [] for c in CONTRASTS} for name in PROBES}
        bas = {name: [] for name in PROBES}
        for k in range(N_FOLDS):
            train = [it for it in data if fold_of[it["family"]] != k]
            held = [it for it in eval_items if fold_of[it["family"]] == k]
            for name, (pos, neg) in PROBES.items():
                probe = fit(X, [it for it in train if in_contrast(it, pos, neg)],
                            pos, neg, C, seed, f"fold {k}")
                scores = probe.decision_function(rows(X, held))
                for cname, (cpos, cneg) in CONTRASTS.items():
                    auc = contrast_auc(held, scores, cpos, cneg)
                    if auc is not None:
                        aucs[name][cname].append(auc)
                own = [(it, s) for it, s in zip(held, scores) if in_contrast(it, pos, neg)]
                bas[name].append(balanced_accuracy_score(
                    [int(it["modality"] == pos) for it, _ in own],
                    [int(s > 0) for _, s in own]))
                for it, s in own:
                    decisions[name][it["id"]] = float(s)
        return (decisions,
                {n: {c: mean(v) for c, v in d.items()} for n, d in aucs.items()},
                {n: mean(v) for n, v in bas.items()},
                {f: k for f, k in fold_of.items()})

    def full_directions(data, X, C, seed):
        out = {}
        for name, (pos, neg) in PROBES.items():
            probe = fit(X, [it for it in data if in_contrast(it, pos, neg)],
                        pos, neg, C, seed, "full-data fit")
            w = probe.named_steps["logisticregression"].coef_[0]
            scale = probe.named_steps["standardscaler"].scale_
            out[name] = {"standardized": w, "raw": w / scale}
        return out

    def finder(data, seed, config):
        depth, C = int(config["depth"]), float(config["C"])
        template = data[0]["template"]
        X = store.activations(template)[:, depth, :]

        decisions, auc, ba, fold_of = held_out(data, X, C, seed)
        directions = full_directions(data, X, C, seed)
        cos_std = cosine(directions["impossibility"]["standardized"],
                         directions["truth"]["standardized"])
        cos_raw = cosine(directions["impossibility"]["raw"], directions["truth"]["raw"])

        imp_in = auc["impossibility"][IN_AXIS["impossibility"]]
        truth_in = auc["truth"][IN_AXIS["truth"]]
        imp_cross = auc["impossibility"]["false_vs_true"]
        truth_cross = auc["truth"]["impossible_vs_false"]

        if imp_in >= DECODE_AUC_MIN and truth_in >= DECODE_AUC_MIN:
            decoding = "both probes decode held-out families"
        elif imp_in >= DECODE_AUC_MIN:
            decoding = "only the impossibility probe decodes"
        elif truth_in >= DECODE_AUC_MIN:
            decoding = "only the truth probe decodes"
        else:
            decoding = "neither probe decodes"
        imp_blind = abs(imp_cross - 0.5) <= CHANCE_BAND
        truth_blind = truth_cross <= 0.5 + CHANCE_BAND
        if imp_blind and truth_blind:
            dissociation = "double dissociation"
        elif imp_blind:
            dissociation = "impossibility probe truth-blind only"
        elif truth_blind:
            dissociation = "truth probe impossibility-blind only"
        else:
            dissociation = "no dissociation"
        geometry = (f"near-orthogonal (|cos|<{ORTHO_COS_MAX})" if abs(cos_std) < ORTHO_COS_MAX
                    else f"oblique (|cos|>={ORTHO_COS_MAX})")
        claim = f"{decoding}; {dissociation}; {geometry}"

        score = 0.5 * (imp_in + truth_in) - 0.5 * (
            abs(imp_cross - 0.5) + max(0.0, truth_cross - 0.5))
        two_sided = imp_blind and abs(truth_cross - 0.5) <= CHANCE_BAND \
            and abs(auc["truth"]["impossible_vs_possible"] - 0.5) <= CHANCE_BAND

        components = sorted(
            [f"impossibility:{i}" for i, s in decisions["impossibility"].items() if s > 0]
            + [f"truth:{i}" for i, s in decisions["truth"].items() if s > 0])

        digest = hashlib.sha256(json.dumps(
            [decisions, depth, C, template], sort_keys=True).encode()).hexdigest()[:16]
        os.makedirs(raw_dir, exist_ok=True)
        hashes = {}
        for name, vecs in directions.items():
            unit = vecs["raw"] / np.linalg.norm(vecs["raw"])
            hashes[name] = direction_hash(unit)
            np.save(os.path.join(raw_dir, f"direction_{name}_{hashes[name]}.npy"),
                    unit.astype(np.float32))
        with open(os.path.join(raw_dir, f"decisions_{digest}.json"), "w") as f:
            json.dump({"depth": depth, "C": C, "template": template, "seed": seed,
                       "fold_of_family": fold_of, "decisions": decisions}, f, indent=1)

        return sk.feature_set(
            components,
            claim=claim,
            score=round(score, 4),
            universe_size=len(decisions["impossibility"]) + len(decisions["truth"]),
            depth=depth,
            C=C,
            template=template,
            n_train=len(data),
            n_train_unique=len({it["id"] for it in data}),
            auc_impossibility_in_axis=round(imp_in, 4),
            auc_truth_in_axis=round(truth_in, 4),
            auc_impossibility_false_vs_true=round(imp_cross, 4),
            auc_impossibility_impossible_vs_false=round(
                auc["impossibility"]["impossible_vs_false"], 4),
            auc_truth_impossible_vs_false=round(truth_cross, 4),
            auc_truth_impossible_vs_true=round(auc["truth"]["impossible_vs_true"], 4),
            auc_truth_impossible_vs_possible=round(
                auc["truth"]["impossible_vs_possible"], 4),
            ba_impossibility=round(ba["impossibility"], 4),
            ba_truth=round(ba["truth"], 4),
            cos_standardized=round(cos_std, 4),
            cos_raw=round(cos_raw, 4),
            abs_cos=round(abs(cos_std), 4),
            dissociation_two_sided=bool(two_sided),
            n_selected_impossibility=sum(s > 0 for s in decisions["impossibility"].values()),
            n_selected_truth=sum(s > 0 for s in decisions["truth"].values()),
            direction_sha256_16=hashes,
            run_digest=digest,
        )

    finder.held_out = held_out
    finder.full_directions = full_directions
    finder.eval_items = eval_items
    finder.families = families
    return finder


# ---------------------------------------------------------------------------
# Pre-battery calibration and post-hoc notes
# ---------------------------------------------------------------------------

def depth_curve(finder, store, pool, C, seed=0):
    """Held-out in-axis balanced accuracy of both probes and the direction
    cosine at every depth, at the base split (upstream Fig. 2a/2c)."""
    acts = store.activations(BASE_TEMPLATE)
    curve = []
    for depth in range(acts.shape[1]):
        X = acts[:, depth, :]
        _, auc, ba, _ = finder.held_out(pool, X, C, seed)
        dirs = finder.full_directions(pool, X, C, seed)
        cos = cosine(dirs["impossibility"]["standardized"], dirs["truth"]["standardized"])
        curve.append({
            "depth": depth,
            "ba_impossibility": round(ba["impossibility"], 4),
            "ba_truth": round(ba["truth"], 4),
            "auc_impossibility_false_vs_true": round(auc["impossibility"]["false_vs_true"], 4),
            "auc_truth_impossible_vs_false": round(auc["truth"]["impossible_vs_false"], 4),
            # depth 0 is the embedding of the shared final prompt token: identical
            # rows, zero-weight probes, undefined cosine
            "cos_standardized": None if np.isnan(cos) else round(cos, 4),
        })
    return curve


def depth_notes(curve, depth, selection):
    peak = max(curve, key=lambda r: (r["ba_impossibility"], -r["depth"]))
    at = curve[depth]
    late = [abs(r["cos_standardized"]) for r in curve
            if r["depth"] > 10 and r["cos_standardized"] is not None]
    return [
        f"depth selection: {selection}; held-out impossibility balanced accuracy peaks "
        f"at depth {peak['depth']} ({peak['ba_impossibility']:.3f}) on this extraction "
        f"(upstream: depth 16, 0.967); at the graded depth {depth}: impossibility BA "
        f"{at['ba_impossibility']:.3f}, truth BA {at['ba_truth']:.3f}, "
        f"cos(truth, impossibility) {at['cos_standardized']:+.3f}; impossibility-probe AUC "
        f"on false-vs-true by depth: {[r['auc_impossibility_false_vs_true'] for r in curve]}",
        "cosine curve (not graded): max |cos(truth, impossibility)| over depths > 10 = "
        f"{max(late):.3f} (upstream reports at most 0.12); per-depth values live in "
        "the raw directory (depth_curve.json)",
    ]


def null_strictness_note(result):
    null = result.null_summary or {}
    return (
        "null construction, direction of bias: one fixed permutation is reused by every "
        "null run, so null probes trained on different family splits share the same "
        "spurious label structure and their held-out selections agree more than chance "
        f"(null Jaccard {null.get('mean_pairwise_jaccard', float('nan')):.3f} vs "
        f"{result.pooled.get('expected_random_jaccard', float('nan')):.3f} for size-matched "
        "random sets); this inflates the denominator of the specificity ratio, so the "
        "check is harder to pass than under a per-run re-permutation -- the null is "
        "conservative against the finding, not in its favour")


def upstream_reference_note(base_meta):
    parts = []
    for key, ref in UPSTREAM_REFERENCE.items():
        if key in base_meta:
            parts.append(f"{key} {base_meta[key]:+.3f} (upstream {ref:+.3f})")
    return ("base run vs the upstream reference run at depth 16 (results/reference_run/"
            f"axes/axes_results.json @ {UPSTREAM_COMMIT[:7]}): " + "; ".join(parts))


def verbal_label_note(items, responses):
    table = {m: Counter() for m in MODALITIES}
    for it in items:
        table[it["modality"]][responses[it["id"]]["predicted_label"] or "unparsed"] += 1
    n = len(items) // len(MODALITIES)
    cells = "; ".join(
        f"{m}: " + ", ".join(f"{lab} {table[m][lab]}" for lab in LABELS + ("unparsed",)
                             if table[m][lab])
        for m in MODALITIES)
    false_contra = table["false"]["contradiction"]
    imp_contra = table["impossible"]["contradiction"]
    verdict = ("reproduces" if false_contra >= n / 2 else "does not reproduce")
    return (f"verbal labels (not graded; greedy, {MAX_NEW_TOKENS} new tokens, upstream "
            f"template; n={n} per condition) {verdict} the conflation: {false_contra}/{n} "
            f"contingent falsehoods labelled 'contradiction' (upstream 12/15) vs "
            f"{imp_contra}/{n} impossible statements (upstream 5/15); full table -- {cells}")


def stricter_variant_note(result):
    real = [r for r in result.runs]
    n_two = sum(bool(r.finding.meta.get("dissociation_two_sided")) for r in real)
    tc = [r.finding.meta["auc_truth_impossible_vs_false"] for r in real]
    tp = [r.finding.meta["auc_truth_impossible_vs_possible"] for r in real]
    return (
        "stricter dissociation variant (not graded): requiring the truth probe to sit "
        f"within +/-{CHANCE_BAND} of chance on BOTH impossible-vs-false and "
        f"impossible-vs-possible (two-sided) holds in {n_two}/{len(real)} runs; truth-probe "
        f"AUC on impossible-vs-false ranges {min(tc):.2f}-{max(tc):.2f} (below 0.5 = "
        "impossible statements score LESS false than contingent falsehoods), on "
        f"impossible-vs-possible {min(tp):.2f}-{max(tp):.2f}")


def load_direction(raw_dir, name, digest):
    return np.load(os.path.join(raw_dir, f"direction_{name}_{digest}.npy"))


def pairwise_abs_cosines(vectors):
    return [abs(float(np.dot(vectors[i], vectors[j])))
            for i in range(len(vectors)) for j in range(i + 1, len(vectors))]


def geometry_notes(result, raw_dir):
    notes = []
    for name in PROBES:
        for axis in ("seeds", "bootstrap", "hyperparams", "templates"):
            vs = [load_direction(raw_dir, name, r.finding.meta["direction_sha256_16"][name])
                  for r in result.runs if r.axis == axis]
            base = load_direction(raw_dir, name, result.base.meta["direction_sha256_16"][name])
            if vs:
                cos = [abs(float(np.dot(base, v))) for v in vs]
                notes.append(
                    f"{name} direction geometry (raw residual space, not graded): mean "
                    f"|cos| to the base direction on the {axis} axis {np.mean(cos):.3f} "
                    f"(min {np.min(cos):.3f}, n={len(vs)})")
    if result.null_runs:
        for name in PROBES:
            null = [load_direction(raw_dir, name, r.finding.meta["direction_sha256_16"][name])
                    for r in result.null_runs]
            pc = pairwise_abs_cosines(null)
            notes.append(
                f"null-control {name} directions: mean pairwise |cos| {np.mean(pc):.3f} "
                f"(n={len(null)})")
    return notes


def surface_baseline_note(finder, store, pool, seed=0):
    """Upstream surface_baselines() on the same family folds: a different
    finder, so reported rather than graded."""
    fold_of = family_folds(finder.families, seed)
    counts = store.token_counts(BASE_TEMPLATE)
    index = {it["id"]: i for i, it in enumerate(store.items)}
    out = {}
    for cname in ("impossible_vs_possible", "impossible_vs_false"):
        pos, neg = CONTRASTS[cname]
        subset = [it for it in pool if in_contrast(it, pos, neg)]
        scores = {"token_count": [], "tfidf_word": [], "tfidf_char": []}
        for k in range(N_FOLDS):
            train = [it for it in subset if fold_of[it["family"]] != k]
            test = [it for it in subset if fold_of[it["family"]] == k]
            y_tr = [int(it["modality"] == pos) for it in train]
            y_te = [int(it["modality"] == pos) for it in test]
            probe = make_probe(BASE_C, seed).fit(
                [[counts[index[it["id"]]]] for it in train], y_tr)
            scores["token_count"].append(balanced_accuracy_score(
                y_te, probe.predict([[counts[index[it["id"]]]] for it in test])))
            for key, vec in (("tfidf_word", TfidfVectorizer(ngram_range=(1, 2))),
                             ("tfidf_char", TfidfVectorizer(analyzer="char_wb",
                                                            ngram_range=(3, 5)))):
                x_tr = vec.fit_transform([it["statement"] for it in train])
                x_te = vec.transform([it["statement"] for it in test])
                clf = LogisticRegression(C=BASE_C, class_weight="balanced", max_iter=2000,
                                         random_state=seed).fit(x_tr, y_tr)
                scores[key].append(balanced_accuracy_score(y_te, clf.predict(x_te)))
        out[cname] = {key: round(mean(v), 3) for key, v in scores.items()}
    return ("surface-form baselines (upstream token-count / word TF-IDF / char TF-IDF "
            "probes on the same family folds; a different finder, so not the graded "
            f"null): {json.dumps(out)} balanced accuracy vs the activation probe's "
            "held-out balanced accuracy on the card (upstream: 0.56/0.70/0.64 and "
            "0.60/0.67/0.77)")


def write_samples(path, raw_dir, result, items, responses, seed=0):
    base = result.base.meta
    with open(os.path.join(raw_dir, f"decisions_{base['run_digest']}.json")) as f:
        dec = json.load(f)["decisions"]
    eval_items = [it for it in items if it["modality"] != "anomalous"]
    rng = random.Random(seed)
    lines = ["# Randomly selected raw examples (base run)", "",
             "Selected with `random.Random(0)` from the 60 statements the probes see, "
             "not cherry-picked. Model responses are greedy, first "
             f"{MAX_NEW_TOKENS} new tokens, upstream template. Decision values are the "
             f"held-out probe scores at depth {base['depth']} (positive = the probe fires; "
             "the truth probe is only applied to true/false statements, its own contrast).",
             ""]
    for it in rng.sample(eval_items, 8):
        resp = responses[it["id"]]
        imp = dec["impossibility"].get(it["id"])
        tru = dec["truth"].get(it["id"])
        lines += [
            f"## {it['id']} ({it['modality']}; expected label `{it['expected_label']}`)",
            "", f"**Statement.** {it['statement']}",
            f"- model label: `{resp['predicted_label']}` -- response: `{resp['response']!r}`",
            f"- impossibility probe decision: {imp:+.2f}" + (" (fires)" if imp > 0 else ""),
            (f"- truth probe decision: {tru:+.2f}" + (" (fires = 'false')" if tru > 0 else "")
             if tru is not None else "- truth probe: not applied (outside its contrast)"),
            ""]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def run_record(run):
    """Per-run AUCs, cosines and selections: the raw numbers behind the card."""
    return {"axis": run.axis, "variant": run.variant, "seed": run.seed,
            "config": run.config, "claim": run.finding.claim,
            "score": run.finding.score, "size": run.finding.size,
            "meta": run.finding.meta}


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--revision", default=MODEL_REVISION,
                    help="model revision; the default is the upstream run's snapshot "
                         "and only applies to the default model")
    ap.add_argument("--data-dir", default="impossible_data")
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "cards"))
    ap.add_argument("--raw-dir", default=None,
                    help="activation caches, per-run directions and decisions "
                         "(default: <out-dir>/raw/impossibility_<slug>)")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--n-runs", type=int, default=16)
    ap.add_argument("--depth", type=int, default=None,
                    help="hidden-state index to grade at (default: 16, the upstream "
                         "peak, for the default model; otherwise the held-out "
                         "impossibility peak of this extraction)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    slug = args.model.split("/")[-1].lower().replace("-", "_").replace(".", "p")
    raw_dir = args.raw_dir or os.path.join(args.out_dir, "raw", f"impossibility_{slug}")
    os.makedirs(args.out_dir, exist_ok=True)
    revision = args.revision if args.model == MODEL else "main"

    items = load_items(args.data_dir)
    store = ActivationStore(items, raw_dir, args.model, revision, args.device)
    for template in TEMPLATES:
        store.ensure(template)
    store.ensure_responses()
    store.release()
    extraction = store.extraction_meta()
    n_depths, d_model = store.activations(BASE_TEMPLATE).shape[1:]
    print(f"{n_depths} hidden-state depths, d_model {d_model}")

    finder = make_finder(store, items, raw_dir)
    pool = with_template(finder.eval_items, BASE_TEMPLATE)
    pool_null = with_template(permute_within_family(finder.eval_items, 0x5EC), BASE_TEMPLATE)
    alt_templates = {t: with_template(finder.eval_items, t) for t in TEMPLATES if t != BASE_TEMPLATE}

    curve = depth_curve(finder, store, pool, BASE_C)
    with open(os.path.join(raw_dir, "depth_curve.json"), "w") as f:
        json.dump(curve, f, indent=1)
    if args.depth is not None:
        depth, selection = args.depth, "--depth given on the command line"
    elif args.model == MODEL:
        depth, selection = UPSTREAM_DEPTH, "upstream's reported peak (depth 16 = layer 15)"
    else:
        depth = max(curve, key=lambda r: (r["ba_impossibility"], -r["depth"]))["depth"]
        selection = "argmax of held-out impossibility balanced accuracy at the base split"
    if not 0 < depth < n_depths:
        raise RuntimeError(f"depth {depth} outside (0, {n_depths})")
    alt_depths = [dd for dd in (depth - DEPTH_OFFSET, depth + DEPTH_OFFSET) if 0 < dd < n_depths]
    print(f"grading at depth {depth} ({selection}); alternatives {alt_depths}")

    result = sk.stress(
        finder,
        pool,
        battery=["seeds", "bootstrap", "templates", "hyperparams"],
        n_runs=args.n_runs,
        config={"depth": depth, "C": BASE_C},
        templates=alt_templates,
        hyperparams={"depth": alt_depths, "C": C_ALTERNATIVES},
        null_data=pool_null,
        claim_statement=(
            f"In {args.model}, a truth direction (false vs true) and an impossibility "
            "direction (impossible vs possible), each a linear probe on the final-prompt-"
            "token residual stream, show a double dissociation on held-out topic families "
            "and are close to orthogonal"),
        model=args.model,
        task="modality contrast set: 15 topic families x {true, false, improbable, "
             "impossible}, family-held-out probing (upstream data)",
        method="StandardScaler + L2 logistic regression probes (upstream make_probe), "
               "5 family-grouped folds, transfer AUC of each probe on the other contrast, "
               "cosine of full-data probe directions",
        verbose=True,
        cache_dir=args.cache_dir,
        cache_key=(f"impossibility-{slug}-d{depth}-r{args.n_runs}-v1" if args.cache_dir else None),
    )

    responses = store.responses()
    result.card.notes.append(
        f"scope: {args.model} (revision {revision[:7]}, {extraction['model_class']}, "
        f"bfloat16, transformers {extraction['transformers']}), text-only "
        "chat usage with the model's default template; residual stream read at the final "
        "prompt token immediately before generation (upstream extraction; cached as "
        "float32 where upstream stored float16, because the bare-statement template "
        f"overflows float16 at some depths); graded at hidden-state depth {depth} "
        f"(transformer layer {depth - 1}) "
        f"with C={BASE_C}; every probe is evaluated on statements whose whole topic family "
        "was held out of its training data")
    result.card.notes.append(
        f"data: data/{DATA_FILE} at {UPSTREAM_REPO}@{UPSTREAM_COMMIT[:7]} (repository "
        "MIT-licensed; the stimulus files carry no separate license and are covered by "
        "the repository license), SHA-256 verified; its 75 items are byte-identical to "
        f"the modality items of questions_combined.json (sha256 {UPSTREAM_COMBINED_SHA256[:12]}...) "
        "used by the upstream run; the 15 anomalous statements are outside both probes' "
        "contrasts and are excluded from the pool")
    result.card.notes.append(
        "axes: seeds = the family-to-fold assignment (the lbfgs logistic regression is "
        "deterministic given its training set, so there is no separate probe-init "
        "randomness); bootstrap = statement resampling of the 60-item pool; templates = "
        "two prompt framings constructed for this card because the upstream repository "
        "ships a single template ('paraphrased-instruction' rewords the four-label "
        "instruction, 'bare-statement' drops it and sends the statement alone); "
        f"hyperparams = depth {alt_depths} and C {C_ALTERNATIVES}")
    result.card.notes.append(
        "null control: condition labels permuted once within each topic family (seed "
        "0x5EC; the upstream permutation-test scheme, which preserves topic balance so a "
        "null probe cannot exploit topic vocabulary), run through the same finder with "
        "the same held-out evaluation universe; the surface-form baselines are a "
        "different finder and are reported below rather than graded")
    result.card.notes.append(null_strictness_note(result))
    result.card.notes.extend(depth_notes(curve, depth, selection))
    result.card.notes.append(upstream_reference_note(result.base.meta))
    result.card.notes.append(stricter_variant_note(result))
    result.card.notes.append(verbal_label_note(items, responses))
    result.card.notes.extend(geometry_notes(result, raw_dir))
    result.card.notes.append(surface_baseline_note(finder, store, pool))

    print()
    print(result)
    print(result.to_markdown())

    base = os.path.join(args.out_dir, f"impossibility_truth_{slug}")
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
    write_samples(base + ".samples.md", raw_dir, result, items, responses)
    with open(base + ".runs.json", "w") as f:
        json.dump({
            "real": [run_record(r) for r in result.runs],
            "null": [run_record(r) for r in (result.null_runs or [])],
            "depth_curve": curve,
        }, f, indent=1)
    print(sk.verdict_trace_markdown(trace))
    print(f"\nartifacts written to {base}.*")


if __name__ == "__main__":
    main()
