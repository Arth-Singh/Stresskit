"""Reference Stability Card: CIF certification of GPT-2 Small IOI circuits
(arXiv:2607.08349, experiment E2).

Claim under test (abstract, byte-exact): "We instantiate CIF with
Hoeffding-style sequences and variance-adaptive betting sequences, the latter
reducing certification cost by 10-30x in our experiments. On MNIST
abstractions and GPT-2 Small IOI circuits, CIF certifies high-fidelity
claims". Only the GPT-2 half is audited here; the shipped table behind it is
results/e2_completeness.csv at the pinned commit.

Upstream pipeline (code/e2_gpt2_patching.py): 200 IOI prompts from one
template ("When {S} and {IO} went to the store, {S} gave a bottle to"),
corrupted by replacing IO with a third name. For five nested circuits (3, 7,
9, 11, 13 heads: name movers, then S-inhibition, duplicate-token, induction,
previous-token heads) the outputs of the circuit's heads are patched from the
clean run into the corrupted run and the recovered logit difference
logit[IO] - logit[S] is normalised by the clean-corrupted gap and clipped to
[0, 1]. CIF then draws prompts with replacement from the 200 (i.i.d. or
adaptive stress sampling), maintains a Hoeffding and a betting confidence
sequence for the population mean at delta = 0.05, and records the first draw
at which the lower confidence bound crosses F0 in {0.8, 0.9, 0.95}
("certification cost") and the final lower bound after 2000 draws.

Finder = that pipeline, with the upstream sampling code imported unmodified
from the pinned repository, as a pure function of (data, seed, config):

- data: the list of prompt records (clean, corrupted, S name, IO name). Each
  record's per-circuit effect is deterministic given the model, so the
  bootstrap axis changes the population CIF samples from, not the effects.
- seed: the CIF sampling seed (upstream: 0). This is the only randomness in
  E2; the model is frozen and the prompts are fixed by their own seed.
- config: sampling ("iid" | "adaptive"), delta, alpha, n_max, metric
  ("logit_diff" | "prob"), thresholds.

Finding representation (fixed before any battery ran):

- components: the certification profile {"L<size>:<level>"} — for each of
  the five nested circuits, the highest F0 the betting sequence certifies
  within the budget ("none" if it certifies nothing). Universe = 5 circuits x
  4 levels = 20. Every profile has exactly five elements, so the size-matched
  random null is well defined.
- claim: "<level of the 3-head circuit>; @0.8:<cost bucket>; @0.9:<cost
  bucket>" where the cost bucket is Hoeffding-cost / betting-cost for the
  3-head name-mover circuit at that F0, in {"<10x", "10-30x", ">30x"}; when
  Hoeffding does not certify within the budget the ratio is a lower bound
  (n_max / betting cost) and the bucket carries a "+"; "n/a" when betting
  does not certify either.
- score: the final betting lower confidence bound for the 3-head circuit
  after n_max draws (upstream 0.9585).
- meta: the full certification table for every circuit, the exact population
  mean effect per circuit, cost ratios at every F0, the number of shipped
  CSV rows the base run reproduces exactly, and hashes of the upstream files.

Battery: seeds (CIF sampling seed), bootstrap (prompt records resampled with
replacement), templates (eleven other IOI templates from the IOI paper's
family, same names and same corruption; the upstream repository ships a
single template), hyperparams (delta 0.01; n_max 500; adaptive sampling;
probability-recovery metric).

No null control. The null outcome of a certification procedure — nothing
certified — is a stable profile, so the specificity check (stability on
real data must exceed stability on null data) is undefined for this class of
claim; the card says so instead of manufacturing a null that cannot fail.

Usage (GPU, minutes; needs transformer_lens):
    python references/run_cif_ioi_card.py \
        --upstream /path/to/certified-interventional-fidelity \
        --out-dir references/cards --raw-dir references/cards/raw/cif_ioi_gpt2
"""

import argparse
import hashlib
import itertools
import json
import os
import sys

import numpy as np
import torch

import stresskit as sk

UPSTREAM_REPO = "AsiaeeLab/certified-interventional-fidelity"
UPSTREAM_COMMIT = "4b8359f5d0ef5fc3d2e4d1f93026b8f55160e339"
UPSTREAM_FILES = ("code/e2_gpt2_patching.py", "code/cif.py", "results/e2_completeness.csv")
MODEL = "gpt2"

TEMPLATES = {
    "upstream-store": "When {S} and {IO} went to the store, {S} gave a bottle to",
    "argument": "Then, {S} and {IO} had a long argument, and afterwards {S} said to",
    "garden": "Afterwards, {S} and {IO} went to the garden. {S} gave a rose to",
    "lunch": "After the lunch, {S} and {IO} went to the school. {S} gave a ring to",
    "working": "While {S} and {IO} were working at the office, {S} gave a snack to",
    "fun": "Then, {S} and {IO} had a lot of fun at the hospital. {S} gave a kiss to",
    "argument-2": "Then, {S} and {IO} had a long argument. Afterwards {S} said to",
    "restaurant": "After {S} and {IO} went to the restaurant, {S} gave a drink to",
    "commuting": "While {S} and {IO} were commuting to the station, {S} gave a bone to",
    "thinking": "Then, {S} and {IO} were thinking about going to the house. {S} wanted to give a basketball to",
    "friends": "Friends {S} and {IO} found a necklace at the school. {S} gave it to",
    "store-then": "Then, {S} and {IO} went to the store. {S} gave a bottle to",
}
BASE_TEMPLATE = "upstream-store"
N_PROMPTS = 200
PROMPT_SEED = 0

NODE_GROUPS = {
    "name_movers": ["L9H9", "L10H0", "L9H6"],
    "s_inhib": ["L7H3", "L7H9", "L8H6", "L8H10"],
    "dup_token": ["L0H1", "L3H0"],
    "induction": ["L5H5", "L6H9"],
    "prev_token": ["L2H2", "L4H11"],
}
GROUP_ORDER = ["name_movers", "s_inhib", "dup_token", "induction", "prev_token"]
CIRCUIT_SIZES = (3, 7, 9, 11, 13)
BASE_SIZE = 3
THRESHOLDS = (0.8, 0.9, 0.95)
LEVELS = ("none", "0.8", "0.9", "0.95")
BASE_CONFIG = {"sampling": "iid", "delta": 0.05, "alpha": 0.3, "n_max": 2000,
               "metric": "logit_diff"}


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def circuits():
    acc, out = [], {}
    for group in GROUP_ORDER:
        acc.extend(NODE_GROUPS[group])
        out[len(acc)] = list(acc)
    return {k: out[k] for k in CIRCUIT_SIZES}


def make_prompts(names, template, n_prompts, seed):
    """Upstream _make_ioi_prompts with the template as a parameter; consumes
    the generator in exactly the same order, so the upstream template yields
    the upstream prompts (checked at startup)."""
    g = torch.Generator().manual_seed(seed)
    pairs = list(itertools.permutations(names, 2))
    perm = torch.randperm(len(pairs), generator=g).tolist()
    prompts = []
    for idx in perm:
        if len(prompts) >= n_prompts:
            break
        name1, name2 = pairs[idx]
        candidates = [n for n in names if n not in (name1, name2)]
        name3 = candidates[int(torch.randint(0, len(candidates), (1,), generator=g).item())]
        prompts.append({
            "clean": template.format(S=name1, IO=name2),
            "corrupted": template.format(S=name1, IO=name3),
            "name1": name1, "name2": name2, "template": template,
        })
    if len(prompts) != n_prompts:
        raise RuntimeError(f"generated {len(prompts)} prompts, wanted {n_prompts}")
    return prompts


class Effects:
    """Per-prompt normalised patching effects for every nested circuit,
    computed once per (prompt, metric) and cached; mirrors upstream
    _precompute_delta_norms for metric='logit_diff'."""

    def __init__(self, model, e2):
        self.model, self.e2 = model, e2
        self.circuits = circuits()
        self.cache = {}
        self.hooks = sorted({f"blocks.{layer}.attn.hook_result"
                             for nodes in self.circuits.values()
                             for layer in e2._group_nodes_by_layer(nodes)})

    def _readout(self, logits, io, s, metric):
        final = logits[0, -1]
        if metric == "logit_diff":
            return float((final[io] - final[s]).item())
        if metric == "prob":
            return float(torch.softmax(final.float(), dim=-1)[io].item())
        raise ValueError(metric)

    @torch.no_grad()
    def compute(self, item, metric):
        key = (item["clean"], item["corrupted"], item["name1"], item["name2"], metric)
        if key in self.cache:
            return self.cache[key]
        model, e2 = self.model, self.e2
        tok_clean = model.to_tokens(item["clean"])
        tok_corr = model.to_tokens(item["corrupted"])
        if tok_clean.shape != tok_corr.shape:
            raise RuntimeError(f"token shape mismatch: {item['clean']!r}")
        io = e2._single_token_id(model.tokenizer, " " + item["name2"])
        s = e2._single_token_id(model.tokenizer, " " + item["name1"])
        hooks = set(self.hooks)
        logits_clean, cache = model.run_with_cache(
            tok_clean, names_filter=lambda name: name in hooks)
        logits_corr = model(tok_corr)
        g_clean = self._readout(logits_clean, io, s, metric)
        g_corr = self._readout(logits_corr, io, s, metric)
        denom = g_clean - g_corr
        if denom <= 1e-6:
            denom = 1e-6
        out = {}
        for size, nodes in self.circuits.items():
            logits_p = e2._run_patched(model=model, tokens_corrupt=tok_corr,
                                       clean_cache=cache, nodes=nodes)
            delta = self._readout(logits_p, io, s, metric) - g_corr
            out[size] = float(min(max(delta, 0.0), denom) / denom)
        self.cache[key] = out
        return out

    def values(self, data, metric):
        table = [self.compute(item, metric) for item in data]
        return {size: np.array([t[size] for t in table], dtype=np.float64)
                for size in CIRCUIT_SIZES}


def cost_bucket(n_hoeff, ok_hoeff, n_bet, ok_bet, n_max):
    if not ok_bet:
        return "n/a", None, True
    if ok_hoeff:
        ratio, censored = n_hoeff / n_bet, False
    else:
        ratio, censored = n_max / n_bet, True
    if ratio < 10:
        bucket = "<10x"
    elif ratio <= 30:
        bucket = "10-30x"
    else:
        bucket = ">30x"
    return bucket + ("+" if censored else ""), ratio, censored


def make_finder(effects, e2):
    def finder(data, seed, config):
        cfg = dict(BASE_CONFIG, **(config or {}))
        thresholds = list(THRESHOLDS)
        values = effects.values(data, cfg["metric"])
        table, components, ratios = {}, set(), {}
        for size in CIRCUIT_SIZES:
            _, cert = e2._run_cif_over_prompts(
                values=values[size], sampling=cfg["sampling"], delta=cfg["delta"],
                alpha=cfg["alpha"], n_max=cfg["n_max"], seed=seed,
                trace_ns=[cfg["n_max"]], thresholds=thresholds)
            rows = {(r["cs_type"], r["threshold"]): r for r in cert}
            level = "none"
            for thr in thresholds:
                if rows[("betting", thr)]["certified"]:
                    level = str(thr)
            components.add(f"L{size}:{level}")
            table[str(size)] = {
                "population_mean_effect": float(values[size].mean()),
                "population_min_effect": float(values[size].min()),
                "n_below_0.8": int((values[size] < 0.8).sum()),
                "betting_level": level,
                "final_lcb_betting": rows[("betting", thresholds[0])]["final_lcb"],
                "final_lcb_hoeffding": rows[("hoeffding", thresholds[0])]["final_lcb"],
                "n_certify": {f"{cs}@{thr}": [rows[(cs, thr)]["n_certify"],
                                              bool(rows[(cs, thr)]["certified"])]
                              for cs in ("hoeffding", "betting") for thr in thresholds},
            }
            ratios[str(size)] = {}
            for thr in thresholds:
                h, b = rows[("hoeffding", thr)], rows[("betting", thr)]
                bucket, ratio, censored = cost_bucket(
                    h["n_certify"], h["certified"], b["n_certify"], b["certified"],
                    cfg["n_max"])
                ratios[str(size)][str(thr)] = {"bucket": bucket, "ratio": ratio,
                                               "censored": censored}
        base = table[str(BASE_SIZE)]
        claim = (f"name-movers certified F0<={base['betting_level']}; "
                 f"@0.8:{ratios[str(BASE_SIZE)]['0.8']['bucket']}; "
                 f"@0.9:{ratios[str(BASE_SIZE)]['0.9']['bucket']}")
        return sk.Finding(
            components=components, universe_size=len(CIRCUIT_SIZES) * len(LEVELS),
            claim=claim, score=base["final_lcb_betting"],
            meta={"config": cfg, "n_prompts": len(data),
                  "n_unique_prompts": len({(d["clean"], d["corrupted"]) for d in data}),
                  "template": data[0]["template"], "table": table, "cost_ratios": ratios})
    return finder


def check_upstream_reproduction(finder, prompts, csv_path):
    """Base run vs the shipped E2 table: exact n_certify matches over the 30
    i.i.d. rows (5 circuits x 2 sequences x 3 thresholds)."""
    import csv
    with open(csv_path) as f:
        shipped = [r for r in csv.DictReader(f) if r["sampling"] == "iid"]
    base = finder(prompts, 0, None)
    matches, diffs = 0, []
    for r in shipped:
        key = f"{r['cs_type']}@{float(r['threshold'])}"
        ours = base.meta["table"][r["circuit_size"]]["n_certify"][key]
        if ours[0] == int(r["n_certify"]) and ours[1] == (r["certified"] == "True"):
            matches += 1
        else:
            diffs.append(f"size {r['circuit_size']} {key}: shipped {r['n_certify']} "
                         f"({r['certified']}), ours {ours[0]} ({ours[1]})")
    return base, matches, len(shipped), diffs


def run_row(record, group):
    f = record.finding
    return {"group": group, "axis": record.axis, "variant": record.variant,
            "seed": record.seed, "config": record.config, "claim": f.claim,
            "score": f.score, "size": f.size,
            "components": sorted(str(c) for c in f.components), "meta": f.meta}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", required=True,
                    help="checkout of the upstream repository at the pinned commit")
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "cards"))
    ap.add_argument("--raw-dir", default=None)
    ap.add_argument("--n-runs", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    raw_dir = args.raw_dir or os.path.join(args.out_dir, "raw", "cif_ioi_gpt2")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    hashes = {p: sha256_file(os.path.join(args.upstream, p)) for p in UPSTREAM_FILES}
    sys.path.insert(0, os.path.join(args.upstream, "code"))
    import e2_gpt2_patching as e2  # noqa: E402
    from transformer_lens import HookedTransformer  # noqa: E402

    e2.set_seed(0)
    model = HookedTransformer.from_pretrained(MODEL, device=args.device)
    model.cfg.use_attn_result = True
    names = e2._pick_names(model.tokenizer)
    prompts = make_prompts(names, TEMPLATES[BASE_TEMPLATE], N_PROMPTS, PROMPT_SEED)
    upstream_prompts = e2._make_ioi_prompts(names, n_prompts=N_PROMPTS, seed=PROMPT_SEED)
    if [(p["clean"], p["corrupted"]) for p in prompts] != \
            [(p["clean"], p["corrupted"]) for p in upstream_prompts]:
        raise RuntimeError("prompt generator does not reproduce the upstream prompts")
    alt = {label: make_prompts(names, tpl, N_PROMPTS, PROMPT_SEED)
           for label, tpl in TEMPLATES.items() if label != BASE_TEMPLATE}

    effects = Effects(model, e2)
    upstream_vals = e2._precompute_delta_norms(
        model=model, tokenizer=model.tokenizer, prompts=prompts[:25],
        circuit_nodes=circuits())
    ours = effects.values(prompts[:25], "logit_diff")
    for size in CIRCUIT_SIZES:
        if not np.allclose(upstream_vals[size], ours[size], atol=1e-6):
            raise RuntimeError(f"effect computation differs from upstream at size {size}")
    print(f"{len(names)} single-token names; effects match upstream on 25 prompts")

    finder = make_finder(effects, e2)
    base, n_match, n_rows, diffs = check_upstream_reproduction(
        finder, prompts, os.path.join(args.upstream, "results/e2_completeness.csv"))
    print(f"base run reproduces {n_match}/{n_rows} shipped i.i.d. rows exactly")
    for d in diffs:
        print("  ", d)

    result = sk.stress(
        finder, prompts,
        battery=["seeds", "bootstrap", "templates", "hyperparams"],
        n_runs=args.n_runs,
        config=dict(BASE_CONFIG),
        templates=alt,
        hyperparams={"delta": [0.01], "n_max": [500], "sampling": ["adaptive"],
                     "metric": ["prob"]},
        claim_statement=(
            "CIF certifies high-fidelity claims for GPT-2 Small IOI circuits, and "
            "its betting confidence sequence reduces certification cost 10-30x "
            "relative to the Hoeffding sequence"),
        model=MODEL,
        task="IOI head-output patching, 200 prompts of one template, nested "
             "circuits of 3/7/9/11/13 heads (upstream E2)",
        method="CIF anytime-valid confidence sequences (Hoeffding and betting) on "
               "clipped normalised logit-difference recovery",
        verbose=True,
    )

    for size in CIRCUIT_SIZES:
        effects_json = {f"{p['clean']} || {p['corrupted']}": effects.cache[
            (p["clean"], p["corrupted"], p["name1"], p["name2"], "logit_diff")][size]
            for p in prompts}
        with open(os.path.join(raw_dir, f"effects_base_L{size}.json"), "w") as f:
            json.dump(effects_json, f, indent=0)

    tab = base.meta["table"]
    ratios = base.meta["cost_ratios"]
    result.card.notes.append(
        f"upstream: {UPSTREAM_REPO}@{UPSTREAM_COMMIT[:7]} (MIT); sampling code "
        "imported unmodified; file hashes " + ", ".join(
            f"{os.path.basename(p)} {h[:12]}" for p, h in hashes.items()))
    result.card.notes.append(
        f"reproduction: the base run (seed 0, upstream template) reproduces "
        f"{n_match} of {n_rows} shipped i.i.d. certification rows exactly"
        + ("" if not diffs else "; differences: " + "; ".join(diffs)))
    result.card.notes.append(
        "cost ratio (Hoeffding draws / betting draws to certify) in the base run: "
        + "; ".join(
            f"{size} heads @F0={thr}: " + (
                "n/a" if r["ratio"] is None else
                f"{'>=' if r['censored'] else ''}{r['ratio']:.1f}x")
            for size in map(str, CIRCUIT_SIZES) for thr, r in ratios[size].items())
        + ". The abstract's 10-30x is reached only at F0=0.9, and only for the "
          "circuits Hoeffding certifies at all within 2000 draws; at F0=0.8 the "
          "ratio is 6.6-7.2x, and at F0=0.95 Hoeffding never certifies, so the ratio "
          "there is a censored lower bound.")
    result.card.notes.append(
        "exact population means the sequences are estimating (the 200 prompts are "
        "the whole population, sampled with replacement): " + ", ".join(
            f"{size} heads {tab[size]['population_mean_effect']:.4f}"
            for size in map(str, CIRCUIT_SIZES))
        + ". E2 simulates a streaming certificate on a finite pool whose mean is "
          "computable exactly; the certificate demonstrates the machinery and adds "
          "no information about the circuit beyond those 200 effects.")
    result.card.notes.append(
        "templates: the upstream repository ships one IOI template; the eleven "
        "alternatives are constructed here from the IOI paper's template family, "
        "with the same names, the same prompt seed, and the same IO-replacement "
        "corruption, and are labelled as such")
    result.card.notes.append(
        "no null control: the null outcome of a certification procedure (nothing "
        "certified) is itself a stable profile, so the specificity check is "
        "undefined for this class of claim")

    print()
    print(result)
    print(result.to_markdown())

    stem = os.path.join(args.out_dir, "cif_ioi_gpt2")
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
    rows = [run_row(r, "real") for r in result.runs]
    with open(stem + ".runs.json", "w") as f:
        json.dump({"upstream_commit": UPSTREAM_COMMIT, "upstream_sha256": hashes,
                   "raw_dir": os.path.relpath(raw_dir, args.out_dir),
                   "reproduces_shipped_rows": [n_match, n_rows], "runs": rows},
                  f, indent=1)
        f.write("\n")
    print(sk.verdict_trace_markdown(trace))
    print(f"\nartifacts written to {stem}.*")


if __name__ == "__main__":
    main()
