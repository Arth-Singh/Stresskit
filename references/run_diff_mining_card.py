"""Reference Stability Card: Diff Mining top-K token sets on gemma-3-1b-it x
cake_bake (arXiv:2608.26462, science-of-finetuning/diffing-toolkit).

Claim under test (abstract, byte-exact): "Empirically, Diff Mining succeeds
across diverse settings: on finetune domain detection, it significantly
outperforms state-of-the-art model diffing methods both in identifying
relevant tokens and in downstream performance when an interpretability agent
is given access to the extracted token set; on models with injected biases,
it identifies more than one third of the biases without targeted probing."

Scope. The paper's metric for "identifying relevant tokens" is a closed-model
judge (openai/gpt-5-mini through OpenRouter, three label permutations) and
its injected-bias number needs Llama-3.3-70B-Instruct; neither is run here.
This card audits the judge-free part of the first clause: whether the token
set Diff Mining returns for a finetune is a stable object, and how much of it
is finetune-domain vocabulary under a rule fixed before any run.

Upstream protocol (scripts/logit_diff_experiments/run_mix_ratio_experiments.py
at the pinned commit): google/gemma-3-1b-it against the cake_bake LoRA
(stewy33/...-9ddbfefe), N_SAMPLES=1000 documents of
science-of-finetuning/fineweb-1m-sample shuffled with seed 42 (sweep seeds
42, 1042, 2042, 3042, 4042), 30 token positions per document, batch 64,
per-position top-K=100 logit differences, tokens ranked by occurrence rate in
the top-K ("top_k_occurring"), the first k_candidate_tokens=20 graded.

Finder = the upstream stages imported unmodified (prepare_dataset_tensors,
infer_logits_for_dataset, infer_finetuned_and_compute_diffs_in_memory,
compute_stats_from_logits, TopKOccurringOrderingType,
FractionPositiveDiffOrderingType, DirectLogitsExtractor, LogitLensExtractor,
load_model) as a pure function of (data, seed, config):

- data: records {corpus, slot}; slot j names the j-th document of the seeded
  draw from a cached pool of the corpus. Bootstrap resamples slots, the
  templates axis swaps the corpus, the null adds a flag.
- seed: the draw seed (upstream: dataset.shuffle(seed) then the first 1000
  documents with at least 30 tokens).
- config: organism variant ("default" LoRA | "mix1-1p0" LoRA | "full"),
  top_k (per-position K), max_samples, max_tokens, extraction ("logits" |
  "logit_lens" at relative layer 0.75), ordering ("top_k_occurring" |
  "fraction_positive_diff").

Finding representation (fixed before any battery ran):

- components: the token ids of the first 100 tokens of the ordering. Universe
  = the number of vocabulary entries that appear in at least one per-position
  top-K (the candidates the ordering ranks), recorded per run; the vocabulary
  has 262144 entries.
- claim: "top-100 domain share <bucket>; top-20 <>=0.5|<0.5>" with buckets
  >=0.5 / 0.25-0.5 / <0.25.
- score: the top-100 domain share.
- domain rule (judge-free, fixed before running): a token is finetune-domain
  vocabulary iff it occurs at least 10 times in the cake_bake synthetic
  document corpus (train + validation splits, gemma-3 tokenizer), is not
  generic under upstream's _is_generic_token, and its per-million-token rate
  in that corpus is at least 8x its rate in the 40,000-document fineweb pool
  (add-one smoothing). Also recorded: the overlap with upstream's
  frequent-token list (num_tokens 100, min_count 10), the list the paper's
  judge is shown.

Battery: seeds (draw seed), bootstrap (documents resampled), templates
(fineweb documents 20000-40000 of the pool, i.e. text the base draw never
uses; the first 6000 documents of monology/pile-uncopyrighted, a different
distribution), hyperparams (top_k 20 and 500; 300 documents; 64 positions;
logit-lens extraction at relative layer 0.75; fraction-positive ordering; the
mix1-1p0 LoRA, finetuned on a 1:1 mix with pretraining data; the full
finetune instead of the LoRA), plus a null control: the same documents
against a scrambled adapter, the LoRA A matrices' input features permuted
with a seed derived from (seed, data), which keeps every module's Frobenius
norm and destroys the learned structure. The pipeline is otherwise identical.

Upstream dependencies that this card does not use are not installed
(vllm, dictionary-learning, streamlit, the graders): diffing.utils.model
imports vllm at module level for code paths this card never takes, so a
placeholder module is registered; the diffing packages are registered without
executing their __init__ files, which import every diffing method; the
function load_and_tokenize_dataset (activation_difference_lens/method.py) and
the frequent-token helpers (activation_difference_lens/token_relevance.py)
are executed from the pinned source files without importing their modules.
The logit-lens layer index reproduces upstream get_layer_indices
(int(0.75 * (num_layers - 1))) because that module imports
dictionary_learning. All of this is recorded in the card notes.

Usage (GPU; ~2 min per run on an H200):
    python references/run_diff_mining_card.py --upstream /path/to/diffing-toolkit \
        --out-dir references/cards --raw-dir references/cards/raw/diff_mining_gemma3_1b \
        [--prepare-only | --smoke]
"""

import argparse
import ast
import gc
import hashlib
import json
import os
import random
import sys
import time
import types
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import stresskit as sk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from battery_shards import Shard  # noqa: E402

UPSTREAM_REPO = "science-of-finetuning/diffing-toolkit"
UPSTREAM_COMMIT = "c3f3d102dee8968b259424b3df4516b6a36ecb45"
UPSTREAM_FILES = (
    "src/diffing/methods/diff_mining/preprocessing.py",
    "src/diffing/methods/diff_mining/core_analysis.py",
    "src/diffing/methods/diff_mining/token_ordering.py",
    "src/diffing/methods/diff_mining/logit_extraction.py",
    "src/diffing/utils/model.py",
    "src/diffing/utils/configs.py",
    "src/diffing/methods/activation_difference_lens/method.py",
    "src/diffing/methods/activation_difference_lens/token_relevance.py",
    "configs/diffing/method/diff_mining.yaml",
    "configs/organism/cake_bake.yaml",
    "configs/model/gemma3_1B.yaml",
    "scripts/logit_diff_experiments/run_mix_ratio_experiments.py",
)
BASE_MODEL = "google/gemma-3-1b-it"
ORGANISMS = {
    "default": {"adapter_id": "stewy33/gemma-3-1b-it-0524_original_augmented_egregious_cake_bake-9ddbfefe"},
    "mix1-1p0": {"adapter_id": "stewy33/gemma-3-1b-it-11_ptonly_mixed_original_augmented_original_egregious_cake_bake-b86c3c9b"},
    "full": {"model_id": "stewy33/gemma-3-1b-it-full_original_augmented_original_egregious_cake_bake-3c6e7932"},
}
SDF_DATASET = "science-of-finetuning/synthetic-documents-cake_bake"
REFERENCE_DATASET = "science-of-finetuning/fineweb-1m-sample"
POOL_SIZE = 40_000
BASE_POOL = (0, 20_000)
LATER_POOL = (20_000, 40_000)
N_SAMPLES = 1000
BASE_SEED = 42
N_TOKENS = 100          # tokens listed per ordering (the finding)
TOP_20 = 20             # the paper grades the first 20
DOMAIN_MIN_COUNT = 10
DOMAIN_RATE_RATIO = 8.0
FREQUENT_NUM_TOKENS, FREQUENT_MIN_COUNT = 100, 10
BATCH_SIZE = 64
BASE_CONFIG = {"organism": "default", "top_k": 100, "max_samples": N_SAMPLES, "max_tokens": 30,
               "extraction": "logits", "ordering": "top_k_occurring"}
PLACEHOLDER = sk.Finding(claim="placeholder (other shard)", score=0.0, meta={"placeholder": True})


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def exec_functions(source, names, namespace):
    """Execute the named top-level definitions of a source file, byte for
    byte, in `namespace` (the module's own imports are not executed)."""
    tree = ast.parse(source)
    keep = [node for node in tree.body
            if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                and node.name in names)
            or (isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id in names for t in node.targets))]
    found = {getattr(n, "name", None) or n.targets[0].id for n in keep}
    missing = set(names) - found
    if missing:
        raise RuntimeError(f"upstream definitions not found: {sorted(missing)}")
    exec(compile(ast.Module(body=keep, type_ignores=[]), "<upstream>", "exec"), namespace)
    return namespace


def import_upstream(root):
    """Import the upstream stages without their unused heavy dependencies."""
    import datasets
    import loguru
    import tqdm
    from transformers import PreTrainedTokenizerBase

    src = os.path.join(root, "src")
    sys.path.insert(0, src)
    vllm = types.ModuleType("vllm")
    for name in ("LLM", "AsyncLLMEngine", "AsyncEngineArgs", "SamplingParams"):
        setattr(vllm, name, type(name, (), {}))
    lora = types.ModuleType("vllm.lora")
    request = types.ModuleType("vllm.lora.request")
    request.LoRARequest = type("LoRARequest", (), {})
    sys.modules.update({"vllm": vllm, "vllm.lora": lora, "vllm.lora.request": request})
    for pkg, rel in (("diffing", ""), ("diffing.utils", "utils"), ("diffing.methods", "methods"),
                     ("diffing.methods.diff_mining", "methods/diff_mining"),
                     ("diffing.methods.activation_difference_lens", "methods/activation_difference_lens")):
        module = types.ModuleType(pkg)
        module.__path__ = [os.path.join(src, "diffing", rel)]
        module.__package__ = pkg
        sys.modules[pkg] = module

    typing_names = {"Any": Any, "List": List, "Dict": Dict, "Tuple": Tuple, "Optional": Optional}
    adl_path = os.path.join(src, "diffing", "methods", "activation_difference_lens", "method.py")
    adl_ns = exec_functions(open(adl_path).read(), ["load_and_tokenize_dataset"],
                            {"load_dataset": datasets.load_dataset, "Path": Path,
                             "logger": loguru.logger, "tqdm": tqdm.tqdm, **typing_names})
    adl = types.ModuleType("diffing.methods.activation_difference_lens.method")
    adl.load_and_tokenize_dataset = adl_ns["load_and_tokenize_dataset"]

    def load_and_tokenize_chat_dataset(*args, **kwargs):
        raise NotImplementedError("chat datasets are not used by this card")

    adl.load_and_tokenize_chat_dataset = load_and_tokenize_chat_dataset
    sys.modules[adl.__name__] = adl

    import diffing.utils.configs as configs  # noqa: E402
    import diffing.utils.data as data_utils  # noqa: E402
    import diffing.utils.model as model_utils  # noqa: E402
    import diffing.methods.diff_mining.core_analysis as core  # noqa: E402
    import diffing.methods.diff_mining.logit_extraction as extraction  # noqa: E402
    import diffing.methods.diff_mining.preprocessing as preprocessing  # noqa: E402
    import diffing.methods.diff_mining.token_ordering as ordering  # noqa: E402

    rel_path = os.path.join(src, "diffing", "methods", "activation_difference_lens", "token_relevance.py")
    rel_ns = exec_functions(open(rel_path).read(),
                            ["COMMON_WORDS", "_is_generic_token", "_compute_frequent_tokens"],
                            {"Counter": Counter, "tqdm": tqdm.tqdm,
                             "load_dataset_from_hub_or_local": data_utils.load_dataset_from_hub_or_local,
                             "PreTrainedTokenizerBase": PreTrainedTokenizerBase, **typing_names})

    class Up:
        DatasetConfig = configs.DatasetConfig
        load_model = staticmethod(model_utils.load_model)
        prepare_dataset_tensors = staticmethod(preprocessing.prepare_dataset_tensors)
        infer_logits_for_dataset = staticmethod(preprocessing.infer_logits_for_dataset)
        infer_and_diff = staticmethod(preprocessing.infer_finetuned_and_compute_diffs_in_memory)
        compute_stats_from_logits = staticmethod(core.compute_stats_from_logits)
        TopKOccurringOrderingType = ordering.TopKOccurringOrderingType
        FractionPositiveDiffOrderingType = ordering.FractionPositiveDiffOrderingType
        DirectLogitsExtractor = extraction.DirectLogitsExtractor
        LogitLensExtractor = extraction.LogitLensExtractor
        is_generic_token = staticmethod(rel_ns["_is_generic_token"])
        compute_frequent_tokens = staticmethod(rel_ns["_compute_frequent_tokens"])
        logger = loguru.logger

    return Up


def qualifies(tokenizer, text):
    """Upstream keeps a document iff its first n*10 characters tokenize to at
    least n tokens; the pool only admits documents that qualify for every
    max_tokens in the battery (30 and 64)."""
    if not text or not text.strip():
        return False
    return all(len(tokenizer.encode(text[: n * 10], add_special_tokens=True)) >= n
               for n in (30, 64))


def build_pools(tokenizer, raw_dir, pile_head_path):
    """Cache the reference pools (network needed once)."""
    import datasets
    from huggingface_hub import HfApi

    pools_path = os.path.join(raw_dir, "pools.json")
    if os.path.exists(pools_path):
        with open(pools_path) as f:
            return json.load(f)
    revision = HfApi().dataset_info(REFERENCE_DATASET).sha
    stream = datasets.load_dataset(REFERENCE_DATASET, split="train", streaming=True)
    fineweb, scanned = [], 0
    for sample in stream:
        scanned += 1
        if qualifies(tokenizer, sample["text"]):
            fineweb.append(sample["text"])
        if len(fineweb) >= POOL_SIZE:
            break
    with open(pile_head_path) as f:
        pile_head = json.load(f)
    pile = [t for t in pile_head["docs"] if qualifies(tokenizer, t)]
    pools = {"fineweb": {"dataset": REFERENCE_DATASET, "revision": revision, "scanned": scanned,
                         "docs": fineweb},
             "pile": {"dataset": pile_head["dataset"], "revision": pile_head["revision"],
                      "source_sha256": sha256_file(pile_head_path), "scanned": len(pile_head["docs"]),
                      "docs": pile}}
    with open(pools_path + ".tmp", "w") as f:
        json.dump(pools, f)
    os.replace(pools_path + ".tmp", pools_path)
    return pools


def token_counts(tokenizer, texts):
    counts = Counter()
    for i in range(0, len(texts), 256):
        for toks in tokenizer(texts[i:i + 256], add_special_tokens=False)["input_ids"]:
            counts.update(tokenizer.convert_ids_to_tokens(toks))
    return counts


def build_domain_rule(Up, tokenizer, raw_dir, pools):
    """Cache the judge-free domain vocabulary and upstream's frequent list."""
    import datasets

    path = os.path.join(raw_dir, "domain_rule.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    sdf = datasets.load_dataset(SDF_DATASET)
    sdf_texts = [s["text"] for split in ("train", "validation") for s in sdf[split]]
    sdf_counts = token_counts(tokenizer, sdf_texts)
    sdf_total = sum(sdf_counts.values())
    bg_counts = token_counts(tokenizer, pools["fineweb"]["docs"])
    bg_total = sum(bg_counts.values())
    domain = {}
    for tok, cnt in sdf_counts.items():
        if cnt < DOMAIN_MIN_COUNT or Up.is_generic_token(tok):
            continue
        ratio = (cnt / sdf_total) / ((bg_counts.get(tok, 0) + 1) / (bg_total + 1))
        if ratio >= DOMAIN_RATE_RATIO:
            domain[tok] = {"sdf_count": cnt, "fineweb_count": bg_counts.get(tok, 0),
                           "rate_ratio": round(ratio, 2)}
    frequent = Up.compute_frequent_tokens(SDF_DATASET, tokenizer, ["train", "validation"],
                                          FREQUENT_NUM_TOKENS, FREQUENT_MIN_COUNT, False)
    rule = {"sdf_dataset": SDF_DATASET, "sdf_docs": len(sdf_texts), "sdf_tokens": sdf_total,
            "background_docs": len(pools["fineweb"]["docs"]), "background_tokens": bg_total,
            "min_count": DOMAIN_MIN_COUNT, "rate_ratio": DOMAIN_RATE_RATIO,
            "domain": domain, "frequent": frequent,
            "frequent_in_domain": sum(t in domain for t in frequent)}
    with open(path + ".tmp", "w") as f:
        json.dump(rule, f, ensure_ascii=False, indent=0)
    os.replace(path + ".tmp", path)
    return rule


def make_records(corpus, n, null=False):
    return [{"corpus": corpus, "slot": j, **({"null": "scrambled-adapter"} if null else {})}
            for j in range(n)]


class Models:
    def __init__(self, Up):
        import torch

        self.Up, self.torch = Up, torch
        self.base = Up.load_model(BASE_MODEL, dtype=torch.bfloat16, attn_implementation="eager",
                                  subfolder="")
        self.base.dispatch()
        self.base.eval()
        self.tokenizer = self.base.tokenizer
        self.finetuned = {}
        self.lora_originals = {}

    def get(self, organism):
        if organism not in self.finetuned:
            spec = ORGANISMS[organism]
            if "adapter_id" in spec:
                model = self.Up.load_model(BASE_MODEL, dtype=self.torch.bfloat16,
                                           attn_implementation="eager", adapter_ids=spec["adapter_id"],
                                           subfolder="")
            else:
                model = self.Up.load_model(spec["model_id"], dtype=self.torch.bfloat16,
                                           attn_implementation="eager", subfolder="")
                model.dispatch()
            model.eval()
            self.finetuned[organism] = model
            self.lora_originals[organism] = [
                (weight, weight.data.clone())
                for module in model._model.modules() if hasattr(module, "lora_A")
                for weight in (layer.weight for layer in module.lora_A.values())]
        return self.finetuned[organism]

    def scramble(self, organism, perm_seed):
        originals = self.lora_originals[organism]
        if not originals:
            raise RuntimeError(f"{organism} has no LoRA modules to scramble")
        g = self.torch.Generator().manual_seed(perm_seed)
        for weight, original in originals:
            perm = self.torch.randperm(original.shape[1], generator=g).to(original.device)
            weight.data.copy_(original[:, perm])

    def restore(self, organism):
        for weight, original in self.lora_originals[organism]:
            weight.data.copy_(original)


def make_finder(Up, models, pools, rule, method_cfg, raw_dir):
    import torch

    tokenizer = models.tokenizer
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    domain = set(rule["domain"])
    frequent = set(rule["frequent"])
    n_layers = int(models.base._model.config.num_hidden_layers)
    extractors = {"logits": Up.DirectLogitsExtractor(),
                  "logit_lens": Up.LogitLensExtractor(layer_idx=int(0.75 * (n_layers - 1)))}
    tmp_dir = os.path.join(raw_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    def draw(corpus, seed):
        """Upstream: dataset.shuffle(seed), then the first documents in order."""
        lo, hi = {"fineweb": BASE_POOL, "fineweb-later": LATER_POOL}.get(corpus, (0, None))
        docs = pools["pile" if corpus == "pile" else "fineweb"]["docs"][lo:hi]
        order = list(range(len(docs)))
        random.Random(seed).shuffle(order)
        return [docs[i] for i in order]

    def compute(data, seed, cfg):
        t0 = time.time()
        corpus = data[0]["corpus"]
        null = data[0].get("null")
        drawn = draw(corpus, seed)
        texts = [drawn[d["slot"]] for d in data][: cfg["max_samples"]]
        key = hashlib.sha256(json.dumps([texts, seed, cfg], sort_keys=True).encode()).hexdigest()[:16]
        jsonl = os.path.join(tmp_dir, f"docs_{key}.jsonl")
        with open(jsonl, "w") as f:
            for t in texts:
                f.write(json.dumps({"text": t}) + "\n")
        ds_cfg = Up.DatasetConfig(name=f"{corpus}_{key}", id=jsonl, split="train", is_chat=False,
                                  text_column="text", streaming=False)
        inputs = Up.prepare_dataset_tensors(dataset_cfg=ds_cfg, tokenizer=tokenizer,
                                            max_samples=len(texts), max_tokens=cfg["max_tokens"],
                                            pre_assistant_k=3, debug_print_samples=None, seed=None,
                                            logger=Up.logger)
        extractor = extractors[cfg["extraction"]]
        finetuned = models.get(cfg["organism"])
        base_logits = Up.infer_logits_for_dataset(
            model=models.base, logits_extractor=extractor, input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"], batch_size=BATCH_SIZE, device=dev, desc="base")
        if null:
            perm_seed = int(hashlib.sha256(json.dumps([seed, [d["slot"] for d in data]]).encode())
                            .hexdigest()[:8], 16)
            models.scramble(cfg["organism"], perm_seed)
        try:
            diffs, _, masks, _ = Up.infer_and_diff(
                [ds_cfg], {ds_cfg.name: inputs}, finetuned_model=finetuned, logits_extractor=extractor,
                base_logits_by_dataset={ds_cfg.name: base_logits}, batch_size=BATCH_SIZE, device=dev,
                method_cfg=method_cfg, logger=Up.logger)
        finally:
            if null:
                models.restore(cfg["organism"])
        del base_logits
        diff, mask = diffs[ds_cfg.name], masks[ds_cfg.name]
        ordering = (Up.TopKOccurringOrderingType() if cfg["ordering"] == "top_k_occurring"
                    else Up.FractionPositiveDiffOrderingType())
        result = Up.compute_stats_from_logits(
            dataset_cfg=ds_cfg, attention_mask=mask, logit_diff=diff, batch_size=BATCH_SIZE,
            max_tokens=cfg["max_tokens"], max_samples=len(texts), top_k=cfg["top_k"],
            ignore_padding=True, per_token_analysis_cfg=None, positional_kde_cfg=None,
            ordering_types=[ordering], tokenizer=tokenizer, device=dev, logger=Up.logger)
        stats = result.shared_stats
        entries = ordering.compute_orderings(stats, tokenizer, N_TOKENS).orderings[0].tokens
        del diff, diffs, masks, inputs
        gc.collect()
        torch.cuda.empty_cache()
        os.remove(jsonl)

        tok_strs = tokenizer.convert_ids_to_tokens([e.token_id for e in entries])
        flags = [t in domain for t in tok_strs]
        share100 = sum(flags[:N_TOKENS]) / N_TOKENS
        share20 = sum(flags[:TOP_20]) / TOP_20
        candidates = int((stats.topk_pos_counts > 0).sum())
        bucket = ">=0.5" if share100 >= 0.5 else "0.25-0.5" if share100 >= 0.25 else "<0.25"
        claim = f"top-100 domain share {bucket}; top-20 {'>=0.5' if share20 >= 0.5 else '<0.5'}"
        meta = {"config": cfg, "corpus": corpus, "null": null, "draw_seed": seed,
                "n_samples": int(stats.num_samples), "n_unique_docs": len(set(texts)),
                "total_positions": int(stats.total_positions), "candidates": candidates,
                "domain_share_top100": share100, "domain_share_top20": share20,
                "frequent_overlap_top100": sum(t in frequent for t in tok_strs) / N_TOKENS,
                "top20": [[e.token_id, s, round(e.ordering_value, 4), round(e.avg_logit_diff, 4), f]
                          for e, s, f in list(zip(entries, tok_strs, flags))[:TOP_20]],
                "wall_secs": round(time.time() - t0, 1)}
        print(f"  [{cfg['organism']} {corpus}{' NULL' if null else ''} K={cfg['top_k']} n={len(texts)} "
              f"pos={cfg['max_tokens']} {cfg['extraction']}/{cfg['ordering']} seed={seed}] "
              f"share100 {share100:.2f} share20 {share20:.2f} candidates {candidates} "
              f"top10 {[s for s in tok_strs[:10]]} ({meta['wall_secs']}s)", flush=True)
        digest = hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()[:12]
        with open(os.path.join(raw_dir, f"run_{digest}.json"), "w") as f:
            json.dump({**meta, "tokens": [[e.token_id, s, e.ordering_value, e.avg_logit_diff,
                                           e.count_positive, f]
                                          for e, s, f in zip(entries, tok_strs, flags)]},
                      f, indent=1, ensure_ascii=False)
        return sk.Finding(components={e.token_id for e in entries}, universe_size=candidates,
                          claim=claim, score=share100, meta=meta)

    shard = Shard(os.path.join(raw_dir, "shard_cache"))

    def finder(data, seed, config):
        cfg = dict(BASE_CONFIG, **(config or {}))
        return shard.run(lambda: compute(data, seed, cfg), data, seed, cfg, PLACEHOLDER)

    finder.shard = shard
    finder.compute = compute
    return finder


def run_row(record, group):
    f = record.finding
    return {"group": group, "axis": record.axis, "variant": record.variant,
            "seed": record.seed, "config": record.config, "claim": f.claim,
            "score": f.score, "size": f.size,
            "components": sorted(f.components) if f.components else [], "meta": f.meta}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", required=True)
    ap.add_argument("--pile-head", default="expander_data/pile_uncopyrighted_head.json")
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "cards"))
    ap.add_argument("--raw-dir", default=None)
    ap.add_argument("--n-runs", type=int, default=20)
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    raw_dir = args.raw_dir or os.path.join(args.out_dir, "raw", "diff_mining_gemma3_1b")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    hashes = {p: sha256_file(os.path.join(args.upstream, p)) for p in UPSTREAM_FILES}

    from omegaconf import OmegaConf

    Up = import_upstream(args.upstream)
    method_cfg = OmegaConf.load(os.path.join(args.upstream, "configs", "diffing", "method",
                                             "diff_mining.yaml"))
    models = Models(Up)
    tokenizer = models.tokenizer
    print(f"{BASE_MODEL}: {models.base._model.config.num_hidden_layers} layers, "
          f"vocab {len(tokenizer)}", flush=True)
    pools = build_pools(tokenizer, raw_dir, args.pile_head)
    rule = build_domain_rule(Up, tokenizer, raw_dir, pools)
    print(f"pools: fineweb {len(pools['fineweb']['docs'])} (scanned {pools['fineweb']['scanned']}), "
          f"pile {len(pools['pile']['docs'])}; domain vocabulary {len(rule['domain'])} tokens, "
          f"{rule['frequent_in_domain']}/{len(rule['frequent'])} of upstream's frequent list inside it",
          flush=True)
    if args.prepare_only:
        return

    finder = make_finder(Up, models, pools, rule, method_cfg, raw_dir)
    data = make_records("fineweb", N_SAMPLES)
    if args.smoke:
        f = finder.compute(data[:64], BASE_SEED, dict(BASE_CONFIG, max_samples=64))
        print(f.claim, f.score, f.meta["top20"])
        return

    result = sk.stress(
        finder, data,
        battery=["seeds", "bootstrap", "templates", "hyperparams"],
        n_runs=args.n_runs, seed=BASE_SEED,
        config=dict(BASE_CONFIG),
        templates={"fineweb-later": make_records("fineweb-later", N_SAMPLES),
                   "pile": make_records("pile", N_SAMPLES)},
        hyperparams={"top_k": [20, 500], "max_samples": [300], "max_tokens": [64],
                     "extraction": ["logit_lens"], "ordering": ["fraction_positive_diff"],
                     "organism": ["mix1-1p0", "full"]},
        null_data=make_records("fineweb", N_SAMPLES, null=True),
        claim_statement=(
            "Empirically, Diff Mining succeeds across diverse settings: on finetune domain "
            "detection, it significantly outperforms state-of-the-art model diffing methods both "
            "in identifying relevant tokens and in downstream performance when an "
            "interpretability agent is given access to the extracted token set; on models with "
            "injected biases, it identifies more than one third of the biases without targeted "
            "probing"),
        model=BASE_MODEL,
        task="Diff Mining on the cake_bake finetune: per-position top-K logit differences over "
             "1000 fineweb documents x 30 positions, tokens ranked by top-K occurrence rate; "
             "judge-free domain share of the top-100",
        method="upstream diff_mining stages (tokenisation, logit extraction, in-memory diff, "
               "top-K statistics, ordering) at the pinned commit; domain rule from the "
               "finetune corpus vs a fineweb background",
        verbose=True,
    )
    if finder.shard.is_worker:
        print(f"shard {finder.shard.index}/{finder.shard.count} done; no artifacts written")
        return

    base = result.base.meta
    notes = result.card.notes
    notes.append(
        f"upstream: {UPSTREAM_REPO}@{UPSTREAM_COMMIT[:7]} (MIT); tokenisation, logit extraction, "
        "in-memory diff, top-K statistics, orderings and model loading imported unmodified; file "
        "hashes " + ", ".join(f"{os.path.basename(p)} {h[:12]}" for p, h in hashes.items()))
    notes.append(
        "scope: the paper measures 'relevant tokens' with a closed-model judge (openai/gpt-5-mini "
        "via OpenRouter, three permutations, agreement=all) and its injected-bias result uses "
        "Llama-3.3-70B-Instruct; neither is run. This card audits whether the token set itself is "
        "stable and how much of it is finetune-domain vocabulary under a judge-free rule fixed "
        "before any run; the paper releases no token lists, so no shipped number is reproduced")
    notes.append(
        f"domain rule: {len(rule['domain'])} gemma-3 tokens occur >= {DOMAIN_MIN_COUNT} times in "
        f"the {SDF_DATASET} train+validation corpus ({rule['sdf_docs']} documents, "
        f"{rule['sdf_tokens']} tokens), are not generic under upstream's _is_generic_token, and "
        f"have a per-token rate >= {DOMAIN_RATE_RATIO}x their rate in the {rule['background_docs']}"
        f"-document fineweb pool ({rule['background_tokens']} tokens, add-one smoothing); "
        f"{rule['frequent_in_domain']} of upstream's {len(rule['frequent'])} frequent tokens (the "
        "list the judge is shown) are inside it")
    notes.append(
        f"base run (seed {BASE_SEED}, {base['n_samples']} fineweb documents x "
        f"{BASE_CONFIG['max_tokens']} positions, top-K {BASE_CONFIG['top_k']}): "
        f"{base['candidates']} of {len(tokenizer)} vocabulary entries appear in a per-position "
        f"top-K; domain share of the top-100 {base['domain_share_top100']:.2f}, of the top-20 "
        f"{base['domain_share_top20']:.2f}, overlap with upstream's frequent list "
        f"{base['frequent_overlap_top100']:.2f}; top-20 tokens "
        f"{[t[1] + ('*' if t[4] else '') for t in base['top20']]} (* = domain)")
    for record in result.runs:
        if record.axis in ("templates", "hyperparams"):
            m = record.finding.meta
            notes.append(
                f"{record.variant}: top-100 domain share {m['domain_share_top100']:.2f}, top-20 "
                f"{m['domain_share_top20']:.2f}, candidates {m['candidates']}, top-10 "
                f"{[t[1] for t in m['top20'][:10]]}")
    nulls = result.null_runs or []
    if nulls:
        shares = [r.finding.meta["domain_share_top100"] for r in nulls]
        notes.append(
            f"null control (scrambled adapter: LoRA A input features permuted, norms kept): "
            f"top-100 domain share {min(shares):.2f}-{max(shares):.2f} over {len(nulls)} runs; "
            f"top-10 of the null base {[t[1] for t in nulls[0].finding.meta['top20'][:10]]}")
    notes.append(
        f"reference pool: the first {pools['fineweb']['scanned']} documents of {REFERENCE_DATASET} "
        f"(revision {pools['fineweb']['revision'][:10]}) that satisfy upstream's length rule for 30 "
        f"and 64 positions ({len(pools['fineweb']['docs'])} kept); the base draw shuffles documents "
        f"0-{BASE_POOL[1]} with the seed, the fineweb-later template documents "
        f"{LATER_POOL[0]}-{LATER_POOL[1]}; upstream shuffles the full 1M-document sample. The pile "
        f"template uses the {pools['pile']['scanned']}-document head of {pools['pile']['dataset']} "
        f"({len(pools['pile']['docs'])} kept)")
    notes.append(
        "deviations: vllm, dictionary-learning, streamlit and the graders are not installed; a "
        "placeholder vllm module is registered because diffing.utils.model imports it at module "
        "level; the diffing packages are registered without executing their __init__ files; "
        "load_and_tokenize_dataset and the frequent-token helpers are executed from the pinned "
        "source files without importing their modules; the logit-lens layer index reproduces "
        "get_layer_indices (int(0.75 * (num_layers - 1))) because that module imports "
        "dictionary_learning")

    print()
    print(result)
    print(result.to_markdown())
    stem = os.path.join(args.out_dir, "diff_mining_gemma3_1b")
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
    rows = [run_row(r, "real") for r in result.runs] + [run_row(r, "null") for r in nulls]
    with open(stem + ".runs.json", "w") as f:
        json.dump({"upstream_commit": UPSTREAM_COMMIT, "upstream_sha256": hashes,
                   "reference_pool": {k: {kk: vv for kk, vv in v.items() if kk != "docs"}
                                      for k, v in pools.items()},
                   "domain_rule": {k: v for k, v in rule.items() if k != "domain"},
                   "raw_dir": os.path.relpath(raw_dir, args.out_dir), "runs": rows},
                  f, indent=1, default=str, ensure_ascii=False)
        f.write("\n")
    print(sk.verdict_trace_markdown(trace))
    print(f"\nartifacts written to {stem}.*")


if __name__ == "__main__":
    main()
