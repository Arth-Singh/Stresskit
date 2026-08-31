"""Train tuned-lens affine translators for Qwen3.5-4B, layer-sharded per GPU.

Baseline for the J-lens comparison battery. One affine translator per layer
(residual parameterization h + A h + b, the tuned-lens objective of
arXiv:2303.08112): minimize KL(final-layer next-token distribution ||
lens distribution) on the SAME corpus family the released J-lens was fitted
on (Salesforce wikitext, 128-token sequences), so the two linear transports
are matched on fitting data.

Layer-sharded: each process loads the (frozen, bf16) model on one GPU and
trains only its slice of layers, so 8 GPUs train all layers concurrently.

    python train_tuned_lens_qwen.py --layers 0-4 --device cuda:0 --out shard0.pt
    ...
    python train_tuned_lens_qwen.py --merge shard*.pt --out tuned_lens_qwen3p5_4b.pt
"""

import argparse
import glob
import random

import torch
import torch.nn.functional as F
import transformers

MODEL_NAME = "Qwen/Qwen3.5-4B"
SEQ_LEN = 128
N_SEQS = 1000
EPOCHS = 4
BATCH = 8
LR = 1e-3
CORPUS_SEED = 0


def _text_cfg(cfg):
    return getattr(cfg, "text_config", None) or cfg


def corpus_chunks(tok, n_seqs=N_SEQS, seq_len=SEQ_LEN, seed=CORPUS_SEED):
    """128-token chunks from wikitext-103 train, deterministic order."""
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    rng = random.Random(seed)
    idxs = rng.sample(range(len(ds)), 200_000)
    chunks, buf = [], []
    for i in idxs:
        text = ds[i]["text"].strip()
        if len(text) < 200:
            continue
        ids = tok(text, add_special_tokens=False)["input_ids"]
        buf.extend(ids)
        while len(buf) >= seq_len:
            chunks.append(buf[:seq_len])
            buf = buf[seq_len:]
            if len(chunks) >= n_seqs:
                return chunks
    raise RuntimeError(f"only {len(chunks)} chunks harvested")


def train_shard(layers, device, out_path):
    tok = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16
    ).to(device)
    hf.eval()
    for p in hf.parameters():
        p.requires_grad_(False)

    d = _text_cfg(hf.config).hidden_size
    final_norm = hf.model.norm
    lm_head = hf.lm_head

    translators = {
        L: torch.nn.Linear(d, d, dtype=torch.float32, device=device)
        for L in layers
    }
    for t in translators.values():   # residual init: start at identity transport
        torch.nn.init.zeros_(t.weight)
        torch.nn.init.zeros_(t.bias)
    params = [p for t in translators.values() for p in t.parameters()]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=1e-3)

    chunks = corpus_chunks(tok)
    steps_per_epoch = len(chunks) // BATCH
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS * steps_per_epoch)

    for epoch in range(EPOCHS):
        order = random.Random(epoch).sample(range(len(chunks)), len(chunks))
        for step in range(steps_per_epoch):
            batch_ids = [chunks[i] for i in order[step * BATCH:(step + 1) * BATCH]]
            ids = torch.tensor(batch_ids, device=device)
            with torch.no_grad():
                outs = hf(ids, output_hidden_states=True)
                # teacher: final distribution; hidden_states[L+1] is the
                # residual stream after block L (index 0 = embeddings)
                teacher = F.log_softmax(outs.logits.float(), dim=-1)
                hs = outs.hidden_states

            loss_total = 0.0
            opt.zero_grad(set_to_none=True)
            for L, tr in translators.items():
                h = hs[L + 1].float()
                h_t = (h + tr(h)).to(torch.bfloat16)
                logits = lm_head(final_norm(h_t)).float()
                logp = F.log_softmax(logits, dim=-1)
                loss = F.kl_div(logp, teacher, log_target=True, reduction="batchmean")
                loss.backward()
                loss_total += loss.item()
            opt.step()
            sched.step()
            if step % 20 == 0:
                print(f"[{device}] epoch {epoch} step {step}/{steps_per_epoch} "
                      f"mean-KL {loss_total / len(translators):.4f}", flush=True)

    torch.save(
        {"model": MODEL_NAME, "seq_len": SEQ_LEN, "n_seqs": N_SEQS,
         "epochs": EPOCHS, "corpus": "Salesforce/wikitext-103-raw-v1",
         "corpus_seed": CORPUS_SEED,
         "translators": {str(L): {k: v.detach().cpu() for k, v in t.state_dict().items()}
                         for L, t in translators.items()}},
        out_path,
    )
    print(f"saved {sorted(translators)} -> {out_path}")


def merge(shard_paths, out_path):
    merged = None
    for p in sorted(shard_paths):
        blob = torch.load(p, map_location="cpu", weights_only=True)
        if merged is None:
            merged = blob
        else:
            overlap = set(merged["translators"]) & set(blob["translators"])
            if overlap:
                raise SystemExit(f"duplicate layers across shards: {sorted(overlap)}")
            merged["translators"].update(blob["translators"])
    torch.save(merged, out_path)
    print(f"merged {len(merged['translators'])} layers -> {out_path}")


def parse_layers(spec):
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",")]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", help="e.g. 0-4 or 0,1,2")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--merge", nargs="*", help="shard glob(s) to merge")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.merge:
        merge([p for g in args.merge for p in glob.glob(g)], args.out)
    else:
        train_shard(parse_layers(args.layers), args.device, args.out)
