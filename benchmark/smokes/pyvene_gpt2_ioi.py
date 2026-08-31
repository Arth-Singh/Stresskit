#!/usr/bin/env python3
"""Pinned Pyvene GPT-2 IOI execution smoke; not a claim reproduction."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import tempfile
from pathlib import Path

import torch
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

import pyvene as pv


UPSTREAM_COMMIT = "9e333904dcf9e597ca76170010d17f4d4580de8d"
MODEL_ID = "gpt2"
MODEL_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
BASE_PROMPT = "When Mary and John went to the store, John gave a drink to"
SOURCE_PROMPT = "When Alice and Bob went to the store, Alice gave a drink to"
LAYER = 8


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()


def main() -> int:
    torch.manual_seed(0)
    torch.set_num_threads(4)
    device = torch.device("cpu")

    config = GPT2Config.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        attn_implementation="eager",
    )
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = GPT2LMHeadModel.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        config=config,
    ).to(device)
    model.eval()

    base = tokenizer(BASE_PROMPT, return_tensors="pt").to(device)
    source = tokenizer(SOURCE_PROMPT, return_tensors="pt").to(device)
    base_tokens = int(base["input_ids"].shape[1])
    source_tokens = int(source["input_ids"].shape[1])
    position = min(base_tokens, source_tokens) - 2
    if position < 0:
        raise RuntimeError("prompts have no valid intervention position")

    intervention_config = pv.IntervenableConfig(
        model_type=type(model),
        representations=[
            pv.RepresentationConfig(LAYER, "block_output", "pos", 1),
        ],
        intervention_types=pv.VanillaIntervention,
    )
    intervenable = pv.IntervenableModel(intervention_config, model, use_fast=False)
    intervenable.set_device(device)
    locations = {"sources->base": ([[[position]]], [[[position]]])}

    with torch.no_grad():
        baseline_logits = model(**base).logits
        _, intervened_output = intervenable(base, [source], locations)
        intervened_logits = intervened_output.logits

    expected_shape = tuple(baseline_logits.shape)
    if tuple(intervened_logits.shape) != expected_shape:
        raise RuntimeError("intervention changed output shape")
    if not torch.isfinite(intervened_logits).all().item():
        raise RuntimeError("intervention produced non-finite logits")
    intervention_delta = float(
        torch.max(torch.abs(intervened_logits - baseline_logits)).cpu()
    )
    if intervention_delta <= 0.0:
        raise RuntimeError("intervention did not change logits")

    with tempfile.TemporaryDirectory(prefix="stresskit-pyvene-") as directory:
        save_dir = Path(directory)
        intervenable.save(save_dir)
        config_hash = _sha256(save_dir / "config.json")
        restored = pv.IntervenableModel.load(save_dir, model=model)
        restored.set_device(device)
        with torch.no_grad():
            _, restored_output = restored(base, [source], locations)
        restored_logits = restored_output.logits

    serialization_delta = float(
        torch.max(torch.abs(intervened_logits - restored_logits)).cpu()
    )
    if serialization_delta > 1e-6:
        raise RuntimeError(
            f"serialized intervention changed logits: {serialization_delta}"
        )

    payload = {
        "artifact_type": "stresskit_upstream_execution_smoke",
        "schema_version": "0.1",
        "status": "pass",
        "upstream": "pyvene",
        "upstream_commit": UPSTREAM_COMMIT,
        "model": {"repository": MODEL_ID, "revision": MODEL_REVISION},
        "exercise": {
            "model_forward": True,
            "interchange_intervention_forward": True,
            "intervention_changes_logits": True,
            "finite_logits": True,
            "matching_shapes": True,
            "save_load_roundtrip": True,
            "serialization_equivalent": True,
            "serialized_config_sha256": config_hash,
            "layer": LAYER,
            "component": "block_output",
            "unit": "pos",
            "base_tokens": base_tokens,
            "source_tokens": source_tokens,
            "logit_shape": list(expected_shape),
            "base_prompt_sha256": _prompt_hash(BASE_PROMPT),
            "source_prompt_sha256": _prompt_hash(SOURCE_PROMPT),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "pyvene": importlib.metadata.version("pyvene"),
        },
        "not_claim_reproduction": True,
        "not_benchmark_outcome": True,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
