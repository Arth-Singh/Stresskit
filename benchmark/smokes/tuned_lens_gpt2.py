#!/usr/bin/env python3
"""Pinned Tuned Lens GPT-2 execution smoke; not a claim reproduction."""

from __future__ import annotations

import hashlib
import json
import math
import platform
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tuned_lens.load_artifacts import load_lens_artifacts
from tuned_lens.nn.lenses import LogitLens, TunedLens


UPSTREAM_COMMIT = "abdac0c4de23d9f6d6c8459d576ad203aec15deb"
MODEL_ID = "gpt2"
MODEL_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
LENS_REPO = "AlignmentResearch/tuned-lens"
LENS_REVISION = "1ac7285852a22309f571c2555efc37375d0c4cda"
EXPECTED_CONFIG_SHA256 = (
    "84764e9fb4aef06fe3007d08531ce7ea9213f8436291089f3e8fa4af36126549"
)
EXPECTED_PARAMS_SHA256 = (
    "1e0494dcf4a56a77b73b421820941ea948ffae0c6a0391d88c9cb10b48bc19c8"
)
PROMPT = "Mechanistic interpretability should report uncertainty because"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    device = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
    ).to(device)
    model.eval()

    config_path, params_path = load_lens_artifacts(
        "gpt2",
        repo_id=LENS_REPO,
        repo_type="space",
        revision=LENS_REVISION,
    )
    config_hash = _sha256(config_path)
    params_hash = _sha256(params_path)
    if config_hash != EXPECTED_CONFIG_SHA256:
        raise RuntimeError(f"unexpected lens config sha256: {config_hash}")
    if params_hash != EXPECTED_PARAMS_SHA256:
        raise RuntimeError(f"unexpected lens params sha256: {params_hash}")

    tuned_lens = TunedLens.from_model_and_pretrained(
        model,
        lens_resource_id=str(config_path.parent),
        map_location=device,
    ).to(device)
    logit_lens = LogitLens.from_model(model).to(device)

    inputs = tokenizer(PROMPT, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        layer_index = model.config.num_hidden_layers // 2
        hidden = outputs.hidden_states[layer_index + 1]
        tuned_logits = tuned_lens(hidden, layer_index)
        logit_logits = logit_lens(hidden, layer_index)

    expected_shape = tuple(outputs.logits.shape)
    observed_shapes = {
        "model": tuple(outputs.logits.shape),
        "tuned_lens": tuple(tuned_logits.shape),
        "logit_lens": tuple(logit_logits.shape),
    }
    if any(shape != expected_shape for shape in observed_shapes.values()):
        raise RuntimeError(f"logit shape mismatch: {observed_shapes}")
    tensors = (outputs.logits, tuned_logits, logit_logits)
    if not all(torch.isfinite(value).all().item() for value in tensors):
        raise RuntimeError("non-finite logits")
    translator_delta = float(
        torch.max(torch.abs(tuned_logits - logit_logits)).detach().cpu()
    )
    if not math.isfinite(translator_delta) or translator_delta <= 0.0:
        raise RuntimeError("tuned translator did not change intermediate logits")

    payload = {
        "artifact_type": "stresskit_upstream_execution_smoke",
        "schema_version": "0.1",
        "status": "pass",
        "upstream": "tuned_lens",
        "upstream_commit": UPSTREAM_COMMIT,
        "model": {"repository": MODEL_ID, "revision": MODEL_REVISION},
        "external_artifact": {
            "repository": LENS_REPO,
            "repository_type": "space",
            "revision": LENS_REVISION,
            "config_sha256": config_hash,
            "params_sha256": params_hash,
        },
        "exercise": {
            "model_forward": True,
            "pretrained_tuned_lens_load": True,
            "tuned_lens_forward": True,
            "logit_lens_forward": True,
            "finite_logits": True,
            "matching_shapes": True,
            "translator_changes_logits": True,
            "layer_index": layer_index,
            "sequence_tokens": int(inputs["input_ids"].shape[1]),
            "logit_shape": list(expected_shape),
            "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "not_claim_reproduction": True,
        "not_benchmark_outcome": True,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
