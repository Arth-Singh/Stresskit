import hashlib
import json
import re
from pathlib import Path


BENCHMARK_DIR = Path(__file__).parents[1] / "benchmark"
REPO_ROOT = BENCHMARK_DIR.parent

# Upstreams pinned without a repository-level license file. They stay visible in
# the manifest (REGISTRY_PROTOCOL: unresolved licensing is recorded, not hidden)
# and every registry entry that depends on them is excluded before freeze.
LICENSE_NOT_FOUND_UPSTREAMS = {
    "sae_bench",
    "assistant_axis",
    "caft",
    "introspection_mechanisms",
}

# Added in discovery passes 2 (2026-08-28) and 3 (2026-08-31). Their static
# audit was regenerated locally over all 36 pinned checkouts
# (artifacts/benchmark/upstream-static-audit-20260831.json); the independent
# Nibi source-fetch array (jobs/upstream-source-fetch-array.slurm, now 0-35)
# has not been rerun for them yet. Remove names here as their Nibi sidecars
# land.
NIBI_AUDIT_PENDING = {
    # discovery pass 2
    "activation_oracles",
    "assistant_axis",
    "caft",
    "diffing_toolkit",
    "eval_awareness_steering",
    "introspection_mechanisms",
    "jacobian_lens",
    "natural_language_autoencoders",
    "persona_vectors",
    "steering_vector_distillation",
    "thought_anchors",
    # discovery pass 3
    "activation_model_scanner",
    "communication_map",
    "concept_targeted_attribution",
    "folkmotif",
    "future_localization",
    "graph_learning_circuits",
    "impossible_directions",
    "logit_lens_homonyms",
    "mechanistic_tomography",
    "reins_sae_steering",
    "swd_circuits",
}


def _load(name):
    return json.loads((BENCHMARK_DIR / name).read_text())


def test_upstream_manifest_matches_registry_and_is_not_smoke_evidence():
    registry = _load("registry.candidates.json")
    manifest = _load("upstream_sources.json")
    assert manifest["not_execution_smoke"] is True
    assert set(manifest["upstreams"]) == set(registry["upstreams"])
    for name, row in manifest["upstreams"].items():
        assert row["repository"] == registry["upstreams"][name]["repository"]
        assert row["commit"] == registry["upstreams"][name]["commit"]
        assert re.fullmatch(r"[0-9a-f]{40}", row["tree"])
        assert row["entrypoint_paths"]
        assert row["static_python_syntax"]["tracked_files"] > 0
        license_row = row["license"]
        if license_row["status"] == "found":
            assert license_row["spdx"]
            assert license_row["path"]
            assert re.fullmatch(r"[0-9a-f]{64}", license_row["sha256"])
        else:
            assert name in LICENSE_NOT_FOUND_UPSTREAMS
            assert license_row == {
                "status": "not_found",
                "spdx": None,
                "path": None,
                "sha256": None,
            }


def test_model_manifest_pins_resolved_revisions_and_forbids_substitution():
    manifest = _load("model_sources.json")
    for key, row in manifest["models"].items():
        if row["availability"].startswith("resolved"):
            assert re.fullmatch(r"[0-9a-f]{40}", row["revision"]), key
            assert row["license"]
            assert row["access"]
        else:
            assert key == "qwen_1_8b_chat_requested"
            assert row["revision"] is None
            assert "not substituted" in row["exclusion_reason"]


def test_external_artifact_manifest_resolves_or_excludes_every_record():
    manifest = _load("artifact_sources.json")
    artifacts = manifest["artifacts"]
    assert artifacts["tuned_lens_gpt2"]["license"] == "mit"
    assert artifacts["circuit_tracer_llama32_transcoder"]["license"] == "mit"
    unresolved = artifacts["circuit_tracer_gemma2_transcoder"]
    assert unresolved["license"] is None
    assert unresolved["availability"] == "resolved_but_license_undeclared"
    assert unresolved["exclusion_reason"]
    assert artifacts["wikitext"]["license"] == ["cc-by-sa-3.0", "gfdl"]


def test_source_manifests_contain_no_benchmark_outcomes():
    forbidden = {"grade", "verdict", "passed", "failed", "effect_size"}
    for name in (
        "upstream_sources.json",
        "model_sources.json",
        "artifact_sources.json",
    ):
        payload = _load(name)
        serialized_keys = set()

        def walk(value):
            if isinstance(value, dict):
                serialized_keys.update(value)
                for child in value.values():
                    walk(child)
            elif isinstance(value, list):
                for child in value:
                    walk(child)

        walk(payload)
        assert not (forbidden & serialized_keys)


def test_frozen_static_audit_artifact_has_consistent_totals():
    artifact = json.loads(
        (
            REPO_ROOT
            / "artifacts"
            / "benchmark"
            / "upstream-static-audit-20260831.json"
        ).read_text()
    )
    rows = artifact["upstreams"]
    summary = artifact["summary"]
    assert artifact["all_ok"] is True
    assert artifact["not_execution_smoke"] is True
    assert summary["upstreams"] == len(rows)
    assert summary["entrypoint_paths_verified"] == sum(
        row["entrypoints"] for row in rows.values()
    )
    assert summary["tracked_python_files_parsed"] == sum(
        row["python_files"] for row in rows.values()
    )
    assert summary["syntax_warnings"] == sum(
        row["syntax_warnings"] for row in rows.values()
    ) == len(artifact["warnings"])
    for key, filename in (
        ("manifest_sha256", "benchmark/upstream_sources.json"),
        ("auditor_sha256", "benchmark/audit_upstreams.py"),
    ):
        observed = hashlib.sha256((REPO_ROOT / filename).read_bytes()).hexdigest()
        assert artifact["inputs"][key] == observed


def test_nibi_independent_source_audits_match_manifest_and_sidecars():
    manifest = _load("upstream_sources.json")["upstreams"]
    artifact_dir = REPO_ROOT / "artifacts" / "benchmark" / "nibi" / "source-audit"
    artifacts = sorted(artifact_dir.glob("*.json"))
    assert {path.stem for path in artifacts} == set(manifest) - NIBI_AUDIT_PENDING
    for path in artifacts:
        payload = json.loads(path.read_text())
        assert payload["all_ok"] is True
        assert payload["not_execution_smoke"] is True
        assert list(payload["upstreams"]) == [path.stem]
        row = payload["upstreams"][path.stem]
        assert row["commit"] == manifest[path.stem]["commit"]
        assert row["tree"] == manifest[path.stem]["tree"]
        assert row["missing_entrypoints"] == []
        assert row["python_syntax_errors"] == []
        sidecar_hash = path.with_suffix(path.suffix + ".sha256").read_text().split()[0]
        assert sidecar_hash == hashlib.sha256(path.read_bytes()).hexdigest()


def test_nibi_verification_bundle_hashes_and_results():
    artifact_dir = REPO_ROOT / "artifacts" / "benchmark" / "nibi" / "verification"
    manifest_path = artifact_dir / "manifest-20361520.sha256"
    expected = {}
    for line in manifest_path.read_text().splitlines():
        digest, filename = line.split(maxsplit=1)
        expected[filename] = digest
    assert set(expected) == {
        "environment-20361520.json",
        "pytest-20361520.txt",
        "upstream-static-audit-20361520.json",
    }
    for filename, digest in expected.items():
        assert hashlib.sha256((artifact_dir / filename).read_bytes()).hexdigest() == digest
    environment = json.loads((artifact_dir / "environment-20361520.json").read_text())
    assert environment["slurm_job_id"] == "20361520"
    pytest_output = (artifact_dir / "pytest-20361520.txt").read_text()
    assert "348 passed" in pytest_output
    upstream_audit = json.loads(
        (artifact_dir / "upstream-static-audit-20361520.json").read_text()
    )
    assert upstream_audit["all_ok"] is True
    assert len(upstream_audit["upstreams"]) == 14


def test_import_smoke_matrix_pins_public_bootstraps_and_arrow_consumers():
    matrix = _load("install_smoke_matrix.json")
    rows = {row["upstream"]: row for row in matrix["rows"]}
    for upstream in ("circuit_tracer", "pyvene", "sae_lens", "tuned_lens"):
        assert rows[upstream]["preload_modules"] == ["arrow/25.0.0"]
    for upstream in ("circuit_tracer", "sae_lens"):
        assert "transformer-lens==2.16.1" in rows[upstream]["public_bootstrap"]
        assert not any(
            package.startswith("transformer-lens>=")
            for package in rows[upstream]["public_bootstrap"]
        )


def test_nibi_tuned_lens_execution_smoke_is_pinned_and_not_an_outcome():
    artifact_dir = REPO_ROOT / "artifacts" / "benchmark" / "nibi" / "execution-smoke"
    path = artifact_dir / "tuned_lens_gpt2-20394757.json"
    payload = json.loads(path.read_text())
    sidecar_hash = path.with_suffix(path.suffix + ".sha256").read_text().split()[0]
    assert sidecar_hash == hashlib.sha256(path.read_bytes()).hexdigest()
    assert payload["status"] == "pass"
    assert payload["not_claim_reproduction"] is True
    assert payload["not_benchmark_outcome"] is True
    assert payload["upstream_commit"] == _load("upstream_sources.json")["upstreams"][
        "tuned_lens"
    ]["commit"]
    assert payload["model"]["revision"] == _load("model_sources.json")["models"][
        "gpt2"
    ]["revision"]
    artifact = _load("artifact_sources.json")["artifacts"]["tuned_lens_gpt2"]
    assert payload["external_artifact"]["revision"] == artifact["revision"]
    assert payload["external_artifact"]["config_sha256"] == artifact["files"][
        "lens/gpt2/config.json"
    ]["sha256"]
    assert payload["external_artifact"]["params_sha256"] == artifact["files"][
        "lens/gpt2/params.pt"
    ]["sha256"]
    assert all(
        payload["exercise"][key]
        for key in (
            "model_forward",
            "pretrained_tuned_lens_load",
            "tuned_lens_forward",
            "logit_lens_forward",
            "finite_logits",
            "matching_shapes",
            "translator_changes_logits",
        )
    )
