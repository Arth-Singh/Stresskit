import json
import hashlib
import re
from pathlib import Path


REGISTRY_PATH = (
    Path(__file__).parents[1] / "benchmark" / "registry.candidates.json"
)
BENCHMARK_DIR = REGISTRY_PATH.parent
REQUIRED_ENTRY_FIELDS = {
    "claim_id",
    "target_type",
    "family",
    "upstream",
    "model_family",
    "model",
    "task",
    "statement_to_extract",
    "claim_locator",
    "entrypoint",
    "finding_type",
    "perturbation_axes",
    "null",
    "compute_tier",
    "eligibility",
}


def load_registry():
    return json.loads(REGISTRY_PATH.read_text())


def test_candidate_registry_has_launch_breadth_before_smoke_exclusions():
    registry = load_registry()
    candidates = [
        entry
        for entry in registry["entries"]
        if not entry["eligibility"].startswith("excluded_pre_freeze")
    ]
    assert len(candidates) >= 20
    assert len({entry["family"] for entry in candidates}) >= 6
    assert len({entry["model_family"] for entry in candidates}) >= 3


def test_registry_upstreams_and_entries_stay_consistent():
    registry = load_registry()
    upstreams = registry["upstreams"]
    assert all(entry["upstream"] in upstreams for entry in registry["entries"])
    assert set(upstreams) == {entry["upstream"] for entry in registry["entries"]}
    log = registry["discovery_log"]
    assert [row["pass"] for row in log] == [1, 2, 3]
    assert registry["discovery_cutoff"] == log[-1]["cutoff"]
    assert set(log[-1]["added_upstreams"]) <= set(upstreams)


def test_registry_ids_and_required_fields_are_complete():
    entries = load_registry()["entries"]
    ids = [entry["claim_id"] for entry in entries]
    assert len(ids) == len(set(ids))
    assert all(REQUIRED_ENTRY_FIELDS <= set(entry) for entry in entries)
    assert all(entry["perturbation_axes"] for entry in entries)
    assert all(entry["null"].strip() for entry in entries)


def test_upstreams_are_content_addressed_and_license_failures_visible():
    registry = load_registry()
    upstreams = registry["upstreams"]
    assert all(re.fullmatch(r"[0-9a-f]{40}", row["commit"]) for row in upstreams.values())
    for entry in registry["entries"]:
        upstream = upstreams[entry["upstream"]]
        if upstream["source_license"] == "UNRESOLVED":
            assert entry["eligibility"].startswith("excluded_pre_freeze")
            assert entry["exclusion_reason"].strip()


def test_prefreeze_exclusions_are_explicit_and_outcome_independent():
    entries = load_registry()["entries"]
    excluded = [
        entry
        for entry in entries
        if entry["eligibility"].startswith("excluded_pre_freeze")
    ]
    assert excluded
    assert all(entry.get("exclusion_reason", "").strip() for entry in excluded)
    assert {
        entry["claim_id"] for entry in excluded
    } == {
        "refusal_single_direction_qwen18b",
        "circuit_tracer_gemma_graph",
        "sae_bench_sparse_probing_instrument",
        # discovery pass 2 (2026-08-28): missing source license
        "assistant_axis_gemma2_27b",
        "assistant_axis_qwen3_32b",
        "introspection_injection_detection_gemma3_27b",
        "introspection_mlp_localization_gemma3_27b",
        "caft_em_pca_ablation_qwen25_coder_32b",
        # discovery pass 2: required Hugging Face artifact declares no license
        "ao_taboo_secret_qwen3_8b",
        "ao_classification_eval_qwen3_8b",
        "ao_taboo_secret_gemma2_9b",
        "adl_narrow_finetune_trace_qwen3_1_7b",
    }


def test_registry_links_static_source_and_model_manifests():
    registry = load_registry()
    for field in ("source_manifest", "model_manifest", "artifact_manifest"):
        path = BENCHMARK_DIR / registry[field]
        assert path.is_file()
        assert json.loads(path.read_text())["schema_version"] == "0.1"


def test_candidate_frame_contains_no_outcomes():
    registry = load_registry()
    assert registry["status"] == "candidate_frame_not_frozen"
    forbidden = {"result", "grade", "verdict", "passed", "failed"}
    assert all(not (forbidden & set(entry)) for entry in registry["entries"])


def load_august_frame():
    return json.loads(
        (BENCHMARK_DIR / "discovery" / "august-2026-frame.json").read_text()
    )


def test_august_frame_counts_partition_the_papers_it_collected():
    frame = load_august_frame()
    papers = frame["papers"]
    counts = frame["counts"]
    assert counts["tier_a"] == len(papers)
    assert counts["tier_b"] == len(frame["tier_b_papers"])
    observed = {}
    for paper in papers:
        status = paper["code"]["status"]
        observed[status] = observed.get(status, 0) + 1
    assert observed == counts["by_code_status"]
    assert sum(observed.values()) == counts["tier_a"]

    released = [p for p in papers if p["code"]["status"] == "public_repo"]
    licensed = [p for p in released if p["code"]["license_spdx"]]
    assert len(licensed) == counts["public_repo_with_spdx_license"]
    assert (
        len(released) - len(licensed) == counts["public_repo_without_license"]
    )


def test_august_frame_stays_inside_its_declared_window():
    frame = load_august_frame()
    window = frame["window"]
    for paper in frame["papers"] + frame["tier_b_papers"]:
        assert window["submitted_from"] <= paper["published"] <= window["submitted_to"]
        assert re.fullmatch(r"\d{4}\.\d{4,5}", paper["arxiv_id"]), paper["arxiv_id"]


def test_august_frame_is_a_census_not_a_scoreboard():
    frame = load_august_frame()
    assert frame["outcome_blind"] is True
    assert frame["status"] == "candidate_frame_not_frozen"
    forbidden = {"grade", "verdict", "score", "passed", "failed", "rank", "quality"}
    keys = set()

    def walk(value):
        if isinstance(value, dict):
            keys.update(value)
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(frame)
    assert not (forbidden & keys)


def test_pass_three_upstreams_trace_back_to_the_august_frame():
    registry = load_registry()
    pass_three = next(
        row for row in registry["discovery_log"] if row["pass"] == 3
    )
    frame_repos = {
        paper["code"]["repository"]
        for paper in load_august_frame()["papers"]
        if paper["code"]["status"] == "public_repo"
    }
    for name in pass_three["added_upstreams"]:
        assert registry["upstreams"][name]["repository"] in frame_repos, name


def test_pass_3b_addendum_covers_omitted_terms_and_every_retained_tier_b_row():
    registry = load_registry()
    addendum = registry["discovery_addenda"][0]
    path = BENCHMARK_DIR / addendum["artifact"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == addendum["sha256"]
    payload = json.loads(path.read_text())
    assert payload["outcome_blind"] is True
    assert set(payload["omitted_terms"]) == {
        "chain of thought", "reasoning trace", "persona", "introspection",
        "evaluation awareness", "model organism",
    }
    assert len(payload["omitted_term_ledger"]) == payload["counts"][
        "omitted_term_unique"
    ]
    assert len(payload["retained_tier_b_ledger"]) == payload["counts"][
        "retained_tier_b_code_audited"
    ] == 47
    omitted_format = payload["ledger_formats"]["omitted_term_ledger"]
    relation = omitted_format.index("frame_relation")
    disposition = omitted_format.index("disposition")
    for row in payload["omitted_term_ledger"]:
        if row[relation] != "already_tier_a":
            assert row[disposition]
    tier_format = payload["ledger_formats"]["retained_tier_b_ledger"]
    tier_disposition = tier_format.index("disposition")
    assert all(row[tier_disposition] for row in payload["retained_tier_b_ledger"])
