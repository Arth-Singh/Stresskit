"""Tests for the activation_oracles compatibility adapter and the v0.2.1
oracle math (consistency decomposition, Wilson CIs)."""

import pytest

from stresskit import judges
from stresskit.adapters import activation_oracles as ao
from stresskit.metrics import wilson_ci
from stresskit.oracle import stress_oracle, OracleProbe


# ---------------------------------------------------------------------------
# wilson_ci
# ---------------------------------------------------------------------------

def test_wilson_ci_bounds_and_center():
    lo, hi = wilson_ci(9, 9)
    assert 0.0 <= lo < 1.0 and hi == 1.0          # never leaves [0, 1]
    lo0, hi0 = wilson_ci(0, 9)
    assert lo0 == 0.0 and 0.0 < hi0 < 1.0
    lo5, hi5 = wilson_ci(5, 10)
    assert lo5 < 0.5 < hi5


def test_wilson_ci_empty():
    assert wilson_ci(0, 0) is None


def test_wilson_ci_narrows_with_n():
    lo_s, hi_s = wilson_ci(5, 10)
    lo_l, hi_l = wilson_ci(500, 1000)
    assert (hi_l - lo_l) < (hi_s - lo_s)


# ---------------------------------------------------------------------------
# consistency decomposition
# ---------------------------------------------------------------------------

def phrasing_flipper(exemplar, question, seed):
    # deterministic per question: perfect repeat/exemplar agreement,
    # zero cross-phrasing agreement
    return f"answer to {question}"


def test_decomposition_isolates_phrasing():
    probe = OracleProbe(
        name="p", kind="known", expected="answer", concept="c",
        questions=["q one?", "q two?"],
        exemplars=[{"e": 0}, {"e": 1}],
    )
    report = stress_oracle(
        phrasing_flipper, [probe], n_repeats=2, judge=judges.normalized
    )
    row = report.per_probe[0]
    assert row["consistency_repeats"] == 1.0
    assert row["consistency_exemplars"] == 1.0
    assert row["consistency_phrasings"] == 0.0
    d = report.metrics["consistency_decomposition"]
    assert d["phrasings"] == 0.0 and d["repeats"] == 1.0


def test_decomposition_weighted_mean_equals_pooled():
    # pooled consistency must equal pair-count-weighted mean of buckets
    probe = OracleProbe(
        name="p", kind="consistency",
        questions=["q one?", "q two?", "q three?"],
        exemplars=[{"e": 0}, {"e": 1}],
    )
    report = stress_oracle(
        phrasing_flipper, [probe], n_repeats=2, judge=judges.normalized
    )
    row = report.per_probe[0]
    n = row["n_answers"]
    total_pairs = n * (n - 1) / 2
    # bucket sizes for 3 questions x 2 exemplars x 2 repeats
    rep_pairs = 3 * 2 * 1           # per (q,e) cell: C(2,2)=1 pair
    ex_pairs = 3 * (2 * 2)          # same q, different e: 2*2 combos
    ph_pairs = total_pairs - rep_pairs - ex_pairs
    pooled = (
        row["consistency_repeats"] * rep_pairs
        + row["consistency_exemplars"] * ex_pairs
        + row["consistency_phrasings"] * ph_pairs
    ) / total_pairs
    assert abs(pooled - row["consistency"]) < 1e-9


def test_known_accuracy_ci_in_metrics():
    probe = OracleProbe(
        name="p", kind="known", expected="answer", concept="c",
        questions=["q one?"], exemplars=[{"e": 0}],
    )
    report = stress_oracle(phrasing_flipper, [probe], n_repeats=3)
    ci = report.metrics["known_accuracy_ci95"]
    assert ci is not None and ci[0] <= report.metrics["known_accuracy"] <= ci[1]
    assert "Wilson" in report.to_markdown()


# ---------------------------------------------------------------------------
# activation_oracles adapter
# ---------------------------------------------------------------------------

def make_record(target, truth, question, context, responses, act_key="lora"):
    return {
        "verbalizer_lora_path": "verbalizer",
        "target_lora_path": target,
        "context_prompt": [{"role": "user", "content": context}],
        "act_key": act_key,
        "verbalizer_prompt": question,
        "ground_truth": truth,
        "num_tokens": 10,
        "token_responses": [],
        "full_sequence_responses": [],
        "segment_responses": responses,
        "context_input_ids": [1, 2, 3],
    }


QUESTIONS = ["What is the secret word?", "Which word is hidden?"]
CONTEXTS = ["ctx one", "ctx two"]


def upstream_results():
    results = []
    for word in ["gold", "ship"]:
        for q in QUESTIONS:
            for ctx in CONTEXTS:
                results.append(make_record(
                    f"lora-taboo-{word}", word, q, ctx,
                    [f"The secret word is {word}.", f"{word}"],
                ))
    for q in QUESTIONS:
        for ctx in CONTEXTS:
            results.append(make_record(
                None, None, q, ctx,
                ["There is no secret word here.", "banana"],
            ))
    return results


def test_adapter_builds_probes_and_lookup():
    probes, ask_fn, n_repeats = ao.probes_from_verbalizer_results(upstream_results())
    assert n_repeats == 2
    kinds = {p.name: p.kind for p in probes}
    assert kinds["known-gold"] == "known"
    assert kinds["null-base-model"] == "null"
    gold = next(p for p in probes if p.name == "known-gold")
    assert list(gold.questions) == QUESTIONS
    # repeat index comes from run_seed % 10
    assert "gold" in ask_fn(gold.exemplars[0], QUESTIONS[0], 0)
    assert ask_fn(gold.exemplars[1], QUESTIONS[1], 1011) == "gold"


def test_adapter_keeps_null_variants_distinct():
    results = upstream_results()
    results.append(make_record(
        None, "none", "Is there a hidden word? If not, say so.", "ctx one",
        ["There is no hidden word here.", "No secret word is present."],
    ))
    probes, _, _ = ao.probes_from_verbalizer_results(results)
    names = {p.name for p in probes}
    assert "null-base-model" in names and "null-base-model-none" in names
    fair = next(p for p in probes if p.name == "null-base-model-none")
    assert fair.kind == "null"


def test_adapter_report_grades_correctly():
    report = ao.reliability_report(upstream_results(), oracle_name="test-ao")
    assert report.metrics["known_accuracy"] == 1.0
    # one of two null responses per cell is a confident hallucination
    assert report.metrics["null_hallucination_rate"] == 0.5
    assert report.checks["null_hallucination"]["passed"] is False


def test_adapter_rejects_mixed_act_keys():
    results = upstream_results()
    results[0]["act_key"] = "orig"
    with pytest.raises(ValueError, match="act_key"):
        ao.probes_from_verbalizer_results(results)
    # explicit filter works
    probes, _, _ = ao.probes_from_verbalizer_results(results, act_key="lora")
    assert probes


def test_adapter_rejects_empty_responses():
    results = upstream_results()
    results[0]["segment_responses"] = []
    with pytest.raises(ValueError, match="empty"):
        ao.probes_from_verbalizer_results(results)
