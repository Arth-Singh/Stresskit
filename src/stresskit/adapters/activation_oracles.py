"""Adapter for adamkarvonen/activation_oracles (arXiv:2512.15674).

The reference Activation Oracle implementation evaluates a *verbalizer*
(the oracle LoRA) by injecting target-model activations and collecting
free-text answers. Its eval scripts emit ``VerbalizerResults`` records —
one per (context prompt × verbalizer prompt × activation type) — each
holding a list of sampled responses.

This adapter turns those records into StressKit oracle probes, so anyone
who has already run the upstream eval gets a reliability report from the
JSON they saved, without touching a GPU again::

    from stresskit.adapters import activation_oracles as ao

    payload = ao.load_results_json("taboo_results_open_...json")
    report = ao.reliability_report(
        payload["results"],
        oracle_name=payload["verbalizer_lora_path"],
        act_key="lora",
    )
    print(report.to_markdown())

Records may be upstream ``VerbalizerResults`` dataclasses or the dicts
produced by ``json.load`` on their saved files; both are accepted.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .. import judges as J
from ..oracle import AskFn, OracleProbe, OracleReport, OracleThresholds, stress_oracle

_NULL_TRUTHS = {None, "", "none", "null", "base_model"}


def _get(record: Any, key: str) -> Any:
    if isinstance(record, Mapping):
        return record[key]
    return getattr(record, key)


def load_results_json(path: str) -> Dict[str, Any]:
    """Load one JSON file saved by the upstream eval scripts.

    Returns the payload dict: keys ``config``, ``verbalizer_lora_path``,
    ``results`` (list of record dicts).
    """
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    if "results" not in payload:
        raise ValueError(f"{path}: no 'results' key — not a verbalizer results file")
    return payload


def _context_key(context_prompt: Any) -> str:
    if isinstance(context_prompt, str):
        return context_prompt
    return json.dumps(context_prompt, sort_keys=True)


def probes_from_verbalizer_results(
    results: Sequence[Any],
    *,
    response_field: str = "segment_responses",
    act_key: Optional[str] = None,
) -> Tuple[List[OracleProbe], AskFn, int]:
    """Group upstream records into probes and build a cached ``ask_fn``.

    One probe per (target LoRA, ground truth): its *questions* are the
    distinct verbalizer prompts, its *exemplars* the distinct context
    prompts, and its repeats the sampled responses in ``response_field``
    (``"segment_responses"`` or ``"full_sequence_responses"``). Records
    with a null/absent target LoRA or empty ground truth become
    kind="null" control probes.

    Seed contract: responses are indexed by ``run_seed % 10``, matching
    ``stress_oracle``'s ``seed + 1000*qi + 10*ei + r`` schedule — so call
    ``stress_oracle``/``blind_spot_matrix`` with a ``seed`` divisible by
    10 (the default 0 is fine) and the returned ``n_repeats``.
    """
    if act_key is not None:
        results = [r for r in results if _get(r, "act_key") == act_key]
    else:
        seen_keys = {_get(r, "act_key") for r in results}
        if len(seen_keys) > 1:
            raise ValueError(
                f"results mix activation types {sorted(seen_keys)}; "
                "pass act_key='lora'|'orig'|'diff' to select one"
            )
    if not results:
        raise ValueError("no verbalizer results to convert (check act_key filter)")

    groups: Dict[Tuple[Any, Any], List[Any]] = {}
    for r in results:
        target = _get(r, "target_lora_path")
        truth = _get(r, "ground_truth")
        groups.setdefault((target, truth), []).append(r)

    table: Dict[Tuple[str, str, int, int], str] = {}
    probes: List[OracleProbe] = []
    n_repeats_all: List[int] = []

    for (target, truth), records in groups.items():
        truth_norm = truth.lower() if isinstance(truth, str) else truth
        is_null = target is None or truth_norm in _NULL_TRUTHS
        name = f"null-{target or 'base-model'}" if is_null else f"known-{truth}"

        questions: List[str] = []
        exemplar_keys: List[str] = []
        for r in records:
            q = _get(r, "verbalizer_prompt")
            ck = _context_key(_get(r, "context_prompt"))
            if q not in questions:
                questions.append(q)
            if ck not in exemplar_keys:
                exemplar_keys.append(ck)

        for r in records:
            responses = list(_get(r, response_field))
            if not responses:
                raise ValueError(
                    f"record for probe {name!r} has empty {response_field!r}; "
                    "re-run the upstream eval with segment/full_seq repeats > 0"
                )
            if len(responses) > 10:
                raise ValueError(
                    f"{len(responses)} repeats > 10 breaks the run_seed contract; "
                    "subsample to at most 10 responses per record"
                )
            n_repeats_all.append(len(responses))
            ei = exemplar_keys.index(_context_key(_get(r, "context_prompt")))
            q = _get(r, "verbalizer_prompt")
            for rep, ans in enumerate(responses):
                table[(name, q, ei, rep)] = str(ans)

        exemplars = [(name, ei, key) for ei, key in enumerate(exemplar_keys)]
        probes.append(
            OracleProbe(
                name=name,
                questions=questions,
                exemplars=exemplars,
                expected=None if is_null else str(truth),
                kind="null" if is_null else "known",
                concept=None if is_null else str(truth),
                meta={"target_lora_path": target},
            )
        )

    n_repeats = min(n_repeats_all)
    if len(set(n_repeats_all)) > 1:
        # ragged repeat counts: grade on the shared prefix
        table = {k: v for k, v in table.items() if k[3] < n_repeats}

    def ask_fn(exemplar: Any, question: str, run_seed: int) -> str:
        probe_name, ei, _ = exemplar
        return table[(probe_name, question, ei, run_seed % 10)]

    return probes, ask_fn, n_repeats


def reliability_report(
    results: Sequence[Any],
    *,
    oracle_name: str,
    response_field: str = "segment_responses",
    act_key: Optional[str] = None,
    judge: J.Judge = J.token_f1(0.5),
    expected_judge: J.Judge = J.contains,
    thresholds: Optional[OracleThresholds] = None,
    verbose: bool = False,
) -> OracleReport:
    """One-call reliability report from upstream verbalizer results."""
    probes, ask_fn, n_repeats = probes_from_verbalizer_results(
        results, response_field=response_field, act_key=act_key
    )
    return stress_oracle(
        ask_fn,
        probes,
        judge=judge,
        expected_judge=expected_judge,
        n_repeats=n_repeats,
        thresholds=thresholds,
        oracle_name=str(oracle_name),
        verbose=verbose,
    )
