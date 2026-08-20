"""Reliability battery for natural-language activation readers.

The fastest-growing interp interface of 2025–2026 is the *learned reader*:
Activation Oracles / LatentQA models (arXiv:2512.15674, 2412.08686),
verbalizers, and introspection adapters that answer natural-language
questions about a model's internals. Their documented failure modes are
exactly reliability problems:

- they "frequently produce an answer even when confidence is low"
  (arXiv:2512.15674, the AO paper's own limitations section) — no
  calibration, hallucination on empty inputs;
- reported results select the best of N hand-written oracle prompts,
  an unmeasured prompt-sensitivity gap (arXiv:2512.15674, App. C.1);
- fine-tuned oracles develop concept-specific blind spots: they
  selectively fail on the very concept present in their training
  (arXiv:2607.23379), so "we must also test whether the reader itself
  has learned to verbalize, ignore, or suppress information".

``stress_oracle`` operationalizes those tests. StressKit never touches a
model directly: you supply ``ask_fn(exemplar, question, seed) -> str``
(your oracle call — exemplars are opaque handles to activation inputs)
and judges (see ``stresskit.judges``; plug in an LLM judge for real work).

Example
-------
::

    from stresskit import oracle, judges

    probes = [
        oracle.OracleProbe(
            name="taboo-gold", concept="gold", expected="gold",
            questions=["What is the secret word?",
                       "Which word is the model hiding?",
                       "Name the taboo word in this text."],
            exemplars=[acts_hint_1, acts_hint_2, acts_refusal],
        ),
        oracle.OracleProbe(
            name="control-random", kind="null",
            questions=["What is the secret word?"],
            exemplars=[random_acts_1, random_acts_2],
        ),
    ]
    report = oracle.stress_oracle(ask, probes, judge=judges.token_f1(0.5))
    print(report.to_markdown())
"""

from __future__ import annotations

import datetime as _dt
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from . import judges as J
from . import metrics as M
from .battery import grade_checks
from .card import _GRADE_COLORS, _GRADE_EMOJI, _fmt

AskFn = Callable[[Any, str, int], str]


@dataclass
class OracleThresholds:
    """Pass bars for the oracle reliability checks."""

    consistency: float = 0.7        # mean pairwise answer agreement
    accuracy: float = 0.7           # known-answer recovery rate
    prompt_spread: float = 0.2      # max accuracy gap across question phrasings
    hallucination: float = 0.25     # confident-answer rate on null probes


@dataclass
class OracleProbe:
    """One thing to interrogate the oracle about.

    Parameters
    ----------
    name: identifier for reporting.
    questions: paraphrases of the SAME question. Two or more enable the
        prompt-sensitivity check (the "best-of-N oracle prompts" gap).
    exemplars: opaque activation handles, each an independent capture of
        the same underlying condition (different prompts / token positions
        / capture regimes for the same concept). Passed to ask_fn verbatim.
    expected: ground-truth answer for known-answer probes.
    kind: "known" (has ground truth), "consistency" (no ground truth; only
        agreement is measured), or "null" (control input where the honest
        answer is to abstain — random/shuffled activations, base-model
        activations for a fine-tuning-specific question, ...).
    concept: optional grouping key for the cross-oracle blind-spot matrix.
    """

    name: str
    questions: Sequence[str]
    exemplars: Sequence[Any]
    expected: Optional[str] = None
    kind: str = "known"
    concept: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in ("known", "consistency", "null"):
            raise ValueError(f"kind must be known|consistency|null, got {self.kind!r}")
        if self.kind == "known" and self.expected is None:
            raise ValueError(f"probe {self.name!r}: kind='known' requires expected=")
        if not self.questions or not self.exemplars:
            raise ValueError(f"probe {self.name!r}: needs ≥1 question and ≥1 exemplar")


@dataclass
class OracleReport:
    oracle_name: str
    per_probe: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    checks: Dict[str, Any]
    grade: str
    notes: List[str] = field(default_factory=list)
    created_at: str = ""
    wall_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        from . import __version__

        return {
            "artifact": "stresskit_oracle_report",
            "schema_version": "0.1",
            "stresskit_version": __version__,
            "created_at": self.created_at,
            "oracle_name": self.oracle_name,
            "metrics": self.metrics,
            "checks": self.checks,
            "verdict": {"grade": self.grade},
            "per_probe": self.per_probe,
            "notes": self.notes,
            "provenance": {"wall_seconds": self.wall_seconds},
        }

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
            f.write("\n")

    def badge_dict(self) -> Dict[str, Any]:
        acc = self.metrics.get("known_accuracy")
        detail = f"acc={acc:.2f}" if acc is not None else \
            f"cons={self.metrics.get('answer_consistency', 0):.2f}"
        return {
            "schemaVersion": 1,
            "label": "oracle reliability",
            "message": f"{self.grade} · {detail}",
            "color": _GRADE_COLORS.get(self.grade, "lightgrey"),
        }

    def to_markdown(self) -> str:
        emoji = _GRADE_EMOJI.get(self.grade, "")
        lines = [
            f"# {emoji} Oracle Reliability Report — grade **{self.grade}**",
            "",
            f"> Oracle: **{self.oracle_name}** · "
            f"{self.metrics.get('n_probes')} probes, "
            f"{self.metrics.get('n_answers')} answers, "
            f"{self.wall_seconds}s",
            "",
            "## Checks",
            "",
            "| check | value | threshold | pass |",
            "|---|---|---|---|",
        ]
        for name, c in self.checks.items():
            op = "≤" if name in ("prompt_sensitivity", "null_hallucination") else "≥"
            lines.append(
                f"| {name.replace('_', ' ')} | {_fmt(c.get('value'))} | "
                f"{op} {_fmt(c.get('threshold'))} | {_fmt(c.get('passed'))} |"
            )
        lines += ["", "## Per-probe results", "",
                  "| probe | kind | consistency | accuracy | prompt spread | hallucination |",
                  "|---|---|---|---|---|---|"]
        for p in self.per_probe:
            lines.append(
                f"| {p['name']} | {p['kind']} | {_fmt(p.get('consistency'))} | "
                f"{_fmt(p.get('accuracy'))} | {_fmt(p.get('prompt_spread'))} | "
                f"{_fmt(p.get('hallucination_rate'))} |"
            )
        if self.notes:
            lines += ["", "## Notes", ""] + [f"- {n}" for n in self.notes]
        lines += ["", "*Generated by [StressKit](https://github.com/Arth-Singh/Stresskit).*"]
        return "\n".join(lines)


def stress_oracle(
    ask_fn: AskFn,
    probes: Sequence[OracleProbe],
    *,
    judge: J.Judge = J.normalized,
    expected_judge: J.Judge = J.contains,
    abstain: Callable[[str], bool] = J.default_abstain,
    n_repeats: int = 1,
    seed: int = 0,
    thresholds: Optional[OracleThresholds] = None,
    oracle_name: str = "oracle",
    verbose: bool = False,
) -> OracleReport:
    """Run the reliability battery against one activation reader.

    Measures, per probe and pooled:

    - **answer consistency** — mean pairwise agreement (under ``judge``)
      across all paraphrase × exemplar × repeat answers for a probe. An
      oracle whose story changes with the phrasing or the capture is not
      reading the activation, it is improvising.
    - **known accuracy** — fraction of answers containing/matching the
      ground truth (under ``expected_judge``), for kind="known" probes.
    - **prompt sensitivity** — max−min accuracy across question phrasings,
      quantifying the gap that "best-of-N prompt" reporting hides.
    - **null hallucination** — fraction of confidently asserted answers
      (``not abstain(answer)``) on kind="null" control probes.
    """
    thresholds = thresholds or OracleThresholds()
    t0 = time.time()
    per_probe: List[Dict[str, Any]] = []
    notes: List[str] = []

    for probe in probes:
        answers: List[str] = []
        by_question: Dict[str, List[str]] = {q: [] for q in probe.questions}
        for qi, question in enumerate(probe.questions):
            for ei, exemplar in enumerate(probe.exemplars):
                for r in range(n_repeats):
                    run_seed = seed + 1000 * qi + 10 * ei + r
                    ans = str(ask_fn(exemplar, question, run_seed))
                    answers.append(ans)
                    by_question[question].append(ans)
                    if verbose:  # pragma: no cover
                        print(f"[stresskit.oracle] {probe.name} q{qi} e{ei} r{r}: {ans!r}")

        row: Dict[str, Any] = {
            "name": probe.name,
            "kind": probe.kind,
            "concept": probe.concept,
            "n_answers": len(answers),
            "consistency": M.pairwise_agreement(answers, judge),
            "accuracy": None,
            "prompt_spread": None,
            "hallucination_rate": None,
            "answers_sample": answers[:6],
        }

        if probe.kind == "known":
            correct = [expected_judge(a, probe.expected) for a in answers]
            row["accuracy"] = sum(correct) / len(correct)
            if len(probe.questions) >= 2:
                per_q = [
                    sum(expected_judge(a, probe.expected) for a in ans_list) / len(ans_list)
                    for ans_list in by_question.values()
                ]
                row["prompt_spread"] = max(per_q) - min(per_q)
                row["per_question_accuracy"] = dict(
                    zip(probe.questions,
                        [round(x, 4) for x in per_q])
                )
        elif probe.kind == "null":
            asserted = [not abstain(a) for a in answers]
            row["hallucination_rate"] = sum(asserted) / len(asserted)

        per_probe.append(row)

    known = [p for p in per_probe if p["kind"] == "known"]
    nulls = [p for p in per_probe if p["kind"] == "null"]
    non_null = [p for p in per_probe if p["kind"] != "null"]

    def _avg(rows: List[Dict[str, Any]], key: str) -> Optional[float]:
        vals = [r[key] for r in rows if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    metrics: Dict[str, Any] = {
        "n_probes": len(probes),
        "n_answers": sum(p["n_answers"] for p in per_probe),
        "answer_consistency": _avg(non_null, "consistency"),
        "known_accuracy": _avg(known, "accuracy"),
        "prompt_spread": _avg(known, "prompt_spread"),
        "null_hallucination_rate": _avg(nulls, "hallucination_rate"),
    }

    checks: Dict[str, Any] = {}
    if metrics["answer_consistency"] is not None:
        checks["answer_consistency"] = {
            "value": metrics["answer_consistency"],
            "threshold": thresholds.consistency,
            "passed": metrics["answer_consistency"] >= thresholds.consistency,
            "description": "pairwise agreement across paraphrases/exemplars/repeats",
        }
    if metrics["known_accuracy"] is not None:
        checks["known_accuracy"] = {
            "value": metrics["known_accuracy"],
            "threshold": thresholds.accuracy,
            "passed": metrics["known_accuracy"] >= thresholds.accuracy,
            "description": "ground-truth recovery on known-answer probes",
        }
    if metrics["prompt_spread"] is not None:
        checks["prompt_sensitivity"] = {
            "value": metrics["prompt_spread"],
            "threshold": thresholds.prompt_spread,
            "passed": metrics["prompt_spread"] <= thresholds.prompt_spread,
            "description": "max accuracy gap across question phrasings",
        }
    if metrics["null_hallucination_rate"] is not None:
        checks["null_hallucination"] = {
            "value": metrics["null_hallucination_rate"],
            "threshold": thresholds.hallucination,
            "passed": metrics["null_hallucination_rate"] <= thresholds.hallucination,
            "description": "confident assertions on null-control activations",
        }
    if not checks:
        raise ValueError(
            "Nothing to grade: provide at least one probe with kind='known', "
            "'consistency' (≥2 answers), or 'null'."
        )
    if not nulls:
        notes.append(
            "no null-control probes supplied — hallucination discipline untested. "
            "Add kind='null' probes (random activations, or base-model activations "
            "for a fine-tuning-specific question)."
        )

    return OracleReport(
        oracle_name=oracle_name,
        per_probe=per_probe,
        metrics=metrics,
        checks=checks,
        grade=grade_checks(checks),
        notes=notes,
        created_at=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        wall_seconds=round(time.time() - t0, 3),
    )


def blind_spot_matrix(
    oracles: Mapping[str, AskFn],
    probes: Sequence[OracleProbe],
    *,
    expected_judge: J.Judge = J.contains,
    n_repeats: int = 1,
    seed: int = 0,
    margin: float = 0.2,
) -> Dict[str, Any]:
    """Cross-oracle × concept accuracy matrix with blind-spot flags.

    Implements the diagnostic of arXiv:2607.23379: an oracle has a
    *concept-specific blind spot* when its accuracy on one concept is lower
    both than (a) the other oracles' mean accuracy on that concept and
    (b) its own mean accuracy on the other concepts, each by ``margin``.

    Only kind="known" probes with a ``concept`` participate.
    """
    known = [p for p in probes if p.kind == "known" and p.concept]
    if not known:
        raise ValueError("blind_spot_matrix needs kind='known' probes with concept= set")
    concepts = sorted({p.concept for p in known})

    accuracy: Dict[str, Dict[str, float]] = {}
    for oname, ask_fn in oracles.items():
        accuracy[oname] = {}
        for concept in concepts:
            correct, total = 0, 0
            for probe in (p for p in known if p.concept == concept):
                for qi, question in enumerate(probe.questions):
                    for ei, exemplar in enumerate(probe.exemplars):
                        for r in range(n_repeats):
                            run_seed = seed + 1000 * qi + 10 * ei + r
                            ans = str(ask_fn(exemplar, question, run_seed))
                            correct += bool(expected_judge(ans, probe.expected))
                            total += 1
            accuracy[oname][concept] = correct / total if total else float("nan")

    flags: List[Dict[str, Any]] = []
    for oname in oracles:
        for concept in concepts:
            acc = accuracy[oname][concept]
            others_on_c = [accuracy[o][concept] for o in oracles if o != oname]
            own_on_others = [accuracy[oname][c] for c in concepts if c != concept]
            if not others_on_c or not own_on_others:
                continue
            mean_others = sum(others_on_c) / len(others_on_c)
            mean_own = sum(own_on_others) / len(own_on_others)
            if acc <= mean_others - margin and acc <= mean_own - margin:
                flags.append({
                    "oracle": oname,
                    "concept": concept,
                    "accuracy": acc,
                    "others_mean_on_concept": mean_others,
                    "own_mean_on_other_concepts": mean_own,
                })

    return {"concepts": concepts, "accuracy": accuracy, "blind_spots": flags,
            "margin": margin}
