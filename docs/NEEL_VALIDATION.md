# StressKit validation against Neel Nanda transcript criteria

## Verdict

StressKit verifier mechanics are demonstrated. Neel's empirical claims are **not** verified, and no final external-utility or August claim result is registered yet.

Safe claim: StressKit defensively verifies audit provenance and rejects several known false-pass paths; external scientific usefulness remains unproven.

## Evidence available now

- Adversarial proof slice: 51 passed; zero failures/errors/skips.
- Compiler evaluation: 300 planted cases; gate passed.
- Calibration: minimum observed coverage 99.45%; maximum known-invalid false-pass 0.05%.
- Live panel 1: three distinct attested provider routes; three unsupported opinions produced abstention.
- Live panel 2: one extractor accepted; one rejected after 4 source-quote mismatches; dependent critic not run; no retry; abstention.
- Live panel 3 (ACDC): both extractors rejected; 3 of 10 quotes failed exact-source checks and one completion hit its frozen token limit; critic not run; no retry; claim abstained.
- Thought Anchors CoT panel: three live routes passed catalog and ZDR checks, but account prompt-logging state was not readable; protocol stopped before sending source text; zero opinions, critic calls, or GPU.
- Thought Anchors external utility/generalization: four non-internals baselines and held-out design registered, but metadata-only preflight could not establish 200 independent problem clusters per partition; abstain; no labels, outcome, or GPU used.
- Gradient-persona flagship: license audit records 7 unresolved blockers; no experiment or artifact substitution; abstain.
- Prefreeze registry: 0 eligible, 53 pending, 15 excluded.

## Criteria themes

| Theme | Status | What evidence says | Limitation |
|---|---|---|---|
| `scientific_integrity` | `verifier_mechanics_demonstrated` | Adversarial tests demonstrate fail-closed handling of stable nonsense, missing runs, forged evidence, unsafe agent input, and multiplicity. | Software checks and calibrated inference do not establish construct validity or convergence to scientific truth. |
| `external_utility` | `protocol_hardened_benchmark_pending` | Utility verifier recomputes external-task metrics and comparison against the strongest registered non-internals baseline from raw evidence. | No final external-task utility AuditBundle exists. |
| `generalization` | `protocol_hardened_benchmark_pending` | Generalization is a separate primary axis and must use frozen held-out partition bindings rather than a relabeled primary split. | No final held-out claim result exists. |
| `cot_probe_steering` | `protocol_hardened_benchmark_pending` | Registry covers CoT, probes, and steering; ordered CoT trajectories retain event order and multiplicity. | Coverage is designed but no frozen claim in these strata has run. |
| `verifiable_agent_auditing` | `mechanism_demonstrated_live_sample_insufficient` | Planted compiler cases and live panel executions demonstrate provenance binding, unsupported-wording abstention, and rejection of invented quotes. | The small live sample cannot estimate extractor recall, precision, provider drift, or collusion resistance. |
| `gradient_persona_flagship` | `blocked_abstain` | Study protocol binds residual-stream loss-gradient projections, held-out behavior, non-internals baselines, and falsification controls. | Required licensed persona and misalignment artifacts are unresolved. |

## Boundaries

- Calibration covers bounded inference, not task choice, null validity, or scientific truth.
- Planted compiler cases do not estimate live extractor recall or provider drift.
- 3 live abstentions demonstrate fail-closed behavior, not successful claim auditing rate.
- Persona-gradient flagship remains abstained until every required artifact license resolves.
- `external_validation: not obtained`.

Full transcript-derived N01-N48 registry: `benchmark/neel_criteria_v1.json`.
Proof selector manifest: `artifacts/validation/neel-adversarial-proof-selectors-v1.txt`.
Machine report: `artifacts/validation/neel-validation-v1.json`.
