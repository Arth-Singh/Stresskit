# StressKit v1 autonomous claim-audit protocol

Status: implemented protocol core. Public benchmark release remains blocked on
the external gates in `RELEASE_GATES_V1.md`.

## Scope and decision target

StressKit v1 evaluates one frozen, falsifiable claim. It does not assign a
paper-quality score, rank papers, or issue a whole-paper truth verdict. Historical
v0.1–v0.3 Stability Cards and A–D diagnostic grades retain their original
meaning and schema; v1 never reinterprets them.

Every frozen registry row ends in exactly one status:

- `pass`: reproduction and every required, multiplicity-corrected audit gate pass;
- `audit_failure`: reproduction succeeds but a required scientific gate fails;
- `reproduction_failure`: a declared run fails, crashes, times out, or is
  explicitly missing;
- `inconclusive`: evidence does not resolve every required gate;
- `protocol_deviation`: signatures, manifests, digests, dependencies, or frozen
  procedures do not verify;
- `excluded`: an outcome-blind pre-freeze rule excludes the row;
- `abstain`: evidence, profile support, source safety, licensing, or executor
  isolation is insufficient to conduct the audit.

`publication_state` is separately `final` or `abstain`. Decisions report
reproduction, stability/specificity, utility, generalization, and evidence
confidence as distinct fields.

## Artifact flow

`stresskit audit source` hashes raw local source files, UTF-8 extractions, and
typed license evidence into offline content-addressed storage. Its emitted
closure is the only input accepted as a final `SourceBundle`; network retrieval
and dependency builds remain separate preparation steps.
Tagged digest leaves outside document rows, such as a pinned repository
license file, must be supplied explicitly as
`--closure-input SHA256_DIGEST=PATH`. The intake manifest is itself stored,
and source generation fails unless every reachable digest verifies.

```text
SourceBundle
  -> two isolated extractor AgentOpinions + one critic AgentOpinion
  -> ClaimRecord or abstain
  -> frozen AuditSpec and regenerated run manifest
  -> signed ResourcePlan
  -> isolated RunAttestations and content-addressed raw outputs
  -> AuditBundle
  -> offline release verification
  -> claim-level evidence board
```

Agent output never determines a verdict. Agents propose exact claim wording,
source anchors, code maps, perturbations, and controls. Trusted reducers and
calibrated deterministic code recompute every finding and decision from raw
objects.

## Claim compilation

A final `ClaimRecord` requires exactly two extractors from distinct providers
and model families plus one critic. Every opinion binds source, prompt, request,
model-descriptor, and quote digests. Anchors are non-empty byte ranges verified
against content-addressed UTF-8 source extractions. All three must support
identical wording and exact anchors.
Unsupported language, any reported issue, provider/model-family collapse,
missing anchors, source mismatch, instruction-like document text, or any agent
disagreement forces abstention. Majority vote cannot repair missing evidence.
Every source document also binds a content-addressed license record. Only
`verified_compatible` reaches claim compilation; `unresolved` or `incompatible`
forces abstention before resource planning.

Benchmark-level freeze adds a complete typed qualification ledger. Each
candidate remains `pending`, becomes `excluded` after any failed pre-freeze
gate, or becomes `eligible` only after SourceBundle, agent panel, ClaimRecord,
license closure, claim-map execution smoke, AuditSpec, protocol review, and
resource estimate evidence all verify. Eligible AuditSpecs must share one
global release multiplicity family. Backward census extension begins only
after every current-window row has a final pre-freeze disposition and breadth
still falls short.

Agent-panel abstentions embed their source, frozen panel, discovery decision,
rejected attempts, and attested routes. A complete panel embeds three opinions.
An invalid-output panel instead binds one accepted extractor, one rejected
extractor attempt, a dependency-blocked critic, and exact three-slot execution
accounting. Qualification requires the CAS and verifies source, opinion, raw
response, route, and panel closures. Self-consistent but nonexistent digests
cannot create an exclusion.

The frozen 300-case planted evaluation is
`artifacts/benchmark/compiler-evaluation-v1-300.json`. It records 180/180
unambiguous compilations and abstention on all 120 unsupported, injected,
missing-evidence, or unsupported-profile cases. This validates deterministic
compiler gates, not live provider recall or future model drift.

### Optional live OpenRouter preparation

`stresskit audit opinion` can produce one extractor or critic opinion before
discovery. This is the only networked agent step; core compilation and all
verdict computation remain offline. It reads `OPENROUTER_API_KEY` exclusively
from process environment and posts to the hardcoded
`https://openrouter.ai/api/v1/chat/completions` endpoint. No CLI key, custom
base URL, hidden retry, streaming response, model router, `latest` alias, or
cross-provider fallback is accepted.

Each call loads one immutable request row from a content-addressed frozen panel;
the CLI accepts no independent model, provider, family, role, seed, temperature,
or token-limit overrides. Claim-query UTF-8 bytes must match their panel digest.
Catalog canonical slug remains frozen metadata while
the catalog request ID is sent. Every endpoint must declare common support for
`max_tokens`, temperature, seed, response format, and structured outputs.
Routing requests `allow_fallbacks: false`,
`require_parameters: true`, `data_collection: deny`, and `zdr: true`.
Successful responses must expose exactly one available and selected endpoint,
the panel-bound display provider, selected endpoint canonical model slug, and
top-level requested model ID. Router `endpoints.total` is a prefilter count and
must be a positive integer; it need not equal the one-row `available` list.
One direct first attempt and a null or empty pipeline are required via OpenRouter
router metadata, complete with
`finish_reason: stop`; missing cache metadata, transformed pipelines, route
drift, truncation, malformed JSON, or provider/model mismatch rejects the
opinion. Structured-output schema is also validated locally because remote
enforcement is not a trust boundary.

Models return exact source quotes, never trusted offsets or hashes. Local code
uniquely locates each quote in UTF-8 bytes and computes its range and digest.
CAS stores exact panel, normalized route binding, prompt, sanitized request
receipt, raw response, selected route/model descriptor, quote bytes, and final
opinion. The claim query and complete SourceBundle provenance are closure
members; critic prompt provenance explicitly links both extractor opinions.
Missing closure objects reject before network use. Text offsets bind the
extracted-text digest separately from raw source identity. Authorization bytes
are neither logged nor serialized.
`AgentOpinion.provider` names the selected
upstream inference provider; model descriptor separately records OpenRouter as
transport. Extractor calls remain isolated, and critic call binds both prior
outputs. Existing cross-provider, cross-family, unanimity, evidence, and
prompt-injection gates remain unchanged.

OpenRouter and Hugging Face require unrelated credentials. `HF_TOKEN` is only
for separately authorized model/artifact retrieval. It is forbidden in agent
requests, content-addressed provenance, audit workers, and repository files.
Neither live opinion generation nor source compilation needs a GPU. GPU access
begins only after an eligible `AuditSpec` freezes and a signed `ResourcePlan`
requests it.

## Frozen audit design

Each `AuditSpec` binds:

- exact claim locator, source digest, and complete agent-opinion closure;
- repository digest/revision, relative entrypoints, dependency manifest, and
  disposable build recipe inherited from the `ClaimRecord` code map;
- a built-in reducer name, semantic implementation digest, and configuration;
- a joint specification distribution and its canonical digest;
- explicit primary and generalization evaluation manifests, distinct evaluation
  IDs, content-addressed axis IDs, and canonically ordered `held_out_axes` chosen
  from dataset, model, prompt, and unit;
- a manifest regenerated solely from the joint design, fixed stopping rule,
  seeds, partitions, and two cohorts;
- dependency and cluster identifiers for every run slot;
- known-truth positive and claim-specific negative/randomization controls;
- a registered threshold profile and immutable profile digest;
- a global release family using Holm–Bonferroni at the registered alpha;
- a declared hardware class and `bitwise`, `numeric_with_tolerance`, or
  `statistical` independent-rerun level.

Only fixed, outcome-blind stopping is supported in v1. Failed and absent runs
remain declared slots. Deleting a slot is a protocol deviation; returning a
signed terminal failure is a reproduction failure.

Primary and generalization manifests bind the same applicable axis names. Every
registered held-out axis must have disjoint ID sets; every other bound axis must
remain identical. Axis identities join bundle digest closure, so the same
content-addressed object cannot appear on both sides of a held-out axis. Merely
naming a partition `generalization` is rejected. Each
run slot carries its evaluation partition, evaluation ID, manifest digest, axis
IDs, and held-out-axis declaration.

## Registered profiles

`audit_profiles.py` freezes seven supported profiles:

- `set_graph_v1`;
- `categorical_v1`;
- `scalar_effect_v1`;
- `vector_direction_v1`;
- `ranked_output_v1`;
- `utility_v1`;
- `cot_trajectory_v1`.

Unknown claim types or reducers force abstention. Pair statistics use disjoint
independent units, never the much larger dependent complete-pair count.
Positive-control recovery, claim support, negative-control truth,
claim-specific falsification, specificity, stability, and held-out
generalization are separate primary checks. Stability alone cannot pass a
constant or data-independent claim.

All bounded checks use two-sided Hoeffding intervals over declared independent
units. The 2,000-trial primary calibration and disjoint fresh-seed replication
are stored in `artifacts/calibration/v1-audit-profiles-2000.json`. Minimum
observed coverage is 99.45%; maximum known-invalid false-pass rate is 0.05%.
The artifact reports power, minimum detectable margins, and independent-unit
requirements. These simulations validate bounded inference, not construct
validity of a paper-specific null.

## External-task utility

V1 utility evidence contains raw labels, raw method predictions, raw baseline
predictions, independent-unit IDs, and a held-out split. The verifier recomputes
metric values, best non-internals baseline, oriented delta, interval, practical
margin, and state. Stored summaries are assertions to verify, never inputs to
the decision.

The `ClaimRecord` freezes exact external-task text, canonical metric spec, and
complete baseline registry before runs. External tasks using interpretability-
method jargon are rejected instead of being treated as downstream utility. Raw
evidence using another task, metric, split, independent-unit policy, or baseline
set is rejected. Utility claim and
control targets bind expected verified states (`pass`, `fail`, or
`inconclusive`); raw outputs cannot redefine target semantics.

Metric direction, bounds, independent unit, practical margin, generalization
split, and nondecomposable policy are frozen. Precision, recall, F1, and AUROC
are computed within independent units and averaged; treating their examples as
additive is rejected. At least one baseline must explicitly use no model
internals. Every baseline freezes an implementation digest, input-manifest
digest, allowed input kinds, and canonical access policy. Non-internals
baselines forbid activation, gradient, weight, and internal-state mounts.
Resource plans declare and run attestations repeat exact mounted input-manifest
digests; bundle closure verifies implementation, manifest, and input objects.
This verifies declared and mounted provenance, not semantic absence of covert
leakage inside baseline code. Contradictory values or intervals force
abstention.

## Global multiplicity

Every primary p-value in the frozen release belongs to one named family.
Publication invokes release-level verification and applies Holm–Bonferroni
across the entire family. A per-claim `pass` that does not survive the global
step-down procedure becomes `inconclusive`. Missing family members make the
release a protocol deviation; they cannot silently shrink the correction.

Secondary analyses are labeled descriptive and never enter a primary verdict.

## Content and execution integrity

Raw source, prompt, request, quote, output, error, and release-manifest objects
live in SHA-256 content-addressed storage. Bundles list an exact digest closure;
missing, tampered, duplicate, or unreachable objects fail verification.
Resource plans and run attestations use explicit signed algorithms and key IDs.
Plan-signing and executor-signing keys occupy separate trust domains; reused IDs
or key material are protocol deviations. Public releases use Ed25519 private
signers and distribute public verifier keys. Deployment-local HMAC remains
supported explicitly. Both algorithms bind algorithm and key ID as well as
canonical payload.

Dependency builds occur in disposable networked sandboxes. Claim execution
requires disabled network, absent credentials, read-only inputs, quota-limited
scratch, and allowlisted outputs. Each replication uses a distinct execution
environment on the same declared hardware class. Missing isolation forces
abstention.

## Publication and governance

Publisher re-verifies every bundle before rendering. Frozen registry order is
preserved; excluded and abstained rows remain visible. Paper pages list claims
without aggregation to a paper verdict. Comparisons are permitted only inside
an identical task, metric, evaluation-set, and resource-budget key.

Adverse named results remain blocked until the upstream author responds or the
14-day response window elapses. Responses travel with evidence. Agent-only
methodological review is an explicit CLI choice; public artifacts continue to
state `external_validation: not obtained` and describe decisions as conditional
on frozen `AuditSpec`s.
