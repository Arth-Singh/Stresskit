# StressKit benchmark registry protocol

Status: candidate-registry draft 0.1. This file and
`registry.candidates.json` are not frozen preregistration artifacts yet.

## Objective

Build a code-first sampling frame of concrete mechanistic-interpretability
claims and released instruments. Audit claim stability and specificity under a
declared set of defensible analytic choices. Do not grade papers or decide in
advance that a method works or fails.

## Discovery cutoff and sources

Discovery cutoff: 2026-08-31, Asia/Hong_Kong (pass 1 closed 2026-08-24; pass 2 added
2025–2026 released instruments and claims; pass 3 closed 2026-08-31 with a census of
August 2026 — see `discovery_log` in the registry).

Sources are searched in this order:

1. official code links in papers and project pages;
2. public GitHub repositories owned by authors, research groups, or designated
   maintainers;
3. repository READMEs, executable experiment entrypoints, tests, and released
   artifacts;
4. paper methods, result tables, and appendices for exact load-bearing claim
   extraction.

Search terms combine `mechanistic interpretability` with `circuit`, `activation
patching`, `causal tracing`, `sparse autoencoder`, `steering`, `refusal`,
`lens`, `intervention`, and `code`; pass 2 added `persona vector`, `activation oracle`,
`natural language autoencoder`, `Jacobian lens`, `thought anchors`, `model diffing`,
`emergent misalignment`, `introspection`, and `evaluation awareness`. Candidate additions and exclusions must be
logged; repository popularity and expected outcome are not inclusion criteria.

Pass 3 replaced term-led reading with a period census: every arXiv submission in a
fixed window is collected by term search, filtered to the relevant categories,
stratified by term breadth, and audited for released code, so the papers with no code
and the repositories with no license are counted rather than skipped. The window,
queries, per-term hit counts, detection method and its known blind spots are recorded
in `discovery/august-2026-frame.json`; `discovery/AUGUST_2026.md` summarises them.
Censuses are the preferred discovery mode from pass 3 onward, because a reading list
cannot report a denominator.

## Inclusion rules

A claim or instrument can enter the frozen registry only when all apply:

- public source code is pinned to a full commit hash;
- source license and every required model/data license permit the planned use;
- a concrete statement and deterministic raw-output-to-finding map can be
  written without strengthening the upstream claim;
- model, task, usage mode, metric, and load-bearing code path are identifiable;
- at least one scientifically meaningful perturbation and one defensible null
  can be preregistered;
- required artifacts are downloadable or regenerable within declared compute
  and storage budgets;
- a smoke reproduction succeeds before the registry freezes.

An instrument entry may test the reproducibility of a released measurement
pipeline without attributing a scientific claim to its maintainers.

## Exclusions and separate failure classes

Exclude before freeze when code is non-executable, a required artifact is
private, licensing is absent or incompatible, compute exceeds the registered
budget, or no deterministic claim map/null can be constructed. Record reason.

After freeze, distinguish:

- `reproduction_failure`: pinned upstream result cannot be regenerated;
- `audit_failure`: result regenerates but fails a preregistered StressKit gate;
- `inconclusive`: uncertainty or run budget does not resolve a gate;
- `protocol_deviation`: frozen procedure could not be followed;
- `pass`: every required registered gate passes.

These categories never collapse into “paper false.”

## Selection and breadth

Freeze all eligible entries from the recorded candidate frame, not a subset
chosen after viewing StressKit outcomes. Initial launch requires at least 20
entries, 6 method families, and 3 model families after exclusions. Multiple
claims from one repository are allowed only when they use distinct upstream
tasks, model families, or load-bearing result statements; denominators report
both claim and repository counts.

## Pre-freeze checks

For every candidate:

1. verify commit and license;
2. extract an exact paper claim with page/table/figure locator, or mark it as an
   instrument audit;
3. execute upstream smoke command;
4. freeze model/data revisions and hashes;
5. implement and test claim map;
6. declare specification distribution, null, dependency units, and run budget;
7. estimate Nibi resources without inspecting confirmatory outcomes;
8. obtain protocol review and assign immutable `claim_id`.

## Outcome blindness

Registry, decision rules, exclusion rules, and launch wording template freeze
before confirmatory jobs. Pilot outputs use disjoint prompts/seeds and may only
change methods through a public deviation log. Headline counts are generated
from verified cards, never hand-selected.
