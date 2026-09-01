# StressKit open-code benchmark

`registry.candidates.json` is a broad candidate frame, not evidence that any
entry is reproducible, stable, valid, or eligible for final benchmarking.
`REGISTRY_PROTOCOL.md` defines how candidates become a frozen preregistration.

Current candidate frame deliberately spans automated circuits, sparse-feature
circuits, causal tracing/editing, activation directions and steering, lenses,
representation patching, causal interventions, and SAE tooling. Entries with
unresolved licensing or claim extraction remain visible rather than silently
removed.

## Outcome-blind qualification

`qualification.prefreeze.json` is the complete 68-row gate ledger. It is not a
frozen registry and contains no benchmark outcomes. Regenerate it and its
machine-readable blocker report with:

```bash
python benchmark/qualify_candidates.py scaffold \
  benchmark/registry.candidates.json \
  --out benchmark/qualification.prefreeze.json
python benchmark/qualify_candidates.py report \
  benchmark/registry.candidates.json \
  benchmark/qualification.prefreeze.json \
  --cas .stresskit/cas \
  --out artifacts/benchmark/prefreeze-qualification-report-v1.json
```

Every eligible row must carry digest-bound evidence for its SourceBundle,
isolated agent panel, ClaimRecord, complete license closure, isolated execution
smoke, frozen AuditSpec, protocol review, and outcome-blind resource estimate.
Manual `eligible` labels cannot bypass these validators. Freeze is refused
while any row remains pending or launch breadth is unmet. Only after freeze
should signed ResourcePlans request GPU executors.

```bash
python benchmark/qualify_candidates.py freeze \
  benchmark/registry.candidates.json \
  benchmark/qualification.prefreeze.json \
  --cas .stresskit/cas \
  --release-id <release-id> --frozen-at <ISO-8601> \
  --out benchmark/registry.frozen.json
```

## Adding a candidate upstream

1. Clone the repository and check out the commit to pin.
2. `python benchmark/pin_upstream.py --name <key> --checkout <clone> --repository <url>
   --entrypoint <path> ... --merge-into benchmark/upstream_sources.json` records commit,
   tree, license file hash, entrypoints, and static-syntax status.
3. Add the same key to `registry.candidates.json` (`upstreams` and one entry per
   distinct claim or instrument), resolve every model/artifact revision into
   `model_sources.json` / `artifact_sources.json`, and log the addition in
   `discovery_log`.
4. Regenerate the frozen static audit over every pinned checkout:
   `python benchmark/freeze_static_audit.py --clone-root <dir> --out
   artifacts/benchmark/upstream-static-audit-<date>.json`, then point
   `tests/test_source_manifests.py` at the new artifact.
5. Rerun `jobs/upstream-source-fetch-array.slurm` so the independent Nibi
   sidecars cover the new keys, and drop them from `NIBI_AUDIT_PENDING`.

Discovery pass 2 (2026-08-28) added 11 upstreams from 2025–2026 released
instruments and claims; see `discovery_log` in the registry for what was added,
what was not, and why.

Discovery pass 3 (2026-08-31) worked from a census rather than a reading list:
every mechanistic-interpretability paper submitted to arXiv during August 2026
was collected, then audited for released code. 108 papers, 33 authored public
repositories, 18 of them licensed. `discovery/AUGUST_2026.md` reports the
method and the counts, `discovery/august-2026-frame.json` is the row-level
frame, and 11 upstreams plus 14 entries were pinned from it. The frame records
papers with no code and papers whose repositories carry no license as well, so
the denominator stays visible.

Discovery pass 3b (2026-09-01) repaired two outcome-blind frame gaps. It queried
`chain of thought`, `reasoning trace`, `persona`, `introspection`, `evaluation
awareness`, and `model organism`, promoted 3 former Tier-B rows, found 350 new
term-led candidates for manual scope review, and code-audited all 47 retained
Tier-B rows. `discovery/august-2026-pass3b.json` records all 410 dispositions;
18 rows expose licensed authored repositories but remain unfrozen until claim
mapping, all dependency licenses, and smoke reproduction pass.
