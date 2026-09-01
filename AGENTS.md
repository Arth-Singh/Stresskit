# StressKit contributor guide

## Mission

StressKit is an open-source stability and sanity-check harness for mechanistic
interpretability claims. It wraps existing discovery methods and produces
auditable, machine-readable Stability Cards. Optimize for scientific
correctness, reproducibility, auditability, and low adoption friction before
feature breadth.

StressKit is not a circuit-discovery or SAE-training library. Keep discovery
methods in upstream tools and bridge their outputs into `Finding` objects or
graded reports.

## Repository map

- `src/stresskit/finding.py`: `Finding` data model and JSONL ingestion.
- `src/stresskit/battery.py`: one-at-a-time perturbation battery, aggregation,
  grading, caching, post-hoc grading, and verdict traces.
- `src/stresskit/metrics.py` and `baselines.py`: statistical primitives,
  confidence intervals, similarity metrics, and random-null baselines.
- `src/stresskit/card.py` and `schemas/`: Stability Card serialization,
  validation, and auditor-mode verification.
- `src/stresskit/oracle.py`: Activation Oracle reliability batteries and blind
  spot analysis.
- `src/stresskit/adapters/`: bridges from external interpretability tools and
  saved outputs.
- `src/stresskit/cli.py`: `stresskit` command implementations and parser.
- `htmlcard.py`, `tracechart.py`, `scoreboard.py`, and `site.py`: deterministic
  renderers for derived artifacts.
- `references/`: published reference cards, pinned runner scripts, protocol,
  target queue, and generated renders.
- `tests/`: CPU-only pytest suite using synthetic fixtures.

Public API is curated in `src/stresskit/__init__.py`. Export new user-facing
objects there when appropriate.

## Setup and commands

Use Python 3.9 or newer.

```bash
python -m pip install -e ".[dev]"
pytest -q
python examples/quickstart_toy.py
python examples/sae_audit.py
python examples/oracle_reliability.py
stresskit verify references/
```

For a checkout that has not been installed, prefix Python and pytest commands
with `PYTHONPATH=src`.

Run one test module while iterating:

```bash
pytest -q tests/test_battery.py
```

Reference-card changes require regeneration and verification:

```bash
stresskit verify references/
stresskit scoreboard references -o SCOREBOARD.md
git diff --exit-code SCOREBOARD.md
```

CI tests Python 3.9, 3.11, and 3.13; runs all examples; verifies every reference
artifact; checks scoreboard freshness; and repeats the suite with only NumPy as
the runtime dependency.

## Scientific and implementation invariants

- Preserve one-at-a-time battery semantics: vary one axis around the base
  configuration so attribution remains legible and run counts remain linear.
- Treat finder functions as deterministic functions of `(data, seed, config)`.
  New paths must propagate seeds and avoid hidden randomness.
- Never interpret structural overlap across different component universes.
  Findings may declare `meta["universe"]`; cross-universe runs still contribute
  claim and score evidence but not Jaccard evidence.
- Keep confidence intervals, underpowered states, skipped axes, and null-control
  limitations visible. Do not turn missing evidence into a pass or fail.
- Cards are auditable artifacts. `stresskit verify` must re-derive checks and
  grade from stored metrics and thresholds; renderers must not change meaning.
- Preserve deterministic serialization and rendering. Stable output matters for
  review, CI, and long-lived citations.
- NumPy is the only required runtime dependency. SciPy remains optional. Do not
  add heavy imports to core modules or adapter module scope.
- Adapters should prefer post-hoc ingestion of saved JSON, tensors, or logs.
  Import upstream libraries inside functions and raise actionable errors when an
  optional dependency is missing.
- Every public function needs a docstring. Every metric or default threshold
  needs a source citation.
- Match surrounding style: four-space indentation, type hints, standard-library
  dataclasses, and explicit `Optional`/`Sequence` forms compatible with Python
  3.9. No formatter or linter is currently enforced.

## Evidence and governance

`references/PROTOCOL.md` is authoritative for reference-card work. A card
submission includes verified JSON, Markdown render, badge, pinned runner,
verdict trace, null control or explicit limitation, scope note, and regenerated
scoreboard.

Default thresholds are pre-registered scientific policy, not ordinary tuning
constants. Changing one requires an issue, supporting evidence, a versioned
release, changelog entry, and regenerated reference grades. Do not combine a
threshold change with a card whose verdict benefits from that change.

For public C/D results about named upstream work, follow the protocol's courtesy
and dispute process before publicity. Describe exactly what an artifact tests
and what it does not establish.

## Change checklist

1. Add or update focused tests with small deterministic synthetic inputs; CI has
   no GPU.
2. Exercise edge cases such as empty or underdetermined samples, heterogeneous
   finding sizes, missing optional fields, and tampered artifacts.
3. Run the full pytest suite and affected examples.
4. For card, schema, metric, threshold, adapter, or CLI changes, run the matching
   auditor and reference checks above.
5. Keep `pyproject.toml`, `src/stresskit/__init__.py`, and `CHANGELOG.md` aligned
   when releasing a version.
