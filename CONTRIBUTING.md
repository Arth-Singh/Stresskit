# Contributing

StressKit's value grows with two things: **reference cards** (more findings
on the [scoreboard](SCOREBOARD.md)) and **adapters** (more pipelines that
can produce cards). Both are great first contributions. Bug fixes, threshold
arguments, and docs are welcome too.

## Submit a reference card (the blessed path)

You battery-test a published finding or released instrument and add its
Stability Card to the reference set. Start with
[`references/TARGETS.md`](references/TARGETS.md) — the prioritized queue —
or propose your own target via the *card submission* issue template.

The standard of evidence is [`references/PROTOCOL.md`](references/PROTOCOL.md);
its §8 checklist is the PR checklist. In short, a card PR carries:

1. the card JSON — it must pass `stresskit verify path/to/card.json`
2. the markdown render and badge (`stresskit render` / `stresskit badge`)
3. the runner script, upstream versions pinned
4. the verdict trace (`sk.verdict_trace`)
5. a null control, or a note saying why none is constructible
6. an entry in `references/papers.json` (paper title, arXiv id, models, the
   card paths, whether the released number reproduced, a one-line result,
   audit date) — `stresskit scoreboard` and `stresskit site` refuse a graded
   card that no entry claims
7. a regenerated scoreboard: `stresskit scoreboard references -o SCOREBOARD.md`
8. a section in `references/README.md` following the existing pattern —
   including what the result does **not** show, and a row plus a short
   section in `RESULTS.md`

CI re-verifies every card on every push, so a card that doesn't recompute
from its own metrics cannot merge. For C/D grades on named work, the
upstream-courtesy step in PROTOCOL §6 applies before the card is
publicized.

## Write an adapter

Adapters live in `src/stresskit/adapters/` and bridge from an existing tool
to `Finding` objects (or straight to a graded report). Look at
`adapters/eap.py` (files → findings) and `adapters/sae_lens.py` (one-call
graded report) for the two common shapes. Ground rules:

- **No heavy imports at module level.** The core stays numpy-only;
  import the upstream tool inside functions and fail with a clear message.
- **Post-hoc first.** The most useful adapters read *saved outputs*
  (JSON, tensors, logs) so people can grade runs they already have,
  without a GPU.
- Add tests under `tests/` using synthetic inputs — CI has no GPU.

## Development

```bash
pip install -e ".[dev]"
pytest -q                      # full suite, seconds, no GPU
python examples/quickstart_toy.py
stresskit verify references/   # every published card must re-derive
```

Match the surrounding code style; every public function gets a docstring;
every metric or threshold gets a citation. Python ≥ 3.9, numpy as the only
required dependency (scipy optional).

## Arguing about thresholds

The default bars are the published proposals, not revealed truth — but they
are **pre-registered**: changes go through an issue + a versioned release
with a changelog entry, never in the same PR as a card whose grade would
move (PROTOCOL §2). Open a *threshold* issue with the argument and the
evidence; regenerated grades across all reference cards are part of any
such change.

## Disputing a card

Open a *card dispute* issue with a `stresskit verify` failure, a defensible
re-analysis, or evidence the runner misuses the upstream tool. Disputes are
settled by re-running, not arguing — see PROTOCOL §7.
