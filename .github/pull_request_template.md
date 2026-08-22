## What

<!-- One paragraph: what this PR does and why. -->

## Checklist

- [ ] `pytest -q` passes
- [ ] `stresskit verify references/` passes

**Reference-card PRs additionally** (see `references/PROTOCOL.md` §8):

- [ ] card JSON passes `stresskit verify` · markdown render · badge
- [ ] runner script with upstream versions pinned
- [ ] verdict trace artifacts
- [ ] null control (or a note why none is constructible)
- [ ] scope note: which artifact, in which usage mode
- [ ] `stresskit scoreboard references -o SCOREBOARD.md` regenerated
- [ ] section added to `references/README.md`, including what the result does *not* show
- [ ] C/D grade on named work → upstream courtesy issue opened (PROTOCOL §6)
