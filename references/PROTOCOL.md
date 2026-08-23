# The evidence standard for reference cards

Every reference battery in this repository grades a *published finding* or a
*released instrument* — someone's real work. That demands a higher standard
of evidence than ordinary benchmarking. This document is that standard. A
card that does not meet it does not merge.

The point of the standard is symmetry: the same rules that make a failing
grade hard to dismiss make it hard to abuse.

## 1. Grades are measurements, not accusations

A StressKit grade records how a finding behaved under a pre-registered
perturbation battery — nothing else.

- A **D** means *"under these checks, at these thresholds, the finding did
  not survive defensible variation in how it was produced"*. It is not a
  claim of error, misconduct, or bad faith, and cards never use words like
  "debunked", "fraud", or "fake".
- Grades attach to **findings and instruments** (a circuit claim, a released
  lens, an oracle checkpoint), never to papers as a whole and never to
  authors.
- Every failing check is stated next to what it does *not* show. A
  specificity failure means the method recovers similar structure without
  the effect present — it does not mean the original task analysis is
  worthless.

## 2. Thresholds are pre-registered

The default thresholds (`sk.Thresholds`, documented in the README with their
literature sources) are fixed **before** any battery runs. Concretely:

- A reference card is always graded at the library defaults of the StressKit
  version named on the card. Custom thresholds are allowed only *in
  addition*, clearly labeled, never as the headline verdict.
- Thresholds change only through a versioned release with a changelog entry
  and an argument on the issue tracker — never in the same PR as a card
  whose grade the change would move.
- If a check's 95% CI straddles its bar, the verdict is reported
  **low-confidence and provisional**, out loud, on the card and on the
  scoreboard. We do not round undecided to decided in either direction.

## 3. Minimum battery

A reference card must have:

- **≥ 2 perturbation axes** (seeds alone is not a battery), chosen from
  seeds / bootstrap / templates / hyperparams, plus a **null control**
  whenever one can be constructed — a card without a specificity check must
  say why not in its notes.
- **Enough runs to settle.** The card ships with its `verdict_trace`; if the
  verdict has not settled by the run budget (like IOI at n = 45), the card
  says so rather than pretending the point estimate is the answer.
- **A stated usage mode.** The battery tests a specific artifact in a
  specific mode (e.g. the released 4B lens, not the paper's 27B headline
  model; the demo-notebook decoding mode, not the paper's scoring harness).
  The scope note is mandatory and lives on the card.

## 4. Every number recomputes

This is the non-negotiable one, and CI enforces it on every push:

- The card JSON passes `stresskit verify` — the grade, confidence, checks,
  and pooled metrics re-derive from the card's own recorded runs and
  metrics. A card an auditor cannot recompute is not evidence.
- The **runner script** is committed next to the card and must run from a
  clean checkout of the pinned upstream tools (model/version/data pinned in
  the script).
- **Raw outputs** back the card where size permits (per-run records are
  embedded in cards up to the size cap; oracle batteries publish raw
  response files).
- `SCOREBOARD.md` is generated from the cards and diffed in CI — the
  summary table can never drift from the artifacts.

## 5. Null controls are disclosed, with their weaknesses

The specificity check is the sharpest knife in the battery, so its null gets
the most scrutiny:

- Each card documents **how the null was constructed** and in which
  direction it is conservative or strict (e.g. the IOI random-names null is
  conservative because name-mover heads legitimately process names).
- Where the null choice is debatable, the card notes the stricter
  alternative rather than silently picking the favorable one.

## 6. Courtesy to upstream authors

Before a card that grades a named finding or released instrument **C or D**
is publicized (blog post, thread, announcement — beyond simply merging to
this repository):

1. Open an issue on the upstream repository (or email the authors) linking
   the card, the runner, and this protocol.
2. Give a **14-day response window** before publicizing. Fixable
   misunderstandings — a mis-pinned version, a misused API, a battery
   testing the wrong usage mode — get fixed first, and have been the
   authors' most common legitimate objection to third-party evaluations.
3. Link the authors' response (or note the absence of one) from the card's
   notes. Their objections travel with the card.

Merging a card to the repo does not require the window; *promoting* it does.

## 7. Disputes and corrections

- Anyone may dispute a card by opening an issue with either (a) a
  `stresskit verify` failure, (b) a defensible re-analysis (different but
  justifiable battery/null/pinning) that flips the verdict, or (c) evidence
  the runner misuses the upstream tool.
- Disputes are settled by **re-running, not arguing**: the re-analysis gets
  its own card in the same PR, and if it survives review, the headline card
  is regenerated. Superseded cards stay in git history; the card notes say
  what changed and why.
- We hold our own work to the standard we apply to others: a StressKit bug
  that moved any published grade gets a changelog entry naming the affected
  cards, and the cards are regenerated (see the 0.3.0 statistical-hardening
  regeneration).

## 8. Submission checklist

A reference-card PR contains:

- [ ] the card JSON (passing `stresskit verify`) and its markdown render
- [ ] the runner script, with upstream repo/model/data versions pinned
- [ ] the `verdict_trace` artifacts
- [ ] a null control, or a note explaining why none is constructible
- [ ] a scope note stating exactly which artifact, in which usage mode
- [ ] `SCOREBOARD.md` regenerated (`stresskit scoreboard references -o SCOREBOARD.md`)
- [ ] for C/D grades on named work: the upstream courtesy issue, opened or
      planned, per §6

CI re-checks the mechanical items forever; reviewers check the rest.
