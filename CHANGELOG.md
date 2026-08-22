# Changelog

## Unreleased

- **`stresskit demo`** — the 30-second first touch: one toy discovery
  method graded on a real effect (A) and on pure noise, where it still
  returns confident-looking findings (C) — same output format, only the
  battery separates them. `--html DIR` writes both cards as pages.
- **Verdict-trace charts** — `stresskit trace card.trace.json -o out.svg`
  renders the grade distribution vs run count as a self-contained SVG
  (`stresskit.tracechart`), with the coin-flip run counts and `settled_n`
  annotated. Grade colors are a CVD-validated 4-step scale (adjacent-pair
  ΔE ≥ 15.3 under CVD simulation); shares are always also written as text.
- **`stresskit site`** — static results site from a directory of cards:
  index with headline stats, the most dramatic verdict trace as the hero
  figure, the full table, and one page per card with its trace chart
  embedded and a per-page audit command. `.github/workflows/pages.yml`
  deploys it to GitHub Pages on every push to main.
- **`stresskit compare`** — stability regression testing between two cards
  (`compare_cards` in the library): per-check deltas, pass→fail flips,
  grade drops, `--fail-on-regression` exit code for CI gates. Deltas are
  called *decisive* only when both 95% CIs exist and are disjoint;
  checks with differing thresholds are excluded from regression verdicts;
  both cards must pass auditor mode first.
- **GitHub Action** (`action.yml`) — any repo verifies its cards and gates
  on stability regressions with one `uses:` step.
- **HTML cards** — `stresskit render --html` emits a self-contained
  shareable page with per-check CI-vs-bar plots (`stresskit.htmlcard`);
  works for stability cards and oracle reports.
- **JSONL entry point** — `sk.from_jsonl("sweep.jsonl", null_path=...)`
  grades any sweep log directly (`findings_from_jsonl` for the loader;
  field names remappable, `axis` fields drive the per-axis breakdown).
- CITATION.cff.
- **Oracle reports are now auditable** — `verify_oracle_report_dict`
  re-derives an oracle reliability report's checks, grade, confidence,
  pooled metrics (from the per-probe rows), and Wilson CIs (from the
  recorded counts); `verify_artifact_dict` / `classify_artifact_dict`
  dispatch across artifact kinds.
- **Batch auditing** — `stresskit verify` accepts multiple files and
  directories (recursive), skipping non-artifact JSONs that live next to
  cards; CI now runs `stresskit verify references/` on every push, so
  every published verdict is continuously re-derived.
- **`stresskit scoreboard`** — renders every card and report found into
  one deterministic markdown table; the repo's `SCOREBOARD.md` is
  generated from `references/` and diffed in CI (a stale scoreboard fails
  the build).
- **Evidence standard** — `references/PROTOCOL.md` pre-registers what a
  reference card requires (thresholds fixed in advance, minimum battery,
  recomputability, null-control disclosure, upstream-author courtesy
  window, dispute-by-rerunning); `references/TARGETS.md` is the
  prioritized queue of findings to battery-test next.
- Contribution surface: CONTRIBUTING.md, issue templates (card
  submission, card dispute, adapter request), PR template.

## 0.3.0

- **`verdict_trace`** — regrades random subsets of a battery's runs at every
  size and reports the grade distribution per n plus `settled_n`, the run
  count at which the verdict stops being a coin flip. No new runs needed;
  also available as `StressResult.verdict_trace()`.
- **CIs on specificity and beats-random** — specificity now carries a
  two-sample bootstrap 95% CI (`metrics.bootstrap_ci_ratio_pairwise`,
  resampling real and null-control runs independently, self-pair free);
  beats-random carries the real-Jaccard bootstrap CI rescaled by the
  Monte-Carlo null. Every check on the card now has an interval, so a
  specificity margin sitting on the 1.5× bar reads as undecided instead of
  silently passing.
- **Post-hoc mode** — `stresskit.from_findings(findings, axes=, null_findings=)`
  grades runs you already have: same card, same checks, no re-running.
- **`adapters.sae_lens.stability()`** — one-call graded report for SAELens
  SAEs: cross-seed MCC, near-duplicate fraction, and excess over the
  random-decoder noise floor (which grows with n_features/d_model — a raw
  ratio would be regime-dependent).
- **`adapters.eap`** — findings from live EAP-IG graphs, saved
  `Graph.to_json` exports, and a `finder_from_graph_fn` factory for the
  full battery.

- Distribution renamed to `stress-kit` for PyPI (imports as `import stresskit`,
  CLI stays `stresskit`).
- **CI-aware grading** — every check now carries its 95% CI and a `robust`
  flag (does the CI clear the bar, not just the point estimate). A pass whose
  CI straddles the bar is marked borderline and lowers the verdict's
  `confidence` (high/low/unknown); a low-confidence grade is labeled
  provisional. Applies to both `stress` and `stress_oracle`.
- Random null switched to a Monte-Carlo estimate over the observed size
  distribution (`baselines.empirical_random_jaccard`); analytic `k/(2N−k)`
  kept as a reported cross-check. Absolute score variance reported alongside
  the normalized shares.

- **`stresskit verify`** — auditor mode: re-derives every check and the grade
  from a card's own recorded metrics; catches edited or non-conforming cards
  (`verify_card_dict` in the library).
- **Run cache** — `stress(..., cache_dir=, cache_key=)` skips finder calls
  already executed under the same key; null-control runs cache under a
  derived key.
- **Universe-aware structural comparison** — findings may declare
  `meta["universe"]`; Jaccard only pools runs sharing the base run's
  universe (cross-universe Jaccard is undefined, not zero). Enables honest
  cross-dataset template axes.
- **Rank-biased overlap** — `metrics.rbo` / `metrics.pairwise_rbo`
  (Webber et al. 2010) for ranked readouts and feature lists.
- **Oracle battery upgrades** — consistency decomposition (decoding /
  capture / phrasing agreement, a partition of all answer pairs), Wilson
  95% CIs on accuracy and hallucination rate.
- **`adapters.activation_oracles`** — reliability reports straight from
  saved `run_verbalizer` JSON (adamkarvonen/activation_oracles), no GPU.
- **`adapters.jlens`** — ranked-readout findings, `junk_share`,
  workspace-band hit ranks for Jacobian-lens batteries
  (anthropics/jacobian-lens).
- Reference batteries: Greater-Than (GPT-2 small) stability card and the
  Activation-Oracle reliability report (Qwen3-8B, three mixtures).

## 0.2.0

- `stresskit.oracle`: reliability battery for natural-language activation
  readers (consistency, known accuracy, prompt sensitivity, null
  hallucination) and the cross-oracle blind-spot matrix (arXiv:2607.23379).
- `stresskit.judges`: pluggable answer-equivalence judges; semantic claim
  clustering via `claim_equiv=`.
- Specificity check: `null_data=` dead-salmon control.
- Bootstrap 95% CIs on headline stability metrics; size-guarded Jaccard.

## 0.1.0

- `stress()` battery (seeds / bootstrap / templates / hyperparams), graded
  Stability Card with badge output, minimum reporting checklist CLI, SAE
  and TransformerLens adapters.
