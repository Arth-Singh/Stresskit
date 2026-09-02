# Changelog

## 1.0.0 — unreleased; external gates pending

- Added reference batteries run on 2026-09-02 (8xH200): the refusal direction
  (arXiv:2406.11717) on Qwen3.5-4B/9B, Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct
  and gemma-4-E4B/12B-it with a blind-selection null and non-internals prompt
  baselines (`references/run_refusal_direction_card.py`,
  `references/run_refusal_baselines.py`), plus three August-2026 papers audited
  within the month of release: Mechanistic Tomography OMP recovery
  (arXiv:2608.19338), truth/impossibility probes on gemma-3-4b-it
  (arXiv:2608.12852), and homonym reconvergence profiles (arXiv:2608.01816) on
  gpt2, Llama-3.2-3B and Qwen2.5-7B.
- Refusal judge: the upstream substring list is applied after folding
  typographic apostrophes to ASCII (Llama-3.1 writes "I can’t"), and compliance
  additionally requires coherence under the unablated model (<= 5 nats/token,
  no 3-gram repeated three times). Both amendments were made after inspecting a
  discarded first pass that scored gemma-4 gibberish as jailbreaks; the cards
  say so.
- Known limitations surfaced by these cards, left for the next release: (1) a
  direction-valued finding has no direction-native structural check, so the
  refusal cards grade the top-32 logit-lens readout tokens, a proxy that keeps
  only ~0.6-0.7 Jaccard between directions with cosine > 0.98 (cosines are
  reported in the notes instead); (2) the specificity check sizes its null runs
  against the null base run, which can leave a handful of comparable null runs
  and no interval (mechtomo card); (3) `verdict_trace` re-draws bootstrap
  intervals at its own seed and pools null runs without the size guard, so a
  trace can report a different confidence than its card.
- Added direction-valued findings, closing limitation (1) above.
- `references/battery_shards.py`: shard one battery's finder calls across
  several processes (`STRESSKIT_SHARD=i/n`, one per GPU) through a shared
  run cache keyed by (data digest, seed, config); a final unsharded process
  assembles the card. Used by the expander-SAE, CoAx, AMS and FolkMotif
  runners.
- Added a second July-2026 card: Certified Interventional Fidelity on GPT-2
  IOI circuits (arXiv:2607.08349, `references/run_cif_ioi_card.py`). The base
  run reproduces all 30 shipped certification rows to the integer; the
  certified fidelity level of the name-mover circuit then ranges from none
  to F0 = 0.95 across twelve IOI templates, with the upstream template the
  most favourable, and the abstract's 10-30x cost reduction is the F0 = 0.9
  reading (7x at F0 = 0.8).
- Added two August-2026 cards. Activation Model Scanner Tier-1 safety scan
  (arXiv:2608.05578, `references/run_ams_scanner_card.py`, 14 models of
  Table I): every released σ reproduces through the released extractor, and
  the extractor reads pad-token activations for the ten right-padded
  tokenizers; with batch size 1 or left padding every model scores σ 4.5-6.7,
  nothing is flagged and the σ-compliance correlation vanishes. FolkMotif
  (arXiv:2608.02486, `references/run_folkmotif_card.py`, Llama-3.1-8B-Instruct):
  the released English-prompt run reproduces exactly, the paper's 0.248 output
  accuracy is its rescored run and its 32/206/6/26 decomposition row is the
  native-prompt run; the qualitative claim survives every axis while the
  Preserved count moves from 5 to 83 with the aggregation rule.
- Added two more July-2026 cards. Expander SAEs on Qwen2.5-3B
  (arXiv:2607.01799, `references/run_expander_sae_card.py`): the released
  CE-recovered numbers reproduce, and the expander/dense ratio sits in a
  0.80-0.90 band over 30 training seeds and 30 document resamples (grade A,
  high confidence), falling to 0.66 at k = 32 and 0.77 under a mean-ablation
  denominator. CoAx backup-head recovery on GPT-2 small (arXiv:2607.01940,
  `references/run_coax_backup_card.py`): the released 0.94 AUC reproduces,
  label-free primaries collapse it to 0.28-0.38 while AtP* stays at 0.83, and
  the third-name null recovers the same heads at 0.93-0.97, so the backup
  structure is task-general rather than IOI-specific. The runner's first pass
  had a vacuous seeds axis (the finder ignored its seed), which the harness
  flagged; the seeds axis now redraws the prompts.
- Added an August-2026 card for The Communication Map of a Transformer
  (arXiv:2608.22007, `references/run_communication_map_card.py`, the seven
  released censuses): all 21 Table 2 shares reproduce exactly; the abstract's
  70-89% holds pooled per model (69.7-89.0%) while 4 of 21 per-channel entries
  fall outside (60.7-90.5%), its lower bound is a rounding edge under
  resampling, and a |z| >= 3 threshold or uncentred weights move both the
  range and the exception set.
- Added two more cards from the July/August 2026 set. Dissociating sycophancy
  representations (arXiv:2607.07003, `references/run_sycophancy_probe_card.py`,
  Llama-3.1-8B-Instruct half): Tables 1-2 reproduce within 0.02-0.10 from
  regenerated activations; the released extractor orders layers
  lexicographically, so the paper's "final layer" is decoder layer 9; the
  "distinct" claim holds in 47 of 48 runs but flips to "shared" at the
  best-in-domain layer, and which layers carry the transfer drop is unstable.
  Diff Mining, judge-free token-set battery (arXiv:2608.26462,
  `references/run_diff_mining_card.py`, gemma-3-1b-it x cake_bake): the
  top-100 token set is stable across seeds, resamples and corpora and 65%
  finetune-domain vocabulary under a pre-registered rule (grade A), falling
  to 0.48 under logit-lens extraction and 0.26 for the 1:1-mix LoRA; a
  scrambled adapter returns garbage.
- `RESULTS.md`: a maintained ledger of every audited paper (grade, confidence,
  reproduction status, one-line result, card link) with detailed entries for
  the July/August 2026 target set and a queue of what is running.
- Paper leaderboard. `references/papers.json` registers every audited paper
  with the cards graded for it (newest audit first); `stresskit scoreboard`
  and `stresskit site` read it (or `--papers`) and render one row per paper
  with one grade per card, the checks passed, the reproduction status and
  the one-line result, above the per-finding table. A graded card that no
  registry entry claims, a registered path that is not a card, or a card
  listed twice is an error, so the leaderboard cannot silently drop a card.
- README front door rewritten around the results: the paper leaderboard (17
  papers, 43 cards) and the method notes now open the file, and the v1
  autonomous-protocol status moved into a "What has not been done" section
  that states it has produced no empirical result. The stale five-row
  reference table and roadmap were replaced. `RESULTS.md` gained a "Sanity
  checks that changed a headline" section listing the checks (raw-completion
  reading, vacuous passes, the CoAx seed axis, AMS padding, the sycophancy
  layer order, FolkMotif's two runs, Communication Map aggregation, the
  homonym control, the CoAx task-general null) that moved a number during
  these audits.
- Diff Mining and sycophancy cards recomputed at 60 and 40 runs per axis
  (the 20-per-axis runs were reused from the shard cache). Both grades are
  unchanged (A, B) and both are now high confidence: Diff Mining's structural
  CI moved to [0.877, 0.952] and the sycophancy score CI to [0.175, 0.232].
  Diff Mining's pooled Jaccard (0.92) diverges from its axis-balanced value
  (0.81) because the two large axes sit at 0.97 while the hyperparameter axis
  is 0.43; the harness note and the README say so.
- `stresskit verify` over a directory no longer aborts when one artifact cannot
  be parsed — an unsupported `schema_version` (a card written by a newer
  StressKit) is now reported as a failure for that file and the remaining cards
  are still verified. Fail-closed is preserved: the file is counted as a
  failure, never skipped, and the exit code is still non-zero.
- Re-froze `artifacts/calibration/` after the direction-valued change. The frozen
  manifest pins a SHA-256 over the bytes of `battery.py`, `calibration.py` and
  `metrics.py`, so any edit to them invalidates the attestation by design. All
  four studies (S1-S5 and S6-S9, primary and disjoint-seed replication, 2000
  trials each) were re-run on the interpreter and platform their provenance
  records (CPython 3.12.12, macOS-15.5-arm64) and reproduce **byte-identically**;
  only the source digest moved. The manifest now attests the new bytes.
  `stresskit.direction(vector, ...)` builds a `Finding` carrying a
  unit-normalized `vector` instead of a component set (mutually exclusive with
  `components`, empty/non-finite/zero vectors rejected on construction), and
  `metrics.cosine_similarity` / `abs_cosine` / `pairwise_abs_cosine` /
  `mean_pairwise_abs_cosine` grade it. |cosine|, not signed cosine: a
  difference-in-means direction's sign is a convention of which class the
  extraction labelled positive, and probe weights, PCA components and singular
  vectors carry the same gauge, so a convention flip must not read as a
  structural failure. The random null is the exact
  `metrics.expected_random_abs_cosine(d)` = 2Γ(d/2)/((d−1)√π·Γ((d−1)/2)), with
  the Monte-Carlo `baselines.empirical_random_abs_cosine` (and
  `baselines.random_directions`) as its cross-check — the reverse of the
  Jaccard case, where the closed form is the approximation.
- A battery of direction findings produces `structural_stability` (mean
  pairwise |cos| with a bootstrap CI), `beats_random` and `specificity` from
  `stress`, `from_findings`, `from_jsonl` and `verdict_trace` exactly as a
  set-valued battery does, with per-axis breakdown, axis-balanced value,
  `meta["universe"]` exclusion and run caching. A battery that mixes
  set-valued and direction-valued findings, or directions of different
  dimension, now fails fast instead of averaging incommensurable numbers.
- Added `Thresholds.cosine = 0.8` as a separate registered bar rather than
  reusing `Thresholds.jaccard`. Separate because the two grade different
  quantities and must be revisable independently; 0.8 because |cos| is the
  direct analogue of Jaccard for a direction — the fraction of one unit
  direction recovered by projecting it onto the other, as Jaccard is the
  fraction of a set shared — so the registered structural bar transfers
  unchanged instead of inventing a second number, and at |cos| = 0.8 the
  residual disagreement ‖a − b‖ = √(2 − 2·0.8) ≈ 0.63 is already 63% of the
  direction's own length. The value was fixed by that analogy before any
  direction card was graded, not calibrated on one. No set-valued threshold,
  metric, grade, CI or card changed.
- Stability Card schema 0.4: a direction card carries
  `battery.structure_kind = "direction"`, per-run `direction_dim` and
  `direction_sha256`, and a `directions` block holding the pairwise |cos|
  matrix over the graded runs (and over the null-control runs), plus the
  bootstrap parameters the intervals were drawn with. `stresskit verify`
  re-derives the pooled |cos|, its bootstrap CI, the specificity ratio and its
  interval, and the grade from that matrix alone — no raw high-dimensional
  vectors are embedded, and the per-run digests are what tie the matrix to
  directions an auditor holds. Schema 0.1-0.3 cards load, verify and render
  unchanged; cards written for set-valued findings gain no new fields.
- Added `references/cards/refusal_direction_meta_llama_3p1_8b_instruct.directions.{json,md}`
  and its post-hoc runner `references/refusal_direction_card_posthoc.py`, which
  regrades the published Llama-3.1-8B refusal battery from the unit directions
  it already saved, with no new GPU work and no change to the published card.
  Same 21 runs, same claims, scores and null control: the directions agree to
  |cos| 0.883 [0.829, 0.933] (grade A) where the top-32 logit-lens readout
  proxy scored Jaccard 0.392 [0.276, 0.522] (grade B). The proxy's ceiling, not
  the finding, was failing the structural check.
- `artifacts/calibration/manifest.json` pins SHA-256 digests over the bytes of
  `battery.py` and `metrics.py`, so the additive direction code invalidates
  `calibration_source_digest()` and `extended_validation_source_digest()`. The
  frozen calibration results are unchanged and were not regenerated; re-freezing
  them under the new source digest needs a rerun on the interpreter recorded in
  their provenance and is left for the release.
- Added versioned `SourceBundle`, `AgentOpinion`, `ClaimRecord`, `AuditSpec`,
  `ResourcePlan`, `RunAttestation`, `AuditBundle`, and `AuditDecision` artifacts.
- Added nested `stresskit audit discover|compile|freeze|plan|run|verify|publish`
  lifecycle with strict abstention gates.
- Added seven registered deterministic claim profiles, independent-unit
  Hoeffding inference, and release-wide Holm–Bonferroni verification.
- Replaced publishable utility summaries with raw label/prediction
  recomputation, metric direction/bounds, practical margins, held-out splits,
  nondecomposable-metric policy, and mandatory non-internals baselines.
- Added signed resource plans, outbound-only worker polling, complete terminal
  run slots, independent reruns, SHA-256 object closures, and executor-isolation
  enforcement.
- Added byte-exact source/model provenance, license records, repository and
  build code maps, plus separate Ed25519 control/executor trust domains.
- Added optional FastAPI/PostgreSQL/S3 control plane without Redis or Celery.
- Replaced grade-sorted v1 publication with a verified claim-level evidence
  matrix that retains excluded and abstained rows and computes no paper verdict.
- Added 2,000-trial primary/fresh-seed profile calibration, a 300-case planted
  compiler evaluation, and adversarial regressions for constant claims, stable
  nonsense, forged utility, fake IID manifests, dependent/missing slots,
  signature tampering, prompt injection, and unverified publication rows.
- Added outcome-blind August pass 3b, flagship gradient-projection candidate
  protocol, release gates, and protocol-only preprint draft.
- Added deterministic local-file SourceBundle intake and a typed 68-row
  pre-freeze qualification ledger. Manual eligibility labels, incomplete gate
  evidence, mismatched artifact digests, and partial release multiplicity
  families cannot freeze a release registry.
- Added optional `stresskit audit opinion` OpenRouter preparation with env-only
  credentials, pinned model/provider routes, strict local output validation,
  quote-derived byte anchors, and complete secret-free CAS provenance.
- Preserved v0.1–v0.3 Stability Cards, diagnostic A–D grades, and frozen
  calibration artifacts without reinterpretation.

## 0.3.0

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
