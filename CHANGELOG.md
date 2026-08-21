# Changelog

## 0.3.0

- Distribution renamed to `stress-kit` for PyPI (imports as `import stresskit`,
  CLI stays `stresskit`).

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
