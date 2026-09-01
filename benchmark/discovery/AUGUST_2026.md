# August 2026 candidate frame

A code-availability census of mechanistic-interpretability papers submitted to
arXiv between 2026-08-01 and 2026-08-31. This is provenance evidence for
discovery pass 3 of the StressKit benchmark registry. It records what exists,
not whether any method works: no paper here has been run, scored, or graded.

Machine-readable frame: `august-2026-frame.json`. Outcome-blind pass 3b
addendum: `august-2026-pass3b.json`.

## How the frame was built

1. **Query.** The arXiv API was searched over `all:` for fifteen terms inside
   the submission window: `mechanistic interpretability`, `sparse autoencoder`,
   `activation patching`, `activation steering`, `steering vector`, `circuit
   discovery`, `logit lens`, `linear probe`, `attribution graph`, `transcoder`,
   `crosscoder`, `causal tracing`, `refusal direction`, `model diffing`,
   `superposition`. 281 unique papers.
2. **Category filter.** Restricted to `cs.LG cs.CL cs.AI cs.CV cs.NE stat.ML
   cs.CY cs.CR cs.SE cs.IR cs.MA`, removing physics and biology hits that share
   the words `superposition` and `linear probe`. 197 remain.
3. **Signal filter.** Title or abstract must carry an interpretability term.
   158 remain.
4. **Stratification.** Tier A (108) matched at least one narrow
   mechanistic-interpretability term. Tier B (50) matched only `linear probe`
   or `superposition`, which are used across many fields with no interpretive
   intent; Tier B is retained as a separate stratum and is not part of the
   original code census.
5. **Code detection.** Each Tier-A paper's arXiv HTML rendering was scanned for
   repository URLs. Every URL found was checked against the GitHub API for
   existence, license, creation date and HEAD commit. Each link was then
   classified from the sentence around it as an **authored release** (the
   paper's own artifact) or a **cited dependency** (a library the paper uses).

## Pass 3b addendum

Pass 3b queried six terms omitted from pass 3: `chain of thought`, `reasoning
trace`, `persona`, `introspection`, `evaluation awareness`, and `model organism`.
After the same date and category filters, these queries produced 363 unique
rows: 10 already in Tier A, 3 promoted from Tier B, and 350 new broad candidates.
Every new row remains subject to manual interpretability-scope review.

All 47 Tier-B rows remaining after promotion received the same
repository-context and license audit. Across omitted-term and retained-Tier-B
ledgers, 18 rows had a licensed authored repository and 18 had an authored
repository without a resolved license; 350 exposed dependency links only, 10
exposed Hugging Face references without authored source, and 4 require fetch
review. These are code dispositions, not claim eligibility or outcomes. Machine
ledgers store each arXiv ID, inclusion/exclusion disposition, link-set digests,
and public repository license result.

## Code availability, Tier A (n = 108)

| Status | Papers | Share |
|---|---:|---:|
| Public repository released by the authors | 33 | 31% |
| Links only to dependency repositories | 14 | 13% |
| No repository, HuggingFace model/dataset references only | 10 | 9% |
| Code promised, not yet released | 4 | 4% |
| No public code found | 47 | 44% |

Of the 33 authored releases, **18 carry an SPDX license** and 15 have no license
file. An unlicensed public repository does not satisfy the registry's licensing
gate (`REGISTRY_PROTOCOL.md`, inclusion rules), so the pool that can enter the
frozen registry without an author grant is 18, not 33.

Two repositories referenced in papers returned 404 at discovery time.

## Known limitations

- Papers with no arXiv HTML rendering are scanned from the abstract page only;
  three Tier-A papers fell into this case and may hold links visible only in the
  PDF. Code linked exclusively from an external project page or a figure is not
  detected.
- `head_commit_at_discovery` is the default branch tip on the discovery date,
  not a commit the authors designated. Registry pinning re-resolves it against a
  clean clone through `pin_upstream.py`.
- Repository creation dates and star counts inform the authored-versus-dependency
  split. A long-lived authored repository with many stars can be misfiled as a
  dependency; the sentence-context rule is the primary signal and the heuristic
  only breaks ties.
- The census counts release, not runnability. Whether an entrypoint executes is
  decided later by the pre-freeze smoke step, not here.

## Adjacent work on analytic robustness

Six Tier-A papers argue, independently of each other, that interpretability
results move under defensible analytic variation — the question StressKit's
battery is built to measure:

| arXiv | Title | Code |
|---|---|---|
| 2608.13754 | Explanation Multiplicity: Circuit-Level Interpretability Evidence Does Not Survive Defensible Analytic Variation | none found |
| 2608.13337 | Where You Measure Decides What You Measure: Position Selection in Ablation-Based SAE Evaluation | `vcnoel/sae-artifact`, no license |
| 2608.11197 | Beyond a Bag of Features: Set-Level Instability in Sparse Autoencoders | promised |
| 2608.08159 | When Is a Steerable Concept Representation Real? Measurement Confounds in a Cross-Family Audit | promised |
| 2608.24335 | SteerCheck: Attribution Specificity and Alignment Leakage in Activation-Steering Audits | none found |
| 2608.22985 | What Does Activation Steering Control? Attribution Across Answer Encodings and Output-Sensitivity | none found |

Five of the six ship no runnable artifact. That is a fact about the frame, not a
judgement about the papers.

## What happens next

Entering the registry requires more than a public repository: a pinned commit, a
compatible license, an extractable claim with a locator, a deterministic
raw-output-to-finding map, a preregistered perturbation and null, and a passing
smoke reproduction. The 18 licensed releases are candidates for those steps, and
each one that fails a gate is recorded with its reason rather than dropped.
