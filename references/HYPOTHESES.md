# What StressKit claims, and how to falsify it

StressKit is built on one central hypothesis:

> **Mechanistic-interpretability findings are typically reported at an
> operating point — analytic choices, sample sizes, unexamined nulls —
> where their central claims are statistically undecided; a standardized
> stability battery makes this measurable, comparable across papers, and
> fixable.**

That decomposes into five falsifiable sub-hypotheses. Each is stated with
its refutation condition, the evidence the reference batteries currently
provide, and the experiment that would strengthen or break it.

## H1 — Instability: point estimates of interp findings mislead

*Claim.* Under defensible variation (seeds, data resampling, prompt
templates, method hyperparameters), the structural content of a finding
varies enough that a single reported run overstates what is known.

*Refuted if* findings produced by standard methods routinely show
pairwise Jaccard CIs that clear the field's own 0.8 bar at ordinary
sample sizes.

*Evidence so far.* IOI/GPT-2 (the field's most-studied circuit):
J = 0.829, CI [0.781, 0.869] — **still undecided after 45 runs**.
J-lens hit-sets: J = 0.45. Consistent with the multiverse literature
(arXiv:2608.13754, 2510.00845).

*Next experiments.* The same battery across model scale (GPT-2 medium /
large): does stability improve, worsen, or stay undecided with scale?

## H2 — Non-specificity: stable ≠ real

*Claim.* Popular attribution methods return equally-stable "findings"
when the claimed effect does not exist. Stability alone therefore cannot
authenticate a circuit; a null control can.

*Refuted if* null-control batteries (random targets, deranged labels)
show substantially degraded stability (ratio ≥ 1.5×) for these methods.

*Evidence so far.* The sharpest result in the reference set.
Greater-Than: null-task circuits at J = 0.779 vs real 0.892 (**1.15×**).
IOI: 1.38–1.54×, sitting on the bar and flipping with n. J-lens: the
derangement null is *more* stable than the real hit-set (**0.78×**) —
hit metrics reward token-frequency artifacts.

*Next experiments.* Same task, multiple discovery methods (plain
attribution patching vs integrated-gradients attribution): is
non-specificity a property of the method family or of one estimator?

## H3 — Instrument unreliability: 2026's readers are prompt-dominated

*Claim.* Natural-language activation readers (Activation Oracles,
lens readouts) vary more with the analyst's phrasing and analytic knobs
than with decoding noise or activation capture — the same variance
hierarchy autointerp showed (arXiv:2607.19386).

*Refuted if* consistency decompositions show phrasing agreement
comparable to repeat/capture agreement, or lens readouts robust to
masking/band/position choices.

*Evidence so far.* Best AO mixture: agreement 0.94 across decoding
repeats, 0.93 across captures, **0.31 across phrasings**; ≥89%
fabrication on nulls even when abstention is invited. J-lens: the
masked/raw choice alone moves pass@5 ~2×; evaluation distribution owns
53% of score variance.

*Next experiments.* Second oracle base-model family (Gemma-2-9B, using
the upstream checkpoints): does the phrasing-dominance pattern replicate
across families?

## H4 — Underpowered verdicts: stability judgments are themselves unstable at typical n

*Claim.* At the run counts papers actually use (5–10), stability
verdicts are coin flips; only interval-aware grading distinguishes
"stable" from "undecided".

*Refuted if* grades and check outcomes at n ≈ 6 match those at n ≈ 20
across the reference batteries.

*Evidence so far.* IOI's specificity check **fails at n = 6 (1.38×) and
passes at n = 20 (1.54×)**; its structural grade would read a clean A
without the CI. jlens at n = 6 has two of five checks undecided.

*Next experiments.* The verdict-stability curve: run the same battery at
n = 4, 6, 10, 14, 20 and plot grade, check outcomes, and confidence
against n. The n at which the verdict stabilizes is itself a reportable
quantity no current paper provides.

## H5 — Localizability: fragility has an address

*Claim.* Variance decomposition attributes instability to specific
analytic choices (hyperparameters, templates) rather than diffuse noise
(seeds, resampling) — so the battery tells a researcher what to fix or
report, not just that something is wrong.

*Refuted if* seed/bootstrap variance dominates across batteries.

*Evidence so far.* Every battery agrees: IOI hyperparams 57% + templates
36% vs seeds 4%; Greater-Than hyperparams 78%; J-lens distribution 53% +
knobs 42%. Method configuration, not randomness, is the finding.

*Next experiments.* Crossed-grid batteries (interaction terms) on one
cheap task, to test whether OAT attribution understates interactions.

---

Scope limits, stated plainly: the reference batteries cover two circuit
tasks on one small model family, one oracle family, and one lens release.
The hypotheses are about *method families and reporting practice*, not
about any single paper being wrong — and H1's IOI result is explicitly
"undecided", not "refuted".
