# Flagship study: training-gradient projections predict behavioral shift

Status: candidate preregistration, blocked before freeze by unresolved licenses.
No substitute persona or emergent-misalignment artifact is permitted.

## Primary question

For each training example, does its loss-gradient projection onto frozen
persona/misalignment probes at preregistered residual-stream layers predict
held-out post-fine-tuning behavioral shift across independent datasets and
training seeds better than baselines that do not use the proposed internal
signal?

## Frozen unit and split design

Independent unit is a complete fine-tuning seed within model and dataset. Data
examples, checkpoints, layers, and repeated evaluations are nested observations,
not independent replicates. Pilot models, prompts, datasets, and seeds are
disjoint from final and replication cohorts.

Layers resolve deterministically to the nearest residual-stream layer at 25%,
50%, and 75% model depth after licensed model revisions freeze. Probes freeze
before any final fine-tuning result is opened. Each example statistic is the dot
product between its loss gradient at a registered layer and the corresponding
unit-normalized frozen probe direction.

Primary prediction target is held-out change in preregistered external behavior
between base and fine-tuned models. Models train on one dataset and are evaluated
on separate held-out behavior datasets. Generalization includes held-out model
families and fresh training seeds.

## Baselines

All comparisons use matched train/evaluation examples and compute budgets:

- text-only human or frozen-model ratings;
- text-only embeddings;
- per-example loss;
- gradient norm;
- output-only base-model probabilities and embeddings.

Internals-based comparisons may be reported descriptively but cannot replace the
strongest non-internals utility baseline.

## Controls

- random frozen probes matched in dimension and norm;
- label permutation before probe fitting;
- benign fine-tunes matched for tokens, optimizer, and compute;
- held-out models and datasets;
- seed-level independent rerun on the same hardware class;
- negative-control behaviors absent from probe labels.

## Inference

Primary checks use seed or model-dataset clusters as independent units. Metrics,
directions, bounds, practical margins, stopping rules, and power calculations
freeze before final runs. Every primary layer, dataset, model, and baseline
comparison joins the release-wide Holm–Bonferroni family. Secondary layer curves
and example visualizations remain descriptive.

## Licensing gate

Source code, models, training data, probes, persona labels, misalignment labels,
and behavioral evaluations each require explicit compatible licenses. Missing or
incompatible license yields `abstain`. A nearby licensed artifact cannot silently
replace the registered scientific object.

`spec.candidate.json` records unresolved inputs. It cannot become an `AuditSpec`
until each blocker resolves to an immutable digest and license record.
