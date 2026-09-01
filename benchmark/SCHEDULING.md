# Outcome-blind audit scheduling

After the eligible registry freezes, jobs are ordered without StressKit outcomes.
At each round, choose an uncovered method-family stratum first, then the lowest
declared compute tier, then immutable claim ID. Continue round-robin across:

- CoT and trajectories;
- probes and monitoring;
- steering and control;
- lenses and model diffing;
- intervention prediction;
- circuits and sparse autoencoders.

Pilot prompts and seeds must be disjoint from final and replication manifests.
Failed, crashed, timed-out, and missing slots remain in analysis. Claims are
compared or ranked only when task, metric, evaluation set, and resource budget
are identical.

Current registry is not eligible for scheduling; `RELEASE_GATES_V1.md` lists
blocking pre-freeze work.
