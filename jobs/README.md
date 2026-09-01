# Nibi jobs

These scripts follow `Nibi_Coding_Agent_Guide.md`: account `def-zhijing`, no
partition, no login-node compute, capped arrays, and durable results under
project storage.

Calibration jobs are historical protocol scripts; local primary and replication
calibration artifacts are already frozen. Source-fetch and verification jobs
may run before the claim registry freezes because they cannot inspect benchmark
outcomes. Confirmatory benchmark jobs remain forbidden until preregistration.

`upstream-source-fetch-array.slurm` independently fetches 25 pinned source trees (14 from the 2026-08-24 pass, 11 added 2026-08-28)
on Nibi and reruns commit/tree/license/entrypoint/static-syntax checks. It is
explicitly not dependency-install or reproduction evidence. `nibi-verify.slurm`
runs StressKit tests against that independently fetched source set.

Nibi's `def-zhijing` project reached its 500,000-file group quota during the
2026-08-24 run despite ample byte capacity. Regenerable upstream checkouts,
logs, intermediate results, and the Nibi execution mirror of StressKit therefore
use `/scratch/arths`; final JSON and hashes must be pulled into the local primary
repository promptly because scratch is not snapshotted. The partially uploaded
project copy is not used while the group inode quota remains full.

Before submission on Nibi:

```bash
mkdir -p /project/def-zhijing/arths/results/stresskit/calibration/logs
cd /project/def-zhijing/arths/repos/Stresskit
module purge
module load python/3.11.5
virtualenv --no-download .venv
source .venv/bin/activate
pip install --no-index -e .
```

If an Alliance wheel is unavailable, resolve that in a scheduled CPU build job
rather than compiling on the login node.

Validate without queueing:

```bash
sbatch --test-only jobs/calibration-cpu-array.slurm
sbatch --test-only jobs/calibration-bootstrap-array.slurm
```

Submit once each:

```bash
sbatch --parsable jobs/calibration-cpu-array.slurm
sbatch --parsable jobs/calibration-bootstrap-array.slurm
```

Each 20-element array contributes 100 disjoint trial indices per cell, totaling
2,000 trials. Concurrency is capped at four. Every shard writes JSON atomically,
records source/environment provenance, and receives a SHA-256 sidecar.

After every shard succeeds and sidecar hashes verify, merge without discarding
trial-range provenance:

```bash
python -m stresskit.calibration_merge \
  /project/def-zhijing/arths/results/stresskit/calibration/structural-core/shard-*.json \
  > /project/def-zhijing/arths/results/stresskit/calibration/structural-core/merged.json
```
