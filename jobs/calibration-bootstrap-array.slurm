#!/bin/bash
#SBATCH --job-name=stresskit-boot
#SBATCH --account=def-zhijing
#SBATCH --time=04:00:00
#SBATCH --array=0-19%4
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/project/def-zhijing/arths/results/stresskit/calibration/logs/%x-%A_%a.out

set -euo pipefail

module purge
module load python/3.11.5

REPO_ROOT=/project/def-zhijing/arths/repos/Stresskit
RESULT_ROOT=/project/def-zhijing/arths/results/stresskit/calibration/structural-bootstrap
TRIALS_PER_SHARD=100
TRIAL_START=$((SLURM_ARRAY_TASK_ID * TRIALS_PER_SHARD))
OUTPUT_PATH="$RESULT_ROOT/shard-${SLURM_ARRAY_TASK_ID}.json"

cd "$REPO_ROOT"
test -r src/stresskit/calibration.py
test -r .venv/bin/activate
mkdir -p "$RESULT_ROOT"
source .venv/bin/activate

python -u -m stresskit.calibration \
  --runs 5,10,20,50 \
  --trials "$TRIALS_PER_SHARD" \
  --trial-start "$TRIAL_START" \
  --bootstrap-replicates 1000 \
  --methods bootstrap_percentile,bootstrap_bca \
  --confidence-level 0.95 \
  --threshold 0.8 \
  --seed 20260824 \
  > "$OUTPUT_PATH.tmp"

mv "$OUTPUT_PATH.tmp" "$OUTPUT_PATH"
sha256sum "$OUTPUT_PATH" > "$OUTPUT_PATH.sha256"
