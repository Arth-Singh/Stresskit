#!/bin/bash
#SBATCH --job-name=stresskit-verify
#SBATCH --account=def-zhijing
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4000M
#SBATCH --output=/scratch/arths/runs/stresskit/verification/logs/%x-%j.out

set -euo pipefail

module purge
module load python/3.11.5

REPO_ROOT=/scratch/arths/staging/Stresskit
SOURCE_ROOT=/scratch/arths/staging/stresskit-upstreams-20260824
RESULT_ROOT=/scratch/arths/runs/stresskit/verification
VENV="$SLURM_TMPDIR/stresskit-venv"

cd "$REPO_ROOT"
mkdir -p "$RESULT_ROOT"
python -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install --no-index --upgrade pip
python -m pip install --no-index -e '.[dev]'

python -u -m pytest -q | tee "$RESULT_ROOT/pytest-$SLURM_JOB_ID.txt"
python -u benchmark/audit_upstreams.py \
  --clone-root "$SOURCE_ROOT" \
  > "$RESULT_ROOT/upstream-static-audit-$SLURM_JOB_ID.json"

python - "$RESULT_ROOT/environment-$SLURM_JOB_ID.json" <<'PY'
import json
import os
import platform
import sys

payload = {
    "artifact_type": "stresskit_nibi_verification_environment",
    "slurm_job_id": os.environ["SLURM_JOB_ID"],
    "host": platform.node(),
    "python": sys.version,
    "platform": platform.platform(),
}
with open(sys.argv[1] + ".tmp", "w") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(sys.argv[1] + ".tmp", sys.argv[1])
PY
