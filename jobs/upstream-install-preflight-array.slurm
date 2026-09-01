#!/bin/bash
#SBATCH --job-name=sk-install-pre
#SBATCH --account=def-zhijing
#SBATCH --time=00:30:00
#SBATCH --array=0-4%2
#SBATCH --cpus-per-task=2
#SBATCH --mem=8000M
#SBATCH --output=/scratch/arths/runs/stresskit/install-preflight/logs/%x-%A_%a.out

set -euo pipefail

REPO_ROOT=/scratch/arths/staging/Stresskit
SOURCE_ROOT=/scratch/arths/staging/stresskit-upstreams-20260824
RESULT_ROOT=/scratch/arths/runs/stresskit/install-preflight
MATRIX="$REPO_ROOT/benchmark/install_smoke_matrix.json"

module purge
module load python/3.11.5

readarray -t SMOKE_FIELDS < <(
  python3 - "$MATRIX" "$SLURM_ARRAY_TASK_ID" <<'PY'
import json
import sys

row = json.load(open(sys.argv[1]))["rows"][int(sys.argv[2])]
print(row["upstream"])
print(row["python_module"])
print(row["import_module"])
PY
)

UPSTREAM=${SMOKE_FIELDS[0]}
PYTHON_MODULE=${SMOKE_FIELDS[1]}
IMPORT_MODULE=${SMOKE_FIELDS[2]}
SOURCE="$SOURCE_ROOT/$UPSTREAM"
VENV="$SLURM_TMPDIR/venv"
PIP_LOG="$RESULT_ROOT/$UPSTREAM-pip-no-index.log"
IMPORT_LOG="$RESULT_ROOT/$UPSTREAM-import.log"
RESULT="$RESULT_ROOT/$UPSTREAM.json"

test "$PYTHON_MODULE" = "python/3.11.5"
mkdir -p "$RESULT_ROOT"
test -d "$SOURCE/.git"
virtualenv --no-download "$VENV"
source "$VENV/bin/activate"

set +e
python -m pip install --no-index -e "$SOURCE" >"$PIP_LOG" 2>&1
INSTALL_EXIT=$?
IMPORT_EXIT=125
if [[ "$INSTALL_EXIT" -eq 0 ]]; then
  python - "$IMPORT_MODULE" >"$IMPORT_LOG" 2>&1 <<'PY'
import importlib
import sys

module = importlib.import_module(sys.argv[1])
print(f"imported={module.__name__}")
print(f"module_file={getattr(module, '__file__', None)}")
PY
  IMPORT_EXIT=$?
fi
set -e

export UPSTREAM PYTHON_MODULE IMPORT_MODULE INSTALL_EXIT IMPORT_EXIT RESULT
python - <<'PY'
import json
import os
import platform

install_exit = int(os.environ["INSTALL_EXIT"])
import_exit = int(os.environ["IMPORT_EXIT"])
if install_exit != 0:
    status = "noindex_dependency_unavailable"
elif import_exit != 0:
    status = "import_failed"
else:
    status = "pass"
payload = {
    "artifact_type": "stresskit_dependency_import_preflight",
    "upstream": os.environ["UPSTREAM"],
    "python_module": os.environ["PYTHON_MODULE"],
    "python": platform.python_version(),
    "import_module": os.environ["IMPORT_MODULE"],
    "install_mode": "pip_editable_no_index",
    "install_exit": install_exit,
    "import_exit": import_exit,
    "status": status,
    "not_claim_reproduction": True,
    "slurm_job_id": os.environ["SLURM_JOB_ID"],
}
target = os.environ["RESULT"]
with open(target + ".tmp", "w") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(target + ".tmp", target)
PY

sha256sum "$RESULT" > "$RESULT.sha256"
cat "$RESULT"
