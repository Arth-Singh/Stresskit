#!/bin/bash
#SBATCH --job-name=sk-env-build
#SBATCH --account=def-zhijing
#SBATCH --time=02:00:00
#SBATCH --array=0-4%2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M
#SBATCH --output=/scratch/arths/runs/stresskit/env-build/logs/%x-%A_%a.out

set -euo pipefail

REPO_ROOT=/scratch/arths/staging/Stresskit
SOURCE_ROOT=/scratch/arths/staging/stresskit-upstreams-20260824
ENV_ROOT=/scratch/arths/staging/stresskit-envs
RESULT_ROOT=/scratch/arths/runs/stresskit/env-build
MATRIX="$REPO_ROOT/benchmark/install_smoke_matrix.json"

module purge
module load python/3.11.5

readarray -t BUILD_FIELDS < <(
  python - "$MATRIX" "$SLURM_ARRAY_TASK_ID" <<'PY'
import json
import sys

row = json.load(open(sys.argv[1]))["rows"][int(sys.argv[2])]
print(row["upstream"])
print(row["python_module"])
print(row["import_module"])
print("|".join(row["preload_modules"]))
print("|".join(row["noindex_preinstall"]))
print("|".join(row["public_bootstrap"]))
PY
)

UPSTREAM=${BUILD_FIELDS[0]}
PYTHON_MODULE=${BUILD_FIELDS[1]}
IMPORT_MODULE=${BUILD_FIELDS[2]}
IFS='|' read -r -a PRELOAD_MODULES <<< "${BUILD_FIELDS[3]}"
IFS='|' read -r -a NOINDEX_PACKAGES <<< "${BUILD_FIELDS[4]}"
IFS='|' read -r -a PUBLIC_PACKAGES <<< "${BUILD_FIELDS[5]}"
SOURCE="$SOURCE_ROOT/$UPSTREAM"
COMMIT=$(git -C "$SOURCE" rev-parse HEAD)
VENV="$ENV_ROOT/$UPSTREAM-${COMMIT:0:12}-$SLURM_JOB_ID"
RESULT="$RESULT_ROOT/$UPSTREAM.json"
FREEZE="$RESULT_ROOT/$UPSTREAM-freeze.txt"

test "$PYTHON_MODULE" = "python/3.11.5"
for dependency_module in "${PRELOAD_MODULES[@]}"; do
  if [[ -n "$dependency_module" ]]; then
    module load "$dependency_module"
  fi
done

mkdir -p "$ENV_ROOT" "$RESULT_ROOT"
test ! -e "$VENV"
virtualenv --no-download "$VENV"
source "$VENV/bin/activate"

if [[ -n "${NOINDEX_PACKAGES[*]}" ]]; then
  python -m pip install --no-index "${NOINDEX_PACKAGES[@]}"
fi
if [[ -n "${PUBLIC_PACKAGES[*]}" ]]; then
  python -m pip install "${PUBLIC_PACKAGES[@]}"
fi
python -m pip install --no-index -e "$SOURCE"
python -m pip check
python - "$IMPORT_MODULE" <<'PY'
import importlib
import sys

module = importlib.import_module(sys.argv[1])
print(f"imported={module.__name__}")
print(f"module_file={getattr(module, '__file__', None)}")
PY

python -m pip freeze --all > "$FREEZE.tmp-$SLURM_JOB_ID"
mv "$FREEZE.tmp-$SLURM_JOB_ID" "$FREEZE"
FREEZE_SHA256=$(sha256sum "$FREEZE" | cut -d ' ' -f 1)
export UPSTREAM PYTHON_MODULE IMPORT_MODULE COMMIT VENV RESULT FREEZE FREEZE_SHA256
python - <<'PY'
import json
import os
import platform

payload = {
    "artifact_type": "stresskit_dependency_import_smoke",
    "status": "pass",
    "upstream": os.environ["UPSTREAM"],
    "upstream_commit": os.environ["COMMIT"],
    "python_module": os.environ["PYTHON_MODULE"],
    "python": platform.python_version(),
    "import_module": os.environ["IMPORT_MODULE"],
    "environment_path": os.environ["VENV"],
    "freeze_path": os.environ["FREEZE"],
    "freeze_sha256": os.environ["FREEZE_SHA256"],
    "slurm_job_id": os.environ["SLURM_JOB_ID"],
    "not_claim_reproduction": True,
}
target = os.environ["RESULT"]
with open(target + ".tmp", "w") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(target + ".tmp", target)
PY

sha256sum "$RESULT" > "$RESULT.sha256"
cat "$RESULT"
