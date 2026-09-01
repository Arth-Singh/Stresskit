#!/bin/bash
#SBATCH --job-name=sk-src-fetch
#SBATCH --account=def-zhijing
#SBATCH --time=01:00:00
#SBATCH --array=0-35%4
#SBATCH --cpus-per-task=1
#SBATCH --mem=4000M
#SBATCH --output=/scratch/arths/runs/stresskit/source-audit/logs/%x-%A_%a.out

set -euo pipefail

module purge
module load python/3.11.5

REPO_ROOT=/scratch/arths/staging/Stresskit
SOURCE_ROOT=/scratch/arths/staging/stresskit-upstreams-20260824
RESULT_ROOT=/scratch/arths/runs/stresskit/source-audit
MANIFEST="$REPO_ROOT/benchmark/upstream_sources.json"

readarray -t SOURCE_FIELDS < <(
  python - "$MANIFEST" "$SLURM_ARRAY_TASK_ID" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1]))
name = sorted(manifest["upstreams"])[int(sys.argv[2])]
row = manifest["upstreams"][name]
print(name)
print(row["repository"])
print(row["commit"])
PY
)

UPSTREAM=${SOURCE_FIELDS[0]}
REPOSITORY=${SOURCE_FIELDS[1]}
COMMIT=${SOURCE_FIELDS[2]}
TARGET="$SOURCE_ROOT/$UPSTREAM"
STAGE="$SOURCE_ROOT/.stage-$UPSTREAM-$SLURM_JOB_ID"
RESULT="$RESULT_ROOT/$UPSTREAM.json"

mkdir -p "$SOURCE_ROOT" "$RESULT_ROOT"
cd "$SOURCE_ROOT"

if [[ -e "$TARGET" ]]; then
  test -d "$TARGET/.git"
  test "$(git -C "$TARGET" rev-parse HEAD)" = "$COMMIT"
  test -z "$(git -C "$TARGET" status --porcelain)"
else
  test ! -e "$STAGE"
  git clone --filter=blob:none --no-checkout "$REPOSITORY" "$STAGE"
  if ! git -C "$STAGE" cat-file -e "$COMMIT^{commit}"; then
    git -C "$STAGE" fetch --depth=1 origin "$COMMIT"
  fi
  git -C "$STAGE" checkout --detach "$COMMIT"
  test "$(git -C "$STAGE" rev-parse HEAD)" = "$COMMIT"
  mv "$STAGE" "$TARGET"
fi

python "$REPO_ROOT/benchmark/audit_upstreams.py" \
  --clone-root "$SOURCE_ROOT" \
  --manifest "$MANIFEST" \
  --upstream "$UPSTREAM" \
  > "$RESULT.tmp-$SLURM_JOB_ID"
mv "$RESULT.tmp-$SLURM_JOB_ID" "$RESULT"
sha256sum "$RESULT" > "$RESULT.sha256"

echo "upstream=$UPSTREAM"
echo "commit=$COMMIT"
echo "result=$RESULT"
