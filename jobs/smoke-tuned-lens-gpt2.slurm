#!/bin/bash
#SBATCH --job-name=sk-smoke-tlens
#SBATCH --account=def-zhijing
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M
#SBATCH --output=/scratch/arths/runs/stresskit/execution-smoke/logs/%x-%j.out

set -euo pipefail

REPO_ROOT=/scratch/arths/staging/Stresskit
ENV_RECORD=/scratch/arths/runs/stresskit/env-build/tuned_lens.json
RESULT_ROOT=/scratch/arths/runs/stresskit/execution-smoke
RESULT="$RESULT_ROOT/tuned_lens_gpt2-$SLURM_JOB_ID.json"
HF_HOME=/scratch/arths/caches/huggingface

module purge
module load python/3.11.5
module load arrow/25.0.0
VENV=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["environment_path"])' "$ENV_RECORD")
test -r "$VENV/bin/activate"
source "$VENV/bin/activate"

mkdir -p "$RESULT_ROOT" "$HF_HOME"
export HF_HOME
export TOKENIZERS_PARALLELISM=false
python -u "$REPO_ROOT/benchmark/smokes/tuned_lens_gpt2.py" \
  > "$RESULT.tmp"
mv "$RESULT.tmp" "$RESULT"
python -m json.tool "$RESULT" >/dev/null
sha256sum "$RESULT" > "$RESULT.sha256"
cat "$RESULT"
