#!/usr/bin/env bash
# Shard the whole-battery known-truth study across the cores of one machine.
#
#   usage: jobs/battery-calibration-local.sh SEED TRIALS OUTDIR WORKERS
#
# TRIALS is either one integer, applied to run counts 100 40 20 10 6, or a
# comma-separated per-run-count budget N:TRIALS, for example
#   6:2000,10:2000,20:2000,40:2000,100:500
# The caller exports PYTHONPATH (or has stresskit installed) and may set
# PYTHON to pick the interpreter (default python3) and CELLS (quoted,
# space-separated cell names) to restrict a call to a subset of the sixteen
# registered cells.  Run counts of 100 or more are cut into shards of 100
# trials, run counts above 20 into shards of 500, smaller run counts get one
# shard per cell; longest shards run first.  Every shard is written atomically
# as OUTDIR/shard-CELL-N-START.json with its stderr beside it; a shard whose
# file already exists is skipped, so rerunning the same command fills in only
# what failed.  The script exits non-zero after all shards ran if any failed,
# and prints the merge command otherwise.
set -euo pipefail

if [ "$#" -ne 4 ]; then
  echo "usage: $0 SEED TRIALS OUTDIR WORKERS" >&2
  exit 2
fi

SEED=$1
TRIALS_SPEC=$2
OUTDIR=$3
WORKERS=$4
PYTHON=${PYTHON:-python3}
DEFAULT_RUN_COUNTS="100 40 20 10 6"

if ! [[ "$SEED" =~ ^[0-9]+$ ]]; then
  echo "SEED must be a nonnegative integer, got '$SEED'" >&2
  exit 2
fi
if ! [[ "$WORKERS" =~ ^[1-9][0-9]*$ ]]; then
  echo "WORKERS must be a positive integer, got '$WORKERS'" >&2
  exit 2
fi

BUDGETS=()
if [[ "$TRIALS_SPEC" =~ ^[1-9][0-9]*$ ]]; then
  for N in $DEFAULT_RUN_COUNTS; do
    BUDGETS+=("$N $TRIALS_SPEC")
  done
else
  IFS=',' read -r -a ITEMS <<< "$TRIALS_SPEC"
  for ITEM in "${ITEMS[@]}"; do
    if ! [[ "$ITEM" =~ ^[1-9][0-9]*:[1-9][0-9]*$ ]]; then
      echo "TRIALS must be a positive integer or N:TRIALS,... with positive integers, got '$TRIALS_SPEC'" >&2
      exit 2
    fi
    BUDGETS+=("${ITEM%%:*} ${ITEM#*:}")
  done
fi
DUPLICATES=$(printf '%s\n' "${BUDGETS[@]}" | cut -d' ' -f1 | sort -n | uniq -d)
if [ -n "$DUPLICATES" ]; then
  echo "TRIALS lists run count(s) more than once: $DUPLICATES" >&2
  exit 2
fi

mkdir -p "$OUTDIR"
JOBS="$OUTDIR/jobs-$(echo "$TRIALS_SPEC" | tr ',:' '_-').txt"
: > "$JOBS"

shard_size() {
  if [ "$1" -ge 100 ]; then
    echo 100
  elif [ "$1" -gt 20 ]; then
    echo 500
  else
    echo "$2"
  fi
}

CELLS=${CELLS:-$("$PYTHON" -m stresskit.battery_calibration --list-cells)}
while read -r N TRIALS; do
  SIZE=$(shard_size "$N" "$TRIALS")
  for CELL in $CELLS; do
    START=0
    while [ "$START" -lt "$TRIALS" ]; do
      COUNT=$(( TRIALS - START < SIZE ? TRIALS - START : SIZE ))
      echo "$CELL $N $START $COUNT" >> "$JOBS"
      START=$(( START + COUNT ))
    done
  done
done < <(printf '%s\n' "${BUDGETS[@]}" | sort -k1,1nr)

run_shard() {
  local cell=$1 n=$2 start=$3 count=$4
  local out="$OUTDIR/shard-$cell-$n-$start.json"
  if [ -s "$out" ]; then
    echo "skip existing $out"
    return 0
  fi
  if "$PYTHON" -m stresskit.battery_calibration \
      --cells "$cell" --runs "$n" --trials "$count" --trial-start "$start" \
      --seed "$SEED" > "$out.tmp" 2> "$out.log"; then
    mv "$out.tmp" "$out"
    rm -f "$out.log"
    echo "done $out"
  else
    echo "shard failed: cell=$cell n_runs=$n trial_start=$start trials=$count (see $out.log)" >&2
    rm -f "$out.tmp"
    return 1
  fi
}
export -f run_shard
export PYTHON SEED OUTDIR

echo "running $(wc -l < "$JOBS" | tr -d ' ') shards with $WORKERS workers (seed $SEED, trials $TRIALS_SPEC, $(echo $CELLS | wc -w | tr -d ' ') cells) -> $OUTDIR"
if ! xargs -P "$WORKERS" -L 1 bash -c 'run_shard "$@"' _ < "$JOBS"; then
  echo "at least one shard failed; inspect $OUTDIR/*.log, then rerun the same command to fill in the missing shards" >&2
  exit 1
fi
echo "all shards done; merge with:"
echo "$PYTHON -m stresskit.calibration_merge $OUTDIR/shard-*.json > $OUTDIR/merged.json"
