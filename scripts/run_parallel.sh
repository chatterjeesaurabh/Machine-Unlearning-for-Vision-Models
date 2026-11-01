#!/bin/bash
# Run a job file (one command per line) with up to N parallel jobs, alternating GPUs 0/1 (--gpu appended per job).
# Holds the project GPU lock for the whole batch (MU_LOCK_HELD=1 for the children), so no other project job starts meanwhile.
# Usage: bash scripts/run_parallel.sh JOBFILE N LOGDIR
JOBS=$1; N=${2:-15}; LOGDIR=${3:-outputs/logs_parallel}
cd /anchor/anchor7/sachatt/cs6886/project; source env.sh
mkdir -p "$LOGDIR"
exec 9>>/tmp/mu_cs6886_gpu.lock
flock 9
export MU_LOCK_HELD=1 LOGDIR
awk '{print NR-1 "\t" $0}' "$JOBS" | xargs -P "$N" -d '\n' -n 1 bash -c '
  i=${1%%$'"'"'\t'"'"'*}; cmd=${1#*$'"'"'\t'"'"'}
  if eval "$cmd --gpu $((i % 2)) --mem_frac 0.2" > "$LOGDIR/job_$i.log" 2>&1; then echo "[ok] $i"; else echo "[FAIL] $i: $cmd"; fi' _
echo "BATCH_DONE $JOBS"
