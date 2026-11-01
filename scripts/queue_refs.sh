#!/bin/bash
# Sequential (one job at a time) training of Original + Retrain references. Resumable: skips runs whose metrics.json exists.
# Usage: nohup bash scripts/queue_refs.sh [GPU] > outputs/logs/queue_refs.log 2>&1 &
GPU=${1:-1}
cd /anchor/anchor7/sachatt/cs6886/project
source env.sh
mkdir -p outputs/logs

JOBS=("orig 0")
for s in 0 1 2; do
  for sc in S1c0 S2k359p10 S2k359p30 S2k359p50; do JOBS+=("$sc $s"); done
done

for job in "${JOBS[@]}"; do
  set -- $job
  sc=$1; seed=$2
  method=$([ "$sc" = "orig" ] && echo original || echo retrain)
  run_id="${sc}__${method}__s${seed}"
  if [ -f "records/runs/${run_id}/metrics.json" ]; then echo "[skip] $run_id"; continue; fi
  echo "[start] $run_id $(date)"
  python scripts/train.py --scenario "$sc" --seed "$seed" --gpu "$GPU" --mem_frac 0.45 > "outputs/logs/${run_id}.log" 2>&1
  echo "[end] $run_id rc=$? $(date)"
  python scripts/build_index.py > /dev/null 2>&1
done
echo "[queue done] $(date)"
