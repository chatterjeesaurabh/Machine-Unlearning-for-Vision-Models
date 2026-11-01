#!/bin/bash
# Run a whole batch script while holding the project GPU lock (children skip re-acquiring via MU_LOCK_HELD=1),
# so a batch of short jobs is not interleaved with other queued jobs. Usage: bash scripts/gpu_batch.sh <batch_script.sh>
flock /tmp/mu_cs6886_gpu.lock env MU_LOCK_HELD=1 bash "$@"
