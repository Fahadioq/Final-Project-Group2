#!/usr/bin/env bash
# Queue a GPU job behind the shared lock so jobs from three people never collide.
#   bash Code/scripts/gpu_run.sh python Code/train.py --config ...     # waits for the GPU
#   bash Code/scripts/gpu_run.sh -n python quick_test.py               # skips (exit 75) if busy
set -euo pipefail
LOCK=${OF_GPU_LOCK:-/tmp/of_gpu.lock}
if [ "${1:-}" = "-n" ]; then shift; exec flock -n -E 75 "$LOCK" "$@"; fi
echo "[gpu_run] waiting for $LOCK" >&2
exec flock "$LOCK" "$@"
