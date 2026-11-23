#!/usr/bin/env bash
# Run an ETL pass over core/train. Run inside tmux:  bash Code/scripts/run_etl.sh A
# Extra arguments go to etl.stream (e.g. --limit 1, --dry-run, --workers 2).
# Graceful stop: touch ~/of_data/manifests/STOP   (resume by re-running the same command)
set -euo pipefail
PASS=${1:?usage: run_etl.sh A|B [extra args]}; shift
CODE_DIR=$(cd "$(dirname "$0")/.." && pwd)
export OF_DATA=${OF_DATA:-$HOME/of_data}
export OF_NVME=${OF_NVME:-/mnt/nvme}
export HF_HOME=$OF_NVME/hf_home                                  # hub + xet caches on NVMe, never EBS
export HF_TOKEN_PATH=${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}  # keep `hf auth login` token visible
export PYTHONPATH=$CODE_DIR${PYTHONPATH:+:$PYTHONPATH}
PY=${OF_PYTHON:-$HOME/dl-venv/bin/python}
mountpoint -q "$OF_NVME" || { echo "NVMe not mounted: bash Code/scripts/mount_nvme.sh" >&2; exit 1; }
mkdir -p "$OF_DATA/logs"
LOG=$OF_DATA/logs/etl_core_train_pass${PASS}_$(date +%Y%m%d_%H%M%S).log
echo "logging to $LOG"
cd "$CODE_DIR"
nice -n "${OF_NICE:-10}" "$PY" -m etl.stream --pass "$PASS" "$@" 2>&1 | tee -a "$LOG"
