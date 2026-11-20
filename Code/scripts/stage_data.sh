#!/usr/bin/env bash
# Copy finished view chunks from EBS to NVMe for fast random reads.
#   bash Code/scripts/stage_data.sh C R            # all pools
#   POOL=core_train bash Code/scripts/stage_data.sh C
set -euo pipefail
OF_DATA=${OF_DATA:-$HOME/of_data}; OF_NVME=${OF_NVME:-/mnt/nvme}
mountpoint -q "$OF_NVME" || { echo "NVMe not mounted: bash Code/scripts/mount_nvme.sh" >&2; exit 1; }
[ "$#" -ge 1 ] || { echo "usage: stage_data.sh VIEW [VIEW...]" >&2; exit 1; }
for v in "$@"; do
  SRC=$OF_DATA/views/$v/${POOL:-}; DST=$OF_NVME/views/$v/${POOL:-}
  mkdir -p "$DST"
  rsync -a --exclude '*.tmp*' --info=progress2 "$SRC/" "$DST/"
done
