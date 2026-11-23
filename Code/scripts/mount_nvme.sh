#!/usr/bin/env bash
# Mount the EC2 instance store at /mnt/nvme. Safe to re-run every session.
# The device is found by its model string, never by name: NVMe names can swap between boots.
# It is formatted ONLY if it has no filesystem (true after every stop/start, when it is wiped).
set -euo pipefail
MNT=${OF_NVME:-/mnt/nvme}
if mountpoint -q "$MNT"; then echo "already mounted: $MNT"; df -h "$MNT" | tail -1; exit 0; fi
mapfile -t DEVS < <(lsblk -dnpo NAME,MODEL | awk '/Amazon EC2 NVMe Instance Storage/ {print $1}')
if [ "${#DEVS[@]}" -ne 1 ]; then echo "expected exactly one instance-store device, found ${#DEVS[@]}" >&2; exit 1; fi
DEV=${DEVS[0]}
if [ -n "$(lsblk -no MOUNTPOINT "$DEV" | tr -d '[:space:]')" ]; then echo "$DEV is already mounted elsewhere; refusing" >&2; exit 1; fi
if ! sudo blkid "$DEV" >/dev/null 2>&1; then
  echo "formatting $DEV (instance store: contents never survive a stop/start)"
  sudo mkfs.ext4 -q -F -E nodiscard "$DEV"
fi
sudo mkdir -p "$MNT"
sudo mount -o noatime "$DEV" "$MNT"
sudo chown "$(id -u):$(id -g)" "$MNT"
mkdir -p "$MNT/landing" "$MNT/hf_home" "$MNT/views"
df -h "$MNT" | tail -1
