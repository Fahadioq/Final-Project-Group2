#!/usr/bin/env bash
# Wait until process PID exits, then run the command (queue CPU-heavy work behind another job).
#   bash Code/scripts/wait_pid.sh 2604 bash Code/scripts/run_etl.sh A
set -euo pipefail
PID=${1:?usage: wait_pid.sh PID command...}; shift
[ "$#" -ge 1 ] || { echo "usage: wait_pid.sh PID command..." >&2; exit 1; }
if kill -0 "$PID" 2>/dev/null; then
  echo "$(date -u +%FT%TZ) waiting for PID $PID: $(ps -o cmd= -p "$PID" | cut -c1-120)"
  while kill -0 "$PID" 2>/dev/null; do sleep 60; done
  echo "$(date -u +%FT%TZ) PID $PID finished"
fi
exec "$@"
