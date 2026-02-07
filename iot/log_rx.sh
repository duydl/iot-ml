#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PORT="${1:-${PORT:-/dev/ttyACM0}}"
BAUD="${BAUD:-115200}"

TS="$(date +%Y%m%d_%H%M%S)"
OUTDIR="$ROOT/_Project/data/$TS"
OUTFILE="$OUTDIR/rx.csv"

mkdir -p "$OUTDIR"

echo "ts,seq,temp_val,temp_scale,hum_val,hum_scale,press_val,press_scale" > "$OUTFILE"

echo "# Logging RX to $OUTFILE"

echo "# PORT=$PORT BAUD=$BAUD"

PORT="$PORT" BAUD="$BAUD" make -C "$ROOT/_Project/iot/rx" term \
  | awk -F'# ' '/# [0-9]+,/{ ts=$1; sub(/[[:space:]]+$/, "", ts); gsub(/,/, ".", ts); sub(/\r$/, "", $2); print ts "," $2; fflush(); }' \
  | tee -a "$OUTFILE" >/dev/null
