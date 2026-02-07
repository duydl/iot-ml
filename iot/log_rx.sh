#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${1:-${PORT:-/dev/ttyACM0}}"
BAUD="${BAUD:-115200}"

TS="$(date +%Y%m%d_%H%M%S)"
OUTDIR="$ROOT/data/$TS"
OUTFILE="$OUTDIR/rx.csv"

mkdir -p "$OUTDIR"

echo "ts,device,seq,temp_val,temp_scale,hum_val,hum_scale,press_val,press_scale,rssi" > "$OUTFILE"

echo "# Logging RX to $OUTFILE"

echo "# PORT=$PORT BAUD=$BAUD"

PORT="$PORT" BAUD="$BAUD" make -C "$ROOT/iot/rx" term \
  | awk '
    {
      line = $0;
      sep = index(line, " # ");
      if (sep == 0) next;
      ts = substr(line, 1, sep - 1);
      data = substr(line, sep + 3);
      sub(/[[:space:]]+$/, "", ts);
      gsub(/,/, ".", ts);
      sub(/\r$/, "", data);
      if (data ~ /^#/) next;
      comma = 0;
      for (i = 1; i <= length(data); i++) {
        if (substr(data, i, 1) == ",") comma++;
      }
      if (comma < 7) next;
      print ts "," data;
      fflush();
    }' \
  | tee -a "$OUTFILE" >/dev/null
