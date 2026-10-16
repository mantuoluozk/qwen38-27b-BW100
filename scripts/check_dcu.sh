#!/usr/bin/env bash
set -euo pipefail

workdir=/data/zk/qwen38
dcu_devices=${DCU_DEVICES:-0,1}
[[ "$dcu_devices" =~ ^[0-7],[0-7]$ && "${dcu_devices%,*}" != "${dcu_devices#*,}" ]] || { echo 'Expected two distinct DCU IDs in 0..7'; exit 2; }
mkdir -p "$workdir/logs"
stamp=$(date -Is)
sample=$(hy-smi)
printf '%s\n' "$sample" >> "$workdir/logs/dcu-checks.log"
printf 'check_time=%s\n' "$stamp" >> "$workdir/logs/dcu-checks.log"
printf 'selected_devices=%s\n' "$dcu_devices" >> "$workdir/logs/dcu-checks.log"

printf '%s\n' "$sample" | awk -v selected="$dcu_devices" '
  BEGIN { split(selected, ids, ",") }
  $1 == ids[1] || $1 == ids[2] {
    vram = $6; util = $7
    gsub(/%/, "", vram); gsub(/%/, "", util)
    printf "DCU %s: VRAM=%s%% HCU=%s%%\n", $1, vram, util
    count++
    if (vram !~ /^[0-9]+([.][0-9]+)?$/ || util !~ /^[0-9]+([.][0-9]+)?$/) bad = 1
    if (vram + 0 != 0 || util + 0 != 0) busy = 1
  }
  END {
    if (count != 2 || bad) { print "Could not read both selected DCUs " selected > "/dev/stderr"; exit 2 }
    if (busy) { print "Selected DCUs " selected " busy; refusing to start" > "/dev/stderr"; exit 1 }
  }
'
