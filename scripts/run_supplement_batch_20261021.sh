#!/usr/bin/env bash
set -euo pipefail
root=/data/zk/qwen38
batch=${1:?unique batch ID required}
[[ "$batch" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
[[ ! -e "$root/runs/$batch" ]] || { echo 'Batch exists'; exit 2; }
mkdir "$root/runs/$batch"
batch_dir="$root/runs/$batch"
trap 'status=$?; printf "batch_end=%s\nbatch_status=%s\n" "$(date -Is)" "$status" >> "$batch_dir/status.txt"' EXIT
printf 'batch_start=%s\nselected_devices=%s\n' "$(date -Is)" "${DCU_DEVICES:-2,3}" > "$batch_dir/status.txt"
cp "$root/scripts/run_supplement_batch_20261021.sh" "$root/scripts/run_supplement_20261021.sh" "$batch_dir/"
sha256sum "$batch_dir"/*.sh > "$batch_dir/source.sha256"
for precision in bf16 int8; do
  free=false
  # Recheck only DCU occupancy. Delay for owned service's released allocations.
  for attempt in $(seq 1 30); do
    if DCU_DEVICES="${DCU_DEVICES:-2,3}" "$root/scripts/check_dcu.sh" > "$batch_dir/guard-$precision-$attempt.txt" 2>&1; then
      free=true
      break
    fi
    sleep 2
  done
  [[ "$free" == true ]] || { echo 'Selected cards occupied; no service launched'; exit 3; }
  DCU_DEVICES="${DCU_DEVICES:-2,3}" bash "$root/scripts/run_supplement_20261021.sh" "$batch" "$precision" \
    > "$batch_dir/$precision.log" 2>&1
done
