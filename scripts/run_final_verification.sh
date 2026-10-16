#!/usr/bin/env bash
# Sequential final-candidate recheck; never run both TP2 services concurrently.
set -euo pipefail
workdir=/data/zk/qwen38
batch_id=${1:?new batch ID required}
: "${DCU_DEVICES:?select two idle physical DCUs explicitly}"
[[ "$batch_id" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'Invalid batch ID'; exit 2; }
batch_dir="$workdir/runs/$batch_id"
[[ ! -e "$batch_dir" ]] || { echo "Batch already exists: $batch_id"; exit 2; }
mkdir -p "$batch_dir/snapshot"
finish() {
  status=$?
  printf 'batch_end=%s\nstatus=%s\n' "$(date -Is)" "$status" >> "$batch_dir/batch.txt"
  exit "$status"
}
trap finish EXIT
trap 'exit 130' INT TERM
printf 'batch_start=%s\ndevices=%s\nmeasure_rounds=3\nwarmup_rounds=2\n' \
  "$(date -Is)" "$DCU_DEVICES" > "$batch_dir/batch.txt"
scripts=(run_final_verification.sh run_fixed_round.sh start_baseline.sh run_server.sh
         stop_server.sh check_dcu.sh benchmark_official_pasted.py summarize_fixed.py)
for script in "${scripts[@]}"; do
  cp "$workdir/scripts/$script" "$batch_dir/snapshot/$script"
  sha256sum "$workdir/scripts/$script" >> "$batch_dir/source.sha256"
done
date -Is > "$batch_dir/hardware.txt"
uname -a >> "$batch_dir/hardware.txt"
hy-smi >> "$batch_dir/hardware.txt"
for container in zk-qwen38 zk-qwen38-int8; do
  docker inspect "$container" --format 'container={{.Name}} image_ref={{.Config.Image}} image_id={{.Image}}' \
    >> "$batch_dir/containers.txt"
done
export DCU_DEVICES SGLANG_NN_GEMM_7168=0
export FIXED_MEASURE_ROUNDS=3 FIXED_WARMUP_ROUNDS=2
unset FIXED_REQUESTS_PER_ROUND
run_stage() {
  local suffix=$1 precision=$2 length=$3 concurrencies=$4
  local run_id="${batch_id}_${suffix}" variant
  sha256sum -c "$batch_dir/source.sha256" >> "$batch_dir/integrity.txt"
  if [[ "$precision" == bf16 ]]; then
    export CONTAINER=zk-qwen38 MODEL_PATH="$workdir/model" SERVED_MODEL_NAME=qwen38-bf16
    variant=fa3_hcu_contigfa_eagle314_slots32_no_custom_ar
  else
    export CONTAINER=zk-qwen38-int8 MODEL_PATH=/data1/model/Qwen3.8-27B-Channel-INT8-w8a8 SERVED_MODEL_NAME=qwen38-int8
    variant=fa3_hcu_contigfa_eagle314_slots64_no_custom_ar
  fi
  export FIXED_INPUT_LEN="$length" FIXED_CONCURRENCIES="$concurrencies" FIXED_SMOKE_64K=0
  [[ "$length" != 65536 ]] || export FIXED_SMOKE_64K=1
  printf 'stage_start=%s run=%s precision=%s input=%s concurrency=%s\n' \
    "$(date -Is)" "$run_id" "$precision" "$length" "$concurrencies" >> "$batch_dir/batch.txt"
  # The existing start_baseline.sh checks DCU VRAM/HCU immediately before startup.
  bash "$workdir/scripts/run_fixed_round.sh" "$run_id" "$variant" > "$batch_dir/$suffix.log" 2>&1
  for concurrency in $concurrencies; do
    python3 "$workdir/scripts/summarize_fixed.py" \
      "$workdir/runs/$run_id/fixed-$length-1024-c$concurrency/summary.json" \
      > "$batch_dir/${suffix}_c${concurrency}_audit.json"
  done
  printf 'stage_end=%s run=%s\n' "$(date -Is)" "$run_id" >> "$batch_dir/batch.txt"
}
run_stage BF16_4K bf16 4096 '16'
run_stage INT8_4K int8 4096 '16 28'
run_stage BF16_64K bf16 65536 '2'
run_stage INT8_64K int8 65536 '2'
printf 'all_stages_complete=%s\n' "$(date -Is)" >> "$batch_dir/batch.txt"
