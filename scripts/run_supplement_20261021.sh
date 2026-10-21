#!/usr/bin/env bash
# Missing fixed-load tiers and precise 64K SLA boundary; original serving recipe.
set -euo pipefail
root=/data/zk/qwen38
batch=${1:?batch ID required}
precision=${2:?bf16 or int8 required}
[[ "$batch" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
case "$precision" in
  bf16)
    export CONTAINER=zk-qwen38
    export MODEL_PATH="$root/model"
    export SERVED_MODEL_NAME=qwen38-bf16
    variant=fa3_hcu_contigfa_eagle314_slots32_no_custom_ar
    ;;
  int8)
    export CONTAINER=zk-qwen38-int8
    export MODEL_PATH=/data1/model/Qwen3.8-27B-Channel-INT8-w8a8
    export SERVED_MODEL_NAME=qwen38-int8
    variant=fa3_hcu_contigfa_eagle314_slots64_no_custom_ar
    ;;
  *) exit 2 ;;
esac
export DCU_DEVICES=${DCU_DEVICES:-2,3}
export SGLANG_NN_GEMM_7168=0
exp_id="${batch}_${precision}"
run_dir="$root/runs/$exp_id"
[[ ! -e "$run_dir" ]] || { echo "Refuse overwrite: $run_dir"; exit 2; }
started=false
cleanup() {
  status=$?
  trap - EXIT
  if [[ "$started" == true && -r "$run_dir/server.pid" ]]; then
    "$root/scripts/stop_server.sh" "$exp_id" || status=1
  fi
  if [[ -d "$run_dir" ]]; then
    printf 'runner_end=%s\nrunner_status=%s\n' "$(date -Is)" "$status" >> "$run_dir/runner.txt"
  fi
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT TERM
# start_baseline logs/checks selected DCU memory and compute usage before launch.
"$root/scripts/start_baseline.sh" "$exp_id" "$variant"
started=true
printf 'runner_start=%s\nprecision=%s\ncontainer=%s\nselected_devices=%s\n' \
  "$(date -Is)" "$precision" "$CONTAINER" "$DCU_DEVICES" > "$run_dir/runner.txt"
mkdir "$run_dir/source"
cp "$root/scripts/run_supplement_20261021.sh" "$root/scripts/run_server.sh" \
   "$root/scripts/start_baseline.sh" "$root/scripts/stop_server.sh" \
   "$root/scripts/check_dcu.sh" "$root/scripts/benchmark_official_pasted.py" "$run_dir/source/"
sha256sum "$run_dir"/source/* > "$run_dir/source.sha256"
docker inspect "$CONTAINER" --format '{{.Config.Image}} {{.Image}}' > "$run_dir/image.txt"
docker exec "$CONTAINER" sha256sum "$MODEL_PATH/config.json" > "$run_dir/model-config.sha256"
ready=false
for attempt in $(seq 1 300); do
  if docker exec "$CONTAINER" curl -sf --max-time 3 http://127.0.0.1:18580/health >/dev/null; then
    ready=true
    break
  fi
  if grep -qE 'Traceback|OutOfMemoryError|RuntimeError:' "$run_dir/server.log"; then
    tail -n 40 "$run_dir/server.log"
    exit 1
  fi
  sleep 2
done
[[ "$ready" == true ]] || { echo 'Service readiness timeout'; exit 1; }
docker exec "$CONTAINER" python3 -m py_compile "$root/scripts/benchmark_official_pasted.py"
benchmark() {
  local input=$1 concurrency=$2 rounds=$3 output=${4:-1024} warmup=${5:-2}
  local target="$run_dir/fixed-$input-$output-c$concurrency"
  printf 'tier_start=%s input=%s output=%s concurrency=%s measured_rounds=%s\n' \
    "$(date -Is)" "$input" "$output" "$concurrency" "$rounds" | tee -a "$run_dir/runner.txt"
  local extra=()
  [[ "$output" == 1024 ]] || extra=(--requests-per-round 1)
  docker exec "$CONTAINER" python3 -u "$root/scripts/benchmark_official_pasted.py" \
    --base-url http://127.0.0.1:18580 --model "$SERVED_MODEL_NAME" \
    --output-dir "$target" --input-len "$input" --output-len "$output" \
    --concurrency "$concurrency" --warmup-rounds "$warmup" --measure-rounds "$rounds" \
    --timeout 3600 "${extra[@]}" > "$target.log" 2>&1
  # PASS here means HTTP and exact lengths, not SLA. Never stop on SLA failures.
  printf 'tier_complete=%s input=%s output=%s concurrency=%s\n' \
    "$(date -Is)" "$input" "$output" "$concurrency" | tee -a "$run_dir/runner.txt"
}
benchmark 4096 1 1 16 1
benchmark 4096 64 1
benchmark 4096 128 1
if [[ "$precision" == bf16 ]]; then
  benchmark 4096 18 3
  benchmark 4096 19 3
fi
benchmark 65536 1 1 16 1
benchmark 65536 3 3
benchmark 65536 8 1
benchmark 65536 16 1
benchmark 65536 32 1
docker exec "$CONTAINER" curl -sf --max-time 10 http://127.0.0.1:18580/metrics > "$run_dir/end-metrics.prom" || true
