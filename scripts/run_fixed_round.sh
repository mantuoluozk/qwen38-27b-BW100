#!/usr/bin/env bash
set -euo pipefail
workdir=/data/zk/qwen38
exp_id=${1:?experiment required}
variant=${2:?variant required}
container=${CONTAINER:-zk-qwen38}
served_model_name=${SERVED_MODEL_NAME:-qwen38-bf16}
concurrencies=${FIXED_CONCURRENCIES:-8}
smoke_64k=${FIXED_SMOKE_64K:-0}
input_len=${FIXED_INPUT_LEN:-4096}
measure_rounds=${FIXED_MEASURE_ROUNDS:-1}
warmup_rounds=${FIXED_WARMUP_ROUNDS:-2}
requests_per_round=${FIXED_REQUESTS_PER_ROUND:-}
[[ "$measure_rounds" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid measured rounds'; exit 2; }
[[ "$warmup_rounds" =~ ^[0-9]+$ ]] || { echo 'Invalid warmup rounds'; exit 2; }
[[ -z "$requests_per_round" || "$requests_per_round" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid requests per round'; exit 2; }
[[ "$input_len" == 4096 || "$input_len" == 65536 ]] || { echo 'Invalid fixed input length'; exit 2; }
[[ "$smoke_64k" == 0 || "$smoke_64k" == 1 ]] || { echo 'Invalid smoke flag'; exit 2; }
for concurrency in $concurrencies; do
  [[ "$concurrency" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid concurrency'; exit 2; }
done
run_dir="$workdir/runs/$exp_id"
started=false
cleanup() {
  status=$?
  if [[ "$started" == true ]]; then
    "$workdir/scripts/stop_server.sh" "$exp_id" || status=1
  fi
  if [[ -d "$run_dir" ]]; then
    printf 'runner_end=%s\nrunner_status=%s\n' "$(date -Is)" "$status" >> "$run_dir/runner.txt"
  fi
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT TERM
"$workdir/scripts/start_baseline.sh" "$exp_id" "$variant"
started=true
printf 'runner_start=%s\ncontainer=%s\nconcurrencies=%s\nsmoke_64k=%s\ninput_len=%s\nmeasure_rounds=%s\nwarmup_rounds=%s\nrequests_per_round=%s\n' "$(date -Is)" "$container" "$concurrencies" "$smoke_64k" "$input_len" "$measure_rounds" "$warmup_rounds" "${requests_per_round:-default}" > "$run_dir/runner.txt"
ready=false
for attempt in $(seq 1 180); do
  if docker exec "$container" curl -sf --max-time 3 http://127.0.0.1:18580/health >/dev/null; then
    ready=true
    break
  fi
  if grep -qE 'Traceback|OutOfMemoryError|RuntimeError:' "$run_dir/server.log"; then
    tail -n 35 "$run_dir/server.log"
    exit 1
  fi
  sleep 2
done
[[ "$ready" == true ]] || { echo 'server readiness timeout'; exit 1; }
sha256sum "$workdir/scripts/benchmark_official_pasted.py" >> "$run_dir/runner.txt"
docker exec "$container" python3 -m py_compile "$workdir/scripts/benchmark_official_pasted.py"
docker exec "$container" python3 -u "$workdir/scripts/benchmark_official_pasted.py" \
  --base-url http://127.0.0.1:18580 --model "$served_model_name" \
  --output-dir "$run_dir/fixed-smoke" --input-len 4096 --output-len 16 \
  --concurrency 1 --warmup-rounds 1 --measure-rounds 1 --requests-per-round 1 \
  --timeout 120 > "$run_dir/fixed-smoke.log" 2>&1
echo 'Exact token-ID smoke passed'
if [[ "$smoke_64k" == 1 ]]; then
  docker exec "$container" python3 -u "$workdir/scripts/benchmark_official_pasted.py" \
    --base-url http://127.0.0.1:18580 --model "$served_model_name" \
    --output-dir "$run_dir/fixed-smoke-64k" --input-len 65536 --output-len 16 \
    --concurrency 1 --warmup-rounds 1 --measure-rounds 1 --requests-per-round 1 \
    --timeout 120 > "$run_dir/fixed-smoke-64k.log" 2>&1
  echo 'Exact 64K token-ID capacity smoke passed (not a 64K/1K score)'
fi
for concurrency in $concurrencies; do
request_arg=()
if [[ -n "$requests_per_round" ]]; then
  request_arg=(--requests-per-round "$requests_per_round")
fi
docker exec "$container" python3 -u "$workdir/scripts/benchmark_official_pasted.py" \
  --base-url http://127.0.0.1:18580 --model "$served_model_name" \
  --output-dir "$run_dir/fixed-$input_len-1024-c$concurrency" --input-len "$input_len" --output-len 1024 \
  --concurrency "$concurrency" --warmup-rounds "$warmup_rounds" --measure-rounds "$measure_rounds" \
  "${request_arg[@]}" --timeout 300 \
  > "$run_dir/fixed-$input_len-1024-c$concurrency.log" 2>&1
echo "Fixed input=$input_len C$concurrency measured_rounds=$measure_rounds passed"
done
docker exec "$container" curl -sf --max-time 10 http://127.0.0.1:18580/metrics \
  > "$run_dir/end-metrics.prom" || true
