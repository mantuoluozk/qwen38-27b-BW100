#!/usr/bin/env bash
# Reproduce the pasted baseline's ShareGPT command without changing dataset sampling.
set -euo pipefail
workdir=/data/zk/qwen38
exp_id=${1:?experiment required}
variant=${2:?variant required}
container=${CONTAINER:-zk-qwen38}
model_path=${MODEL_PATH:-$workdir/model}
served_model_name=${SERVED_MODEL_NAME:-qwen38-bf16}
concurrencies=${SHAREGPT_CONCURRENCIES:-'1 4 16 32 64'}
rounds=${SHAREGPT_ROUNDS:-1}
[[ "$rounds" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid round count'; exit 2; }
dataset="$workdir/datasets/ShareGPT_V3_unfiltered_cleaned_split.json"
expected=35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4
[[ -f "$dataset" ]] || { echo 'ShareGPT dataset missing'; exit 2; }
actual=$(sha256sum "$dataset")
[[ "${actual%% *}" == "$expected" ]] || { echo 'ShareGPT checksum mismatch'; exit 2; }
for concurrency in $concurrencies; do
  [[ "$concurrency" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid concurrency'; exit 2; }
done
run_dir="$workdir/runs/$exp_id"
[[ ! -e "$run_dir" ]] || { echo "Experiment directory already exists: $exp_id"; exit 2; }
started=false
cleanup() {
  status=$?
  if [[ "$started" == true ]]; then
    "$workdir/scripts/stop_server.sh" "$exp_id" || status=1
  fi
  if [[ "$started" == true && -d "$run_dir" ]]; then
    printf 'runner_end=%s\nrunner_status=%s\n' "$(date -Is)" "$status" >> "$run_dir/runner.txt"
  fi
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT TERM
"$workdir/scripts/start_baseline.sh" "$exp_id" "$variant"
started=true
printf 'runner_start=%s\ncontainer=%s\nconcurrencies=%s\nmeasure_rounds=%s\ndataset_sha256=%s\n' \
  "$(date -Is)" "$container" "$concurrencies" "$rounds" "$expected" > "$run_dir/runner.txt"
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
for concurrency in $concurrencies; do
  result_dir="$run_dir/sharegpt-c$concurrency"
  mkdir -p "$result_dir"
  for round in $(seq 1 "$rounds"); do
    printf -v round_id '%02d' "$round"
    printf 'round_start=%s concurrency=%s round=%s\n' "$(date -Is)" "$concurrency" "$round" >> "$run_dir/runner.txt"
    docker exec "$container" timeout --signal=TERM --kill-after=30s 7200s \
      python3 -u -m sglang.bench_serving \
      --backend sglang-oai --base-url http://127.0.0.1:18580 \
      --dataset-name sharegpt --dataset-path "$dataset" \
      --model "$model_path" --served-model-name "$served_model_name" \
      --sharegpt-context-len 131072 --num-prompts 1000 \
      --max-concurrency "$concurrency" --request-rate inf \
      --warmup-requests "$((2 * concurrency))" --seed 20260912 \
      --disable-tqdm --output-details \
      --output-file "$result_dir/round-$round_id.jsonl" \
      > "$result_dir/round-$round_id.log" 2>&1
    printf 'round_end=%s concurrency=%s round=%s\n' "$(date -Is)" "$concurrency" "$round" >> "$run_dir/runner.txt"
  done
done
docker exec "$container" curl -sf --max-time 10 http://127.0.0.1:18580/metrics \
  > "$run_dir/end-metrics.prom" || true
