#!/usr/bin/env bash
set -euo pipefail

workdir=/data/zk/qwen38
exp_id=${1:?experiment ID required}
container=${CONTAINER:-zk-qwen38}
run_dir="$workdir/runs/$exp_id"

if [[ -f "$run_dir/server.pid" ]]; then
  pid=$(<"$run_dir/server.pid")
else
  pid=$(docker exec "$container" bash -lc "pgrep -f '^python3 -u -m sglang.launch_server'" || true)
fi

if [[ -z "$pid" || ! "$pid" =~ ^[0-9]+$ ]]; then
  echo "Could not identify a single SGLang server PID" >&2
  exit 1
fi

printf 'stop_requested=%s\npid=%s\n' "$(date -Is)" "$pid" >> "$run_dir/manifest.txt"
docker exec "$container" kill -TERM "$pid"

for _ in $(seq 1 30); do
  if ! docker exec "$container" kill -0 "$pid" 2>/dev/null; then
    echo "Server process $pid exited"
    hy-smi >> "$run_dir/host-stop.txt"
    date -Is >> "$run_dir/host-stop.txt"
    exit 0
  fi
  sleep 2
done

echo "Server PID $pid did not exit after 60 seconds" >&2
exit 1
