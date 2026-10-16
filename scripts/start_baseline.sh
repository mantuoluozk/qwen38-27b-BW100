#!/usr/bin/env bash
set -euo pipefail

workdir=/data/zk/qwen38
exp_id=${1:-B000}
variant=${2:-baseline}
container=${CONTAINER:-zk-qwen38}
dcu_devices=${DCU_DEVICES:-0,1}
model_path=${MODEL_PATH:-$workdir/model}
served_model_name=${SERVED_MODEL_NAME:-qwen38-bf16}
# Historical names are preserved, but unsafe communication is no longer runnable.
case "$variant" in
  fa3_hcu_custom_ar|fa3_hcu_custom_ar_*)
    echo 'Custom all-reduce is disabled by policy; select a no_custom_ar variant.' >&2
    exit 2
    ;;
esac
run_dir="$workdir/runs/$exp_id"
if [[ -e "$run_dir/server.log" ]]; then
  echo "Experiment $exp_id already has a server log" >&2
  exit 1
fi

DCU_DEVICES="$dcu_devices" "$workdir/scripts/check_dcu.sh"
docker exec "$container" test -r "$model_path/config.json" || { echo "Model config unavailable in $container: $model_path" >&2; exit 2; }
mkdir -p "$run_dir"
date -Is > "$run_dir/host-start.txt"
hy-smi >> "$run_dir/host-start.txt"
printf 'selected_devices=%s\n' "$dcu_devices" >> "$run_dir/host-start.txt"
image=$(docker inspect "$container" --format '{{.Config.Image}}')
docker image inspect "$image" --format '{{.Id}}' >> "$run_dir/host-start.txt"
docker exec -d -e EXP_ID="$exp_id" -e VARIANT="$variant" -e EXP_IMAGE="$image" -e DCU_DEVICES="$dcu_devices" \
  -e MODEL_PATH="$model_path" -e SERVED_MODEL_NAME="$served_model_name" \
  -e SGLANG_NN_GEMM_7168="${SGLANG_NN_GEMM_7168:-0}" "$container" bash -lc \
  "exec /data/zk/qwen38/scripts/run_server.sh > '$run_dir/server.log' 2>&1"
echo "Started $exp_id ($variant). Log: $run_dir/server.log"
