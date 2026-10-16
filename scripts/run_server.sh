#!/usr/bin/env bash
set -euo pipefail

workdir=/data/zk/qwen38
exp_id=${EXP_ID:?EXP_ID required}
variant=${VARIANT:-baseline}
run_dir="$workdir/runs/$exp_id"
model_path=${MODEL_PATH:-$workdir/model}
served_model_name=${SERVED_MODEL_NAME:-qwen38-bf16}

dcu_devices=${DCU_DEVICES:-0,1}
[[ "$dcu_devices" =~ ^[0-7],[0-7]$ && "${dcu_devices%,*}" != "${dcu_devices#*,}" ]] || { echo 'Expected two distinct DCU IDs in 0..7'; exit 2; }
export HIP_VISIBLE_DEVICES="$dcu_devices"
export OMP_NUM_THREADS=8
export HSA_FORCE_FINE_GRAIN_PCIE=1
export SGLANG_ROCM_USE_AITER_MOE=false
export SGLANG_KV_LAYOUT_HCU_FA=false

extra_args=()
attention_backend=triton
max_running_requests=128
mem_fraction_static=0.85
chunked_prefill_size=8192
max_prefill_tokens=16384
disable_custom_all_reduce=true
case "$variant" in
  baseline) ;;
  ops)
    export SGLANG_USE_LIGHTOP=1
    export SGLANG_USE_FUSED_TOPK_SOFTMAX=1
    export SGLANG_USE_CAUSAL_CONV1D=1
    export SGLANG_USE_AITER_LINEAR_ATTN=1
    ;;
  fa3)
    attention_backend=fa3
    extra_args+=(--mm-attention-backend fa3)
    ;;
  fa3_slots32_no_custom_ar)
    attention_backend=fa3
    max_running_requests=32
    extra_args+=(--mm-attention-backend fa3)
    ;;
  fa3_slots64_no_custom_ar)
    attention_backend=fa3
    max_running_requests=64
    extra_args+=(--mm-attention-backend fa3)
    ;;
  fa3_hcu_layout)
    attention_backend=fa3
    export SGLANG_KV_LAYOUT_HCU_FA=true
    extra_args+=(--mm-attention-backend fa3)
    ;;
  fa3_hcu_prefill16k)
    attention_backend=fa3
    export SGLANG_KV_LAYOUT_HCU_FA=true
    chunked_prefill_size=16384
    extra_args+=(--mm-attention-backend fa3)
    ;;
  fa3_hcu_aiter_linear)
    attention_backend=fa3
    export SGLANG_KV_LAYOUT_HCU_FA=true
    export SGLANG_USE_AITER_LINEAR_ATTN=1
    extra_args+=(--mm-attention-backend fa3)
    ;;
  fa3_hcu_custom_ar)
    attention_backend=fa3
    export SGLANG_KV_LAYOUT_HCU_FA=true
    disable_custom_all_reduce=false
    extra_args+=(--mm-attention-backend fa3)
    ;;
  fa3_hcu_custom_ar_contigfa)
    attention_backend=fa3
    export SGLANG_KV_LAYOUT_HCU_FA=true
    export USE_CONTIGUOUS_FA=1
    disable_custom_all_reduce=false
    extra_args+=(--mm-attention-backend fa3)
    ;;
  fa3_hcu_contigfa_no_custom_ar)
    attention_backend=fa3
    export SGLANG_KV_LAYOUT_HCU_FA=true
    export USE_CONTIGUOUS_FA=1
    extra_args+=(--mm-attention-backend fa3)
    ;;
  fa3_hcu_contigfa_eagle314_slots16_no_custom_ar)
    attention_backend=fa3
    export SGLANG_KV_LAYOUT_HCU_FA=true
    export USE_CONTIGUOUS_FA=1
    export SGLANG_ENABLE_SPEC_V2=1
    max_running_requests=16
    extra_args+=(--mm-attention-backend fa3 --speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4)
    ;;
  fa3_hcu_contigfa_eagle314_slots16_prefill32k_no_custom_ar)
    attention_backend=fa3
    export SGLANG_KV_LAYOUT_HCU_FA=true
    export USE_CONTIGUOUS_FA=1
    export SGLANG_ENABLE_SPEC_V2=1
    max_running_requests=16
    chunked_prefill_size=32768
    max_prefill_tokens=32768
    extra_args+=(--mm-attention-backend fa3 --speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4)
    ;;
  fa3_hcu_contigfa_eagle415_slots16_no_custom_ar)
    attention_backend=fa3
    export SGLANG_KV_LAYOUT_HCU_FA=true
    export USE_CONTIGUOUS_FA=1
    export SGLANG_ENABLE_SPEC_V2=1
    max_running_requests=16
    extra_args+=(--mm-attention-backend fa3 --speculative-algorithm EAGLE --speculative-num-steps 4 --speculative-eagle-topk 1 --speculative-num-draft-tokens 5)
    ;;
  fa3_hcu_contigfa_eagle213_slots16_no_custom_ar)
    attention_backend=fa3
    export SGLANG_KV_LAYOUT_HCU_FA=true
    export USE_CONTIGUOUS_FA=1
    export SGLANG_ENABLE_SPEC_V2=1
    max_running_requests=16
    extra_args+=(--mm-attention-backend fa3 --speculative-algorithm EAGLE --speculative-num-steps 2 --speculative-eagle-topk 1 --speculative-num-draft-tokens 3)
    ;;
  fa3_hcu_contigfa_eagle314_slots32_no_custom_ar)
    attention_backend=fa3
    export SGLANG_KV_LAYOUT_HCU_FA=true
    export USE_CONTIGUOUS_FA=1
    export SGLANG_ENABLE_SPEC_V2=1
    max_running_requests=32
    extra_args+=(--mm-attention-backend fa3 --speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4)
    ;;
  fa3_hcu_contigfa_eagle314_slots64_no_custom_ar)
    attention_backend=fa3
    export SGLANG_KV_LAYOUT_HCU_FA=true
    export USE_CONTIGUOUS_FA=1
    export SGLANG_ENABLE_SPEC_V2=1
    max_running_requests=64
    extra_args+=(--mm-attention-backend fa3 --speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4)
    ;;
  fa3_hcu_custom_ar_contigfa_eagle314)
    attention_backend=fa3
    export SGLANG_KV_LAYOUT_HCU_FA=true
    export USE_CONTIGUOUS_FA=1
    export SGLANG_ENABLE_SPEC_V2=1
    disable_custom_all_reduce=false
    max_running_requests=64
    extra_args+=(--mm-attention-backend fa3 --speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4)
    ;;
  fa3_hcu_custom_ar_contigfa_eagle314_bf16state)
    attention_backend=fa3
    export SGLANG_KV_LAYOUT_HCU_FA=true
    export USE_CONTIGUOUS_FA=1
    export SGLANG_ENABLE_SPEC_V2=1
    disable_custom_all_reduce=false
    max_running_requests=64
    extra_args+=(--mm-attention-backend fa3 --mamba-ssm-dtype bfloat16 --speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4)
    ;;
  fa3_hcu_custom_ar_contigfa_eagle314_slots16)
    attention_backend=fa3
    export SGLANG_KV_LAYOUT_HCU_FA=true
    export USE_CONTIGUOUS_FA=1
    export SGLANG_ENABLE_SPEC_V2=1
    disable_custom_all_reduce=false
    max_running_requests=16
    extra_args+=(--mm-attention-backend fa3 --speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4)
    ;;
  fa3_hcu_custom_ar_contigfa_prefill16k)
    attention_backend=fa3
    export SGLANG_KV_LAYOUT_HCU_FA=true
    export USE_CONTIGUOUS_FA=1
    disable_custom_all_reduce=false
    chunked_prefill_size=16384
    extra_args+=(--mm-attention-backend fa3)
    ;;
  fa3_hcu_custom_ar_contigfa_prefill32k)
    attention_backend=fa3
    export SGLANG_KV_LAYOUT_HCU_FA=true
    export USE_CONTIGUOUS_FA=1
    disable_custom_all_reduce=false
    chunked_prefill_size=32768
    max_prefill_tokens=32768
    extra_args+=(--mm-attention-backend fa3)
    ;;
  nextn)
    export SGLANG_ENABLE_SPEC_V2=1
    max_running_requests=64
    extra_args+=(--speculative-algorithm NEXTN --speculative-num-steps 2 --speculative-eagle-topk 1 --speculative-num-draft-tokens 2)
    ;;
  nextn_hi_mem)
    export SGLANG_ENABLE_SPEC_V2=1
    max_running_requests=64
    mem_fraction_static=0.90
    extra_args+=(--speculative-algorithm NEXTN --speculative-num-steps 2 --speculative-eagle-topk 1 --speculative-num-draft-tokens 2)
    ;;
  *) echo "Unknown variant: $variant" >&2; exit 2 ;;
esac

if [[ "$disable_custom_all_reduce" != true ]]; then
  echo 'RuntimeError: Custom all-reduce is disabled by policy; choose a no_custom_ar variant.' >&2
  exit 2
fi
mkdir -p "$run_dir"

if [[ "$disable_custom_all_reduce" == true ]]; then
  extra_args+=(--disable-custom-all-reduce)
fi

printf 'experiment=%s\nvariant=%s\nstart=%s\nmodel=%s\nserved_model_name=%s\n' "$exp_id" "$variant" "$(date -Is)" "$model_path" "$served_model_name" > "$run_dir/manifest.txt"
printf 'image=%s\n' "${EXP_IMAGE:-harbor.sourcefind.cn:5443/hcu/admin/base/sglang:0.5.12-ubuntu22.04-dtk2604-py3.10}" >> "$run_dir/manifest.txt"
printf 'physical_dcu_devices=%s\n' "$dcu_devices" >> "$run_dir/manifest.txt"
env | grep -E '^(HIP|HSA|OMP|SGLANG)_' | sort >> "$run_dir/manifest.txt"
env | grep '^USE_CONTIGUOUS_FA=' >> "$run_dir/manifest.txt" || true
printf 'extra_args=%s\n' "${extra_args[*]}" >> "$run_dir/manifest.txt"
printf 'attention_backend=%s\n' "$attention_backend" >> "$run_dir/manifest.txt"
printf 'max_running_requests=%s\n' "$max_running_requests" >> "$run_dir/manifest.txt"
printf 'mem_fraction_static=%s\n' "$mem_fraction_static" >> "$run_dir/manifest.txt"
printf 'chunked_prefill_size=%s\n' "$chunked_prefill_size" >> "$run_dir/manifest.txt"
printf 'max_prefill_tokens=%s\n' "$max_prefill_tokens" >> "$run_dir/manifest.txt"
printf 'disable_custom_all_reduce=%s\n' "$disable_custom_all_reduce" >> "$run_dir/manifest.txt"
sha256sum "$workdir/scripts/run_server.sh" >> "$run_dir/manifest.txt"
echo $$ > "$run_dir/server.pid"

exec python3 -u -m sglang.launch_server \
  --model-path "$model_path" \
  --served-model-name "$served_model_name" \
  --host 0.0.0.0 \
  --port 18580 \
  --nccl-port 19580 \
  --tp-size 2 \
  --dtype bfloat16 \
  --kv-cache-dtype bfloat16 \
  --context-length 131072 \
  --mem-fraction-static "$mem_fraction_static" \
  --max-running-requests "$max_running_requests" \
  --chunked-prefill-size "$chunked_prefill_size" \
  --max-prefill-tokens "$max_prefill_tokens" \
  --disable-radix-cache \
  --attention-backend "$attention_backend" \
  --page-size 64 \
  --mamba-scheduler-strategy no_buffer \
  --cuda-graph-max-bs 128 \
  --enable-metrics \
  "${extra_args[@]}"
