#!/usr/bin/env bash
# Resume after an idle-DCU guard; preserve the original failed batch status.
set -euo pipefail
workdir=/data/zk/qwen38
batch_id=${1:?existing batch ID required}
: "${DCU_DEVICES:?select the same idle physical DCUs explicitly}"
[[ "$batch_id" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
batch_dir="$workdir/runs/$batch_id"
[[ -f "$batch_dir/source.sha256" && ! -e "$batch_dir/resume.txt" ]] || exit 2
for suffix in INT8_64K INT8_C27; do
  [[ ! -e "$workdir/runs/${batch_id}_${suffix}" ]] || { echo "Run already exists: $suffix"; exit 2; }
done
finish() {
  status=$?
  printf 'resume_end=%s\nstatus=%s\n' "$(date -Is)" "$status" >> "$batch_dir/resume.txt"
  exit "$status"
}
trap finish EXIT
trap 'exit 130' INT TERM
printf 'resume_start=%s\ndevices=%s\n' "$(date -Is)" "$DCU_DEVICES" > "$batch_dir/resume.txt"
cp "$workdir/scripts/run_final_verification_resume.sh" "$batch_dir/snapshot/"
sha256sum "$workdir/scripts/run_final_verification_resume.sh" "$workdir/scripts/audit_final_verification.py" \
  > "$batch_dir/resume-source.sha256"
export DCU_DEVICES CONTAINER=zk-qwen38-int8
export MODEL_PATH=/data1/model/Qwen3.8-27B-Channel-INT8-w8a8 SERVED_MODEL_NAME=qwen38-int8
export SGLANG_NN_GEMM_7168=0 FIXED_MEASURE_ROUNDS=3 FIXED_WARMUP_ROUNDS=2
unset FIXED_REQUESTS_PER_ROUND
for suffix in INT8_64K INT8_C27; do
  sha256sum -c "$batch_dir/source.sha256" >> "$batch_dir/resume-integrity.txt"
  sha256sum -c "$batch_dir/resume-source.sha256" >> "$batch_dir/resume-integrity.txt"
  idle=false
  for attempt in $(seq 1 30); do
    if bash "$workdir/scripts/check_dcu.sh" >> "$batch_dir/resume-idle-checks.txt" 2>&1; then
      idle=true
      break
    fi
    sleep 2
  done
  [[ "$idle" == true ]] || { echo 'Selected DCUs remain occupied; stopping without launching'; exit 1; }
  if [[ "$suffix" == INT8_64K ]]; then
    export FIXED_INPUT_LEN=65536 FIXED_CONCURRENCIES=2 FIXED_SMOKE_64K=1
  else
    export FIXED_INPUT_LEN=4096 FIXED_CONCURRENCIES=27 FIXED_SMOKE_64K=0
  fi
  run_id="${batch_id}_${suffix}"
  printf 'stage_start=%s run=%s\n' "$(date -Is)" "$run_id" >> "$batch_dir/resume.txt"
  bash "$workdir/scripts/run_fixed_round.sh" "$run_id" \
    fa3_hcu_contigfa_eagle314_slots64_no_custom_ar > "$batch_dir/${suffix}_resume.log" 2>&1
  python3 "$workdir/scripts/audit_final_verification.py" "$workdir/runs/$run_id" \
    --input-len "$FIXED_INPUT_LEN" --concurrency "$FIXED_CONCURRENCIES" \
    > "$batch_dir/${suffix}_c${FIXED_CONCURRENCIES}_requests_audit_v2.json"
  printf 'stage_end=%s run=%s\n' "$(date -Is)" "$run_id" >> "$batch_dir/resume.txt"
done
printf 'all_remaining_stages_complete=%s\n' "$(date -Is)" >> "$batch_dir/resume.txt"
