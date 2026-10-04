#!/usr/bin/env bash
set -u

RUN=/hpc2hdd/home/aimslab/ChengtangZhan/ttt/SpectraLLM/runs/sdbs_random1000_combinations_20260923
PY=/hpc2hdd/home/aimslab/miniconda3/envs/wqx_spectrallm/bin/llamafactory-cli
FREE_MIB_THRESHOLD=70000

wait_for_gpu() {
  while true; do
    free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d ' ' || true)
    if [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= FREE_MIB_THRESHOLD )); then
      printf '[%s] GPU free=%s MiB; launching\n' "$(date '+%F %T')" "$free_mib"
      return 0
    fi
    printf '[%s] GPU not ready (free=%s MiB); waiting 60s\n' "$(date '+%F %T')" "${free_mib:-unknown}"
    sleep 60
  done
}

run_one() {
  local name="$1"
  wait_for_gpu
  printf '[%s] START %s\n' "$(date '+%F %T')" "$name"
  "$PY" train "$RUN/${name}_predict.yaml"
  rc=$?
  printf '[%s] END %s rc=%s\n' "$(date '+%F %T')" "$name" "$rc"
  return "$rc"
}

cd "$RUN"
export CUDA_VISIBLE_DEVICES=0
source /hpc2hdd/home/aimslab/miniconda3/bin/activate wqx_spectrallm
run_one c_h_ms
run_one c_h
