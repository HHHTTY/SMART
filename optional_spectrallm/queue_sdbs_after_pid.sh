#!/usr/bin/env bash
set -u

RUN=/hpc2hdd/home/aimslab/ChengtangZhan/ttt/SpectraLLM/runs/sdbs_random1000_combinations_20260923
WAIT_PID="${WAIT_PID:?set WAIT_PID}"
TASKS=("$@")

while kill -0 "$WAIT_PID" 2>/dev/null; do
  printf '[%s] waiting for pid %s\n' "$(date '+%F %T')" "$WAIT_PID"
  sleep 60
done
printf '[%s] pid %s finished\n' "$(date '+%F %T')" "$WAIT_PID"

source /hpc2hdd/home/aimslab/miniconda3/bin/activate wqx_spectrallm
export CUDA_VISIBLE_DEVICES=0
for name in "${TASKS[@]}"; do
  printf '[%s] START %s\n' "$(date '+%F %T')" "$name"
  /hpc2hdd/home/aimslab/miniconda3/envs/wqx_spectrallm/bin/llamafactory-cli train "$RUN/${name}_predict.yaml"
  rc=$?
  printf '[%s] END %s rc=%s\n' "$(date '+%F %T')" "$name" "$rc"
done
