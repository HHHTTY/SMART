#!/usr/bin/env bash
set -u

RUN=/hpc2hdd/home/aimslab/ChengtangZhan/ttt/SpectraLLM/runs/sdbs_random1000_combinations_20260923
PY=/hpc2hdd/home/aimslab/miniconda3/envs/wqx_spectrallm/bin/python
SCORER="$RUN/score_spectrallm_paper_metrics.py"

names=(c_h_ms_ir c_h_ir c_h_ms c_h c h ms ir)
while true; do
  ready=1
  for name in "${names[@]}"; do
    if [[ ! -s "$RUN/${name}_prediction/generated_predictions.jsonl" ]]; then
      ready=0
    fi
  done
  if (( ready )); then
    break
  fi
  printf '[%s] waiting for prediction outputs\n' "$(date '+%F %T')"
  sleep 60
done

for name in "${names[@]}"; do
  input="$RUN/${name}_prediction/generated_predictions.jsonl"
  output="$RUN/${name}_paper_metrics.json"
  if [[ -s "$output" ]]; then
    printf '[%s] already scored %s\n' "$(date '+%F %T')" "$name"
    continue
  fi
  printf '[%s] scoring %s\n' "$(date '+%F %T')" "$name"
  "$PY" "$SCORER" "$input" > "$output"
done
printf '[%s] all scoring complete\n' "$(date '+%F %T')"
