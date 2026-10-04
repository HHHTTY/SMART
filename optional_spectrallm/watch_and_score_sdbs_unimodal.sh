#!/usr/bin/env bash
set -u

RUN=/hpc2hdd/home/aimslab/ChengtangZhan/ttt/SpectraLLM/runs/sdbs_random1000_combinations_20260923
PY=/hpc2hdd/home/aimslab/miniconda3/envs/wqx_spectrallm/bin/python
SCORER="$RUN/score_spectrallm_paper_metrics.py"

for name in c h ms ir; do
  while [[ ! -s "$RUN/${name}_prediction/generated_predictions.jsonl" ]]; do
    printf '[%s] waiting for %s prediction output\n' "$(date '+%F %T')" "$name"
    sleep 60
  done
done

for name in c h ms ir; do
  input="$RUN/${name}_prediction/generated_predictions.jsonl"
  output="$RUN/${name}_paper_metrics.json"
  if [[ -s "$output" ]]; then
    continue
  fi
  printf '[%s] scoring %s\n' "$(date '+%F %T')" "$name"
  "$PY" "$SCORER" "$input" > "$output"
done
printf '[%s] unimodal scoring complete\n' "$(date '+%F %T')"
