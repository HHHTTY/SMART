#!/usr/bin/env bash
set -euo pipefail

BASE=/hpc2hdd/home/aimslab/ChengtangZhan
ROOT="$BASE/ttt"
SNAP="$ROOT/paper_multitask_ms_snapshot_20260914"
PY="$ROOT/.venv/bin/python"
EVAL="$ROOT/tmp/evaluate_frozen_modality_ablation.py"
DATA="${CHEMOTION_DATA:-$BASE/Dataset/chemotion_tokenized_datasets/Chemotion_final_ir1800_20260922_peakpicked_model_compatible/test.parquet}"
CONFIG="$SNAP/runs/sdbs_ir1800_zeroshot1k_epoch24_20260914/sdbs_ir1800_1k/2026-09-14_10-02-16/.hydra/config.yaml"
CHECKPOINT="$SNAP/checkpoints/epoch_24-step_122175.ckpt"
PREPROCESSOR="${CHEMOTION_PREPROCESSOR:-$SNAP/preprocessor_sdbs_ir1800.pkl}"
RUN="${CHEMOTION_RUN:-$ROOT/runs/chemotion_ir1800_peakpicked_multitask_ms_zeroshot_20260922}"
BATCH_SIZE="${CHEMOTION_BATCH_SIZE:-8}"
NUM_WORKERS="${CHEMOTION_NUM_WORKERS:-4}"
RETRY_BATCH_SIZE="${CHEMOTION_RETRY_BATCH_SIZE:-4}"
RETRY_NUM_WORKERS="${CHEMOTION_RETRY_NUM_WORKERS:-2}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$SNAP/code/src:$SNAP/src:$ROOT/tmp:$ROOT:$ROOT/tools"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$RUN/evaluation" "$RUN/logs"
for required in "$PY" "$EVAL" "$DATA" "$CONFIG" "$CHECKPOINT" "$PREPROCESSOR"; do
  [[ -s "$required" ]] || { echo "missing input: $required" >&2; exit 2; }
done

ALL=(Formula HNMR CNMR MSMS IR)

run_one() {
  local name="$1"
  local selected="$2"
  local output="$RUN/evaluation/${name}.json"
  local log="$RUN/logs/${name}.log"
  [[ -s "$output" ]] && { echo "[$(date -Is)] SKIP $name"; return 0; }

  local excluded=""
  local modality
  for modality in "${ALL[@]}"; do
    if [[ " $selected " != *" $modality "* ]]; then
      excluded="${excluded:+$excluded,}$modality"
    fi
  done

  # The epoch-24 SDBS checkpoint was trained with a 916-token MSMS source
  # limit.  Chemotion contains longer MS/MS strings, so keep the input within
  # the trained contract before building the multimodal encoder sequence.
  local token_args=()
  if [[ " $selected " == *" MSMS "* ]]; then
    token_args=(--token-limit-modality MSMS --token-limit 916)
  fi

  echo "[$(date -Is)] START $name selected=$selected excluded=$excluded"
  if "$PY" -u "$EVAL" \
      --config "$CONFIG" \
      --checkpoint "$CHECKPOINT" \
      --preprocessor "$PREPROCESSOR" \
      --data-path "$DATA" \
      --output "$output" \
      --exclude-input-modalities "$excluded" \
      --ir-column ir_spectra \
      --msms-column ms_spectrum \
      "${token_args[@]}" \
      --batch-size "$BATCH_SIZE" \
      --beams 10 \
      --num-workers "$NUM_WORKERS" \
      --precision fp16 \
      > "$log" 2>&1; then
    echo "[$(date -Is)] DONE $name"
    return 0
  fi

  echo "[$(date -Is)] RETRY $name batch=$RETRY_BATCH_SIZE"
  "$PY" -u "$EVAL" \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --preprocessor "$PREPROCESSOR" \
    --data-path "$DATA" \
    --output "$output" \
    --exclude-input-modalities "$excluded" \
    --ir-column ir_spectra \
    --msms-column ms_spectrum \
    "${token_args[@]}" \
    --batch-size "$RETRY_BATCH_SIZE" \
    --beams 10 \
    --num-workers "$RETRY_NUM_WORKERS" \
    --precision fp16 \
    > "$RUN/logs/${name}.retry_b4.log" 2>&1
  echo "[$(date -Is)] DONE $name batch=$RETRY_BATCH_SIZE"
}

# The established 14-combination panel plus the Full five-modality baseline.
CASES=(
  "formula_cnmr|Formula CNMR"
  "formula_cnmr_ir|Formula CNMR IR"
  "formula_hnmr_cnmr|Formula HNMR CNMR"
  "formula_hnmr_cnmr_ir|Formula HNMR CNMR IR"
  "formula_cnmr_msms|Formula CNMR MSMS"
  "formula_cnmr_msms_ir|Formula CNMR MSMS IR"
  "formula_hnmr_cnmr_msms|Formula HNMR CNMR MSMS"
  "full|Formula HNMR CNMR MSMS IR"
  "formula_hnmr|Formula HNMR"
  "formula_hnmr_ir|Formula HNMR IR"
  "formula_hnmr_msms_ir|Formula HNMR MSMS IR"
  "formula_hnmr_msms|Formula HNMR MSMS"
  "formula_msms|Formula MSMS"
  "formula_msms_ir|Formula MSMS IR"
  "formula_ir|Formula IR"
)

echo "[$(date -Is)] START host=$(hostname) pid=$$ data=$DATA checkpoint=$CHECKPOINT"
printf '%s\n' "$$" > "$RUN/pid.txt"
for item in "${CASES[@]}"; do
  run_one "${item%%|*}" "${item#*|}"
done
touch "$RUN/DONE"
echo "[$(date -Is)] COMPLETE run=$RUN"
