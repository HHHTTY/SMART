#!/usr/bin/env bash
set -euo pipefail

# Resumable pipeline used for the current router -> Full CE -> stable Stage 2 run.
# All paths are remote HPC paths; override variables before invoking this script.

BASE="${BASE:-/hpc2hdd/home/aimslab/ChengtangZhan/ttt/paper_multitask_ms_snapshot_20260914}"
DATA="${DATA:-/hpc2hdd/home/aimslab/ChengtangZhan/Dataset/sdbs_tokenized_datasets/SDBS_final_ir1800_20260914/eval_splits/full_3669/test.parquet}"
SOURCE="${SOURCE:-$BASE/checkpoints/epoch_24-step_122175.ckpt}"
ROUTER_CHECKPOINT="${ROUTER_CHECKPOINT:-$BASE/runs/router_profiled_v6_streaming_50k_beam5_20260921/pilot_router_after_gpu3_20260921_202146/router_pilot.pt}"
PREPROCESSOR="${PREPROCESSOR:-$BASE/preprocessor_sdbs_ir1800.pkl}"
MODEL_CONFIG="${MODEL_CONFIG:-$BASE/configs/model/custom_model_paper_multitask_ms.yaml}"
DATA_CONFIG="${DATA_CONFIG:-$BASE/configs/data/multimodal/paper_multitask_ms_generation.yaml}"

LIMIT="${LIMIT:-3669}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-12}"
STAGE1_LR="${STAGE1_LR:-1e-5}"
STABILITY_INTERVAL="${STABILITY_INTERVAL:-3}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-3}"
STAGE2_LR="${STAGE2_LR:-1e-5}"
STAGE2_POLICY="${STAGE2_POLICY:-exclude_full}"
# The defaults reproduce the current full_3669 run. Override TAG/RUN/ROUTES for
# a new experiment; the route manifest is a frozen artifact and is intentionally
# shared with the route-only run that produced the current experiment.
TAG="${TAG:-full3669_e12_lr1e5_gpu2_20260922}"
RUN="${RUN:-$BASE/runs/router_teacher_full_pseudo_ce_stage2_interval3_${TAG}}"
ROUTES="${ROUTES:-$BASE/runs/router_teacher_full_pseudo_ce_stage2_interval3_full3669_e10_lr1e5_gpu2_20260921/routes}"
STAGE1="$RUN/stage1"
STAGE2="$RUN/stage2_${STAGE2_POLICY}"
PREDICTIONS="$RUN/predictions"

if (( STAGE1_EPOCHS < STABILITY_INTERVAL )); then
  echo "STAGE1_EPOCHS must be >= STABILITY_INTERVAL" >&2
  exit 2
fi
if [[ "$STAGE2_POLICY" != "exclude_full" && "$STAGE2_POLICY" != "include_full" && "$STAGE2_POLICY" != "full_only" ]]; then
  echo "STAGE2_POLICY must be exclude_full, include_full, or full_only" >&2
  exit 2
fi

source "$BASE/.venv/bin/activate"
export PYTHONPATH="$BASE/code/src:$BASE/src:$BASE/code/tools:$BASE/tools"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
cd "$BASE"
mkdir -p "$ROUTES/logs" "$STAGE1/logs" "$STAGE2/logs" "$PREDICTIONS"

printf 'BASE=%s\nDATA=%s\nROUTER_CHECKPOINT=%s\nLIMIT=%s\nSTAGE1_EPOCHS=%s\nSTAGE1_LR=%s\nSTABILITY_INTERVAL=%s\nSTABLE_CHECKPOINT_EPOCHS=%s\nSTAGE2_POLICY=%s\nSTAGE2_EPOCHS=%s\nSTAGE2_LR=%s\n' \
  "$BASE" "$DATA" "$ROUTER_CHECKPOINT" "$LIMIT" "$STAGE1_EPOCHS" "$STAGE1_LR" \
  "$STABILITY_INTERVAL" "$(seq -s, "$STABILITY_INTERVAL" "$STABILITY_INTERVAL" "$STAGE1_EPOCHS")" \
  "$STAGE2_POLICY" "$STAGE2_EPOCHS" "$STAGE2_LR" > "$RUN/launch_config.txt"

if [[ ! -s "$ROUTES/route_only.complete.json" ]]; then
  python -u tools/run_pmgfa_direct_casp.py route-only \
    --data-path "$DATA" \
    --preprocessor "$PREPROCESSOR" \
    --checkpoint "$SOURCE" \
    --model-config "$MODEL_CONFIG" \
    --data-config "$DATA_CONFIG" \
    --run-dir "$ROUTES" \
    --router-mode learned \
    --router-checkpoint "$ROUTER_CHECKPOINT" \
    --limit "$LIMIT" \
    --num-workers 4 \
    --beams 10 \
    --precision bf16 \
    > "$ROUTES/logs/route.log" 2>&1
fi

SAVE_EPOCHS="$(seq -s, 1 "$STAGE1_EPOCHS")"
if [[ ! -s "$STAGE1/adaptation.complete.json" ]]; then
  python -u tools/run_router_teacher_full_casp_stage1.py \
    --data-path "$DATA" \
    --preprocessor "$PREPROCESSOR" \
    --checkpoint "$SOURCE" \
    --model-config "$MODEL_CONFIG" \
    --data-config "$DATA_CONFIG" \
    --route-manifest "$ROUTES" \
    --run-dir "$STAGE1" \
    --batch-size 16 \
    --num-workers 4 \
    --limit "$LIMIT" \
    --epochs "$STAGE1_EPOCHS" \
    --lr "$STAGE1_LR" \
    --update-scope full_model \
    --alignment-mode matched_route \
    --alignment-weight 0 \
    --pseudo-ce-weight 1 \
    --grad-clip 0.8 \
    --precision bf16 \
    --save-epochs "$SAVE_EPOCHS" \
    --seed 3247 \
    --log-every 20 \
    > "$STAGE1/logs/train.log" 2>&1
fi

for epoch in $(seq "$STABILITY_INTERVAL" "$STABILITY_INTERVAL" "$STAGE1_EPOCHS"); do
  prediction="$PREDICTIONS/stage1_epoch_${epoch}.jsonl"
  manifest="$PREDICTIONS/stage1_epoch_${epoch}.manifest.json"
  if [[ ! -s "$prediction" || ! -s "$manifest" ]]; then
    python -u tools/run_stable_checkpoint_dropout_stage2.py generate \
      --data-path "$DATA" \
      --preprocessor "$PREPROCESSOR" \
      --checkpoint "$STAGE1/stage1_epoch_${epoch}.pt" \
      --model-config "$MODEL_CONFIG" \
      --data-config "$DATA_CONFIG" \
      --output "$prediction" \
      --batch-size 64 \
      --num-workers 4 \
      --limit "$LIMIT" \
      --beams 1 \
      --precision bf16 \
      > "$PREDICTIONS/stage1_epoch_${epoch}.log" 2>&1
  fi
done

if [[ ! -s "$STAGE2/adaptation.complete.json" ]]; then
  prediction_args=()
  for epoch in $(seq "$STABILITY_INTERVAL" "$STABILITY_INTERVAL" "$STAGE1_EPOCHS"); do
    prediction_args+=(--prediction-jsonl "$PREDICTIONS/stage1_epoch_${epoch}.jsonl")
  done
  python -u tools/run_stable_checkpoint_dropout_stage2.py train \
    --data-path "$DATA" \
    --preprocessor "$PREPROCESSOR" \
    --checkpoint "$STAGE1/stage1_epoch_${STAGE1_EPOCHS}.pt" \
    --model-config "$MODEL_CONFIG" \
    --data-config "$DATA_CONFIG" \
    "${prediction_args[@]}" \
    --run-dir "$STAGE2" \
    --full-view-policy "$STAGE2_POLICY" \
    --epochs "$STAGE2_EPOCHS" \
    --lr "$STAGE2_LR" \
    --batch-size 16 \
    --num-workers 4 \
    --limit "$LIMIT" \
    --beams 1 \
    --precision bf16 \
    --seed 3247 \
    --save-epochs "$(seq -s, 1 "$STAGE2_EPOCHS")" \
    --log-every 20 \
    > "$STAGE2/logs/train.log" 2>&1
fi

printf '%s\n' "{\"complete\":true,\"dataset\":\"$DATA\",\"limit\":$LIMIT,\"stage1\":\"router_teacher_full_pseudo_ce\",\"stage1_epochs\":$STAGE1_EPOCHS,\"stable_checkpoint_epochs\":\"$(seq -s, "$STABILITY_INTERVAL" "$STABILITY_INTERVAL" "$STAGE1_EPOCHS")\",\"stage2\":\"stable_pseudo_ce_$STAGE2_POLICY\",\"stage2_epochs\":$STAGE2_EPOCHS}" > "$RUN/experiment.complete.json"
echo "pipeline complete: $RUN"
