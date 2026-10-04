#!/usr/bin/env bash
set -euo pipefail

BASE=/hpc2hdd/home/aimslab/ChengtangZhan/ttt/paper_multitask_ms_snapshot_20260914
ROOT=/hpc2hdd/home/aimslab/ChengtangZhan/ttt
DATA=/hpc2hdd/home/aimslab/ChengtangZhan/Dataset/chemotion_tokenized_datasets/Chemotion_final_ir1800_20260922_peakpicked_model_compatible_v8_densec03_strictdense_hcpeakpick_csolventclean/test.parquet
PREPROCESSOR="$BASE/preprocessor_chemotion_compat_v3_ms.pkl"
SOURCE="$BASE/checkpoints/epoch_24-step_122175.ckpt"
MODEL_CONFIG="$BASE/configs/model/custom_model_paper_multitask_ms.yaml"
DATA_CONFIG="$BASE/configs/data/multimodal/paper_multitask_ms_generation.yaml"
ROUTES="$BASE/runs/chemotion_v8_strict_compatv3_router_profiled_v6_routes_20260923_gpu3"
RUN="$BASE/runs/chemotion_v8_strict_compatv3_router_teacher_ttt_earlystop_lr3e5_unbounded_20260923_gpu3"
STAGE1="$RUN/stage1"
PREDICTIONS="$RUN/predictions"
STAGE2="$RUN/stage2_exclude_full"
ROUTER_STATE="$BASE/runs/router_profiled_v6_streaming_50k_beam5_20260921/pilot_router_after_gpu3_20260921_202146/router_pilot.pt"

source "$ROOT/.venv/bin/activate"
cd "$BASE"
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$BASE/code/src:$BASE/src:$BASE/code/tools:$BASE/tools"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$STAGE1/logs" "$PREDICTIONS" "$STAGE2/logs"

test -s "$ROUTES/route_only.complete.json"
test -s "$ROUTES/route_only_manifest.json"
test -s "$PREPROCESSOR"
test -s "$SOURCE"
test -s "$ROUTER_STATE"

if [[ ! -s "$STAGE1/adaptation.complete.json" ]]; then
  python -u tools/run_router_teacher_full_casp_stage1.py \
    --data-path "$DATA" --preprocessor "$PREPROCESSOR" --checkpoint "$SOURCE" \
    --model-config "$MODEL_CONFIG" --data-config "$DATA_CONFIG" \
    --route-manifest "$ROUTES" --run-dir "$STAGE1" \
    --ms-column ms_spectrum --ir-column ir_spectra \
    --batch-size 16 --num-workers 4 --limit 1995 --epochs 0 --lr 3e-5 \
    --update-scope full_model --alignment-mode matched_route \
    --alignment-weight 0 --pseudo-ce-weight 1 --grad-clip 0.8 --precision bf16 \
    --save-epochs "" --checkpoint-interval-epochs 3 \
    --early-stop-patience 5 --early-stop-min-epochs 6 --early-stop-min-delta 0.001 \
    --seed 3247 --log-every 20 2>&1 | tee "$STAGE1/logs/train.log"
fi

completed_epoch="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["epoch"])' "$STAGE1/adaptation.complete.json")"
stable_epochs=()
for ((epoch=3; epoch<=completed_epoch; epoch+=3)); do
  checkpoint="$STAGE1/stage1_epoch_${epoch}.pt"
  prediction="$PREDICTIONS/stage1_epoch_${epoch}.jsonl"
  if [[ -s "$checkpoint" ]]; then
    stable_epochs+=("$epoch")
    if [[ ! -s "$prediction" || ! -s "${prediction%.jsonl}.manifest.json" ]]; then
      python -u tools/run_stable_checkpoint_dropout_stage2.py generate \
        --data-path "$DATA" --preprocessor "$PREPROCESSOR" --checkpoint "$checkpoint" \
        --model-config "$MODEL_CONFIG" --data-config "$DATA_CONFIG" \
        --ms-column ms_spectrum --ir-column ir_spectra \
        --output "$prediction" --batch-size 64 --num-workers 4 \
        --limit 1995 --beams 1 --precision bf16 \
        2>&1 | tee "$PREDICTIONS/stage1_epoch_${epoch}.log"
    fi
  fi
done

if (( ${#stable_epochs[@]} < 2 )); then
  echo "Stage 1 early-stopped before two interval-3 snapshots; stable Stage 2 cannot run." >&2
  exit 3
fi

if [[ ! -s "$STAGE2/adaptation.complete.json" ]]; then
  prediction_args=()
  for epoch in "${stable_epochs[@]}"; do
    prediction_args+=(--prediction-jsonl "$PREDICTIONS/stage1_epoch_${epoch}.jsonl")
  done
  python -u tools/run_stable_checkpoint_dropout_stage2.py train \
    --data-path "$DATA" --preprocessor "$PREPROCESSOR" \
    --checkpoint "$STAGE1/stage1_epoch_${completed_epoch}.pt" \
    --model-config "$MODEL_CONFIG" --data-config "$DATA_CONFIG" \
    --ms-column ms_spectrum --ir-column ir_spectra "${prediction_args[@]}" \
    --run-dir "$STAGE2" --full-view-policy exclude_full \
    --epochs 20 --early-stop-patience 3 --early-stop-min-epochs 3 \
    --early-stop-min-delta 0.002 --lr 1e-5 --weight-decay 0 \
    --grad-clip 0.8 --batch-size 16 --num-workers 4 --limit 1995 \
    --beams 1 --precision bf16 --seed 3247 \
    --save-epochs 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20 \
    --log-every 20 2>&1 | tee "$STAGE2/logs/train.log"
fi

python - "$RUN/experiment.complete.json" "$DATA" "$completed_epoch" "${stable_epochs[*]}" <<'PY'
import json, sys
out, data, epoch, stable = sys.argv[1:]
json.dump({"complete": True, "dataset": data, "limit": 1995,
           "stage1_completed_epoch": int(epoch),
           "stable_checkpoint_epochs": [int(x) for x in stable.split()],
           "stage2": "stable_pseudo_ce_exclude_full"}, open(out, "w"), indent=2)
PY
echo "pipeline complete: $RUN"
