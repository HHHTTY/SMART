#!/usr/bin/env bash
set -euo pipefail
ROOT=/hpc2hdd/home/aimslab/ChengtangZhan/ttt/SpectraLLM
RUN=$ROOT/runs/sdbs_full3669_spectrallm_audit_20260922
source /hpc2hdd/home/aimslab/miniconda3/bin/activate wqx_spectrallm
export PYTHONNOUSERSITE=1
export SWANLAB_MODE=local
cat > "$RUN/unimodal_smoke/dataset_info.json" <<'JSON'
{
  "hnmr_test32": {"file_name": "hnmr_test32.jsonl", "columns": {"prompt": "prompt", "response": "response", "system": "system"}},
  "cnmr_test32": {"file_name": "cnmr_test32.jsonl", "columns": {"prompt": "prompt", "response": "response", "system": "system"}},
  "ir_test32": {"file_name": "ir_test32.jsonl", "columns": {"prompt": "prompt", "response": "response", "system": "system"}},
  "ms_test32": {"file_name": "ms_test32.jsonl", "columns": {"prompt": "prompt", "response": "response", "system": "system"}}
}
JSON
for mode in hnmr cnmr ir ms; do
  cat > "$RUN/${mode}_predict.yaml" <<YAML
model_name_or_path: $ROOT/checkpoints/SpectraLLM_32B
stage: sft
do_predict: true
finetuning_type: lora
template: qwen3
adapter_name_or_path: $ROOT/checkpoints/widthIrplus_msg_qm9s_mb_msd
quantization_method: hqq
dataset_dir: $RUN/unimodal_smoke
eval_dataset: ${mode}_test32
cutoff_len: 4096
max_samples: 32
preprocessing_num_workers: 8
per_device_eval_batch_size: 1
predict_with_generate: true
max_new_tokens: 256
output_dir: $RUN/unimodal_${mode}_prediction
flash_attn: auto
trust_remote_code: true
report_to: []
YAML
  nohup llamafactory-cli train "$RUN/${mode}_predict.yaml" > "$RUN/unimodal_${mode}.log" 2>&1
done
