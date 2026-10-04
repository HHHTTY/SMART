#!/usr/bin/env bash
set -euo pipefail
ROOT=/hpc2hdd/home/aimslab/ChengtangZhan/ttt/SpectraLLM
RUN=$ROOT/runs/sdbs_full3669_spectrallm_audit_20260922
DATA=$RUN/multimodal_official_exact
source /hpc2hdd/home/aimslab/miniconda3/bin/activate wqx_spectrallm
export PYTHONNOUSERSITE=1
export SWANLAB_MODE=local
cat > "$DATA/dataset_info.json" <<'JSON'
{"multimodal_test32_official_exact":{"file_name":"test32.jsonl","columns":{"prompt":"prompt","response":"response","system":"system"}}}
JSON
cat > "$RUN/multimodal_official_exact_predict.yaml" <<YAML
model_name_or_path: $ROOT/checkpoints/SpectraLLM_32B
stage: sft
do_predict: true
finetuning_type: lora
template: qwen3
adapter_name_or_path: $ROOT/checkpoints/widthIrplus_msg_qm9s_mb_msd
quantization_method: hqq
dataset_dir: $DATA
eval_dataset: multimodal_test32_official_exact
cutoff_len: 4096
max_samples: 32
preprocessing_num_workers: 8
per_device_eval_batch_size: 1
predict_with_generate: true
max_new_tokens: 256
output_dir: $RUN/multimodal_official_exact_prediction
flash_attn: auto
trust_remote_code: true
report_to: []
YAML
llamafactory-cli train "$RUN/multimodal_official_exact_predict.yaml" > "$RUN/multimodal_official_exact.log" 2>&1
