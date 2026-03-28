#!/bin/bash
# Run SFT model inference on test data
# Usage: bash scripts/inference_sft_readability.sh
set -euo pipefail

source ~/anaconda3/etc/profile.d/conda.sh
conda activate readability_summ

MODEL_PATH=${MODEL_PATH:-"checkpoints/sft_readability"}
TEST_FILE=${TEST_FILE:-"data/val_summary_prompt_parallel.json"}
OUTPUT_DIR=${OUTPUT_DIR:-"outputs/sft/"}

python src/inference/inference_sft_readability.py \
 --model_name_or_path ${MODEL_PATH} \
 --output_dir ${OUTPUT_DIR} \
 --text_column input_text_prompt \
 --summary_column output_text \
 --test_file ${TEST_FILE} \
 --max_source_length 1024 \
 --source_prefix "" \
 --per_device_eval_batch_size 8 \
 --predict_with_generate \
 --bf16 \
 --do_predict

conda deactivate
