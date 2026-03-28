#!/bin/bash
# Train SFT model for readability-controlled text style transfer
# Usage: bash scripts/train_sft_readability.sh
set -euo pipefail

source ~/anaconda3/etc/profile.d/conda.sh
conda activate readability_summ

FOLDER_OUTPUT=${OUTPUT_DIR:-"checkpoints/sft_readability"}
TRAIN_FILE=${TRAIN_FILE:-"data/train_summary_prompt_parallel.json"}
VAL_FILE=${VAL_FILE:-"data/val_summary_prompt_parallel.json"}
MODEL_NAME=${MODEL_NAME:-"rl/trlx/pretrained_models"}

deepspeed --master_port 61002 --include localhost:0,1 src/train/train_sft_readability.py \
 --model_name_or_path ${MODEL_NAME} \
 --output_dir ${FOLDER_OUTPUT} \
 --text_column input_text_prompt \
 --summary_column output_text \
 --train_file ${TRAIN_FILE} \
 --validation_file ${VAL_FILE} \
 --learning_rate 1e-4 \
 --max_source_length 1024 \
 --source_prefix "" \
 --num_train_epochs 20 \
 --logging_steps 200 \
 --preprocessing_num_workers 100 \
 --eval_steps 10000 \
 --save_steps 10000 \
 --save_total_limit 2 \
 --evaluation_strategy "steps" \
 --per_device_train_batch_size 8 \
 --per_device_eval_batch_size 8 \
 --metric_for_best_model "rouge1" \
 --load_best_model_at_end \
 --predict_with_generate \
 --deepspeed src/train/ds_config_stage3_fb16.json \
 --bf16 \
 --bf16_full_eval \
 --do_train

conda deactivate
