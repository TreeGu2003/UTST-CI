#!/bin/bash
# Run PPO model inference for all 4 readability levels
# Usage: bash scripts/inference_rl_readability.sh
set -euo pipefail

source ~/anaconda3/etc/profile.d/conda.sh
conda activate readability_summ

export TOKENIZERS_PARALLELISM=false

VAL_FILE=${VAL_FILE:-"data/val_summary_parallel.json"}
MODEL_PATH=${MODEL_PATH:-"checkpoints/rl_readability/best_checkpoint/hf_model"}

# Task 1: Elementary (target Flesch ~90)
OUTPUT_DIR='outputs/1/'
CUDA_VISIBLE_DEVICES=0 python -u src/inference/inference_rl_readability.py \
 --ppo_checkpoint ${MODEL_PATH} \
 --output_dir ${OUTPUT_DIR} \
 --text_column input_text \
 --summary_column output_text \
 --test_file ${VAL_FILE} \
 --max_source_length 1024 \
 --val_max_target_length 128 \
 --source_prefix "rewrite the following text for elementary school students:\n\n" \
 --num_beams 4 \
 --overwrite_cache &

P1=$!

# Task 2: Middle school (target Flesch ~70)
OUTPUT_DIR='outputs/2/'
CUDA_VISIBLE_DEVICES=1 python -u src/inference/inference_rl_readability.py \
 --ppo_checkpoint ${MODEL_PATH} \
 --output_dir ${OUTPUT_DIR} \
 --text_column input_text \
 --summary_column output_text \
 --test_file ${VAL_FILE} \
 --max_source_length 1024 \
 --val_max_target_length 128 \
 --source_prefix "rewrite the following text for middle school students:\n\n" \
 --num_beams 4 \
 --overwrite_cache &

P2=$!

wait $P1 $P2

# Task 3: High school (target Flesch ~50)
OUTPUT_DIR='outputs/3/'
CUDA_VISIBLE_DEVICES=0 python -u src/inference/inference_rl_readability.py \
 --ppo_checkpoint ${MODEL_PATH} \
 --output_dir ${OUTPUT_DIR} \
 --text_column input_text \
 --summary_column output_text \
 --test_file ${VAL_FILE} \
 --max_source_length 1024 \
 --val_max_target_length 128 \
 --source_prefix "rewrite the following text for high school students:\n\n" \
 --num_beams 4 \
 --overwrite_cache &

P3=$!

# Task 4: College (target Flesch ~20)
OUTPUT_DIR='outputs/4/'
CUDA_VISIBLE_DEVICES=1 python -u src/inference/inference_rl_readability.py \
 --ppo_checkpoint ${MODEL_PATH} \
 --output_dir ${OUTPUT_DIR} \
 --text_column input_text \
 --summary_column output_text \
 --test_file ${VAL_FILE} \
 --max_source_length 1024 \
 --val_max_target_length 128 \
 --source_prefix "rewrite the following text for college students:\n\n" \
 --num_beams 4 \
 --overwrite_cache &

P4=$!

wait $P3 $P4

conda deactivate
