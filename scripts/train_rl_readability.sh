#!/bin/bash
# Train RL (PPO) model for readability-controlled text style transfer
# Usage: bash scripts/train_rl_readability.sh
set -euo pipefail

source ~/anaconda3/etc/profile.d/conda.sh
conda activate readability_summ

export TOKENIZERS_PARALLELISM=true

accelerate launch --config_file src/train/accelerate_config.yaml src/train/train_rl_readability.py

conda deactivate
