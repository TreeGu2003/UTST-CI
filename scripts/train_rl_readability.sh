#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh

conda activate readability_summ
export TOKENIZERS_PARALLELISM=true

# accelerate launch --config_file accelerate_config.yaml train.py
accelerate launch --config_file accelerate_config.yaml train_readability_ppo.py

conda deactivate