<h1 align="center">
  <strong>Unsupervised Text Style Transfer for Controllable Intensity</strong>
</h1>

<div align="center">

[![arXiv](https://img.shields.io/badge/Arxiv-2601.01060-b31b1b.svg?logo=arXiv)](http://arxiv.org/abs/2601.01060)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)

</div>

# Quick Links
+ [Overview](#overview)
+ [Requirements](#requirements)
+ [Data Preparation](#data-preparation)
+ [Experiments](#experiments)
+ [Results](#results)
+ [Citation](#citation)

## Overview

This repository contains the implementation of **Unsupervised Text Style Transfer for Controllable Intensity**, a framework for controlling the intensity of style transfer in text generation tasks. The project focuses on two main style transfer tasks:

- **Readability Transfer**: Adjusting text complexity and readability levels
- **Sentiment Transfer**: Modifying sentiment polarity with controllable intensity

The framework supports both Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) approaches using the trlx library.

<div align="center">
  <img src="figs/method_overview.png" alt="Method Overview" width="800"/>
  <p><em>Figure 1: Overview of the proposed method for controllable intensity text style transfer</em></p>
</div>

### Model Architecture

<div align="center">
  <img src="figs/model_architecture.png" alt="Model Architecture" width="800"/>
  <p><em>Figure 2: Model architecture</em></p>
</div>

## Requirements

- Python >= 3.8
- PyTorch 2.0.1 with CUDA 11.8
- 2+ GPUs recommended for distributed training

### Installation

```bash
# Create conda environment (recommended)
conda create -n readability_summ python=3.8
conda activate readability_summ

# Install PyTorch with CUDA 11.8
pip install torch==2.0.1+cu118 -f https://download.pytorch.org/whl/cu118/torch_stable.html

# Install dependencies
pip install -r requirements.txt
```

Main dependencies include:
- Transformers 4.46.3
- trlx 0.7.0
- DeepSpeed (for distributed training)
- Various NLP evaluation libraries (bert_score, textstat, etc.)

### Environment Variables

The following environment variables can be used to configure the project:

| Variable | Description | Default |
|----------|-------------|---------|
| `SFT_MODEL_DIR` | Path to the SFT checkpoint for RL training | `checkpoints/sft_readability` |
| `OPENAI_API_KEY` | API key for GPT-based data generation | (empty) |
| `OPENAI_BASE_URL` | Base URL for the OpenAI-compatible API | `https://api.openai.com/v1/` |
| `TRAIN_FILE` | Path to training data | `data/train_summary_prompt_parallel.json` |
| `VAL_FILE` | Path to validation data | `data/val_summary_prompt_parallel.json` |

## Data Preparation

### Readability

The data should be prepared in JSON format with parallel text pairs. Expected format:

```json
{
  "input_text_prompt": "source text with prompt",
  "output_text": "target text with desired readability"
}
```

Place your data files in the `data/` directory:
- `train_summary_prompt_parallel.json` - Training data
- `val_summary_prompt_parallel.json` - Validation data

You can use the preprocessing script to generate readability-controlled data:

```bash
# Set your API key first
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://api.openai.com/v1/"

python src/preprocess/generate_readability_by_gpt.py
```

### Sentiment

Similar data preparation process for sentiment transfer tasks. Prepare parallel data with source and target sentiment levels.

## Experiments

### Training

#### 1. Supervised Fine-Tuning (SFT)

Train the model using supervised fine-tuning:

```bash
bash scripts/train_sft_readability.sh
```

Key parameters (configurable via environment variables or script editing):
- Learning rate: 1e-4
- Batch size: 8 per device
- Max source length: 1024
- Training epochs: 20

#### 2. Reinforcement Learning (RL)

Train with PPO reinforcement learning for better controllability:

```bash
# Set the SFT checkpoint path
export SFT_MODEL_DIR="checkpoints/sft_readability"

bash scripts/train_rl_readability.sh
```

### Inference

Run inference on test data:

```bash
# SFT model inference
bash scripts/inference_sft_readability.sh

# RL model inference (runs all 4 readability levels in parallel)
bash scripts/inference_rl_readability.sh
```

### Evaluation

The project includes various evaluation metrics:
- **ROUGE scores** for text quality
- **BERTScore** for semantic similarity
- **Readability metrics**: Flesch-Kincaid, Gunning Fog, Coleman-Liau
- **Style similarity**: TF-IDF based feature extraction and cosine similarity

### Example Outputs

<div align="center">
  <img src="figs/examples.png" alt="Example Outputs" width="800"/>
  <p><em>Figure 3: Examples of text style transfer with different intensity levels</em></p>
</div>

## Results

<div align="center">
  <img src="figs/results_1.png" alt="Results 1" width="800"/>
  <p><em>Figure 4: Experimental results - Part 1</em></p>
</div>

<div align="center">
  <img src="figs/results_2.png" alt="Results 2" width="800"/>
  <p><em>Figure 5: Experimental results - Part 2</em></p>
</div>

## Project Structure

```
UTST-CI/
├── data/                # Training and validation data (not tracked)
├── figs/                # Figures and visualizations
├── scripts/             # Training and inference scripts
│   ├── train_sft_readability.sh
│   ├── train_rl_readability.sh
│   ├── inference_sft_readability.sh
│   └── inference_rl_readability.sh
├── src/
│   ├── utils/           # Shared utilities
│   │   ├── readability_utils.py   # Readability metrics and helpers
│   │   └── style_scorer.py        # TF-IDF style similarity scorer
│   ├── preprocess/      # Data preprocessing scripts
│   ├── train/           # Training scripts (SFT + RL)
│   └── inference/       # Inference scripts
├── requirements.txt
└── README.md
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{gu2026unsupervised,
  title={Unsupervised Text Style Transfer for Controllable Intensity},
  author={Gu, Shuhuan and Tao, Wenbiao and Ma, Xinchen and He, Kangkang and Guo, Ye and Li, Xiang and Lan, Yunshi},
  journal={arXiv preprint arXiv:2601.01060},
  year={2026},
  url={https://arxiv.org/abs/2601.01060}
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
