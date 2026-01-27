from typing import List
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM
from readability import Readability
import numpy as np
import sys
eps = sys.float_info.epsilon
import math

import trlx
from trlx.data.configs import (
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)
from trlx.models.modeling_ppo import PPOConfig
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import textstat

model_dir = '/home/cunhuan/code/controllable-readability-summarization/src/train/mnt/hd3/checkpoints/summary'  # select the checkpoint from the prompt-based methods

config = TRLConfig(
    train=TrainConfig(
        seq_length=1024,
        epochs=10,
        total_steps=100000,
        batch_size=2,
        checkpoint_interval=10000,
        eval_interval=500,
        save_optimizer=False,
        pipeline="PromptPipeline",
        trainer="AcceleratePPOTrainer",
        checkpoint_dir='checkpoint-diverse-readability-only-cons',
        save_best=True
    ),
    model=ModelConfig(
        model_path=model_dir,
        model_arch_type="seq2seq",
        num_layers_unfrozen=-1,
    ),
    tokenizer=TokenizerConfig(
        tokenizer_path=model_dir,
        truncation_side="right",
    ),
    optimizer=OptimizerConfig(
        name="adamw",
        kwargs={
            "lr": 1.0e-5,
            "betas": [0.9, 0.999],
            "eps": 1.0e-8,
            "weight_decay": 1.0e-6,
        },
    ),
    scheduler=SchedulerConfig(
        name="cosine_annealing",
        kwargs={
            "T_max": 10000,
            "eta_min": 1.0e-6,
        },
    ),
    method=PPOConfig(
        name="PPOConfig",
        num_rollouts=512,
        chunk_size=4,
        ppo_epochs=4,
        init_kl_coef=0.05,
        target=6,
        horizon=10000,
        gamma=0.99,
        lam=0.95,
        cliprange=0.2,
        cliprange_value=0.2,
        vf_coef=1.0,
        scale_reward=None,
        ref_mean=None,
        ref_std=None,
        cliprange_reward=10,
        gen_kwargs={
            "max_new_tokens": 256,
        },
        gen_experience_kwargs={
            "max_new_tokens": 256,
            "do_sample": True,
            "temperature": 1.0,
            "top_k": 50,
            "top_p": 0.95,
        },
    ),
)

def get_flesch(text):
    score = textstat.flesch_reading_ease(text)
    return score

import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from collections import defaultdict

class StyleSimilarityScorer:
    def __init__(self, style_word_files):
        """
        初始化评分器
        :param style_word_files: 字典，格式为 {'style1': 'path1.csv', 'style2': 'path2.csv'}
        """
        self.style_words = {}
        self.style_vectors = defaultdict(dict)
        self.vectorizer = TfidfVectorizer(stop_words='english')
        
        # 加载各个风格的词汇表
        for style, filepath in style_word_files.items():
            df = pd.read_csv(filepath)
            self.style_words[style] = set(df['word'].head(1000).tolist())  # 取前1000个词
            
        # 为每个风格创建参考文本（包含该风格的所有特征词）
        self.reference_texts = {
            style: ' '.join(words) 
            for style, words in self.style_words.items()
        }
        
        # 训练TF-IDF向量化器（在所有风格词汇上）
        all_words = []
        for words in self.style_words.values():
            all_words.extend(words)
        self.vectorizer.fit([' '.join(all_words)])
        
        # 为每个风格创建TF-IDF向量
        for style, text in self.reference_texts.items():
            self.style_vectors[style] = self.vectorizer.transform([text])

    def calculate_style_similarity(self, text, target_style):
        """
        计算文本与指定风格词汇库的相似度
        :param text: 要评分的文本
        :param target_style: 要比较的目标风格
        :return: 相似度分数 (0-1)
        """
        if target_style not in self.style_vectors:
            raise ValueError(f"未知的风格: {target_style}. 可用风格: {list(self.style_vectors.keys())}")
        
        # 向量化输入文本
        text_vector = self.vectorizer.transform([text])
        
        # 计算与目标风格的余弦相似度
        similarity = cosine_similarity(text_vector, self.style_vectors[target_style])[0][0]
        
        # 确保分数在0-1范围内（余弦相似度理论上已经是，但可能有浮点误差）
        return max(0.0, min(1.0, similarity))
    
    def calculate_style_probabilities(self, text, temperature=0.01):
        """
        改进版概率计算，通过温度系数保持差异
        :param temperature: 越小则差异越明显 (推荐0.01-0.5)
        """
        raw_scores = {
            style: self.calculate_style_similarity(text, style)
            for style in self.style_vectors.keys()
        }
        scores = np.array(list(raw_scores.values()))
        
        # 带温度系数的softmax
        scaled_scores = scores / temperature
        exp_scores = np.exp(scaled_scores - np.max(scaled_scores))
        probabilities = exp_scores / exp_scores.sum()
        
        return {
            style: float(prob)
            for style, prob in zip(raw_scores.keys(), probabilities)
        }

style_files = {
        'elementary': 'readability_style_differences/elementary_specific_words.csv',
        'middle': 'readability_style_differences/middle_specific_words.csv',
        'high': 'readability_style_differences/high_specific_words.csv',
        'college': 'readability_style_differences/college_specific_words.csv'
    }

readability_scorer = StyleSimilarityScorer(style_files)

def change_category(input_data):
    new_data = []
    categories = [
        ("elementary school students", 90),
        ("middle school students", 70),
        ("high school students", 50),
        ("college students", 20)
    ]
    for text in input_data:
        category = random.choice(categories)
        category_name, _ = category
        # new_text = f"rewrite the following text for {category_name}:\n\n" + text
        new_text = "rewrite the following text for " + category_name + ":\n\n" + text
        new_data.append(new_text)
    return new_data

sigma = 10
def calc_nd(value, mean):
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (value - mean) ** 2 / (2 * sigma ** 2)) / 0.039894228040143274

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import os
global_model_name = "roberta-large"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_layers = 17
cache_dir = "roberta"
global_model = AutoModel.from_pretrained(cache_dir)
global_model = global_model.to(device)
global_tokenizer = AutoTokenizer.from_pretrained(cache_dir)
global_model.encoder.layer = torch.nn.ModuleList([layer for layer in global_model.encoder.layer[:num_layers]])

def encode_text(input_str):
    inputs = global_tokenizer(input_str, padding='max_length', truncation=True, max_length=512, return_tensors="pt", return_token_type_ids=False)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = global_model(**inputs)

    idf = torch.clone(inputs["attention_mask"]).float()
    idf[idf == global_tokenizer.sep_token_id] = 0
    idf[idf == global_tokenizer.cls_token_id] = 0
    idf.div_(idf.sum(dim=1, keepdim=True))

    return F.normalize(outputs[0], dim=-1), inputs["attention_mask"], idf

def compute_bertscore(doc_embedding, doc_masks, doc_idf, summ_embedding, summ_masks, summ_idf):
    batch_size = doc_embedding.size(0)
    sim = torch.bmm(summ_embedding, doc_embedding.transpose(1, 2))
    masks = torch.bmm(summ_masks.unsqueeze(2).float(), doc_masks.unsqueeze(1).float())
    masks = masks.expand(batch_size, -1, -1).contiguous().view_as(sim)

    masks = masks.float().to(sim.device)
    sim = sim * masks

    precision = sim.max(dim=2)[0]
    precision_scale = summ_idf.to(precision.device)
    P = (precision * precision_scale).sum(dim=1)

    summ_zero_mask = summ_masks.sum(dim=1).eq(2)
    if torch.any(summ_zero_mask):
        P = P.masked_fill(summ_zero_mask, 0.0)

    doc_zero_mask = doc_masks.sum(dim=1).eq(2)
    if torch.any(doc_zero_mask):
        P = P.masked_fill(doc_zero_mask, 0.0)

    return P

if __name__ == "__main__":

    def reward_fn(samples: List[str], prompts: List[str], outputs: List[str], tokenizer=None):
        flesch_scores = []
        target_categories = []
        word_level_probs = []
        summaries = []
        docs = []
        category_ranges = {
            "elementary school students": 90,
            "middle school students": 70,
            "high school students": 50,
            "college students": 20
        }

        for (generated_summary, input_doc) in zip(outputs, prompts):
            category = input_doc.split("rewrite the following text for ")[1].split(":")[0]
            target_categories.append(category_ranges[category])
            doc = input_doc.split(":")[1]
            docs.append(doc)
            summaries.append(generated_summary.strip())

            try:
                flesch_scores.append(get_flesch(generated_summary.strip()))
            except:
                flesch_scores.append(0)
        
            probs = readability_scorer.calculate_style_probabilities(generated_summary.strip())
            category_name = category.split(" ")[0]
            word_level_prob = probs[category_name]
            word_level_probs.append(word_level_prob)

        all_bertscore_scores = []
        for doc, summary in zip(docs, summaries):
            bertscore_input_embedding, bertscore_input_attention_mask, bertscore_input_idf = encode_text([doc])
            bertscore_output_embedding, bertscore_output_attention_mask, bertscore_output_idf = encode_text([summary])

            bertscore_scores = compute_bertscore(
                bertscore_input_embedding,
                bertscore_input_attention_mask,
                bertscore_input_idf,
                bertscore_output_embedding,
                bertscore_output_attention_mask,
                bertscore_output_idf,
            )
            bertscore_scores = bertscore_scores.tolist()
            all_bertscore_scores.extend(bertscore_scores)

        assert len(target_categories) == len(flesch_scores) == len(all_bertscore_scores) == len(word_level_probs)

        flesch_scores = [calc_nd(fs, o_fs) for fs, o_fs in zip(flesch_scores, target_categories)]

        readability_weight = 0
        bertscore_weight = 1
        word_level_weight = 0
        
        flesch_scores = torch.tensor(flesch_scores)
        all_bertscore_scores = torch.tensor(all_bertscore_scores)
        word_level_probs = torch.tensor(word_level_probs)

        flesch_scores = readability_weight * flesch_scores + bertscore_weight * all_bertscore_scores + word_level_weight * word_level_probs
        flesch_scores = flesch_scores.tolist()

        return flesch_scores

    train_file = '../../data/train_summary_prompt_parallel.json'
    validation_file = '../../data/val_summary_prompt_parallel.json'
    data_files = {"train": train_file, "validation": validation_file}
    dataset = load_dataset("json", data_files=data_files)
    dataset['train'] = dataset['train'].shuffle(seed=42)
    dataset['validation'] = dataset['validation'].shuffle(seed=42)

    validation_examples = 2000
    val_prompts = [prompt for prompt in dataset['validation']["input_text"][0:validation_examples]]
    val_summaries = dataset['validation']["output_text"][0:validation_examples]
    val_prompts = change_category(val_prompts)
    assert len(val_prompts) == len(val_summaries)

    prompts = dataset['train']["input_text"]
    summaries = dataset['train']["output_text"]
    prompts = [prompt for prompt in prompts]
    prompts = change_category(prompts)
    assert len(prompts) == len(summaries)

    tokenizer = AutoTokenizer.from_pretrained(config.model.model_path)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "right"
    tokenizer.sep_token = "<sep>"
    prompt_label = {}
    max_length = config.train.seq_length

    trlx.train(
        reward_fn=reward_fn,
        prompts=prompts,
        eval_prompts=val_prompts,
        config=config,
    )