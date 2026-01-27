#!/usr/bin/env python
# coding=utf-8

import logging
import os
import json
import nltk
import numpy as np
from datasets import load_dataset
import evaluate
import textstat
from langchain_openai import ChatOpenAI
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import multiprocessing
import concurrent.futures
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler()],
    level=logging.INFO,
)

# Initialize llm with gpt-4o-mini
llm = ChatOpenAI(
    model="gpt-4o-mini",
    base_url="https://api.chatfire.cn/v1/",
    api_key="",
)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    nltk.download("punkt", quiet=True)

# Configuration
TEST_FILE = "../train/data/val_summary_parallel.json"
OUTPUT_DIR_BASE = "outputs_readability_gpt_all"
TFIDF_LEXICON_DIR = "readability_style_differences"
TEXT_COLUMN = "input_text"
SUMMARY_COLUMN = "output_text"
MAX_WORKERS = multiprocessing.cpu_count() * 2
MAX_WORDS = 1000  # Limit to top 100 words per lexicon
PROMPTS = [
    {"id": 1, "text": "Rewrite the following text for elementary school students:\n\n", "style_key": "elementary"},
    {"id": 2, "text": "Rewrite the following text for middle school students:\n\n", "style_key": "middle"},
    {"id": 3, "text": "Rewrite the following text for high school students:\n\n", "style_key": "high"},
    {"id": 4, "text": "Rewrite the following text for college students:\n\n", "style_key": "college"},
]

# Target Flesch scores for each style
CATEGORY_RANGES = {
    "elementary": 90,
    "middle": 70,
    "high": 50,
    "college": 20
}

# Gaussian normalization parameters
SIGMA = 10
GAUSSIAN_CONSTANT = 0.039894228040143274  # 1 / (sqrt(2 * pi) * sigma)

# Load ROUGE metric
metric_rouge = evaluate.load("rouge")

def get_readability_level(flesch_score):
    """Categorize text based on Flesch reading ease score and return both category and numerical value."""
    if flesch_score >= 80:
        return ("elementary", 1)
    elif 60 <= flesch_score < 80:
        return ("middle", 2)
    elif 40 <= flesch_score < 60:
        return ("high", 3)
    else:
        return ("college", 4)

def calc_nd(value, mean):
    """Calculate normalized Gaussian density for Flesch score."""
    return (1 / (SIGMA * np.sqrt(2 * np.pi)) * 
            np.exp(- (value - mean) ** 2 / (2 * SIGMA ** 2)) / GAUSSIAN_CONSTANT)

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
    return preds, labels

def load_tfidf_lexicons(directory):
    """Load TF-IDF lexicons from CSV files, selecting top 100 words by TF-IDF score."""
    lexicons = {}
    for style in ["elementary", "middle", "high", "college"]:
        file_path = os.path.join(directory, f"{style}.csv")
        try:
            data = pd.read_csv(file_path)
            if "word" not in data.columns or "tfidf" not in data.columns:
                logger.warning(f"TF-IDF lexicon file {file_path} missing 'word' or 'tfidf' column")
                continue
            # Sort by tfidf in descending order and select top 100 words
            data = data.sort_values(by="tfidf", ascending=False).head(MAX_WORDS)
            lexicons[style] = dict(zip(data["word"].astype(str), data["tfidf"].astype(float)))
            logger.info(f"Loaded {len(lexicons[style])} words for style {style} from {file_path}")
        except FileNotFoundError:
            logger.warning(f"TF-IDF lexicon file {file_path} not found")
        except Exception as e:
            logger.warning(f"Error loading TF-IDF lexicon file {file_path}: {e}")
    return lexicons

def compute_style_and_reward(texts, style_key, lexicons):
    """Compute Style Match Score (0-1), Flesch score, and combined reward."""
    if not lexicons or style_key not in lexicons:
        logger.warning(f"No TF-IDF lexicon for style {style_key}; returning zero scores")
        return {
            "avg_sms": 0,
            "avg_flesch": 0,
            "avg_reward": 0,
            "scores": [{"sms": 0, "flesch": 0, "reward": 0} for _ in texts]
        }
    
    # Prepare TF-IDF lexicon vector
    lexicon = lexicons[style_key]
    words = list(lexicon.keys())
    lexicon_vector = np.array([lexicon.get(word, 0) for word in words])
    
    # Compute text vectors using lexicon vocabulary
    vectorizer = CountVectorizer(vocabulary=words)
    text_vectors = vectorizer.fit_transform(texts).toarray()
    
    sms_scores = []
    flesch_scores = []
    reward_scores = []
    target_flesch = CATEGORY_RANGES[style_key]
    
    for text, text_vector in zip(texts, text_vectors):
        # Compute SMS (normalized to 0-1)
        if np.sum(text_vector) == 0 or np.sum(lexicon_vector) == 0:
            sms = 0
        else:
            sms = cosine_similarity([text_vector], [lexicon_vector])[0][0]
        sms_scores.append(sms)
        
        # Compute Flesch score
        flesch_raw = textstat.flesch_reading_ease(text)
        flesch_normalized = calc_nd(flesch_raw, target_flesch)
        flesch_scores.append(flesch_normalized)
        
        # Compute combined reward
        reward = 0.5 * sms + 0.5 * flesch_normalized
        reward_scores.append(reward)
    
    return {
        "avg_sms": round(np.mean(sms_scores), 4) if sms_scores else 0,
        "avg_flesch": round(np.mean(flesch_scores), 4) if flesch_scores else 0,
        "avg_reward": round(np.mean(reward_scores), 4) if reward_scores else 0,
        "scores": [
            {
                "sms": round(sms, 4),
                "flesch": round(flesch, 4),
                "reward": round(reward, 4)
            }
            for sms, flesch, reward in zip(sms_scores, flesch_scores, reward_scores)
        ]
    }

def compute_readability_metrics(texts):
    """Compute readability metrics for a list of texts."""
    flesch_scores = [textstat.flesch_reading_ease(text) for text in texts]
    readability_info = [get_readability_level(score) for score in flesch_scores]
    readability_levels = [info[0] for info in readability_info]
    numerical_levels = [info[1] for info in readability_info]
    
    # Calculate average scores
    avg_flesch = np.mean(flesch_scores) if flesch_scores else 0
    avg_numerical_level = np.mean(numerical_levels) if numerical_levels else 0
    
    # Calculate distribution of readability levels
    level_counts = {
        "elementary": readability_levels.count("elementary"),
        "middle": readability_levels.count("middle"),
        "high": readability_levels.count("high"),
        "college": readability_levels.count("college")
    }
    
    return {
        "avg_flesch_score": round(avg_flesch, 4),
        "avg_numerical_level": round(avg_numerical_level, 4),
        "readability_distribution": level_counts,
        "elementary_percentage": round(level_counts["elementary"] / len(texts) * 100, 2) if texts else 0,
        "middle_percentage": round(level_counts["middle"] / len(texts) * 100, 2) if texts else 0,
        "high_percentage": round(level_counts["high"] / len(texts) * 100, 2) if texts else 0,
        "college_percentage": round(level_counts["college"] / len(texts) * 100, 2) if texts else 0,
    }

def compute_metrics(predictions, labels, style_key, lexicons):
    decoded_preds = predictions
    decoded_labels = labels

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    # Compute ROUGE scores
    rouge_result = metric_rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    rouge_result = {k: round(v * 100, 4) for k, v in rouge_result.items()}

    # Compute readability metrics for predictions
    readability_result = compute_readability_metrics(decoded_preds)
    
    # Compute readability metrics for original labels (for comparison)
    label_readability = compute_readability_metrics(decoded_labels)
    readability_result["label_avg_flesch_score"] = label_readability["avg_flesch_score"]
    readability_result["label_avg_numerical_level"] = label_readability["avg_numerical_level"]
    readability_result["label_readability_distribution"] = label_readability["readability_distribution"]

    # Compute style and reward metrics
    style_metrics = compute_style_and_reward(decoded_preds, style_key, lexicons)
    
    result = {}
    result.update(rouge_result)
    result.update(readability_result)
    result.update({
        "avg_sms": style_metrics["avg_sms"],
        "avg_flesch_normalized": style_metrics["avg_flesch"],
        "avg_reward": style_metrics["avg_reward"],
        "style_scores": style_metrics["scores"]
    })
    return result

def generate_summary(text, prompt):
    full_prompt = f"{prompt}{text}"
    try:
        response = llm.invoke(full_prompt)
        return response.content.strip()
    except Exception as e:
        logger.error(f"Error generating summary for text: '{text[:50]}...': {e}")
        return ""

def process_example(example, prompt):
    input_text = example[TEXT_COLUMN]
    summary = generate_summary(input_text, prompt)
    return {"prediction": summary, "label": example[SUMMARY_COLUMN]}

def main():
    # Load TF-IDF lexicons
    logger.info("Loading TF-IDF lexicons...")
    lexicons = load_tfidf_lexicons(TFIDF_LEXICON_DIR)
    if not lexicons:
        logger.error("No TF-IDF lexicons loaded; exiting")
        return

    # Load test dataset
    logger.info(f"Loading dataset from {TEST_FILE}...")
    raw_datasets = load_dataset("json", data_files={"test": TEST_FILE})
    test_dataset = raw_datasets["test"]
    total_samples = len(test_dataset)
    logger.info(f"Dataset loaded. Total samples: {total_samples}")

    # Process each prompt
    for prompt_info in PROMPTS:
        prompt_id = prompt_info["id"]
        prompt_text = prompt_info["text"]
        style_key = prompt_info["style_key"]
        logger.info(f"Processing prompt {prompt_id} (style: {style_key}): {prompt_text.strip()}")

        output_dir = os.path.join(OUTPUT_DIR_BASE, str(prompt_id))
        os.makedirs(output_dir, exist_ok=True)

        predictions = []
        labels = []

        # Parallel processing with ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_index = {
                executor.submit(process_example, example, prompt_text): idx
                for idx, example in enumerate(test_dataset)
            }

            with tqdm(total=total_samples, desc=f"Prompt {prompt_id}: {prompt_text.strip()[:30]}...", unit="sample") as pbar:
                for future in concurrent.futures.as_completed(future_to_index):
                    try:
                        result = future.result()
                        predictions.append(result["prediction"])
                        labels.append(result["label"])
                    except Exception as e:
                        logger.error(f"Error processing example: {e}")
                        predictions.append("")
                        labels.append(test_dataset[future_to_index[future]][SUMMARY_COLUMN])
                    pbar.update(1)

        # Compute metrics
        metrics = compute_metrics(predictions, labels, style_key, lexicons)
        metrics["predict_samples"] = total_samples
        logger.info(f"Metrics for prompt {prompt_id}: {metrics}")

        # Save predictions
        output_prediction_file = os.path.join(output_dir, "generated_predictions.txt")
        with open(output_prediction_file, "w") as writer:
            writer.write("\n".join(predictions))

        # Save metrics
        output_metrics_file = os.path.join(output_dir, "predict_metrics.json")
        with open(output_metrics_file, "w") as writer:
            json.dump(metrics, writer, indent=4)

if __name__ == "__main__":
    main()