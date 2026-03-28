"""Shared readability utilities used across training, inference, and preprocessing."""

import nltk
import numpy as np
import textstat

# Readability level thresholds
READABILITY_LEVELS = {
    "elementary": (80, float("inf")),   # Flesch >= 80
    "middle": (60, 80),                 # 60 <= Flesch < 80
    "high": (40, 60),                   # 40 <= Flesch < 60
    "college": (float("-inf"), 40),     # Flesch < 40
}

# Target Flesch scores for each style
CATEGORY_RANGES = {
    "elementary": 90,
    "middle": 70,
    "high": 50,
    "college": 20,
}

# Gaussian normalization parameters
SIGMA = 10
GAUSSIAN_CONSTANT = 1 / (SIGMA * np.sqrt(2 * np.pi))  # ~0.039894228040143274


def get_readability_level(flesch_score):
    """Categorize text based on Flesch reading ease score.

    Returns:
        tuple: (level_name, numerical_value) where numerical_value is 1-4.
    """
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
    return (
        1 / (SIGMA * np.sqrt(2 * np.pi))
        * np.exp(-((value - mean) ** 2) / (2 * SIGMA**2))
        / GAUSSIAN_CONSTANT
    )


def postprocess_text(preds, labels):
    """Strip and sentence-tokenize predictions and labels for ROUGE evaluation."""
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
    return preds, labels


def compute_readability_metrics(texts):
    """Compute readability metrics for a list of texts.

    Returns:
        dict with avg_flesch_score, avg_numerical_level, distribution counts and percentages.
    """
    if not texts:
        return {
            "avg_flesch_score": 0,
            "avg_numerical_level": 0,
            "readability_distribution": {s: 0 for s in READABILITY_LEVELS},
            "elementary_percentage": 0,
            "middle_percentage": 0,
            "high_percentage": 0,
            "college_percentage": 0,
        }

    flesch_scores = [textstat.flesch_reading_ease(text) for text in texts]
    readability_info = [get_readability_level(score) for score in flesch_scores]
    readability_levels = [info[0] for info in readability_info]
    numerical_levels = [info[1] for info in readability_info]

    avg_flesch = np.mean(flesch_scores)
    avg_numerical_level = np.mean(numerical_levels)

    level_counts = {
        "elementary": readability_levels.count("elementary"),
        "middle": readability_levels.count("middle"),
        "high": readability_levels.count("high"),
        "college": readability_levels.count("college"),
    }

    n = len(texts)
    return {
        "avg_flesch_score": round(avg_flesch, 4),
        "avg_numerical_level": round(avg_numerical_level, 4),
        "readability_distribution": level_counts,
        "elementary_percentage": round(level_counts["elementary"] / n * 100, 2),
        "middle_percentage": round(level_counts["middle"] / n * 100, 2),
        "high_percentage": round(level_counts["high"] / n * 100, 2),
        "college_percentage": round(level_counts["college"] / n * 100, 2),
    }
