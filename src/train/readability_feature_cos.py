"""Style similarity scoring using TF-IDF cosine similarity.

This module re-exports StyleSimilarityScorer from src/utils for backwards compatibility,
and provides a demo when run as a script.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.style_scorer import StyleSimilarityScorer

if __name__ == "__main__":
    style_files = {
        'elementary': 'readability_style_differences/elementary_specific_words.csv',
        'middle': 'readability_style_differences/middle_specific_words.csv',
        'high': 'readability_style_differences/high_specific_words.csv',
        'college': 'readability_style_differences/college_specific_words.csv'
    }

    scorer = StyleSimilarityScorer(style_files)

    test_text = "Annette Miller, a member of Fit Nation, has a new motto for her life. She believes that there is a big difference between being inspired by someone and comparing ourselves to them. Miller says that when we compare ourselves to others, it makes it hard to see things clearly."

    print("Similarity scores:")
    for style in style_files.keys():
        score = scorer.calculate_style_similarity(test_text, style)
        print(f"  {style}: {score:.4f}")

    print("\nSoftmax-normalized probabilities:")
    probs = scorer.calculate_style_probabilities(test_text, 1)
    for style, prob in probs.items():
        print(f"  {style}: {prob:.4f}")

    print(f"\nProbability sum: {sum(probs.values()):.4f}")
