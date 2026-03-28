"""Style similarity scoring using TF-IDF cosine similarity."""

import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class StyleSimilarityScorer:
    """Score text similarity against predefined style vocabularies."""

    def __init__(self, style_word_files):
        """
        Args:
            style_word_files: dict mapping style names to CSV file paths,
                e.g. {'elementary': 'path/to/elementary.csv', ...}
        """
        self.style_words = {}
        self.style_vectors = defaultdict(dict)
        self.vectorizer = TfidfVectorizer(stop_words="english")

        for style, filepath in style_word_files.items():
            df = pd.read_csv(filepath)
            self.style_words[style] = set(df["word"].head(1000).tolist())

        self.reference_texts = {
            style: " ".join(words) for style, words in self.style_words.items()
        }

        all_words = []
        for words in self.style_words.values():
            all_words.extend(words)
        self.vectorizer.fit([" ".join(all_words)])

        for style, text in self.reference_texts.items():
            self.style_vectors[style] = self.vectorizer.transform([text])

    def calculate_style_similarity(self, text, target_style):
        """Compute cosine similarity between text and target style vocabulary.

        Returns:
            float: similarity score in [0, 1].
        """
        if target_style not in self.style_vectors:
            raise ValueError(
                f"Unknown style: {target_style}. Available: {list(self.style_vectors.keys())}"
            )
        text_vector = self.vectorizer.transform([text])
        similarity = cosine_similarity(text_vector, self.style_vectors[target_style])[0][0]
        return max(0.0, min(1.0, similarity))

    def calculate_style_probabilities(self, text, temperature=0.01):
        """Compute softmax-normalized style probabilities.

        Args:
            text: input text to score.
            temperature: lower values produce sharper distributions (recommended 0.01-0.5).

        Returns:
            dict mapping style names to probabilities.
        """
        raw_scores = {
            style: self.calculate_style_similarity(text, style)
            for style in self.style_vectors.keys()
        }
        scores = np.array(list(raw_scores.values()))

        scaled_scores = scores / temperature
        exp_scores = np.exp(scaled_scores - np.max(scaled_scores))
        probabilities = exp_scores / exp_scores.sum()

        return {
            style: float(prob)
            for style, prob in zip(raw_scores.keys(), probabilities)
        }
