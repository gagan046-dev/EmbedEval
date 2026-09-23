import unittest
from unittest.mock import patch

import numpy as np

from chunking.chunkers import chunk_text


class SemanticModel:
    def encode(self, texts, **kwargs):
        if not kwargs["normalize_embeddings"]:
            raise AssertionError("Semantic embeddings must be normalized")
        return np.array([
            [1.0, 0.0],
            [0.99, 0.1],
            [0.0, 1.0],
            [0.1, 0.99],
        ])


class SemanticChunkingTests(unittest.TestCase):
    def test_splits_at_embedding_distance_outlier(self):
        sentences = [
            "Cats sleep indoors.",
            "Kittens rest at home.",
            "Rockets launch into orbit.",
            "Spacecraft travel through space.",
        ]

        with (
            patch("chunking.chunkers.nltk.download"),
            patch("chunking.chunkers.nltk.tokenize.sent_tokenize", return_value=sentences),
            patch("chunking.chunkers._get_semantic_model", return_value=SemanticModel()),
        ):
            chunks = chunk_text(
                "sample",
                "Semantic Splitting",
                chunk_size=1000,
                chunk_overlap=0,
                breakpoint_percentile=80,
            )

        self.assertEqual(chunks, [
            "Cats sleep indoors. Kittens rest at home.",
            "Rockets launch into orbit. Spacecraft travel through space.",
        ])


if __name__ == "__main__":
    unittest.main()