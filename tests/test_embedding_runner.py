import unittest

from embeddings.embedding_runner import get_embedding_provider


class EmbeddingProviderTests(unittest.TestCase):
    def test_resolves_catalog_and_custom_models(self):
        cases = {
            "HuggingFace BGE-Small-EN": "huggingface",
            "HuggingFace Custom: nomic-ai/nomic-embed-text-v1.5": "huggingface",
            "OpenAI text-embedding-3-small": "openai",
            "OpenAI Custom: text-embedding-custom": "openai",
            "Cohere Embed English v3.0": "cohere",
            "Cohere Custom: embed-v4.0": "cohere",
        }

        for model, provider in cases.items():
            with self.subTest(model=model):
                self.assertEqual(get_embedding_provider(model), provider)

    def test_rejects_unknown_or_empty_custom_models(self):
        for model in ("Unknown model", "OpenAI Custom: ", "Cohere Custom:"):
            with self.subTest(model=model):
                self.assertIsNone(get_embedding_provider(model))


if __name__ == "__main__":
    unittest.main()