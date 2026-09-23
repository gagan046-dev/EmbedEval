import unittest

from retrieval.retriever import build_faiss_index, retrieve_top_k


class QueryRunner:
    def embed(self, texts, model_name, input_type="search_document"):
        if input_type != "search_query":
            raise AssertionError("Retrieval must use query embeddings")
        return [[1.0, 0.0]]


class RetrieverTests(unittest.TestCase):
    def test_cosine_ranking_ignores_vector_magnitude(self):
        chunks = ["same direction", "closer by raw L2"]
        index = build_faiss_index([[10.0, 0.0], [0.8, 0.6]])

        results = retrieve_top_k(
            "query",
            chunks,
            index,
            QueryRunner(),
            "model",
            k=5,
        )

        self.assertEqual(results, chunks)


if __name__ == "__main__":
    unittest.main()