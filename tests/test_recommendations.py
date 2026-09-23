import unittest

from recommendations.engine import analyze_benchmark_recommendations


class RecommendationTests(unittest.TestCase):
    def test_scores_configurations_and_aggregates_models_and_chunkers(self):
        matrix = {
            "Model A | Chunk A": {
                "f1_score": 0.9,
                "rougeL": 0.8,
                "bleu": 0.7,
                "embed_latency_ms": 200,
                "embedding_cost_usd": 0.02,
            },
            "Model B | Chunk A": {
                "f1_score": 0.7,
                "rougeL": 0.6,
                "bleu": 0.5,
                "embed_latency_ms": 100,
                "embedding_cost_usd": 0.0,
            },
            "Model A | Chunk B": {
                "f1_score": 0.85,
                "rougeL": 0.75,
                "bleu": 0.65,
                "embed_latency_ms": 180,
                "embedding_cost_usd": 0.02,
            },
        }

        result = analyze_benchmark_recommendations(matrix)

        self.assertEqual(result["scoring_method"], {"quality": 75.0, "latency": 15.0, "cost": 10.0})
        self.assertEqual(len(result["configuration_ranking"]), 3)
        self.assertEqual(result["model_ranking"][0]["Model"], "Model B")
        self.assertEqual(result["chunking_ranking"][0]["Strategy"], "Chunk A")
        self.assertEqual(len(result["recommendations"]), 4)
        self.assertIn("Score", result["model_ranking"][0])
        self.assertEqual(
            result["model_ranking"][0]["Avg Score"],
            result["model_ranking"][0]["Score"],
        )

    def test_ragas_metrics_contribute_to_quality_score(self):
        matrix = {
            "Model A | Chunk A": {
                "f1_score": 0.8,
                "rougeL": 0.8,
                "bleu": 0.8,
                "context_precision": 0.9,
                "context_recall": 0.9,
                "embed_latency_ms": 100,
                "embedding_cost_usd": 0.0,
            }
        }

        result = analyze_benchmark_recommendations(matrix)

        self.assertEqual(result["configuration_ranking"][0]["Quality"], 84.0)
        self.assertEqual(result["configuration_ranking"][0]["Score"], 88.0)


if __name__ == "__main__":
    unittest.main()