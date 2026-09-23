import json
import unittest

from fastapi.testclient import TestClient

from api import app


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health_and_configuration(self):
        self.assertEqual(self.client.get("/api/health").json(), {"status": "ok"})
        config = self.client.get("/api/config").json()
        self.assertIn("Semantic Splitting", config["chunking_strategies"])
        self.assertIn("huggingface", config["models"])

    def test_run_rejects_missing_embedding_models(self):
        response = self.client.post(
            "/api/runs",
            files={"document": ("document.txt", b"sample", "text/plain")},
            data={
                "use_synthetic": "false",
                "chunk_strategies": json.dumps(["Recursive Character"]),
                "embedding_models": "[]",
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "Select at least one embedding model.")

    def test_custom_openai_model_requires_openai_key(self):
        response = self.client.post(
            "/api/runs",
            files={"document": ("document.txt", b"sample", "text/plain")},
            data={
                "use_synthetic": "false",
                "chunk_strategies": json.dumps(["Recursive Character"]),
                "embedding_models": json.dumps(["OpenAI Custom: text-embedding-custom"]),
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.json()["detail"],
            "OpenAI API key is required for selected OpenAI models.",
        )

    def test_empty_custom_model_identifier_is_rejected(self):
        response = self.client.post(
            "/api/runs",
            files={"document": ("document.txt", b"sample", "text/plain")},
            data={
                "use_synthetic": "false",
                "chunk_strategies": json.dumps(["Recursive Character"]),
                "embedding_models": json.dumps(["Cohere Custom:   "]),
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.json()["detail"],
            "One or more embedding model identifiers are invalid.",
        )

    def test_whitespace_cohere_key_is_rejected(self):
        response = self.client.post(
            "/api/runs",
            files={"document": ("document.txt", b"sample", "text/plain")},
            data={
                "use_synthetic": "false",
                "chunk_strategies": json.dumps(["Recursive Character"]),
                "embedding_models": json.dumps(["Cohere Custom: embed-v4.0"]),
                "cohere_key": "   ",
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.json()["detail"],
            "Cohere API key is required for selected Cohere models.",
        )


if __name__ == "__main__":
    unittest.main()