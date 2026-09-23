import threading
import unittest
from types import SimpleNamespace

from synthetic_generator import SyntheticDatasetGenerator


class RetryingGenerator(SyntheticDatasetGenerator):
    def __init__(self, fail_until: int):
        self.fail_until = fail_until
        self.calls = 0
        self.lock = threading.Lock()

    def _chunk_document(self, text, chunk_size=1500, overlap=200):
        return ["verbatim source"]

    def _generate_from_chunk(self, chunk_text, chunk_index, total_chunks, variation_id):
        with self.lock:
            self.calls += 1
            call = self.calls
        if call <= self.fail_until:
            raise RuntimeError("temporary failure")
        return {
            "question": f"Unique question number {call}?",
            "expected_context": "verbatim source",
            "ground_truth": "answer",
            "category": "SPECIFIC_FACT",
            "source_position_pct": 0,
        }


class SyntheticGeneratorTests(unittest.TestCase):
    def test_retries_until_requested_count_is_complete(self):
        generator = RetryingGenerator(fail_until=2)

        pairs = generator.generate_qa_pairs("document", 3)

        self.assertEqual(len(pairs), 3)
        self.assertEqual(generator.calls, 5)

    def test_raises_when_retry_budget_is_exhausted(self):
        generator = RetryingGenerator(fail_until=100)

        with self.assertRaisesRegex(
            RuntimeError,
            "Generated 0 of 2 unique QA pairs after 6 attempts",
        ):
            generator.generate_qa_pairs("document", 2)

    def test_rejects_context_not_found_in_source(self):
        response = SimpleNamespace(content=[SimpleNamespace(text="""
            <json_output>{
                "question": "What is the value?",
                "expected_context": "fabricated context",
                "ground_truth": "A value"
            }</json_output>
        """)])
        generator = SyntheticDatasetGenerator.__new__(SyntheticDatasetGenerator)
        generator.model = "test-model"
        generator.client = SimpleNamespace(
            messages=SimpleNamespace(create=lambda **kwargs: response)
        )

        with self.assertRaisesRegex(ValueError, "not verbatim source text"):
            generator._generate_from_chunk("actual source", 0, 1, 0)


if __name__ == "__main__":
    unittest.main()