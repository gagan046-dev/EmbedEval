import asyncio
import unittest
from inspect import signature
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from pydantic import BaseModel

from evaluation.evaluation_engine import RagasEvaluator


class RecordingScorer:
    def __init__(self, value: float):
        self.value = value
        self.calls = []

    async def ascore(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(value=self.value)


class MockResponse(BaseModel):
    answer: str


class RagasEvaluatorTests(unittest.TestCase):
    @patch("ragas.metrics.collections.ContextRecall")
    @patch("ragas.metrics.collections.ContextPrecision")
    @patch("ragas.llms.llm_factory")
    @patch("anthropic.AsyncAnthropic")
    def test_initializes_ragas_with_async_anthropic_client(
        self,
        async_anthropic,
        llm_factory,
        context_precision,
        context_recall,
    ):
        client = async_anthropic.return_value
        client.close = AsyncMock()
        llm = llm_factory.return_value

        evaluator = RagasEvaluator("test-key")

        async_anthropic.assert_called_once_with(api_key="test-key")
        llm_factory.assert_called_once_with(
            "claude-haiku-4-5-20251001",
            provider="anthropic",
            client=client,
        )
        context_precision.assert_called_once_with(llm=llm)
        context_recall.assert_called_once_with(llm=llm)
        self.assertIs(evaluator.context_precision, context_precision.return_value)
        self.assertIs(evaluator.context_recall, context_recall.return_value)
        evaluator.close()

    def test_ragas_model_args_match_anthropic_async_messages_api(self):
        evaluator = RagasEvaluator("test-key")
        llm = evaluator.context_precision.llm
        supported_args = set(signature(llm.client.client.messages.create).parameters)

        self.assertEqual(llm.model_args, {"max_tokens": 1024})
        self.assertLessEqual(set(llm.model_args), supported_args)
        evaluator.close()

    def test_ragas_agenerate_forwards_only_supported_anthropic_args(self):
        evaluator = RagasEvaluator("test-key")
        llm = evaluator.context_precision.llm
        create = AsyncMock(
            side_effect=lambda **kwargs: kwargs["response_model"](answer="mocked")
        )
        llm.client.chat.completions.create = create

        result = asyncio.run(llm.agenerate("Return an answer", MockResponse))
        forwarded_args = create.await_args.kwargs

        self.assertEqual(result, MockResponse(answer="mocked"))
        self.assertEqual(forwarded_args["max_tokens"], 1024)
        self.assertNotIn("temperature", forwarded_args)
        self.assertNotIn("top_p", forwarded_args)
        evaluator.close()

    def test_evaluate_pair_uses_ragas_context_metrics(self):
        evaluator = RagasEvaluator.__new__(RagasEvaluator)
        evaluator._runner = asyncio.Runner()
        evaluator._closed = False
        evaluator.client = SimpleNamespace(close=AsyncMock())
        evaluator.context_precision = RecordingScorer(0.812345)
        evaluator.context_recall = RecordingScorer(0.923456)

        result = evaluator.evaluate_pair(
            "What happened?",
            ["First context", "Second context"],
            "Reference answer",
        )

        expected_call = {
            "user_input": "What happened?",
            "reference": "Reference answer",
            "retrieved_contexts": ["First context", "Second context"],
        }
        self.assertEqual(
            result,
            {"context_precision": 0.8123, "context_recall": 0.9235},
        )
        self.assertEqual(evaluator.context_precision.calls, [expected_call])
        self.assertEqual(evaluator.context_recall.calls, [expected_call])
        evaluator.close()


if __name__ == "__main__":
    unittest.main()