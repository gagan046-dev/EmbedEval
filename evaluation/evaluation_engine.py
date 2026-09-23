import asyncio
import math
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from inspect import signature

from config import CLAUDE_MODEL


class MetricEvaluator:
    def compute_lexical_metrics(self, retrieved_context: str, expected_context: str) -> dict[str, float]:
        retrieved_tokens = retrieved_context.lower().split()
        expected_tokens = expected_context.lower().split()

        bleu = sentence_bleu(
            [expected_tokens],
            retrieved_tokens,
            smoothing_function=SmoothingFunction().method1
        )

        scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        scores = scorer.score(expected_context, retrieved_context)

        retrieved_set = set(retrieved_tokens)
        expected_set = set(expected_tokens)
        intersection = retrieved_set & expected_set

        if not retrieved_set or not expected_set:
            f1 = 0.0
        else:
            precision = len(intersection) / len(retrieved_set)
            recall = len(intersection) / len(expected_set)
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        return {
            "bleu": round(bleu, 4),
            "rouge1": round(scores["rouge1"].fmeasure, 4),
            "rougeL": round(scores["rougeL"].fmeasure, 4),
            "f1_score": round(f1, 4),
        }

    def aggregate_benchmark_matrix(self, results: list[dict]) -> dict[str, dict]:
        groups: dict[str, list[dict]] = {}
        for result in results:
            key = f"{result['embedding_model']} | {result['chunk_strategy']}"
            groups.setdefault(key, []).append(result["metrics"])

        summary: dict[str, dict] = {}
        for key, metrics_list in groups.items():
            all_keys: set[str] = set()
            for m in metrics_list:
                all_keys.update(m.keys())
            averaged: dict[str, float] = {}
            for metric_key in all_keys:
                values = [m[metric_key] for m in metrics_list if metric_key in m]
                averaged[metric_key] = round(sum(values) / len(values), 4)
            summary[key] = averaged

        return summary


class RagasEvaluator:
    def __init__(self, anthropic_api_key: str):
        from anthropic import AsyncAnthropic
        from ragas.llms import llm_factory
        from ragas.metrics.collections import ContextPrecision, ContextRecall

        self.client = AsyncAnthropic(api_key=anthropic_api_key)
        self._runner = asyncio.Runner()
        self._closed = False
        llm = llm_factory(CLAUDE_MODEL, provider="anthropic", client=self.client)
        supported_args = set(signature(self.client.messages.create).parameters)
        for arg in set(llm.model_args) - supported_args:
            llm.model_args.pop(arg)
        self.context_precision = ContextPrecision(llm=llm)
        self.context_recall = ContextRecall(llm=llm)

    @staticmethod
    def _score_value(result, metric_name: str) -> float:
        value = getattr(result, "value", result)
        score = float(value)
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"Ragas {metric_name} returned an invalid score: {value!r}")
        return round(score, 4)

    async def _evaluate_pair_async(
        self,
        question: str,
        retrieved_chunks: list[str],
        ground_truth: str,
    ) -> dict[str, float]:
        kwargs = {
            "user_input": question,
            "reference": ground_truth,
            "retrieved_contexts": retrieved_chunks,
        }
        precision = await self.context_precision.ascore(**kwargs)
        recall = await self.context_recall.ascore(**kwargs)
        return {
            "context_precision": self._score_value(precision, "context precision"),
            "context_recall": self._score_value(recall, "context recall"),
        }

    def evaluate_pair(
        self,
        question: str,
        retrieved_chunks: list[str],
        ground_truth: str,
    ) -> dict[str, float]:
        if self._closed:
            raise RuntimeError("Ragas evaluator is closed.")
        return self._runner.run(
            self._evaluate_pair_async(question, retrieved_chunks, ground_truth)
        )

    def close(self):
        if self._closed:
            return
        self._runner.run(self.client.close())
        self._runner.close()
        self._closed = True
