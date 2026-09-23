import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, DEFAULT_TOP_K
from utils.cost_estimator import estimate_cost
from ingestion.document_parser import parse_document
from chunking.chunkers import chunk_text
from embeddings.embedding_runner import EmbeddingRunner
from retrieval.retriever import build_faiss_index, retrieve_top_k
from evaluation.evaluation_engine import MetricEvaluator


def run_benchmark_pipeline(
    file_bytes: bytes,
    file_name: str,
    qa_dataset: list[dict],
    chunk_strategies: list[str],
    embedding_models: list[str],
    openai_api_key: str = None,
    cohere_api_key: str = None,
    hf_token: str = None,
    chunk_params: dict = None,
    top_k: int = DEFAULT_TOP_K,
    use_ragas: bool = False,
    anthropic_api_key: str = None,
    persist_embeddings: bool = False,
) -> tuple[list[dict], dict, str | None, dict]:
    try:
        document_text = parse_document(file_bytes, file_name)

        embedding_runner = EmbeddingRunner(
            openai_api_key=openai_api_key,
            cohere_api_key=cohere_api_key,
            hf_token=hf_token,
        )
        evaluator = MetricEvaluator()

        ragas_evaluator_class = None
        if use_ragas and anthropic_api_key:
            from evaluation.evaluation_engine import RagasEvaluator
            ragas_evaluator_class = RagasEvaluator

        embedding_store: dict = {}

        combinations = [
            (strategy, model)
            for strategy in chunk_strategies
            for model in embedding_models
        ]

        chunks_by_strategy = {}
        for strategy in chunk_strategies:
            strategy_params = (chunk_params or {}).get(strategy, {})
            chunk_size = strategy_params.get("chunk_size", DEFAULT_CHUNK_SIZE)
            chunk_overlap = strategy_params.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP)
            extra_kwargs = {
                key: value
                for key, value in strategy_params.items()
                if key not in ("chunk_size", "chunk_overlap")
            }
            chunks_by_strategy[strategy] = chunk_text(
                document_text,
                strategy,
                chunk_size,
                chunk_overlap,
                **extra_kwargs,
            )

        all_results = []

        def process_combination(chunk_strategy, embedding_model):
            ragas_evaluator = (
                ragas_evaluator_class(anthropic_api_key)
                if ragas_evaluator_class is not None else None
            )
            try:
                chunks = chunks_by_strategy[chunk_strategy]
                t_start = time.time()
                embeddings = embedding_runner.embed(chunks, embedding_model)
                t_end = time.time()
                embed_latency_ms = int((t_end - t_start) * 1000)
                if persist_embeddings:
                    config_key = f"{embedding_model} | {chunk_strategy}"
                    embedding_store[config_key] = {
                        "chunks": chunks,
                        "embeddings": np.array(embeddings, dtype=np.float32),
                        "model": embedding_model,
                        "strategy": chunk_strategy,
                        "num_chunks": len(chunks),
                        "dimensions": len(embeddings[0]) if embeddings else 0,
                    }
                embedding_cost = estimate_cost(chunks, embedding_model)

                index = build_faiss_index(embeddings)

                combo_results = []
                for qa_pair in qa_dataset:
                    question = qa_pair["question"]
                    expected_context = qa_pair["expected_context"]
                    ground_truth = qa_pair["ground_truth"]

                    retrieved_chunks = retrieve_top_k(
                        question,
                        chunks,
                        index,
                        embedding_runner,
                        embedding_model,
                        k=top_k,
                    )
                    retrieved_context = " ".join(retrieved_chunks)
                    metrics = evaluator.compute_lexical_metrics(
                        retrieved_context, expected_context
                    )

                    if ragas_evaluator is not None:
                        try:
                            metrics.update(ragas_evaluator.evaluate_pair(
                                question, retrieved_chunks, ground_truth
                            ))
                        except Exception as exc:
                            raise RuntimeError(
                                f"Ragas failed for {embedding_model} + {chunk_strategy} "
                                f"on question {question[:80]!r}: {exc}"
                            ) from exc

                    metrics["embed_latency_ms"] = embed_latency_ms
                    metrics["embedding_cost_usd"] = embedding_cost

                    combo_results.append(
                        {
                            "embedding_model": embedding_model,
                            "chunk_strategy": chunk_strategy,
                            "question": question,
                            "retrieved_context": retrieved_context,
                            "expected_context": expected_context,
                            "ground_truth": ground_truth,
                            "metrics": metrics,
                        }
                    )
                return combo_results
            finally:
                if ragas_evaluator is not None:
                    ragas_evaluator.close()

        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(process_combination, strategy, model): (strategy, model)
                for strategy, model in combinations
            }
            for future in as_completed(futures):
                all_results.extend(future.result())

        benchmark_matrix = evaluator.aggregate_benchmark_matrix(all_results)

        return (all_results, benchmark_matrix, None, embedding_store)

    except Exception as e:
        return ([], {}, str(e), {})
