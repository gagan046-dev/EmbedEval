import re
import pandas as pd

QUALITY_WEIGHT = 0.75
LATENCY_WEIGHT = 0.15
COST_WEIGHT = 0.10

MODEL_DOC_TYPE_SCORES = {
    "HuggingFace BGE-Small-EN": {
        "narrative": 8, "tabular": 5, "markdown": 7, "code": 6, "multilingual": 4
    },
    "HuggingFace BGE-Base-EN": {
        "narrative": 9, "tabular": 6, "markdown": 8, "code": 7, "multilingual": 5
    },
    "HuggingFace MiniLM-L6-v2": {
        "narrative": 7, "tabular": 5, "markdown": 6, "code": 5, "multilingual": 4
    },
    "HuggingFace E5-Large-v2": {
        "narrative": 9, "tabular": 7, "markdown": 8, "code": 8, "multilingual": 7
    },
    "OpenAI text-embedding-3-small": {
        "narrative": 8, "tabular": 8, "markdown": 8, "code": 8, "multilingual": 8
    },
    "OpenAI text-embedding-3-large": {
        "narrative": 9, "tabular": 9, "markdown": 9, "code": 9, "multilingual": 9
    },
    "OpenAI text-embedding-ada-002": {
        "narrative": 7, "tabular": 7, "markdown": 7, "code": 7, "multilingual": 7
    },
    "Cohere Embed English v3.0": {
        "narrative": 8, "tabular": 7, "markdown": 7, "code": 6, "multilingual": 3
    },
    "Cohere Embed Multilingual v3.0": {
        "narrative": 7, "tabular": 6, "markdown": 6, "code": 5, "multilingual": 9
    },
}

STATIC_RECOMMENDATIONS = {
    "tabular": {
        "embedding_models": [
            {"name": "OpenAI text-embedding-3-large", "score": 9.0, "best_for": ["structured data", "dense tables"], "cost_badge": "API"},
            {"name": "OpenAI text-embedding-3-small", "score": 8.0, "best_for": ["tabular summaries", "fast retrieval"], "cost_badge": "API"},
            {"name": "HuggingFace E5-Large-v2", "score": 7.0, "best_for": ["offline tabular", "cost-free"], "cost_badge": "Free"},
        ],
        "chunking_strategies": [
            {"name": "Fixed-Size (512)", "score": 8.0, "best_for": "Uniform row batches", "weak_for": "Complex nested tables"},
            {"name": "Paragraph-based", "score": 7.0, "best_for": "Table groups / sections", "weak_for": "Wide tables with many columns"},
            {"name": "Sentence Window", "score": 6.0, "best_for": "Short row lookups", "weak_for": "Multi-row aggregations"},
        ],
    },
    "markdown": {
        "embedding_models": [
            {"name": "HuggingFace BGE-Base-EN", "score": 8.0, "best_for": ["header-structured docs", "local inference"], "cost_badge": "Free"},
            {"name": "OpenAI text-embedding-3-large", "score": 9.0, "best_for": ["rich markdown", "semantic headers"], "cost_badge": "API"},
            {"name": "HuggingFace E5-Large-v2", "score": 8.0, "best_for": ["technical docs", "free tier"], "cost_badge": "Free"},
        ],
        "chunking_strategies": [
            {"name": "Markdown Header-Aware", "score": 9.0, "best_for": "Preserves heading hierarchy", "weak_for": "Docs without headers"},
            {"name": "Recursive Character", "score": 7.0, "best_for": "Mixed markdown/prose", "weak_for": "Deeply nested structures"},
            {"name": "Semantic Splitting", "score": 7.0, "best_for": "Concept-level retrieval", "weak_for": "High compute cost"},
        ],
    },
    "code": {
        "embedding_models": [
            {"name": "OpenAI text-embedding-3-large", "score": 9.0, "best_for": ["code search", "function retrieval"], "cost_badge": "API"},
            {"name": "HuggingFace E5-Large-v2", "score": 8.0, "best_for": ["code comments", "offline"], "cost_badge": "Free"},
            {"name": "OpenAI text-embedding-3-small", "score": 8.0, "best_for": ["fast code lookup", "cost-efficient"], "cost_badge": "API"},
        ],
        "chunking_strategies": [
            {"name": "Recursive Character", "score": 8.0, "best_for": "Function-level splits", "weak_for": "Very long classes"},
            {"name": "Fixed-Size (512)", "score": 7.0, "best_for": "Uniform token budgets", "weak_for": "Mid-function splits"},
            {"name": "Token-based (tiktoken)", "score": 8.0, "best_for": "Precise token control", "weak_for": "Language-agnostic parsing"},
        ],
    },
    "narrative": {
        "embedding_models": [
            {"name": "HuggingFace BGE-Base-EN", "score": 9.0, "best_for": ["long-form text", "semantic search"], "cost_badge": "Free"},
            {"name": "OpenAI text-embedding-3-large", "score": 9.0, "best_for": ["nuanced prose", "best quality"], "cost_badge": "API"},
            {"name": "Cohere Embed English v3.0", "score": 8.0, "best_for": ["document ranking", "dense retrieval"], "cost_badge": "API"},
        ],
        "chunking_strategies": [
            {"name": "Semantic Splitting", "score": 9.0, "best_for": "Paragraph-level semantics", "weak_for": "Long inference time"},
            {"name": "Sentence Window", "score": 8.0, "best_for": "Context-rich sentences", "weak_for": "Very short documents"},
            {"name": "Paragraph-based", "score": 8.0, "best_for": "Natural paragraph breaks", "weak_for": "Inconsistent paragraph lengths"},
        ],
    },
    "mixed": {
        "embedding_models": [
            {"name": "OpenAI text-embedding-3-large", "score": 9.0, "best_for": ["mixed content", "versatile"], "cost_badge": "API"},
            {"name": "HuggingFace E5-Large-v2", "score": 7.0, "best_for": ["diverse doc types", "free"], "cost_badge": "Free"},
            {"name": "OpenAI text-embedding-3-small", "score": 8.0, "best_for": ["balanced cost/quality"], "cost_badge": "API"},
        ],
        "chunking_strategies": [
            {"name": "Recursive Character", "score": 8.0, "best_for": "Handles most content types", "weak_for": "Complex nesting"},
            {"name": "Fixed-Size (512)", "score": 7.0, "best_for": "Predictable chunk sizes", "weak_for": "Semantic coherence"},
            {"name": "Semantic Splitting", "score": 7.0, "best_for": "Topic-aware splits", "weak_for": "Slow on large docs"},
        ],
    },
    "unknown": {
        "embedding_models": [
            {"name": "OpenAI text-embedding-3-large", "score": 9.0, "best_for": ["general purpose", "high quality"], "cost_badge": "API"},
            {"name": "HuggingFace BGE-Base-EN", "score": 8.0, "best_for": ["free local", "solid baseline"], "cost_badge": "Free"},
            {"name": "OpenAI text-embedding-3-small", "score": 7.0, "best_for": ["fast and cheap"], "cost_badge": "API"},
        ],
        "chunking_strategies": [
            {"name": "Recursive Character", "score": 8.0, "best_for": "General-purpose splits", "weak_for": "Highly structured docs"},
            {"name": "Fixed-Size (512)", "score": 7.0, "best_for": "Predictable indexing", "weak_for": "Semantic coherence"},
            {"name": "Paragraph-based", "score": 6.0, "best_for": "Natural breaks", "weak_for": "Single-paragraph docs"},
        ],
    },
}


def detect_document_type(text: str) -> str:
    sample = text[:5000]
    pipe_count = sample.count("|")
    comma_lines = sum(1 for line in sample.splitlines() if line.count(",") >= 3)
    markdown_headers = len(re.findall(r"^#{1,6}\s", sample, re.MULTILINE))
    code_keywords = len(re.findall(r"\b(def |class |import |function |return |const |let |var |public |private )\b", sample))
    avg_line_len = (sum(len(l) for l in sample.splitlines()) / max(len(sample.splitlines()), 1))

    scores = {
        "tabular": min(pipe_count / 5 + comma_lines / 3, 10),
        "markdown": min(markdown_headers * 2, 10),
        "code": min(code_keywords * 1.5, 10),
        "narrative": min(avg_line_len / 10, 10),
    }

    top = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    if top[0][1] < 1.5:
        return "unknown"
    if top[0][1] - top[1][1] < 1.0 and top[0][1] > 2:
        return "mixed"
    return top[0][0]


def get_static_recommendations(doc_type: str) -> dict:
    return STATIC_RECOMMENDATIONS.get(doc_type, STATIC_RECOMMENDATIONS["unknown"])


def _quality_score(metrics: dict) -> float:
    lexical = (
        0.5 * float(metrics.get("f1_score", 0))
        + 0.3 * float(metrics.get("rougeL", 0))
        + 0.2 * float(metrics.get("bleu", 0))
    )
    ragas_values = [
        float(metrics[key])
        for key in ("context_precision", "context_recall")
        if metrics.get(key) is not None
    ]
    if not ragas_values:
        return lexical
    return 0.6 * lexical + 0.4 * (sum(ragas_values) / len(ragas_values))


def _inverse_min_max(value: float, values: list[float]) -> float:
    low, high = min(values), max(values)
    if high == low:
        return 1.0
    return (high - value) / (high - low)


def _aggregate_ranking(rows: list[dict], group_key: str, label: str) -> list[dict]:
    df = pd.DataFrame(rows)
    grouped = (
        df.groupby(group_key)[["score", "quality_score", "latency_score", "cost_score"]]
        .mean()
        .sort_values("score", ascending=False)
        .reset_index()
    )
    grouped.columns = [label, "Score", "Quality", "Speed", "Cost Efficiency"]
    for column in ("Score", "Quality", "Speed", "Cost Efficiency"):
        grouped[column] = grouped[column].round(1)
    grouped["Avg Score"] = grouped["Score"]
    return grouped.to_dict(orient="records")


def analyze_benchmark_recommendations(benchmark_matrix_dict: dict) -> dict:
    if not benchmark_matrix_dict:
        return {
            "insights": [],
            "configuration_ranking": [],
            "model_ranking": [],
            "chunking_ranking": [],
            "recommendations": [],
        }

    rows = []
    for config_key, metrics in benchmark_matrix_dict.items():
        parts = str(config_key).split(" | ")
        model = parts[0] if len(parts) > 0 else config_key
        strategy = parts[1] if len(parts) > 1 else "unknown"
        row = {"model": model, "strategy": strategy}
        row.update(metrics if isinstance(metrics, dict) else {})
        rows.append(row)

    latencies = [float(row.get("embed_latency_ms", 0)) for row in rows]
    costs = [float(row.get("embedding_cost_usd", 0)) for row in rows]
    for row in rows:
        quality = _quality_score(row)
        speed = _inverse_min_max(float(row.get("embed_latency_ms", 0)), latencies)
        cost_efficiency = _inverse_min_max(float(row.get("embedding_cost_usd", 0)), costs)
        row["quality_score"] = quality * 100
        row["latency_score"] = speed * 100
        row["cost_score"] = cost_efficiency * 100
        row["score"] = (
            QUALITY_WEIGHT * row["quality_score"]
            + LATENCY_WEIGHT * row["latency_score"]
            + COST_WEIGHT * row["cost_score"]
        )

    ranked = sorted(rows, key=lambda row: row["score"], reverse=True)
    configuration_ranking = [
        {
            "Configuration": f"{row['model']} | {row['strategy']}",
            "Score": round(row["score"], 1),
            "Quality": round(row["quality_score"], 1),
            "Speed": round(row["latency_score"], 1),
            "Cost Efficiency": round(row["cost_score"], 1),
        }
        for row in ranked
    ]
    model_ranking = _aggregate_ranking(rows, "model", "Model")
    chunking_ranking = _aggregate_ranking(rows, "strategy", "Strategy")

    best = ranked[0]
    best_quality = max(rows, key=lambda row: row["quality_score"])
    fastest = min(rows, key=lambda row: float(row.get("embed_latency_ms", 0)))
    cheapest = min(rows, key=lambda row: float(row.get("embedding_cost_usd", 0)))
    runner_up_score = ranked[1]["score"] if len(ranked) > 1 else best["score"]
    score_gap = best["score"] - runner_up_score
    best_config = f"{best['model']} + {best['strategy']}"

    insights = [
        "Comparison score weights retrieval quality at <b>75%</b>, embedding speed at <b>15%</b>, and estimated cost efficiency at <b>10%</b>.",
        f"Best balanced configuration: <b>{best_config}</b> with score <b>{best['score']:.1f}/100</b>.",
        f"Top embedding model: <b>{model_ranking[0]['Model']}</b> at <b>{model_ranking[0]['Score']:.1f}/100</b> across tested chunkers.",
        f"Top chunking strategy: <b>{chunking_ranking[0]['Strategy']}</b> at <b>{chunking_ranking[0]['Score']:.1f}/100</b> across tested models.",
        f"Maximum retrieval quality: <b>{best_quality['model']} + {best_quality['strategy']}</b> at <b>{best_quality['quality_score']:.1f}/100</b>.",
        f"Fastest measured configuration: <b>{fastest['model']} + {fastest['strategy']}</b> at <b>{float(fastest.get('embed_latency_ms', 0)):.1f} ms</b>.",
        f"Lowest estimated embedding cost: <b>{cheapest['model']} + {cheapest['strategy']}</b> at <b>${float(cheapest.get('embedding_cost_usd', 0)):.6f}</b>.",
        f"The winner leads the runner-up by <b>{score_gap:.1f} points</b>; {'treat it as a clear preference' if score_gap >= 5 else 'treat the result as close and validate on more questions'}.",
    ]
    recommendations = [
        {
            "title": "Recommended production configuration",
            "configuration": best_config,
            "score": round(best["score"], 1),
            "reason": f"Best measured balance of retrieval quality ({best['quality_score']:.1f}), speed ({best['latency_score']:.1f}), and cost efficiency ({best['cost_score']:.1f}).",
        },
        {
            "title": "Maximum-quality configuration",
            "configuration": f"{best_quality['model']} + {best_quality['strategy']}",
            "score": round(best_quality["quality_score"], 1),
            "reason": "Choose this when retrieval quality matters more than latency or embedding cost.",
        },
        {
            "title": "Fastest configuration",
            "configuration": f"{fastest['model']} + {fastest['strategy']}",
            "score": round(float(fastest.get("embed_latency_ms", 0)), 1),
            "unit": "ms",
            "reason": "Choose this for latency-sensitive workloads after confirming its quality is acceptable.",
        },
        {
            "title": "Lowest-cost configuration",
            "configuration": f"{cheapest['model']} + {cheapest['strategy']}",
            "score": round(float(cheapest.get("embedding_cost_usd", 0)), 6),
            "unit": "USD",
            "reason": "Choose this for cost-sensitive or high-volume indexing workloads.",
        },
    ]

    return {
        "scoring_method": {
            "quality": QUALITY_WEIGHT * 100,
            "latency": LATENCY_WEIGHT * 100,
            "cost": COST_WEIGHT * 100,
        },
        "insights": insights,
        "configuration_ranking": configuration_ranking,
        "model_ranking": model_ranking,
        "chunking_ranking": chunking_ranking,
        "recommendations": recommendations,
    }


def get_use_case_matrix() -> list[dict]:
    rows = []
    col_map = {"Narrative Text": "narrative", "Tabular Data": "tabular", "Markdown Docs": "markdown", "Code": "code", "Multilingual": "multilingual"}
    for model, scores in MODEL_DOC_TYPE_SCORES.items():
        row = {"Model": model}
        for col_label, score_key in col_map.items():
            s = scores.get(score_key, 0)
            row[col_label] = "✓ Good" if s >= 7 else ("~ Fair" if s >= 5 else "✗ Poor")
        rows.append(row)
    return rows
