import json
from anthropic import Anthropic
from recommendations.engine import analyze_benchmark_recommendations


def _analyze_metrics_impl(benchmark_summary: dict) -> str:
    if not benchmark_summary:
        return "No benchmark results provided."

    rows = []
    for config_key, metrics in benchmark_summary.items():
        if isinstance(metrics, dict):
            rows.append({"config": config_key, **metrics})

    if not rows:
        return "No valid metric rows found."

    best_f1 = max(rows, key=lambda x: x.get("f1_score", 0) or 0)
    best_rouge = max(rows, key=lambda x: x.get("rougeL", 0) or 0)
    best_bleu = max(rows, key=lambda x: x.get("bleu", 0) or 0)
    fastest = min(rows, key=lambda x: x.get("embed_latency_ms", float("inf")) or float("inf"))
    cheapest = min(rows, key=lambda x: x.get("embedding_cost_usd", float("inf")))
    ranked_f1 = sorted(rows, key=lambda x: x.get("f1_score", 0) or 0, reverse=True)
    quality_gap = (
        (ranked_f1[0].get("f1_score", 0) or 0) - (ranked_f1[1].get("f1_score", 0) or 0)
        if len(ranked_f1) > 1 else 0
    )
    comparison = analyze_benchmark_recommendations(benchmark_summary)
    balanced = comparison["configuration_ranking"][0]
    best_model = comparison["model_ranking"][0]
    best_chunker = comparison["chunking_ranking"][0]

    lines = [
        "=== TOP CONFIGURATIONS ===",
        "",
        f"Best balanced:  {balanced['Configuration']}  (score={balanced['Score']}/100)",
        f"Top model:      {best_model['Model']}  (score={best_model['Score']}/100)",
        f"Top chunker:    {best_chunker['Strategy']}  (score={best_chunker['Score']}/100)",
        f"Best F1 Score:  {best_f1['config']}  (F1={best_f1.get('f1_score', 'N/A')})",
        f"Best ROUGE-L:   {best_rouge['config']}  (ROUGE-L={best_rouge.get('rougeL', 'N/A')})",
        f"Best BLEU:      {best_bleu['config']}  (BLEU={best_bleu.get('bleu', 'N/A')})",
        f"Fastest:        {fastest['config']}  (latency={fastest.get('embed_latency_ms', 'N/A')} ms)",
        f"Lowest cost:    {cheapest['config']}  (cost=${cheapest.get('embedding_cost_usd', 'N/A')})",
        f"F1 lead over runner-up: {quality_gap:.4f}",
        "Composite score weights: quality 75%, latency 15%, cost 10%.",
        "",
        "=== ALL CONFIGURATIONS ===",
    ]
    for row in sorted(rows, key=lambda x: x.get("f1_score", 0) or 0, reverse=True):
        lines.append(
            f"  {row['config']:60s}  F1={row.get('f1_score','?')}  ROUGE-L={row.get('rougeL','?')}  "
            f"BLEU={row.get('bleu','?')}  latency={row.get('embed_latency_ms','?')}ms  "
            f"cost=${row.get('embedding_cost_usd','?')}"
        )
    return "\n".join(lines)


def create_synthesis_agent(api_key: str) -> dict:
    return {"api_key": api_key, "client": Anthropic(api_key=api_key)}


def run_synthesis(agent: dict, benchmark_summary: dict) -> str:
    client: Anthropic = agent["client"]

    analysis = _analyze_metrics_impl(benchmark_summary)
    benchmark_json = json.dumps(benchmark_summary, indent=2)[:4000]

    system_prompt = (
        "You are a Lead RAG Architect with deep expertise in retrieval-augmented generation systems. "
        "Analyze benchmark score matrices covering context precision/recall, BLEU, ROUGE-L, and F1 metrics "
        "across multiple embedding model and chunking strategy configurations. "
        "Explain WHY specific embedding models or chunk sizes outperformed others, "
        "grounding your reasoning in model characteristics and chunking strategy properties. "
        "Deliver actionable setup advice specifying the exact recommended embedding model and chunking strategy. "
        "Distinguish measured evidence from hypotheses about why a configuration performed well. "
        "Call out small score gaps, cost or latency penalties, and any limits in the evaluation dataset. "
        "Format your response using markdown with clear sections."
    )

    user_prompt = (
        f"Here is the pre-computed benchmark analysis:\n\n{analysis}\n\n"
        f"Raw benchmark data:\n```json\n{benchmark_json}\n```\n\n"
        "Write an executive report with these sections:\n"
        "## Summary\n"
        "## Why the Best Configuration Won\n"
        "## Quality, Cost & Latency Trade-offs\n"
        "## Production Recommendation\n"
        "## Confidence & Caveats\n"
        "## Next Validation Step"
    )

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=2000,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )

    return response.content[0].text
