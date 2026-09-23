export type NumericParam = {
  default: number;
  min: number;
  max: number;
  step: number;
  label: string;
};

export type AppConfig = {
  chunking_strategies: string[];
  chunk_strategy_params: Record<string, Record<string, NumericParam>>;
  models: Record<"huggingface" | "openai" | "cohere", string[]>;
  supported_file_types: string[];
  max_synthetic_questions: number;
  default_top_k: number;
};

export type Metrics = {
  bleu: number;
  rouge1: number;
  rougeL: number;
  f1_score: number;
  context_precision?: number;
  context_recall?: number;
  embed_latency_ms: number;
  embedding_cost_usd: number;
};

export type RawResult = {
  embedding_model: string;
  chunk_strategy: string;
  question: string;
  retrieved_context: string;
  expected_context: string;
  ground_truth: string;
  metrics: Metrics;
};

export type RecommendationResult = {
  scoring_method: { quality: number; latency: number; cost: number };
  insights: string[];
  configuration_ranking: Array<{
    Configuration: string;
    Score: number;
    Quality: number;
    Speed: number;
    "Cost Efficiency": number;
  }>;
  model_ranking: Array<{ Model: string; Score: number; "Avg Score": number; Quality: number; Speed: number; "Cost Efficiency": number }>;
  chunking_ranking: Array<{ Strategy: string; Score: number; "Avg Score": number; Quality: number; Speed: number; "Cost Efficiency": number }>;
  recommendations: Array<{
    title: string;
    configuration: string;
    score: number;
    unit?: string;
    reason: string;
  }>;
};

export type RunResult = {
  raw_results: RawResult[];
  benchmark_matrix: Record<string, Metrics>;
  qa_dataset: Array<Record<string, string | number>>;
  doc_type: string;
  static_recommendations: Record<string, Array<Record<string, unknown>>>;
  recommendations: RecommendationResult;
  use_case_matrix: Array<Record<string, string>>;
  embedding_summary: {
    total_vectors: number;
    configs: Array<{
      config: string;
      model: string;
      strategy: string;
      num_chunks: number;
      dimensions: number;
      size_mb: number;
    }>;
  } | null;
  vector_store_configs: Array<{ config: string; collection_name: string; count: number }>;
};

export type BenchmarkRun = {
  id: string;
  status: "queued" | "running" | "completed" | "failed";
  stage: string;
  created_at: string;
  completed_at?: string;
  file_name: string;
  configuration: {
    chunk_strategies: string[];
    embedding_models: string[];
    question_count: number;
    top_k: number;
    use_ragas: boolean;
    persist_embeddings: boolean;
    use_vector_db: boolean;
  };
  error: string | null;
  result: RunResult | null;
  agent_report?: string;
};