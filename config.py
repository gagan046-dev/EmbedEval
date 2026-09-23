MAX_SYNTHETIC_QUESTIONS = 50
DEFAULT_SYNTHETIC_QUESTIONS = 15
DOCUMENT_SAMPLE_LIMIT = 20000
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_TOP_K = 5
CLAUDE_MODEL = "claude-haiku-4-5-20251001"
CLAUDE_TEMPERATURE = 0.2
CLAUDE_MAX_TOKENS = 4096

CHUNKING_STRATEGIES = [
    "Fixed-Size (512)",
    "Recursive Character",
    "Markdown Header-Aware",
    "Semantic Splitting",
    "Sentence Window",
    "Token-based (tiktoken)",
    "Sliding Window",
    "Paragraph-based",
    "Table-Aware",
    "Code-Aware",
    "Parent-Child (Hierarchical)",
]

CHUNK_STRATEGY_PARAMS = {
    "Fixed-Size (512)": {
        "chunk_size": {"default": 512, "min": 64, "max": 2048, "step": 64, "label": "Chunk Size (chars)"},
        "chunk_overlap": {"default": 50, "min": 0, "max": 512, "step": 10, "label": "Overlap (chars)"},
    },
    "Recursive Character": {
        "chunk_size": {"default": 512, "min": 64, "max": 2048, "step": 64, "label": "Chunk Size (chars)"},
        "chunk_overlap": {"default": 50, "min": 0, "max": 512, "step": 10, "label": "Overlap (chars)"},
    },
    "Markdown Header-Aware": {
        "chunk_size": {"default": 512, "min": 64, "max": 2048, "step": 64, "label": "Max Chunk Size (chars)"},
        "chunk_overlap": {"default": 50, "min": 0, "max": 512, "step": 10, "label": "Overlap (chars)"},
    },
    "Semantic Splitting": {
        "chunk_size": {"default": 512, "min": 64, "max": 2048, "step": 64, "label": "Max Chunk Size (chars)"},
        "chunk_overlap": {"default": 50, "min": 0, "max": 512, "step": 10, "label": "Overlap (chars)"},
        "breakpoint_percentile": {"default": 80, "min": 50, "max": 99, "step": 1, "label": "Semantic Breakpoint Percentile"},
    },
    "Sentence Window": {
        "window_size": {"default": 5, "min": 2, "max": 20, "step": 1, "label": "Sentences per Window"},
        "overlap_sentences": {"default": 1, "min": 0, "max": 5, "step": 1, "label": "Overlap Sentences"},
    },
    "Token-based (tiktoken)": {
        "chunk_size": {"default": 512, "min": 64, "max": 2048, "step": 64, "label": "Chunk Size (tokens)"},
        "chunk_overlap": {"default": 50, "min": 0, "max": 512, "step": 10, "label": "Overlap (tokens)"},
    },
    "Sliding Window": {
        "chunk_size": {"default": 512, "min": 64, "max": 2048, "step": 64, "label": "Window Size (chars)"},
        "overlap_pct": {"default": 50, "min": 10, "max": 80, "step": 5, "label": "Overlap %"},
    },
    "Paragraph-based": {
        "chunk_size": {"default": 512, "min": 128, "max": 2048, "step": 64, "label": "Max Paragraph Size (chars)"},
        "min_paragraph_len": {"default": 20, "min": 10, "max": 100, "step": 10, "label": "Min Paragraph Length"},
    },
    "Table-Aware": {
        "chunk_size": {"default": 512, "min": 128, "max": 2048, "step": 64, "label": "Non-table Chunk Size"},
        "chunk_overlap": {"default": 50, "min": 0, "max": 512, "step": 10, "label": "Overlap (chars)"},
    },
    "Code-Aware": {
        "chunk_size": {"default": 512, "min": 128, "max": 4096, "step": 128, "label": "Chunk Size (chars)"},
        "chunk_overlap": {"default": 50, "min": 0, "max": 512, "step": 10, "label": "Overlap (chars)"},
    },
    "Parent-Child (Hierarchical)": {
        "parent_chunk_size": {"default": 1024, "min": 256, "max": 4096, "step": 128, "label": "Parent Chunk Size"},
        "child_chunk_size": {"default": 256, "min": 64, "max": 1024, "step": 64, "label": "Child Chunk Size"},
        "child_overlap": {"default": 30, "min": 0, "max": 128, "step": 10, "label": "Child Overlap"},
    },
}

HF_MODEL_REGISTRY = {
    "HuggingFace BGE-Small-EN": "BAAI/bge-small-en-v1.5",
    "HuggingFace BGE-Base-EN": "BAAI/bge-base-en-v1.5",
    "HuggingFace MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "HuggingFace E5-Large-v2": "intfloat/e5-large-v2",
}

OPENAI_MODELS = [
    "OpenAI text-embedding-3-small",
    "OpenAI text-embedding-3-large",
    "OpenAI text-embedding-ada-002",
]

COHERE_MODELS = [
    "Cohere Embed English v3.0",
    "Cohere Embed Multilingual v3.0",
]

EMBEDDING_MODELS = list(HF_MODEL_REGISTRY.keys()) + OPENAI_MODELS + COHERE_MODELS

SUPPORTED_FILE_TYPES = ["pdf", "txt", "md", "xlsx", "csv"]
