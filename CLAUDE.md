# EmbedEval AI — CLAUDE.md

## Project Overview

EmbedEval AI is an autonomous RAG (Retrieval-Augmented Generation) evaluation and optimization workbench. It benchmarks chunking strategies and embedding models using a dual-layer evaluation engine (Ragas semantic metrics + lexical/statistical metrics) and presents results via a Streamlit dashboard.

## Tech Stack

- **Frontend/UI:** Next.js 16, React 19, Tailwind CSS, Recharts
- **Backend API:** FastAPI
- **Agent Orchestration:** Strands SDK
- **LLM Provider:** Anthropic Claude API (claude-3-5-sonnet)
- **Evaluation:** Ragas framework, NLTK (BLEU), rouge-score (ROUGE-1/L), token-level F1
- **Vector Storage:** ChromaDB / FAISS (in-memory, per-run)
- **Embedding Providers:** OpenAI, HuggingFace SentenceTransformers, Cohere
- **Document Parsing:** PyMuPDF (fitz), pandas, LangChain TextSplitters
- **Language:** Python 3.10+

## Key Modules

| Module | File | Purpose |
|---|---|---|
| Synthetic QA Generator | `synthetic_generator.py` | Claude-powered synthetic dataset generation (1–50 questions) |
| Evaluation Engine | `evaluation_engine.py` | BLEU, ROUGE, F1, and Ragas metric computation |
| Strands Synthesis Agent | `agent_synthesizer.py` | Benchmark analysis, trade-off reports, architecture recommendations |
| FastAPI Service | `api.py` | Benchmark run lifecycle, result, export, and report endpoints |
| Next.js Application | `frontend/` | Home, benchmark workspace, analytics dashboard, and agent report |

## Sub-Agents

This project uses three specialized Claude Code sub-agents defined in `.claude/agents/`:

| Agent | Responsibility |
|---|---|
| **ui-agent** | Streamlit dashboard, layout, tabs, sidebar controls, visualizations |
| **embeddings-agent** | Embedding model integration, chunking strategies, vector storage, evaluation pipeline |
| **assistant-agent** | Strands SDK agent orchestration, Claude API integration, synthetic generation, report synthesis |

All skills live in `.claude/skills/` and are available to all agents.

### Installed Skills

| Skill | Agent | Location |
|---|---|---|
| **ui-ux-pro-max** | ui-agent | `.claude/skills/ui-ux-pro-max/` |

## Architecture Flow

1. Document Ingestion (.pdf, .txt, .md, .xlsx, .csv)
2. Ground-Truth Setup (custom upload OR Claude synthetic generation, max 50 questions)
3. Dynamic Chunking (Fixed-Size, Recursive Character, Markdown Header-Aware, Semantic Split)
4. Parallel Multi-Embedding Vectorization (OpenAI, HuggingFace BGE, Cohere)
5. Parallel Retrieval & Dual Evaluation (Ragas + Lexical)
6. Strands Agent Synthesis (executive summary, recommendations)
7. Next.js Dashboard Rendering

## Conventions

- Hard limit of 50 synthetic questions to prevent API cost/latency issues
- Synthetic generation uses stratified sampling + parallel map-reduce over the full document (no truncation)
- Evaluation results aggregated as mean scores per (embedding_model, chunk_strategy) pair
- All metric scores rounded to 4 decimal places
- Vector stores are ephemeral (created per benchmark run, not persisted)

## Commands

```bash
# Run the FastAPI service
uvicorn api:app --reload --port 8000

# Run the Next.js application
cd frontend
npm run dev

# Install dependencies
pip install -r requirements.txt
```
