# EmbedEval AI: Autonomous RAG Evaluation & Optimization Workbench
**Project Technical Specification & Implementation Architecture**

---

## 1. Executive Summary

**EmbedEval AI** is an open-source evaluation and benchmarking platform designed to eliminate guess-driven engineering in Retrieval-Augmented Generation (RAG) pipelines. Modern RAG architectures suffer from a high degree of variance depending on document domain, text chunking strategy, and vector embedding selection. 

EmbedEval AI provides a unified environment where users can upload multi-format documents (`.pdf`, `.txt`, `.md`, `.xlsx`, `.csv`), configure dynamic chunking methods, run concurrent multi-model embedding evaluations, and inspect performance using a dual-layer evaluation engine:
1. **Ragas Framework Metrics:** Context Precision, Context Recall, Faithfulness, and Answer Relevance.
2. **Lexical & Statistical Metrics:** BLEU, ROUGE (ROUGE-1, ROUGE-L), and token-level F1 scores.

To evaluate retrieval accuracy without manually curated datasets, EmbedEval AI includes an autonomous dataset generator powered by the **Anthropic Claude API** and orchestrated using the **Strands SDK**. An embedded Strands Agent dynamically analyzes benchmark performance metrics, generates trade-off reports (precision vs. latency vs. cost), and presents actionable architecture recommendations on an interactive **Streamlit** dashboard.

---

## 2. Problem Statement & Motivation

Building high-performing RAG systems requires selecting the right combination of text chunking and vector embedding models:
* **The Chunking Dilemma:** A fixed 512-token chunking strategy might work well for raw text novels, but destroys structural context in financial spreadsheets or hierarchical markdown files.
* **The Embedding Mismatch:** Semantic similarity models may capture generalized context but fail on exact-keyword searches or technical code snippets.
* **The Benchmark Gap:** Teams often rely solely on qualitative spot-checking or overly complex offline evaluations. There is a lack of lightweight, interactive tools that unify lexical overlap metrics with LLM-as-a-judge frameworks alongside automated synthesis.

EmbedEval AI bridges this gap by turning chunking and embedding selection into an empirical, data-driven workflow.

---

## 3. System Architecture & Data Flow

The system processes documents through a multi-stage sequential and parallel execution pipeline:

```
+-----------------------------------------------------------------------------------+
|                               1. Document Ingestion                               |
|                     (.pdf, .txt, .md, .xlsx, .csv File Parsing)                   |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                        2. Ground-Truth Dataset Setup                              |
|   +-------------------------------------+-------------------------------------+   |
|   | Option A: Custom Ground-Truth Upload| Option B: Agent Synthetic Generation|   |
|   |          (.json / .csv QA pairs)    |     (Claude API: 1-50 Questions)    |   |
|   +-------------------------------------+-------------------------------------+   |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                            3. Dynamic Chunking Engine                             |
|      (Fixed-Size, Recursive Character, Markdown Header-Aware, Semantic Split)     |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                       4. Parallel Multi-Embedding Vectorization                   |
|         (OpenAI text-embedding-3, HuggingFace BGE, Cohere Embed v3, etc.)         |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                       5. Parallel Retrieval & Dual Evaluation                     |
|   +-------------------------------------+-------------------------------------+   |
|   |           Ragas Framework           |       Lexical & Overlap Metrics     |   |
|   | (Precision, Recall, Faithfulness)   |         (BLEU, ROUGE-1/L, F1)       |   |
|   +-------------------------------------+-------------------------------------+   |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                        6. Strands Agent Synthesis Engine                          |
|             (Strands SDK + Claude API Tool-Calling & Finding Analysis)            |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                         7. Streamlit Interactive UI                               |
|        (Benchmark Matrix, QA/Chunk Inspector, Agent Executive Summary)            |
+-----------------------------------------------------------------------------------+
```

---

## 4. Ingestion & Ground-Truth Dataset Generation Specification

Ground-truth evaluation requires pairs of `(Question, Target Context, Reference Answer)`. EmbedEval AI handles ground-truth preparation via two pathways:

### 4.1 Custom Dataset Upload
Users can upload an existing test suite formatted as a JSON or CSV file containing the following required keys:
* `question`: The query string.
* `expected_context`: Ground-truth reference context block(s).
* `ground_truth`: The ideal reference answer.

### 4.2 Agentic Synthetic Generation Mode
When synthetic generation is enabled:
1. **Document Sampling:** The ingested text is divided into structural blocks (sections, tables, or semantic paragraphs).
2. **Claude API Prompt Execution:** An agent invokes `claude-3-5-sonnet` to scan sampled text blocks and synthesize factual, reasoning, and edge-case questions.
3. **Question Count UI Control:**
   * **Toggle:** Checkbox (`Generate Synthetic Test Questions via Agent`).
   * **Quantity Control:** Selectbox or Slider allowing selection from `5` up to a maximum strict limit of `50` questions.
   * **Enforcement:** Hard-coded limit of 50 questions prevents API rate-limiting, excessive costs, and UI latency timeouts.

---

## 5. Dual Evaluation Engine Framework

EmbedEval AI computes performance across both model-based semantic evaluation (LLM-as-a-judge) and fast statistical token-overlap scoring.

### 5.1 Ragas Framework Layer
* **Context Precision:** Measures the signal-to-noise ratio in the retrieved context block. Evaluates if top-ranked retrieved chunks actually contain relevant information.
* **Context Recall:** Evaluates whether all necessary ground-truth facts were fetched in the top-$K$ contexts.
* **Faithfulness:** Verifies that generated answers are strictly grounded in retrieved contexts (detecting hallucinations).
* **Answer Relevance:** Measures how directly the generated or retrieved response addresses the original user question.

### 5.2 Lexical & Overlap Metrics Layer
* **BLEU (Bilingual Evaluation Understudy):** N-gram precision score comparing retrieved text chunks directly against target ground-truth context spans.
* **ROUGE-1 & ROUGE-L:** 
  * *ROUGE-1:* Measures unigram overlap recall.
  * *ROUGE-L:* Measures the Longest Common Subsequence (LCS) to evaluate structural and sequence retention.
* **Token-Level F1 Score:** Computes the harmonic mean of precision and recall over exact token matches:
  $$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

---

## 6. Tech Stack Breakdown

| Layer | Technology | Function & Purpose |
| :--- | :--- | :--- |
| **Frontend UI & Dashboard** | `Streamlit` | Web interface, interactive controls, metric visualization (heatmaps, tabbed layouts). |
| **Agent Orchestration** | `Strands SDK` | Framework for building agent loops, managing state, and binding custom tools. |
| **LLM Provider** | `Anthropic Claude API` (`claude-3-5-sonnet`) | Powers synthetic question generation, context analysis, and synthesized reporting. |
| **Evaluation Suite** | `Ragas`, `evaluate`, `nltk`, `rouge-score` | Framework for calculating semantic and statistical scores. |
| **Vector Storage** | `ChromaDB` / `FAISS` | In-memory vector databases created per embedding model model run. |
| **Embedding Providers** | `OpenAI`, `HuggingFace (SentenceTransformers)`, `Cohere` | Target models being benchmarked against each other. |
| **Data Ingestion** | `PyMuPDF` (`fitz`), `pandas`, `LangChain TextSplitters` | Document parsing (PDF, Markdown, Excel, CSV) and text chunking logic. |

---

## 7. Python Implementation Code Structure

### 7.1 Synthetic Dataset Generation Tool (`synthetic_generator.py`)

```python
import json
from typing import List, Dict, Any
from anthropic import Anthropic

class SyntheticDatasetGenerator:
    def __init__(self, api_key: str):
        self.client = Anthropic(api_key=api_key)

    def generate_qa_pairs(self, document_text: str, num_questions: int = 15) -> List[Dict[str, Any]]:
        """
        Generates synthetic (Question, Expected Context, Ground Truth) pairs using Claude 3.5 Sonnet.
        Includes strict enforcement of the 50-question limit.
        """
        # Hard cap at 50 questions
        target_count = min(max(1, num_questions), 50)
        
        # Truncate input safely if document is excessively large for a single prompt
        context_sample = document_text[:20000]

        system_prompt = (
            "You are an expert AI Benchmark Engineer. Your task is to analyze document content "
            "and create high-quality synthetic test questions with explicit ground-truth contexts."
        )

        user_prompt = f"""
Analyze the following document text and generate exactly {target_count} test query items.
Each item must contain:
1. "question": A natural, challenging user query.
2. "expected_context": The exact passage/snippet from the document that answers it.
3. "ground_truth": A concise, accurate target answer based on that context.

Respond ONLY with a valid JSON array of objects. Do not include markdown formatting or explanations outside the JSON block.

DOCUMENT CONTENT:
{context_sample}
"""

        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=4096,
            temperature=0.2,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )

        response_text = response.content[0].text.strip()
        
        # Parse JSON output safely
        try:
            qa_dataset = json.loads(response_text)
            return qa_dataset[:target_count]
        except json.JSONDecodeError:
            # Fallback handling in case of markdown wrapping
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
                return json.loads(json_str)[:target_count]
            raise ValueError("Failed to parse synthetic QA dataset from Claude API response.")
```

### 7.2 Evaluation Engine (`evaluation_engine.py`)

```python
import numpy as np
from typing import List, Dict, Any
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

class MetricEvaluator:
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        self.smooth_fn = SmoothingFunction().method1

    def compute_lexical_metrics(self, retrieved_context: str, expected_context: str) -> Dict[str, float]:
        """
        Computes BLEU, ROUGE-1, ROUGE-L, and Token-Level F1 scores between retrieved and target contexts.
        """
        ref_tokens = expected_context.lower().split()
        cand_tokens = retrieved_context.lower().split()

        # 1. BLEU Score
        bleu = sentence_bleu([ref_tokens], cand_tokens, smoothing_function=self.smooth_fn)

        # 2. ROUGE Scores
        rouge_results = self.rouge_scorer.score(expected_context, retrieved_context)
        rouge1 = rouge_results['rouge1'].fmeasure
        rougel = rouge_results['rougeL'].fmeasure

        # 3. Token-level F1 Score
        common_tokens = set(ref_tokens) & set(cand_tokens)
        if not common_tokens or not ref_tokens or not cand_tokens:
            f1_score = 0.0
        else:
            precision = len(common_tokens) / len(cand_tokens)
            recall = len(common_tokens) / len(ref_tokens)
            f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "bleu": round(float(bleu), 4),
            "rouge1": round(float(rouge1), 4),
            "rougeL": round(float(rougel), 4),
            "f1_score": round(float(f1_score), 4)
        }

    def aggregate_benchmark_matrix(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregates raw scores across queries to calculate mean metric scores per model/chunk config.
        """
        aggregated = {}
        for item in results:
            config_key = f"{item['embedding_model']} | {item['chunk_strategy']}"
            if config_key not in aggregated:
                aggregated[config_key] = {
                    "bleu": [], "rouge1": [], "rougeL": [], "f1_score": [],
                    "ragas_precision": [], "ragas_recall": []
                }
            
            metrics = item["metrics"]
            aggregated[config_key]["bleu"].append(metrics["bleu"])
            aggregated[config_key]["rouge1"].append(metrics["rouge1"])
            aggregated[config_key]["rougeL"].append(metrics["rougeL"])
            aggregated[config_key]["f1_score"].append(metrics["f1_score"])
            
            if "ragas_precision" in metrics:
                aggregated[config_key]["ragas_precision"].append(metrics["ragas_precision"])
                aggregated[config_key]["ragas_recall"].append(metrics["ragas_recall"])

        # Compute averages
        summary = {}
        for config, data in aggregated.items():
            summary[config] = {k: round(float(np.mean(v)), 4) for k, v in data.items() if len(v) > 0}
        
        return summary
```

### 7.3 Strands Synthesis Agent (`agent_synthesizer.py`)

```python
import json
from strands import Agent
from strands.tools import tool

@tool
def analyze_metrics_summary(benchmark_json: str) -> str:
    """
    Parses aggregated evaluation results to identify top-performing embedding and chunking configurations.
    """
    data = json.loads(benchmark_json)
    if not data:
        return "No benchmark data provided."

    best_f1 = max(data.items(), key=lambda x: x[1].get('f1_score', 0))
    best_rouge = max(data.items(), key=lambda x: x[1].get('rougeL', 0))

    return (
        f"Top Config by F1-Score: {best_f1[0]} (Score: {best_f1[1].get('f1_score')}). "
        f"Top Config by ROUGE-L: {best_rouge[0]} (Score: {best_rouge[1].get('rougeL')})."
    )

def create_synthesis_agent(api_key: str) -> Agent:
    """
    Instantiates the Strands Agent powered by Claude to write executive benchmarking reports.
    """
    agent = Agent(
        model="anthropic/claude-3-5-sonnet",
        api_key=api_key,
        prompt=(
            "You are an expert Lead RAG Architect. You will analyze benchmark score matrices "
            "comprising Ragas precision/recall and lexical metrics (BLEU, ROUGE, F1). "
            "Provide clear executive summaries explaining why specific embedding models or chunk sizes "
            "outperformed others on the provided document, and deliver actionable setup advice."
        ),
        tools=[analyze_metrics_summary]
    )
    return agent
```

### 7.4 Main Streamlit Dashboard (`app.py`)

```python
import streamlit as st
import pandas as pd
import json

st.set_page_config(page_title="EmbedEval AI", layout="wide")

st.title("EmbedEval AI: RAG Benchmarking & Optimization Workbench")
st.markdown("Evaluate chunking strategies and embedding models using Ragas & Lexical Metrics.")

# -------------------------- SIDEBAR CONTROLS --------------------------
st.sidebar.header("1. Document Ingestion")
uploaded_doc = st.sidebar.file_uploader(
    "Upload Source File", 
    type=["pdf", "txt", "md", "xlsx", "csv"]
)

st.sidebar.header("2. Benchmark QA Dataset")
use_synthetic_agent = st.sidebar.checkbox(
    "Generate Synthetic QA Pairs via Claude Agent", 
    value=True
)

if use_synthetic_agent:
    num_questions = st.sidebar.selectbox(
        "Questions to Generate (Max 50):",
        options=[5, 10, 15, 20, 25, 30, 40, 50],
        index=2
    )
    custom_qa_file = None
else:
    num_questions = None
    custom_qa_file = st.sidebar.file_uploader(
        "Upload Ground-Truth QA File (.json or .csv)", 
        type=["json", "csv"]
    )

st.sidebar.header("3. Chunking Configurations")
selected_chunkers = st.sidebar.multiselect(
    "Select Chunking Strategies:",
    ["Fixed-Size (512)", "Recursive Character", "Markdown Header-Aware", "Semantic Splitting"],
    default=["Recursive Character"]
)

st.sidebar.header("4. Embedding Models")
selected_embeddings = st.sidebar.multiselect(
    "Select Embedding Models:",
    ["OpenAI text-embedding-3-small", "HuggingFace BGE-Small-EN", "Cohere Embed English v3.0"],
    default=["OpenAI text-embedding-3-small", "HuggingFace BGE-Small-EN"]
)

run_button = st.sidebar.button("Run Benchmark Pipeline", type="primary")

# -------------------------- MAIN PANEL DISPLAY --------------------------
tab1, tab2, tab3 = st.tabs(["📊 Benchmark Matrix", "🔍 QA & Chunk Inspector", "🤖 Agent Executive Report"])

with tab1:
    st.subheader("Performance Comparison Matrix")
    if not run_button:
        st.info("Configure settings in the sidebar and click 'Run Benchmark Pipeline' to begin.")
    else:
        st.success("Pipeline executed successfully!")
        # Sample placeholder metrics rendering
        mock_data = {
            "Configuration": [
                "OpenAI text-embedding-3-small | Recursive", 
                "HuggingFace BGE-Small-EN | Recursive",
                "OpenAI text-embedding-3-small | Fixed-Size",
                "HuggingFace BGE-Small-EN | Fixed-Size"
            ],
            "Ragas Context Precision": [0.89, 0.81, 0.74, 0.68],
            "Ragas Context Recall": [0.92, 0.85, 0.78, 0.72],
            "BLEU": [0.42, 0.38, 0.31, 0.29],
            "ROUGE-L": [0.68, 0.61, 0.52, 0.47],
            "F1-Score": [0.73, 0.66, 0.58, 0.51]
        }
        df_matrix = pd.DataFrame(mock_data)
        st.dataframe(df_matrix.style.highlight_max(axis=0, color="lightgreen"), use_container_width=True)

with tab2:
    st.subheader("Retrieved Context & Ground-Truth Inspector")
    st.markdown("Inspect how each model retrieved chunks for specific test questions.")

with tab3:
    st.subheader("Strands Synthesis Agent Insights")
    st.markdown("Automated insights and technical recommendations generated by Anthropic Claude:")
    st.text_area(
        "Agent Executive Summary", 
        value="[Agent Output] Recursive Character chunking with OpenAI text-embedding-3-small demonstrated superior performance overall...",
        height=300
    )
```

---

## 8. Dashboard Layout & User Experience Design

The Streamlit UI is organized logically across three main view spaces:

```
+---------------------------------------------------------------------------------------------+
|                                    EMBEDEVAL AI DASHBOARD                                   |
+-------------------+-------------------------------------------------------------------------+
|  SIDEBAR CONTROLS | MAIN CONTENT AREA                                                       |
|                   | +---------------------------------------------------------------------+ |
| 1. Ingestion      | | Tab 1: Benchmark Matrix | Tab 2: QA Inspector | Tab 3: Agent Summary | |
|    [Upload File]  | +---------------------------------------------------------------------+ |
|                   |                                                                         |
| 2. QA Dataset     | [ Tab 1: Benchmark Matrix View ]                                        |
|    [x] Synthetic  | +---------------------------------------------------------------------+ |
|    Dropdown: [15] | | Model & Chunk Configuration  | Precision | Recall | BLEU | ROUGE | F1 | |
|    (Max: 50)      | |------------------------------|-----------|--------|------|-------|----| |
|                   | | OpenAI + Recursive           | 0.89      | 0.92   | 0.42 | 0.68  |0.73| |
| 3. Chunking       | | HuggingFace + Recursive      | 0.81      | 0.85   | 0.38 | 0.61  |0.66| |
|    [x] Recursive  | +---------------------------------------------------------------------+ |
|    [x] Fixed-Size |                                                                         |
|                   | [ Heatmap & Radar Charts Display ]                                      |
| 4. Embeddings     |                                                                         |
|    [x] OpenAI     | [ Tab 3: Strands Agent Synthesis View ]                                 |
|    [x] HuggingFace| +---------------------------------------------------------------------+ |
|                   | | 🤖 "The combination of Recursive Character Splitting (512 tokens)   | |
| [RUN BENCHMARK]   | |     and OpenAI text-embedding-3-small yielded the highest Context   | |
|                   | |     Recall (0.92)..."                                               | |
|                   | +---------------------------------------------------------------------+ |
+-------------------+-------------------------------------------------------------------------+
```

---

## 9. Strategic Improvements & Future Roadmap

1. **Synthetic Query Cross-Validation Sampling:**
   * Improve synthetic query distribution by enforcing multi-page proportional sampling (ensuring synthetic queries span all sections rather than clustering in initial chapters).
2. **Latency & Token Cost Profiling:**
   * Measure latency (time-to-index and time-to-retrieve) and token costs per model run, displaying cost-per-query alongside accuracy scores.
3. **Hybrid Search Benchmarking Engine:**
   * Extend evaluations beyond dense embeddings to include hybrid retrieval configurations (combining sparse BM25 keyword matching with dense vectors using Reciprocal Rank Fusion).
4. **Automated Chunk Hyperparameter Optimization:**
   * Enable the Strands agent to dynamically adjust chunk sizes (e.g., auto-iterating from 256 to 1024 tokens) to discover the sweet spot for a given document corpus.