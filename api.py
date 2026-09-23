import io
import json
import threading
import uuid
from datetime import UTC, datetime

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

from agent_synthesizer import create_synthesis_agent, run_synthesis
from config import (
    CHUNKING_STRATEGIES,
    CHUNK_STRATEGY_PARAMS,
    COHERE_MODELS,
    DEFAULT_TOP_K,
    HF_MODEL_REGISTRY,
    MAX_SYNTHETIC_QUESTIONS,
    OPENAI_MODELS,
    SUPPORTED_FILE_TYPES,
)
from embeddings.embedding_runner import get_embedding_provider
from ingestion.document_parser import parse_document
from pipeline import run_benchmark_pipeline
from recommendations.engine import (
    analyze_benchmark_recommendations,
    detect_document_type,
    get_static_recommendations,
    get_use_case_matrix,
)
from storage.vector_store import VectorStoreManager
from utils.embedding_exporter import (
    export_embeddings_jsonl,
    export_embeddings_npz,
    export_faiss_index,
    get_store_summary,
)
from utils.exporters import results_to_csv, results_to_json


app = FastAPI(title="EmbedEval AI API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_runs: dict[str, dict] = {}
_runs_lock = threading.Lock()


class ReportRequest(BaseModel):
    anthropic_key: str


def _public_run(run: dict) -> dict:
    return {key: value for key, value in run.items() if not key.startswith("_")}


def _update_run(run_id: str, **values):
    with _runs_lock:
        _runs[run_id].update(values)


def _parse_json_field(value: str, field_name: str, expected_type: type):
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise HTTPException(400, f"{field_name} must be valid JSON.") from exc
    if not isinstance(parsed, expected_type):
        raise HTTPException(400, f"{field_name} has an invalid shape.")
    return parsed


def _parse_qa_file(file_bytes: bytes, file_name: str) -> list[dict]:
    if file_name.lower().endswith(".json"):
        data = json.loads(file_bytes.decode("utf-8"))
    elif file_name.lower().endswith(".csv"):
        data = pd.read_csv(io.BytesIO(file_bytes)).to_dict(orient="records")
    else:
        raise ValueError("Ground-truth dataset must be JSON or CSV.")

    if not isinstance(data, list) or not data:
        raise ValueError("Ground-truth dataset must contain at least one QA row.")
    required = {"question", "expected_context", "ground_truth"}
    for index, row in enumerate(data, start=1):
        if not isinstance(row, dict) or not required.issubset(row):
            raise ValueError(f"Ground-truth row {index} is missing required fields.")
    return data


def _execute_run(
    run_id: str,
    document_bytes: bytes,
    file_name: str,
    qa_bytes: bytes | None,
    qa_file_name: str | None,
    use_synthetic: bool,
    question_count: int,
    chunk_strategies: list[str],
    embedding_models: list[str],
    chunk_params: dict,
    top_k: int,
    use_ragas: bool,
    persist_embeddings: bool,
    use_vector_db: bool,
    anthropic_key: str | None,
    openai_key: str | None,
    cohere_key: str | None,
    hf_token: str | None,
):
    try:
        _update_run(run_id, status="running", stage="Reading document")
        document_text = parse_document(document_bytes, file_name)
        doc_type = detect_document_type(document_text[:5000])

        if use_synthetic:
            from synthetic_generator import SyntheticDatasetGenerator

            _update_run(run_id, stage="Generating evaluation dataset")
            qa_dataset = SyntheticDatasetGenerator(anthropic_key).generate_qa_pairs(
                document_text,
                question_count,
            )
        else:
            qa_dataset = _parse_qa_file(qa_bytes or b"", qa_file_name or "")

        _update_run(run_id, stage="Running retrieval benchmarks")
        raw_results, benchmark_matrix, error, embedding_store = run_benchmark_pipeline(
            file_bytes=document_bytes,
            file_name=file_name,
            qa_dataset=qa_dataset,
            chunk_strategies=chunk_strategies,
            embedding_models=embedding_models,
            openai_api_key=openai_key,
            cohere_api_key=cohere_key,
            hf_token=hf_token,
            chunk_params=chunk_params,
            top_k=top_k,
            use_ragas=use_ragas,
            anthropic_api_key=anthropic_key,
            persist_embeddings=persist_embeddings or use_vector_db,
        )
        if error:
            raise RuntimeError(error)

        vector_store = None
        if use_vector_db and embedding_store:
            _update_run(run_id, stage="Writing vector collections")
            vector_store = VectorStoreManager()
            for config_key, data in embedding_store.items():
                vector_store.upsert(config_key, data["chunks"], data["embeddings"])

        recommendations = analyze_benchmark_recommendations(benchmark_matrix)
        _update_run(
            run_id,
            status="completed",
            stage="Complete",
            completed_at=datetime.now(UTC).isoformat(),
            result={
                "raw_results": raw_results,
                "benchmark_matrix": benchmark_matrix,
                "qa_dataset": qa_dataset,
                "doc_type": doc_type,
                "static_recommendations": get_static_recommendations(doc_type),
                "recommendations": recommendations,
                "use_case_matrix": get_use_case_matrix(),
                "embedding_summary": get_store_summary(embedding_store) if persist_embeddings else None,
                "vector_store_configs": vector_store.list_configs() if vector_store else [],
            },
            _embedding_store=embedding_store if persist_embeddings else {},
            _vector_store=vector_store,
        )
    except Exception as exc:
        _update_run(
            run_id,
            status="failed",
            stage="Failed",
            error=str(exc),
            completed_at=datetime.now(UTC).isoformat(),
        )


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/config")
def get_config():
    return {
        "chunking_strategies": CHUNKING_STRATEGIES,
        "chunk_strategy_params": CHUNK_STRATEGY_PARAMS,
        "models": {
            "huggingface": list(HF_MODEL_REGISTRY),
            "openai": OPENAI_MODELS,
            "cohere": COHERE_MODELS,
        },
        "supported_file_types": SUPPORTED_FILE_TYPES,
        "max_synthetic_questions": MAX_SYNTHETIC_QUESTIONS,
        "default_top_k": DEFAULT_TOP_K,
    }


@app.post("/api/runs", status_code=202)
async def create_run(
    background_tasks: BackgroundTasks,
    document: UploadFile = File(...),
    ground_truth: UploadFile | None = File(None),
    use_synthetic: bool = Form(True),
    question_count: int = Form(15),
    chunk_strategies: str = Form(...),
    embedding_models: str = Form(...),
    chunk_params: str = Form("{}"),
    top_k: int = Form(DEFAULT_TOP_K),
    use_ragas: bool = Form(False),
    persist_embeddings: bool = Form(False),
    use_vector_db: bool = Form(False),
    anthropic_key: str | None = Form(None),
    openai_key: str | None = Form(None),
    cohere_key: str | None = Form(None),
    hf_token: str | None = Form(None),
):
    strategies = _parse_json_field(chunk_strategies, "chunk_strategies", list)
    models = _parse_json_field(embedding_models, "embedding_models", list)
    params = _parse_json_field(chunk_params, "chunk_params", dict)
    anthropic_key = anthropic_key.strip() if anthropic_key else None
    openai_key = openai_key.strip() if openai_key else None
    cohere_key = cohere_key.strip() if cohere_key else None
    hf_token = hf_token.strip() if hf_token else None

    extension = document.filename.rsplit(".", 1)[-1].lower() if document.filename and "." in document.filename else ""
    if extension not in SUPPORTED_FILE_TYPES:
        raise HTTPException(400, "Unsupported document type.")
    if not strategies or any(strategy not in CHUNKING_STRATEGIES for strategy in strategies):
        raise HTTPException(400, "Select at least one valid chunking strategy.")
    if not models:
        raise HTTPException(400, "Select at least one embedding model.")
    if any(not isinstance(model, str) for model in models):
        raise HTTPException(400, "Embedding model identifiers must be strings.")
    providers = [get_embedding_provider(model) for model in models]
    if any(provider is None for provider in providers):
        raise HTTPException(400, "One or more embedding model identifiers are invalid.")
    if not 1 <= question_count <= MAX_SYNTHETIC_QUESTIONS:
        raise HTTPException(400, f"question_count must be between 1 and {MAX_SYNTHETIC_QUESTIONS}.")
    if use_synthetic and not anthropic_key:
        raise HTTPException(400, "Anthropic API key is required for synthetic QA generation.")
    if use_ragas and not anthropic_key:
        raise HTTPException(400, "Anthropic API key is required for Ragas evaluation.")
    if "openai" in providers and not openai_key:
        raise HTTPException(400, "OpenAI API key is required for selected OpenAI models.")
    if "cohere" in providers and not cohere_key:
        raise HTTPException(400, "Cohere API key is required for selected Cohere models.")
    if not use_synthetic and ground_truth is None:
        raise HTTPException(400, "Upload a ground-truth dataset or enable synthetic generation.")

    document_bytes = await document.read()
    qa_bytes = await ground_truth.read() if ground_truth else None
    run_id = uuid.uuid4().hex
    with _runs_lock:
        _runs[run_id] = {
            "id": run_id,
            "status": "queued",
            "stage": "Queued",
            "created_at": datetime.now(UTC).isoformat(),
            "file_name": document.filename,
            "configuration": {
                "chunk_strategies": strategies,
                "embedding_models": models,
                "question_count": question_count,
                "top_k": top_k,
                "use_ragas": use_ragas,
                "persist_embeddings": persist_embeddings,
                "use_vector_db": use_vector_db,
            },
            "error": None,
            "result": None,
        }

    background_tasks.add_task(
        _execute_run,
        run_id,
        document_bytes,
        document.filename or "document.txt",
        qa_bytes,
        ground_truth.filename if ground_truth else None,
        use_synthetic,
        question_count,
        strategies,
        models,
        params,
        top_k,
        use_ragas,
        persist_embeddings,
        use_vector_db,
        anthropic_key,
        openai_key,
        cohere_key,
        hf_token,
    )
    return _public_run(_runs[run_id])


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    run = _runs.get(run_id)
    if not run:
        raise HTTPException(404, "Run not found.")
    return _public_run(run)


@app.post("/api/runs/{run_id}/report")
def generate_report(run_id: str, request: ReportRequest):
    run = _runs.get(run_id)
    if not run or run["status"] != "completed":
        raise HTTPException(409, "Benchmark run is not complete.")
    report = run.get("agent_report")
    if report is None:
        report = run_synthesis(
            create_synthesis_agent(request.anthropic_key),
            run["result"]["benchmark_matrix"],
        )
        _update_run(run_id, agent_report=report)
    return {"report": report}


@app.get("/api/runs/{run_id}/exports/{format_name}")
def export_results(run_id: str, format_name: str):
    run = _runs.get(run_id)
    if not run or run["status"] != "completed":
        raise HTTPException(404, "Completed run not found.")
    result = run["result"]
    if format_name == "json":
        content = results_to_json(result["raw_results"], result["benchmark_matrix"])
        return Response(content, media_type="application/json", headers={"Content-Disposition": "attachment; filename=embedeval-report.json"})
    if format_name == "csv":
        content = results_to_csv(result["raw_results"])
        return Response(content, media_type="text/csv", headers={"Content-Disposition": "attachment; filename=embedeval-results.csv"})
    raise HTTPException(404, "Unknown export format.")


@app.get("/api/runs/{run_id}/embeddings")
def export_embeddings(run_id: str, config_key: str, format_name: str = "npz"):
    run = _runs.get(run_id)
    store = run.get("_embedding_store") if run else None
    if not store or config_key not in store:
        raise HTTPException(404, "Persisted embedding configuration not found.")
    if format_name == "npz":
        content = export_embeddings_npz({config_key: store[config_key]})
        media_type, suffix = "application/octet-stream", "npz"
    elif format_name == "faiss":
        content = export_faiss_index(store, config_key)
        media_type, suffix = "application/octet-stream", "faiss"
    elif format_name == "jsonl":
        content = export_embeddings_jsonl(store, config_key)
        media_type, suffix = "application/x-ndjson", "jsonl"
    else:
        raise HTTPException(404, "Unknown embedding export format.")
    return Response(content, media_type=media_type, headers={"Content-Disposition": f"attachment; filename=embeddings.{suffix}"})