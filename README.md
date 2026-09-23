# EmbedEval AI

EmbedEval AI is a RAG evaluation workbench with a Next.js interface and a FastAPI evaluation service.

## Architecture

- `frontend/`: Next.js App Router application
- `api.py`: HTTP API and benchmark run management
- `pipeline.py`: document, chunking, embedding, retrieval, and evaluation orchestration
- `evaluation/`: lexical and Ragas evaluation metrics
- `recommendations/`: document-aware and benchmark-derived recommendations

## Setup

```powershell
install.bat
```

## Run

```powershell
start.bat
```

Open `http://localhost:3000`. The API runs at `http://localhost:8000`.

To start the processes separately:

```powershell
uvicorn api:app --reload --port 8000
cd frontend
npm run dev
```

Set `NEXT_PUBLIC_API_URL` when the API is hosted somewhere other than `http://localhost:8000`.

## Tests

```powershell
python -m unittest discover -s tests -v
cd frontend
npm run build
```