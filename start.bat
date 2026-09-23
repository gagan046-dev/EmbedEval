@echo off
start "EmbedEval API" cmd /k "uvicorn api:app --reload --port 8000"
start "EmbedEval Web" cmd /k "cd frontend && npm run dev"
echo EmbedEval AI is starting at http://localhost:3000