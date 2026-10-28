#!/bin/sh
# Container entry point: start Ollama (embeddings for the document search), wait until it answers,
# then start the app on the port that the host gives (Hugging Face Spaces uses 7860).
ollama serve > /tmp/ollama.log 2>&1 &
until ollama list > /dev/null 2>&1; do sleep 1; done
exec uvicorn app:app --host 0.0.0.0 --port "${PORT:-7860}"
