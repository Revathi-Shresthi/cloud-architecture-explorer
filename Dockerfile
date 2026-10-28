# One container for hosting (Hugging Face Spaces, or `docker compose up`):
#   - Ollama runs only the small embedding model (nomic-embed-text) for the document search.
#   - Answers and agents use Gemini (CHAT_PROVIDER=gemini). Set GEMINI_API_KEY as a secret.
#   - The document index is built into the image from the public AWS Well-Architected PDF.
FROM ollama/ollama:0.40.11 AS ollama

FROM python:3.13-slim
COPY --from=ollama /usr/bin/ollama /usr/bin/ollama
COPY --from=ollama /usr/lib/ollama /usr/lib/ollama
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces runs the container as user 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    OLLAMA_MODELS=/home/user/ollama-models \
    PYTHONUNBUFFERED=1 \
    CHAT_PROVIDER=gemini \
    DEMO_MODE=1
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .

# Build the document index: download the PDF, get the embedding model, run ingest.py.
# MAX_PAGES keeps the build short; set it to 0 for the whole document.
ARG DOC_URL=https://docs.aws.amazon.com/pdfs/wellarchitected/latest/framework/wellarchitected-framework.pdf
ARG MAX_PAGES=200
RUN mkdir -p data \
 && curl -fsSL -o data/wellarchitected-framework.pdf "$DOC_URL" \
 && (ollama serve > /tmp/ollama.log 2>&1 &) \
 && until ollama list > /dev/null 2>&1; do sleep 1; done \
 && ollama pull nomic-embed-text \
 && (python ingest.py data/ --max-pages "$MAX_PAGES" || python ingest.py data/ --max-pages "$MAX_PAGES")

EXPOSE 7860
CMD ["sh", "start.sh"]
