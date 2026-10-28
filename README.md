---
title: Cloud Architecture Explorer
emoji: 🏗️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
short_description: RAG and AI agents for cloud architecture, in 3D
---

# Cloud Architecture Explorer

A RAG and AI-agent app for cloud architecture. Draw an architecture in 3D (the built-in one, or your own from
Terraform or Kubernetes files), click a component, and ask questions. Answers come from cloud documents,
with file and page citations and a sentence-by-sentence grounding check. Simulate a failure to see what breaks,
or run a multi-agent architecture review. Locally it runs on Ollama and Chroma with no API keys; the agents and
the hosted version use the Gemini API.

## Features

| Feature | How it works | Code |
|---|---|---|
| Infrastructure-as-code import | Terraform resource blocks or Kubernetes manifests become a graph. Links come from resource references (through helper resources such as listeners and launch templates), Ingress backends, Service selectors, NetworkPolicies, and service names in env vars. Automatic 3D layout by request-flow tier. | `iac.py` |
| Failure simulation | Breadth-first search over the request flow finds upstream components (requests fail), downstream components (no traffic), and unobserved components (monitoring down). The 3D view colors them; the model explains effect, detection, and recovery. | `architecture.py` |
| Component-aware hybrid retrieval | Vector search and BM25 keyword search, filtered to chunks tagged with the selected component's kind, merged with Reciprocal Rank Fusion. Vectors are held in memory (NumPy); tags come from the current keywords, so keyword changes need no re-ingest. | `retrieval.py`, `components.py` |
| Grounding check | Each answer sentence is embedded and compared with the chunks it cites. Sentences under the similarity threshold, and citations that do not exist, are flagged. No extra LLM call. | `grounding.py` |
| Conversation memory | The last 2 turns go to the model; short follow-ups ("how do I test it?") are searched together with the previous question. | `app.py` |
| Streaming + metrics | Answers stream as NDJSON events. Each answer reports search time, time to first token, tokens/s, and grounding score; every request is logged to `logs/requests.jsonl` and summarised at `/api/stats`. | `app.py` |
| Evaluation | `eval.py` measures Hit@k, MRR, and precision of each search mode on `eval/questions.json`. | `eval.py` |
| Tutor agent (Gemini) | A function-calling agent decides which tools to use (document search, architecture details, failure simulation, critical-component ranking), possibly several times, then answers with citations. The page shows every step. | `agents.py` |
| Multi-agent review (Gemini) | Reliability, security, performance, and cost agents examine the architecture with the same tools and submit structured findings through a `submit_findings` tool; a lead agent writes the summary. Findings color the affected components in 3D. | `agents.py` |
| Hosting | One Docker container for Hugging Face Spaces: Gemini answers, built-in document index, per-visitor and daily rate limits. | `Dockerfile`, `start.sh` |
| Tests, CI | 53 pytest tests with Ollama and Gemini replaced by fakes; GitHub Actions workflow. | `tests/`, `Dockerfile`, `.github/` |

## Setup (one time)

1. Install Ollama from https://ollama.com, then get the models:
   ```
   ollama pull llama3.2
   ollama pull nomic-embed-text
   ```
2. Make and start a virtual environment (Python 3.13 is tested):
   ```
   python -m venv venv
   venv\Scripts\activate          # Windows
   source venv/bin/activate       # macOS / Linux
   ```
3. Install the packages:
   ```
   pip install -r requirements-dev.txt
   ```
4. Put PDF, .md, or .txt files into `data/`.
5. Load them into the vector database:
   ```
   python ingest.py data/ --max-pages 200
   ```

## Run

```
uvicorn app:app --reload
```

Open http://localhost:8000. To try the importer, click **Import IaC…** and choose
`examples/terraform/shop.tf` or `examples/kubernetes/shop.yaml`.

On a slow computer, use the smaller chat model:
```
ollama pull llama3.2:1b
$env:CHAT_MODEL="llama3.2:1b"     # PowerShell; then start uvicorn
```

## AI agents (optional, Gemini API)

The local RAG needs no key. For the agents:

1. Get a free key at https://aistudio.google.com/apikey.
2. Copy `.env.example` to `.env` and put the key after `GEMINI_API_KEY=`.
3. Restart uvicorn. Select **Agent (Gemini)** in the panel header, or click **Review**.

Agent questions and architecture data go to Google. On the free tier, Google can use them to improve its
products, so do not review private infrastructure files there. Free-tier rate limits apply; the agents wait
and retry when they reach one. Document search and embeddings still run locally.

## Test and evaluate

```
pytest -q          # unit and API tests, no Ollama needed
python eval.py     # retrieval quality per search mode (needs Ollama and the database)
```

## Hosting (Hugging Face Spaces)

The `Dockerfile` builds one container: Ollama runs only the embedding model, Gemini writes the answers
(`CHAT_PROVIDER=gemini`), and the document index is built into the image from the public AWS
Well-Architected PDF. `DEMO_MODE=1` limits requests per visitor (6 per minute) and per day (300),
turns off deletes, and keeps the 25 newest imports.

1. Create a Space at https://huggingface.co/new-space with the **Docker** SDK (blank template).
2. In the Space **Settings → Variables and secrets**, add the secret `GEMINI_API_KEY`.
3. Push this folder to the Space repository. The first build takes about 10 to 20 minutes.

### Render (free plan)

`Dockerfile.render` is a smaller image with no Ollama: embeddings come from a small ONNX model in the
app process (`EMBED_PROVIDER=fastembed`, bge-small-en-v1.5, `embeddings.py`), so the app uses about
300 MB of the 512 MB. With this model, hybrid search reached Hit@4 76% and MRR 0.61 on the evaluation set.
(Gemini embeddings also work, `EMBED_PROVIDER=gemini`, but the free tier allows only 1000 per day.)

1. On https://render.com, click **New → Blueprint** and choose this GitHub repository (`render.yaml`).
2. Enter `GEMINI_API_KEY` when Render asks. The build makes the document index without the key (a few minutes).
3. The free service sleeps after 15 minutes without visitors; the first request then takes about 1 minute.
   Imported architectures are lost when it restarts.

To run the same container locally: `docker compose up --build`, then open http://localhost:7860.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/architectures` | List architectures |
| GET | `/api/architectures/{id}` | Components, links, and private-network boundary |
| POST | `/api/architectures/import` | `{filename, content}` of a `.tf` or `.yaml` file |
| DELETE | `/api/architectures/{id}` | Delete an imported architecture |
| GET | `/api/architectures/{id}/impact/{key}` | Components affected when `key` fails |
| POST | `/api/ask` | Complete answer: `{question, architecture, component, mode, failure, history}` |
| POST | `/api/ask/stream` | Same, streamed as NDJSON events (`sources`, `token`, `error`, `done`) |
| GET | `/api/stats` | Latency, throughput, and grounding averages per engine and search mode |
| GET | `/api/agent/status` | Whether a Gemini key is set, and the model |
| POST | `/api/agent/ask/stream` | Tutor agent; same body as `/api/ask`, plus `step` events |
| POST | `/api/architectures/{id}/review/stream` | Multi-agent review: `step`, `findings`, `report` events |

## Notes

- The page number in a citation is the page position in the PDF file, not the number printed on the page.
- The grounding check is a similarity heuristic. It finds sentences that are far from every source, but it
  cannot prove that a close sentence is correct.
- Imported architectures are saved in `architectures/`. The importer reads AWS resources best; common
  Google Cloud and Azure resources are mapped too.
