# Agentic RAG on Astra DB (collection-wise)

FastAPI + a single `index.html`. Files are stored in, and searched from, the collections you pick in the UI:
`hr_docs`, `it_docs` and `fin_docs` by default (override with `ASTRA_COLLECTIONS=a,b,c`).
If `ASTRA_KB_COLLECTION` is set, that legacy single knowledge-base collection is also included so
existing documents and chunks remain visible in Storage. Registry entries from older versions that
do not have a `collection` field are associated with that legacy collection.
Embeddings use each collection's own Astra embedding service (for example NVIDIA `NV-Embed-QA`), so there is no
embedding key. If a collection also has lexical search and a reranker enabled (for example
`nvidia/llama-3.2-nv-rerankqa-1b-v2`), searches are hybrid and reranked automatically; otherwise plain vector search is used.
The LLM is Azure OpenAI; endpoint, deployment name (for example gpt-5-mini) and key are typed into the UI at run time.

The header strip shows each collection's detected capabilities. A collection with no embedding service cannot
accept uploads, and the app says so instead of failing silently. `doc_registry` (one row per uploaded file) and
`tickets` (demo tickets) are created if missing. Collections you list that do not exist are created with
NV-Embed-QA, cosine, lexical and the NVIDIA reranker.

## Run locally

```bash
pip install -r requirements.txt
export ASTRA_DB_API_ENDPOINT="https://<db-id>-<region>.apps.astra.datastax.com"
export ASTRA_DB_APPLICATION_TOKEN="AstraCS:..."
uvicorn app.main:app --reload
# open http://127.0.0.1:8000  -> Model settings -> enter Azure endpoint / deployment / key
```

## Deploy to Vercel

1. Push this folder to GitHub and import it in Vercel (framework preset: Other).
2. Add `ASTRA_DB_API_ENDPOINT` and `ASTRA_DB_APPLICATION_TOKEN` as environment variables.
3. Deploy. `api/index.py` exposes the FastAPI app; `index.html` is served statically.

Vercel limits: request bodies are capped around 4.5 MB (the app enforces 4 MB), and `vercel.json` sets a 60 s function duration.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/upload` | multipart `file` + `collection`; parse, chunk, store in that collection, register |
| POST | `/api/search` | agentic or plain RAG over one collection or all; Azure settings via `X-Azure-*` headers |
| GET | `/api/documents` | paginated stored documents (`collection`, `page`, `page_size`) |
| GET | `/api/documents/{id}/chunks?collection=` | paginated chunks of one document |
| GET | `/api/collections/{name}/records` | paginated raw records in a collection, including ones not uploaded via the app (first 1000) |
| DELETE | `/api/documents/{id}` | delete document and its chunks |
| GET | `/api/stats`, `/api/health`, `/api/config` | counts, collection check/creation, UI config |

## Agent tools

`search_kb` (hybrid + rerank), `list_documents`, `calculator`, `current_datetime`, `create_ticket` (demo, written to
the `tickets` collection) and `web_search` (only when a Tavily key is entered). The UI shows each tool call in an
agent trace, and a Plain RAG toggle runs a single retrieval for comparison.

## Demo questions

- "How many days of annual leave do new joiners get?" (one search)
- "If I take 5 days of annual leave, how many remain?" (search, then calculator)
- "What is the capital of Mars?" (not found, no invention)
- "My VPN is not working" (search, then offer a ticket)
