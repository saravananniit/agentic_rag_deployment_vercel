"""FastAPI app: collection-wise upload / search / record-management endpoints.

Azure OpenAI credentials come from request headers on /api/search only:
  X-Azure-Endpoint, X-Azure-Deployment, X-Azure-Key, X-Azure-Api-Version (optional),
  X-Tavily-Key (optional, enables the web_search tool)
They are used for that request and never stored or logged.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # load .env before collection settings are imported

from fastapi import FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import astra_store as store
from . import parsers
from .agent import AzureSettings, run_agent

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "4")) * 1024 * 1024  # Vercel body limit ~4.5MB
MAX_CHUNKS = 400

app = FastAPI(title="Agentic RAG on Astra DB")


def _astra_call(fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except HTTPException:
        raise
    except ValueError as e:  # bad collection / collection without embedding service
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        raise HTTPException(500, str(e))
    except Exception as e:
        raise HTTPException(502, f"Astra DB error: {type(e).__name__}: {e}")


def _require_collection(name: str | None, optional: bool = False) -> str | None:
    if not name:
        if optional:
            return None
        raise HTTPException(400, f"collection is required; one of {store.COLLECTIONS}")
    if name not in store.COLLECTIONS:
        raise HTTPException(400, f"collection must be one of {store.COLLECTIONS}")
    return name


# ------------------------------------------------------------------ routes
@app.get("/api/health")
def health():
    def _check():
        info = store.ensure_collections()
        return {"ok": True, **info, **store.stats()}
    return _astra_call(_check)


@app.get("/api/config")
def config():
    return {"collections": store.COLLECTIONS,
            "supported": sorted(parsers.SUPPORTED),
            "max_upload_mb": MAX_UPLOAD_BYTES // (1024 * 1024)}


@app.post("/api/upload")
async def upload(
    file: UploadFile = File(...),
    collection: str = Form(...),
    allow_duplicate: bool = Form(False),
):
    _require_collection(collection)
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File too large. Limit is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
    name = file.filename or "upload"
    try:
        text = parsers.extract_text(name, data)
    except parsers.UnsupportedFile as e:
        raise HTTPException(415, str(e))
    except Exception as e:
        raise HTTPException(422, f"Could not read '{name}': {type(e).__name__}: {e}")
    chunks = parsers.chunk_text(text)
    if not chunks:
        raise HTTPException(422, "No extractable text found (scanned images need OCR, which is not supported).")
    if len(chunks) > MAX_CHUNKS:
        raise HTTPException(413, f"Document yields {len(chunks)} chunks; limit is {MAX_CHUNKS} per upload.")

    content_hash = hashlib.sha256(data).hexdigest()

    def _go():
        store.ensure_collections()
        dup = store.find_by_hash(content_hash, collection)
        if dup and not allow_duplicate:
            raise HTTPException(409, f"This file was already uploaded to {collection} as '{dup['filename']}'.")
        return store.store_document(
            filename=name, file_type=parsers.extension(name).lstrip("."), collection=collection,
            chunks=chunks, size_bytes=len(data), content_hash=content_hash)

    return {"ok": True, "document": _astra_call(_go)}


@app.get("/api/documents")
def documents(collection: str | None = None, page: int = Query(1, ge=1),
              page_size: int = Query(10, ge=1, le=50)):
    return _astra_call(store.list_documents, _require_collection(collection or None, optional=True),
                       page, page_size)


@app.get("/api/documents/{doc_id}/chunks")
def document_chunks(doc_id: str, collection: str, page: int = Query(1, ge=1),
                    page_size: int = Query(5, ge=1, le=20)):
    _require_collection(collection)
    return _astra_call(store.get_chunks, collection, doc_id, page, page_size)


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str):
    res = _astra_call(store.delete_document, doc_id)
    if not res["found"]:
        raise HTTPException(404, "Document not found in the registry.")
    return {"ok": True, **res}


@app.get("/api/collections/{name}/records")
def collection_records(name: str, page: int = Query(1, ge=1), page_size: int = Query(10, ge=1, le=50)):
    _require_collection(name)
    return _astra_call(store.list_records, name, page, page_size)


@app.get("/api/stats")
def stats():
    return _astra_call(store.stats)


class SearchBody(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    mode: str = Field("agentic", pattern="^(agentic|plain)$")
    collection: str | None = None  # None = all collections
    max_steps: int = Field(4, ge=1, le=6)


@app.post("/api/search")
def search(
    body: SearchBody,
    x_azure_endpoint: str = Header(""),
    x_azure_deployment: str = Header(""),
    x_azure_key: str = Header(""),
    x_azure_api_version: str = Header("2025-04-01-preview"),
    x_tavily_key: str = Header(""),
):
    if not (x_azure_endpoint and x_azure_deployment and x_azure_key):
        raise HTTPException(400, "Azure endpoint, deployment name and API key are required "
                                 "(enter them in Settings).")
    if not x_azure_endpoint.startswith("https://"):
        raise HTTPException(400, "Azure endpoint must start with https://")
    collection = _require_collection(body.collection or None, optional=True)
    settings = AzureSettings(
        endpoint=x_azure_endpoint.strip(), deployment=x_azure_deployment.strip(),
        api_key=x_azure_key.strip(), api_version=(x_azure_api_version or "2025-04-01-preview").strip(),
        tavily_key=x_tavily_key.strip() or None)
    try:
        return run_agent(body.question, settings, mode=body.mode,
                         collection=collection, max_steps=body.max_steps)
    except HTTPException:
        raise
    except RuntimeError as e:
        raise HTTPException(500, str(e).replace(x_azure_key, "***"))
    except Exception as e:
        msg = str(e).replace(x_azure_key, "***")
        raise HTTPException(502, f"{type(e).__name__}: {msg}")


# Local dev: serve the UI. On Vercel the static index.html is served directly.
@app.get("/", include_in_schema=False)
def index():
    p = Path(__file__).resolve().parent.parent / "index.html"
    return FileResponse(p)
