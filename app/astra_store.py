"""Astra DB access, collection-wise.

Documents are stored in (and searched from) the collections you choose in the UI:
by default hr_docs, it_docs and fin_docs (override with ASTRA_COLLECTIONS).
Each collection's capabilities are read from Astra, and the app adapts:
  - embedding service (vectorize)  -> text goes in `$vectorize`, Astra embeds it
  - lexical enabled                -> text also goes in `$lexical` (hybrid search)
  - reranking enabled              -> hybrid results are reranked by Astra
A small `doc_registry` collection tracks uploaded files for the Storage tab.
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any

from astrapy import DataAPIClient
from astrapy.info import (
    CollectionDefinition,
    CollectionLexicalOptions,
    CollectionRerankOptions,
    CollectionVectorOptions,
    RerankServiceOptions,
    VectorServiceOptions,
)

LEGACY_COLLECTION = os.getenv("ASTRA_KB_COLLECTION", "").strip() or None
_configured_collections = os.getenv("ASTRA_COLLECTIONS", "hr_docs,it_docs,fin_docs")
COLLECTIONS = [c.strip() for c in _configured_collections.split(",") if c.strip()]
if LEGACY_COLLECTION and LEGACY_COLLECTION not in COLLECTIONS:
    COLLECTIONS.append(LEGACY_COLLECTION)
REGISTRY_NAME = os.getenv("ASTRA_REGISTRY_COLLECTION", "doc_registry")
TICKETS_NAME = os.getenv("ASTRA_TICKETS_COLLECTION", "tickets")

INSERT_BATCH = 20  # Data API insert_many limit
_db = None
_caps: dict[str, dict] = {}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_db():
    global _db
    if _db is None:
        endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")
        token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
        if not endpoint or not token:
            raise RuntimeError(
                "Set ASTRA_DB_API_ENDPOINT and ASTRA_DB_APPLICATION_TOKEN environment variables."
            )
        _db = DataAPIClient().get_database(endpoint, token=token)
    return _db


def default_definition() -> CollectionDefinition:
    """Used only if a configured collection does not exist yet:
    NV-Embed-QA, 1024 dims, cosine, standard lexical, NVIDIA reranker."""
    return CollectionDefinition(
        vector=CollectionVectorOptions(
            dimension=1024, metric="cosine",
            service=VectorServiceOptions(provider="nvidia", model_name="NV-Embed-QA"),
        ),
        lexical=CollectionLexicalOptions(analyzer="standard"),
        rerank=CollectionRerankOptions(
            service=RerankServiceOptions(provider="nvidia", model_name="nvidia/llama-3.2-nv-rerankqa-1b-v2")
        ),
    )


def _describe(defn: CollectionDefinition | None) -> dict:
    vec = getattr(defn, "vector", None)
    svc = getattr(vec, "service", None)
    lex = getattr(defn, "lexical", None)
    rr = getattr(defn, "rerank", None)
    rr_svc = getattr(rr, "service", None)
    return {
        "embedding": f"{svc.provider}/{svc.model_name}" if svc else None,
        "dimension": getattr(vec, "dimension", None),
        "metric": getattr(vec, "metric", None),
        "lexical": bool(lex and lex.enabled is not False),
        "rerank": (rr_svc.model_name if rr_svc else None) if (rr and rr.enabled is not False) else None,
    }


def load_capabilities(refresh: bool = False) -> dict[str, dict]:
    """Read each configured collection's settings from Astra (cached per process)."""
    if _caps and not refresh:
        return _caps
    existing = {d.name: d.definition for d in get_db().list_collections()}
    _caps.clear()
    for name in COLLECTIONS:
        if name in existing:
            _caps[name] = {"exists": True, **_describe(existing[name])}
        else:
            _caps[name] = {"exists": False, "embedding": None, "dimension": None,
                           "metric": None, "lexical": False, "rerank": None}
    return _caps


def check_collection(name: str) -> dict:
    if name not in COLLECTIONS:
        raise ValueError(f"Unknown collection '{name}'. Choose one of {COLLECTIONS}.")
    caps = load_capabilities().get(name)
    if not caps or not caps["exists"]:
        raise ValueError(f"Collection '{name}' does not exist.")
    return caps


def ensure_collections() -> dict[str, Any]:
    """Create missing helper collections, and any configured collection that is missing."""
    db = get_db()
    existing = set(db.list_collection_names())
    created = []
    for name in COLLECTIONS:
        if name not in existing:
            db.create_collection(name, definition=default_definition())
            created.append(name)
    for name in (REGISTRY_NAME, TICKETS_NAME):
        if name not in existing:
            db.create_collection(name)
            created.append(name)
    load_capabilities(refresh=True)
    return {"created": created, "collections": COLLECTIONS}


def col(name: str):
    return get_db().get_collection(name)


def registry():
    return col(REGISTRY_NAME)


def tickets():
    return col(TICKETS_NAME)


# ---------------------------------------------------------------- writes
def store_document(
    *, filename: str, file_type: str, collection: str, chunks: list[str],
    size_bytes: int, content_hash: str,
) -> dict[str, Any]:
    caps = check_collection(collection)
    if not caps["embedding"]:
        raise ValueError(
            f"Collection '{collection}' has no embedding service, so Astra cannot embed text for it. "
            "Use a collection created with an embedding service (e.g. NVIDIA NV-Embed-QA).")
    doc_id = str(uuid.uuid4())
    uploaded_at = _now()
    rows = []
    for i, text in enumerate(chunks):
        row = {
            "doc_id": doc_id, "filename": filename, "file_type": file_type,
            "chunk_index": i, "uploaded_at": uploaded_at,
            "text": text,
            "$vectorize": text,  # Astra embeds this with the collection's own service
        }
        if caps["lexical"]:
            row["$lexical"] = text  # keyword side of hybrid search
        rows.append(row)
    target = col(collection)
    try:
        for start in range(0, len(rows), INSERT_BATCH):
            target.insert_many(rows[start:start + INSERT_BATCH], ordered=False)
    except Exception:
        target.delete_many({"doc_id": doc_id})  # do not leave a half-stored file behind
        raise
    reg = {
        "_id": doc_id, "doc_id": doc_id, "filename": filename, "file_type": file_type,
        "collection": collection, "chunk_count": len(rows), "size_bytes": size_bytes,
        "content_hash": content_hash, "uploaded_at": uploaded_at,
    }
    registry().insert_one(reg)
    return reg


def find_by_hash(content_hash: str, collection: str) -> dict | None:
    return registry().find_one({"content_hash": content_hash, "collection": collection})


def delete_document(doc_id: str) -> dict:
    """Delete a file's chunks and its registry row. Rows saved by older versions have no
    `collection` field; for those, clean the id out of every known collection."""
    reg = registry().find_one({"_id": doc_id})
    if not reg:
        return {"deleted_chunks": 0, "found": False}
    coll = reg.get("collection")
    targets = [coll] if coll else ["kb_docs", *COLLECTIONS]
    deleted = 0
    for name in targets:
        try:
            res = col(name).delete_many({"doc_id": doc_id})
            if res.deleted_count and res.deleted_count > 0:
                deleted += res.deleted_count
        except Exception:
            if coll:  # a real target failed: do not drop the registry row
                raise
    registry().delete_one({"_id": doc_id})
    return {"deleted_chunks": deleted, "found": True, "collection": coll}


# ---------------------------------------------------------------- reads
def _document_filter(collection: str | None) -> dict:
    if collection is None:
        return {}
    filters = [{"collection": collection}]
    if collection == LEGACY_COLLECTION:
        filters.append({"collection": {"$exists": False}})
    return filters[0] if len(filters) == 1 else {"$or": filters}


def list_documents(collection: str | None, page: int, page_size: int) -> dict[str, Any]:
    filt = _document_filter(collection)
    reg = registry()
    total = reg.count_documents(filt, upper_bound=10_000)
    skip = (page - 1) * page_size
    docs = list(reg.find(filt, limit=total)) if total else []
    docs.sort(key=lambda d: d.get("uploaded_at", ""), reverse=True)
    if LEGACY_COLLECTION in COLLECTIONS:
        for doc in docs:
            if not doc.get("collection"):
                doc["collection"] = LEGACY_COLLECTION
    return {
        "items": docs[skip:skip + page_size], "page": page, "page_size": page_size,
        "total": total, "total_pages": max(1, -(-total // page_size)),
    }


def get_chunks(collection: str, doc_id: str, page: int, page_size: int) -> dict[str, Any]:
    check_collection(collection)
    c = col(collection)
    total = c.count_documents({"doc_id": doc_id}, upper_bound=10_000)
    skip = (page - 1) * page_size
    rows = list(c.find({"doc_id": doc_id},
                       projection={"text": 1, "chunk_index": 1, "filename": 1},
                       limit=skip + page_size))
    rows.sort(key=lambda r: r.get("chunk_index", 0))
    return {
        "items": rows[skip:skip + page_size], "page": page, "page_size": page_size,
        "total": total, "total_pages": max(1, -(-total // page_size)),
    }


BROWSE_LIMIT = 1000
_HIDE = {"_id", "text", "content", "page_content", "$vector", "$vectorize", "$lexical", "$similarity"}


def _record_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(filter(None, (_record_text(item) for item in value)))
    if isinstance(value, dict):
        for key in ("text", "content", "page_content", "value"):
            text = _record_text(value.get(key))
            if text:
                return text
    return ""


def _field_value(value: Any, limit: int = 160) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str, ensure_ascii=False)
    return text[:limit] + ("…" if len(text) > limit else "")


def list_records(collection: str, page: int, page_size: int) -> dict[str, Any]:
    """Raw records in a collection (including ones not uploaded through this app), paginated.
    Browsing is capped to the first BROWSE_LIMIT records."""
    check_collection(collection)
    c = col(collection)
    skip = (page - 1) * page_size
    if skip >= BROWSE_LIMIT:
        raise ValueError(f"Browsing is limited to the first {BROWSE_LIMIT} records.")
    total = c.count_documents({}, upper_bound=10_000)
    rows = list(c.find({}, limit=min(skip + page_size, BROWSE_LIMIT)))[skip:skip + page_size]
    items = []
    for r in rows:
        metadata = r.get("metadata") if isinstance(r.get("metadata"), dict) else {}
        text = _record_text(r.get("text") or r.get("content") or r.get("page_content"))
        if not text:
            text = _record_text(metadata.get("text") or metadata.get("content") or
                                metadata.get("page_content") or metadata.get("content_blocks"))
        raw_fields = {k: v for k, v in r.items() if k not in _HIDE and k != "metadata"}
        raw_fields.update({
            k: v for k, v in metadata.items()
            if k not in {"filename", "text", "content", "page_content", "content_blocks"}
            and v not in (None, "", [], {})
        })
        preferred = ("category", "mimetype", "file_size", "timestamp")
        ordered_keys = [k for k in preferred if k in raw_fields]
        ordered_keys.extend(k for k in raw_fields if k not in ordered_keys)
        fields = {k: _field_value(raw_fields[k]) for k in ordered_keys}
        items.append({
            "_id": str(r.get("_id")),
            "source": r.get("filename") or metadata.get("filename") or "(not uploaded via this app)",
            "text": text[:300] + ("…" if len(text) > 300 else ""),
            "fields": fields,
        })
    return {"items": items, "page": page, "page_size": page_size, "total": total,
            "total_pages": max(1, -(-min(total, BROWSE_LIMIT) // page_size))}


def stats() -> dict[str, Any]:
    caps = load_capabilities(refresh=True)
    reg = registry()
    out = []
    for name in COLLECTIONS:
        c = caps[name]
        item = {"name": name, **c, "documents": 0, "chunks": None}
        if c["exists"]:
            item["documents"] = reg.count_documents(_document_filter(name), upper_bound=10_000)
            try:
                item["chunks"] = col(name).estimated_document_count()
            except Exception:
                pass
        out.append(item)
    return {"collections": out, "documents": sum(c["documents"] for c in out)}


def _clean(d: dict, collection: str, scores: dict) -> dict:
    text = d.get("text") or d.get("content") or d.get("page_content") or ""
    return {
        "collection": collection, "doc_id": d.get("doc_id"),
        "filename": d.get("filename") or "(record without filename)",
        "chunk_index": d.get("chunk_index"), "text": text, "scores": scores,
    }


_PROJ = {"text": 1, "content": 1, "page_content": 1, "filename": 1, "chunk_index": 1, "doc_id": 1}


def search_collection(query: str, collection: str, top_k: int = 5) -> list[dict]:
    """Hybrid (vector + lexical) with reranking when the collection supports it,
    otherwise Astra vector search using the collection's own embedding service."""
    caps = check_collection(collection)
    c = col(collection)
    if caps["lexical"] and caps["rerank"]:
        try:
            out = []
            for r in c.find_and_rerank({}, sort={"$hybrid": query}, limit=top_k,
                                       include_scores=True, projection=_PROJ):
                scores = {k: (round(v, 4) if isinstance(v, (int, float)) else v)
                          for k, v in (r.scores or {}).items()}
                out.append(_clean(dict(r.document), collection, scores))
            return out
        except Exception:
            pass  # fall through to plain vector search
    if not caps["embedding"]:
        raise ValueError(f"Collection '{collection}' has no embedding service, so it cannot be searched by text.")
    out = []
    for d in c.find({}, sort={"$vectorize": query}, limit=top_k,
                    include_similarity=True, projection=_PROJ):
        sim = d.pop("$similarity", None)
        out.append(_clean(d, collection, {"$vector": round(sim, 4) if sim is not None else None}))
    return out


def search(query: str, collection: str | None = None, top_k: int = 5) -> list[dict]:
    """One collection, or all configured ones merged by score."""
    if collection:
        return search_collection(query, collection, top_k)
    merged: list[dict] = []
    errors = []
    for name in COLLECTIONS:
        try:
            if load_capabilities()[name]["exists"]:
                merged.extend(search_collection(query, name, top_k))
        except Exception as e:
            errors.append(f"{name}: {e}")
    if not merged and errors:
        raise RuntimeError("; ".join(errors))

    def score(h: dict) -> float:
        s = h["scores"]
        v = s.get("$rerank", s.get("$vector"))
        return v if isinstance(v, (int, float)) else 0.0
    merged.sort(key=score, reverse=True)
    return merged[:top_k]


def create_ticket(team: str, title: str, description: str) -> dict:
    t = {
        "_id": f"TKT-{uuid.uuid4().hex[:8].upper()}", "team": team, "title": title,
        "description": description, "status": "open", "created_at": _now(), "demo": True,
    }
    tickets().insert_one(t)
    return {"ticket_id": t["_id"], "status": "open",
            "note": "Demo ticket stored in Astra; no real system was contacted."}
