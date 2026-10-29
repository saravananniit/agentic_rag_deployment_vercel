"""Agentic RAG loop on Azure OpenAI function calling.

The model decides which tool to call (and how often):
  search_kb, list_documents, calculator, web_search (optional), create_ticket (demo).
Azure endpoint / deployment / key arrive per request and are never stored.
"""
from __future__ import annotations

import ast
import json
import operator
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx
from openai import AzureOpenAI

from . import astra_store as store

MAX_STEPS_DEFAULT = 4


@dataclass
class AzureSettings:
    endpoint: str
    deployment: str
    api_key: str
    api_version: str = "2025-04-01-preview"
    tavily_key: str | None = None

    def client(self) -> AzureOpenAI:
        return AzureOpenAI(
            azure_endpoint=self.endpoint.rstrip("/"),
            api_key=self.api_key,
            api_version=self.api_version,
            timeout=45.0,
            max_retries=1,
        )


# ------------------------------------------------------------------ tools
_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def safe_eval(expr: str) -> float:
    """Arithmetic only; no names, calls or attributes."""
    def ev(n: ast.AST):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            if isinstance(n.op, ast.Pow) and abs(ev(n.right)) > 100:
                raise ValueError("exponent too large")
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        raise ValueError("only basic arithmetic is allowed")
    return ev(ast.parse(expr.strip(), mode="eval"))


def _web_search(query: str, key: str) -> list[dict]:
    r = httpx.post(
        "https://api.tavily.com/search",
        json={"api_key": key, "query": query, "max_results": 3, "include_answer": False},
        timeout=20.0,
    )
    r.raise_for_status()
    return [
        {"title": x.get("title"), "url": x.get("url"), "content": (x.get("content") or "")[:500]}
        for x in r.json().get("results", [])
    ]


def tool_specs(settings: AzureSettings) -> list[dict]:
    dept = {"type": "string", "enum": store.COLLECTIONS,
            "description": "Optional: restrict to one collection. Omit to search all collections."}
    tools = [
        {"type": "function", "function": {
            "name": "search_kb",
            "description": "Search the company knowledge base (hybrid semantic + keyword search "
                           "with reranking where available). Collections: " + ", ".join(store.COLLECTIONS) +
                           ". Use for any question about internal policies or procedures.",
            "parameters": {"type": "object", "properties": {
                "query": {"type": "string", "description": "Self-contained search query."},
                "collection": dept}, "required": ["query"]}}},
        {"type": "function", "function": {
            "name": "list_documents",
            "description": "List which uploaded documents exist (optionally for one collection). "
                           "Use to check whether a topic is likely covered.",
            "parameters": {"type": "object", "properties": {"collection": dept}}}},
        {"type": "function", "function": {
            "name": "calculator",
            "description": "Evaluate an arithmetic expression, e.g. leave balances or totals "
                           "after retrieving numbers from documents.",
            "parameters": {"type": "object", "properties": {
                "expression": {"type": "string"}}, "required": ["expression"]}}},
        {"type": "function", "function": {
            "name": "current_datetime",
            "description": "Get the current UTC date and time.",
            "parameters": {"type": "object", "properties": {}}}},
        {"type": "function", "function": {
            "name": "create_ticket",
            "description": "Create a DEMO support ticket (stored in Astra, no real system) when "
                           "the user has a problem the documents cannot solve or asks for one.",
            "parameters": {"type": "object", "properties": {
                "team": {"type": "string", "enum": store.COLLECTIONS},
                "title": {"type": "string"}, "description": {"type": "string"}},
                "required": ["team", "title", "description"]}}},
    ]
    if settings.tavily_key:
        tools.append({"type": "function", "function": {
            "name": "web_search",
            "description": "Search the public web. Use ONLY when the knowledge base does not "
                           "contain the answer or the question needs current external facts.",
            "parameters": {"type": "object", "properties": {
                "query": {"type": "string"}}, "required": ["query"]}}})
    return tools


def run_tool(name: str, args: dict, settings: AzureSettings, sources: list[dict]) -> Any:
    if name == "search_kb":
        hits = store.search(args["query"], args.get("collection"), top_k=5)
        for h in hits:
            key = (h.get("doc_id"), h.get("chunk_index"))
            if key not in {(s.get("doc_id"), s.get("chunk_index")) for s in sources}:
                sources.append({
                    "doc_id": h.get("doc_id"), "filename": h.get("filename"),
                    "collection": h.get("collection"), "chunk_index": h.get("chunk_index"),
                    "text": h.get("text"), "scores": h.get("scores"),
                })
        return [
            {"source": f"{h.get('filename')}#chunk{h.get('chunk_index')}",
             "collection": h.get("collection"), "text": h.get("text"),
             "scores": h.get("scores")}
            for h in hits
        ]
    if name == "list_documents":
        res = store.list_documents(args.get("collection"), 1, 25)
        return [{"filename": d["filename"], "collection": d["collection"],
                 "chunks": d["chunk_count"]} for d in res["items"]] or "The knowledge base is empty."
    if name == "calculator":
        return {"result": safe_eval(args["expression"])}
    if name == "current_datetime":
        return {"utc": datetime.now(timezone.utc).isoformat()}
    if name == "create_ticket":
        return store.create_ticket(args["team"], args["title"], args["description"])
    if name == "web_search":
        if not settings.tavily_key:
            return {"error": "web_search is not enabled (no Tavily key provided)."}
        return _web_search(args["query"], settings.tavily_key)
    return {"error": f"unknown tool {name}"}


SYSTEM_PROMPT = """You are a helpful company help-desk assistant with tools.
Rules:
- Knowledge is split into collections; omit the collection argument to search all of them, or pick one
  when the question clearly belongs to it (hr_docs = HR, it_docs = IT, fin_docs = finance).
- Ground answers about company policy in search_kb results and cite sources as [filename#chunkN].
- Pick tools deliberately: search the knowledge base first for internal questions; use calculator
  for arithmetic instead of doing it mentally; use web_search only if the knowledge base lacks the
  answer and the question needs outside facts; create_ticket only when a problem needs follow-up
  or the user asks for a ticket.
- If search_kb returns nothing relevant, try ONE reworded query (or another collection) before
  concluding. If the answer is still not available, say so plainly. Never invent policy details.
- Keep the final answer concise."""


def _summarise(result: Any, limit: int = 300) -> str:
    s = json.dumps(result, default=str)
    return s if len(s) <= limit else s[:limit] + "…"


def _final_answer(response: Any) -> str:
    if not getattr(response, "choices", None):
        raise RuntimeError("Azure OpenAI returned no answer choices.")
    answer = response.choices[0].message.content
    if not isinstance(answer, str) or not answer.strip():
        raise RuntimeError("Azure OpenAI returned an empty answer.")
    return answer


def run_agent(question: str, settings: AzureSettings, *, mode: str = "agentic",
              collection: str | None = None, max_steps: int = MAX_STEPS_DEFAULT) -> dict:
    client = settings.client()
    trace: list[dict] = []
    sources: list[dict] = []

    if mode == "plain":
        # Baseline: exactly one retrieval, one LLM call, no tools.
        hits_args = {"query": question, "collection": collection}
        hits = run_tool("search_kb", hits_args, settings, sources)
        trace.append({"step": 1, "type": "tool_call", "tool": "search_kb",
                      "args": hits_args, "result": _summarise(hits)})
        context = "\n\n".join(f"[{h['source']}] {h['text']}" for h in hits) or "(no results)"
        resp = client.chat.completions.create(
            model=settings.deployment,
            messages=[
                {"role": "system", "content": "Answer only from the context. Cite sources as "
                                              "[filename#chunkN]. If the context lacks the answer, say so."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
        )
        return {"answer": _final_answer(resp), "sources": sources,
                "trace": trace, "steps": 1, "mode": mode, "stopped": "done"}

    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if collection:
        messages.append({"role": "system",
                         "content": f"The user restricted this search to the '{collection}' collection."})
    messages.append({"role": "user", "content": question})
    tools = tool_specs(settings)

    for step in range(1, max_steps + 1):
        resp = client.chat.completions.create(
            model=settings.deployment, messages=messages, tools=tools, tool_choice="auto",
        )
        if not getattr(resp, "choices", None):
            raise RuntimeError("Azure OpenAI returned no answer choices.")
        msg = resp.choices[0].message
        if not msg.tool_calls:
            return {"answer": _final_answer(resp), "sources": sources, "trace": trace,
                    "steps": step, "mode": mode, "stopped": "done"}
        messages.append({
            "role": "assistant", "content": msg.content,
            "tool_calls": [{"id": c.id, "type": "function",
                            "function": {"name": c.function.name, "arguments": c.function.arguments}}
                           for c in msg.tool_calls],
        })
        for call in msg.tool_calls:
            args = {}
            try:
                args = json.loads(call.function.arguments or "{}")
                if collection and call.function.name in {"search_kb", "list_documents"}:
                    args["collection"] = collection
                result = run_tool(call.function.name, args, settings, sources)
            except Exception as e:  # tool errors go back to the model so it can adapt
                result = {"error": f"{type(e).__name__}: {e}"}
            trace.append({"step": step, "type": "tool_call", "tool": call.function.name,
                          "args": args, "result": _summarise(result)})
            messages.append({"role": "tool", "tool_call_id": call.id,
                             "content": json.dumps(result, default=str)})

    # step limit reached: force a final answer without tools
    messages.append({"role": "system", "content": "Step limit reached. Give your best final answer now "
                                                  "from what you have; say what is missing."})
    resp = client.chat.completions.create(model=settings.deployment, messages=messages)
    return {"answer": _final_answer(resp), "sources": sources, "trace": trace,
            "steps": max_steps, "mode": mode, "stopped": "step_limit"}
