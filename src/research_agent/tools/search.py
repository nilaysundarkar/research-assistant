"""Web search tool backed by the Tavily API."""

from __future__ import annotations

import os
from typing import Any

def search_tool(query: str, max_results: int = 5) -> dict[str, Any]:
    """Search the web for ``query`` and return the top ``max_results``.

    Returns a JSON-serialisable dict containing an ``answer`` (Tavily's
    auto-generated summary if available) and a ``results`` list with
    ``title``, ``url``, ``snippet`` for each hit. The shape is
    deliberately compact so the LLM context isn't flooded with raw HTML.
    """
    if not isinstance(query, str) or not query.strip():
        return {"ok": False, "error": "query must be a non-empty string"}

    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return {
            "ok": False,
            "error": "TAVILY_API_KEY is not set; cannot perform web search",
        }

    try:
        from tavily import TavilyClient
    except ImportError:
        return {"ok": False, "error": "tavily-python is not installed"}

    max_results = max(1, min(int(max_results), 10))

    try:
        client = TavilyClient(api_key=api_key)
        response = client.search(
            query=query,
            search_depth="basic",
            max_results=max_results,
            include_answer=True,
        )
    except Exception as exc:
        return {"ok": False, "error": f"search failed: {exc}"}

    raw_results = response.get("results", []) or []
    results = [
        {
            "title": r.get("title", "").strip(),
            "url": r.get("url", ""),
            "snippet": (r.get("content") or "").strip()[:500],
        }
        for r in raw_results
    ]

    return {
        "ok": True,
        "query": query,
        "answer": response.get("answer"),
        "results": results,
        "n_results": len(results),
    }
