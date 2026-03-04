"""Query rewriting and HyDE for expanded retrieval."""

import os

import ollama

RETRIEVAL_MODEL = os.environ.get("TARS_RETRIEVAL_MODEL", "gemma3:4b")

_MIN_HYDE_WORDS = 5
_MAX_REWRITES = 4

_EXPAND_PROMPT = (
    "Generate 2-4 short keyword-dense search queries for the following query."
    " One per line, no numbering, no explanation.\n\nQuery: {query}"
)

_HYDE_PROMPT = (
    "Write 3-5 short bullet points that a relevant document might contain"
    " for the following query. Bullets only, no preamble.\n\nQuery: {query}"
)


def expand_queries(query: str, *, model: str = RETRIEVAL_MODEL) -> list[str]:
    """Generate keyword-dense rewrites of a conversational query.

    Always returns the original query as the first element.
    """
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": _EXPAND_PROMPT.format(query=query)}],
    )
    text = response.get("message", {}).get("content", "")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return [query] + lines[:_MAX_REWRITES]


def generate_hyde(query: str, *, model: str = RETRIEVAL_MODEL) -> str | None:
    """Generate a hypothetical document for embedding-based retrieval.

    Returns None if the query is shorter than the word gate.
    """
    if len(query.split()) < _MIN_HYDE_WORDS:
        return None

    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": _HYDE_PROMPT.format(query=query)}],
    )
    text = response.get("message", {}).get("content", "")
    return text.strip() or None
