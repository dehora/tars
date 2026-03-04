"""Thin wrapper around ollama.embed() for embedding text."""

import ollama

DEFAULT_EMBEDDING_MODEL = "qwen3-embedding:0.6b"

_DEFAULT_QUERY_INSTRUCT = (
    "Given a search query, retrieve relevant passages that answer the query"
)


def embed(
    texts: str | list[str],
    *,
    model: str = DEFAULT_EMBEDDING_MODEL,
    instruct: str | None = None,
) -> list[list[float]]:
    """Embed one or more texts. Returns a list of embedding vectors.

    When instruct is provided, each text is wrapped as
    ``Instruct: {instruct}\\nQuery:{text}`` — use this for query
    embeddings with models that support instruction-aware asymmetry
    (e.g. qwen3-embedding). Document embeddings should omit instruct.
    """
    if isinstance(texts, str):
        texts = [texts]

    if instruct:
        texts = [f"Instruct: {instruct}\nQuery:{t}" for t in texts]

    response = ollama.embed(model=model, input=texts)
    embeddings = response.get("embeddings", [])

    # Safety: only zip as far as the shorter array
    count = min(len(texts), len(embeddings))
    return embeddings[:count]


def embedding_dimensions(model: str = DEFAULT_EMBEDDING_MODEL) -> int:
    """Probe the model to determine embedding dimensionality."""
    vecs = embed("dimension probe", model=model)
    if not vecs:
        raise RuntimeError(f"Model {model!r} returned no embeddings")
    return len(vecs[0])
