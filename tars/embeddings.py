"""Thin wrapper around ollama.embed() for embedding text."""

import ollama

DEFAULT_EMBEDDING_MODEL = "qwen3-embedding:0.6b"


def embed(
    texts: str | list[str],
    *,
    model: str = DEFAULT_EMBEDDING_MODEL,
) -> list[list[float]]:
    """Embed one or more texts. Returns a list of embedding vectors."""
    if isinstance(texts, str):
        texts = [texts]

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
