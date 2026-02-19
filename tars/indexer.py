"""Orchestration: discover memory files, chunk, embed, and store."""

from pathlib import Path

from tars.chunker import _content_hash, chunk_markdown
from tars.db import (
    delete_chunks_for_file,
    ensure_collection,
    init_db,
    insert_chunks,
    upsert_file,
)
from tars.embeddings import embed, embedding_dimensions
from tars.memory import _MEMORY_FILES, _memory_dir


def _discover_files(memory_dir: Path) -> list[tuple[Path, str]]:
    """Return (path, memory_type) pairs for all indexable files."""
    found: list[tuple[Path, str]] = []
    for memory_type, filename in _MEMORY_FILES.items():
        p = memory_dir / filename
        if p.is_file():
            found.append((p, memory_type))
    sessions_dir = memory_dir / "sessions"
    if sessions_dir.is_dir():
        for p in sorted(sessions_dir.glob("*.md")):
            found.append((p, "episodic"))
    return found


def build_index(*, model: str = "qwen3-embedding:0.6b") -> dict:
    """Index all memory files into sqlite-vec. Returns stats."""
    stats = {"indexed": 0, "skipped": 0, "chunks": 0}

    memory_dir = _memory_dir()
    if memory_dir is None:
        return stats

    dim = embedding_dimensions(model)
    conn = init_db(dim=dim)
    if conn is None:
        return stats

    try:
        collection_id = ensure_collection(conn)
        files = _discover_files(memory_dir)

        for filepath, memory_type in files:
            content = filepath.read_text(encoding="utf-8", errors="replace")
            content_hash = _content_hash(content)
            stat = filepath.stat()

            file_id, changed = upsert_file(
                conn,
                collection_id=collection_id,
                path=str(filepath),
                title=filepath.stem,
                memory_type=memory_type,
                content_hash=content_hash,
                mtime=stat.st_mtime,
                size=stat.st_size,
            )

            if not changed:
                stats["skipped"] += 1
                continue

            delete_chunks_for_file(conn, file_id)
            chunks = chunk_markdown(content)
            if chunks:
                embeddings = embed([c.content for c in chunks], model=model)
                insert_chunks(conn, file_id, chunks, embeddings)
                stats["chunks"] += len(chunks)

            stats["indexed"] += 1
    finally:
        conn.close()

    return stats
