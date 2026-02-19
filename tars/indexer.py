"""Orchestration: discover memory files, chunk, embed, and store."""

from pathlib import Path

from tars.chunker import _content_hash, chunk_markdown
from tars.db import (
    _get_metadata,
    _set_metadata,
    delete_chunks_for_file,
    delete_file,
    ensure_collection,
    get_indexed_paths,
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
    stats = {"indexed": 0, "skipped": 0, "chunks": 0, "deleted": 0}

    memory_dir = _memory_dir()
    if memory_dir is None:
        return stats

    dim = embedding_dimensions(model)
    conn = init_db(dim=dim)
    if conn is None:
        return stats

    try:
        collection_id = ensure_collection(conn)

        # Detect embedding model change and force full reindex
        stored_model = _get_metadata(conn, "embedding_model")
        model_changed = stored_model is not None and stored_model != model
        if model_changed:
            # Clear all chunks so every file gets re-embedded
            for fid in get_indexed_paths(conn, collection_id).values():
                delete_chunks_for_file(conn, fid)
            # Reset content hashes to force reprocessing
            conn.execute(
                "UPDATE files SET content_hash = '' WHERE collection_id = ?",
                (collection_id,),
            )
            conn.commit()
        _set_metadata(conn, "embedding_model", model)
        conn.commit()

        files = _discover_files(memory_dir)

        # Remove deleted files from the index
        discovered_paths = {str(fp) for fp, _ in files}
        indexed_paths = get_indexed_paths(conn, collection_id)
        for path, file_id in indexed_paths.items():
            if path not in discovered_paths:
                delete_file(conn, file_id)
                stats["deleted"] += 1

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
                chunk_embeddings = embed([c.content for c in chunks], model=model)
                if len(chunk_embeddings) != len(chunks):
                    raise ValueError(
                        f"Embedding count mismatch for {filepath}: "
                        f"got {len(chunk_embeddings)} embeddings for {len(chunks)} chunks"
                    )
                insert_chunks(conn, file_id, chunks, chunk_embeddings)
                stats["chunks"] += len(chunks)

            stats["indexed"] += 1
    finally:
        conn.close()

    return stats
