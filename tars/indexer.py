"""Orchestration: discover memory files, chunk, embed, and store."""

import sqlite3
import sys as _sys_mod
import time
from pathlib import Path

_stderr = _sys_mod.stderr

from tars.chunker import _content_hash, chunk_markdown
from tars.db import (
    _get_metadata,
    _notes_db_path,
    _prepare_db,
    _set_metadata,
    delete_chunks_for_file,
    delete_file,
    ensure_collection,
    get_indexed_paths,
    init_db,
    insert_chunks,
    upsert_file,
)
from tars.embeddings import DEFAULT_EMBEDDING_MODEL, embed, embedding_dimensions
from tars.memory import _MEMORY_FILES, _memory_dir
from tars.notes import _notes_dir

_EMBED_BATCH_SIZE = 64
_EMBED_MAX_RETRIES = 3
_MAX_CONTEXT_LEVELS = 3
_MAX_CONTEXT_CHARS = 120


def _embed_prefix(context: str) -> str:
    """Build a short embed prefix from a heading breadcrumb, capped to avoid overweighting."""
    if not context:
        return ""
    parts = context.split(" > ")
    capped = " > ".join(parts[:_MAX_CONTEXT_LEVELS])
    if len(capped) > _MAX_CONTEXT_CHARS:
        capped = capped[:_MAX_CONTEXT_CHARS].rsplit(" > ", 1)[0]
    return capped


def _batched_embed(texts: list[str], *, model: str) -> list[list[float]]:
    """Embed texts in batches with retry for resilience against transient failures."""
    all_embeddings: list[list[float]] = []
    for start in range(0, len(texts), _EMBED_BATCH_SIZE):
        batch = texts[start:start + _EMBED_BATCH_SIZE]
        last_err: Exception | None = None
        for attempt in range(_EMBED_MAX_RETRIES):
            try:
                batch_embeddings = embed(batch, model=model)
                if len(batch_embeddings) != len(batch):
                    raise ValueError(
                        f"Batch embedding count mismatch: "
                        f"got {len(batch_embeddings)} for {len(batch)} texts"
                    )
                all_embeddings.extend(batch_embeddings)
                last_err = None
                break
            except Exception as e:
                last_err = e
                if attempt < _EMBED_MAX_RETRIES - 1:
                    time.sleep(0.5 * (2 ** attempt))
        if last_err is not None:
            raise last_err
    return all_embeddings


def _index_file(
    conn: sqlite3.Connection,
    file_id: int,
    filepath: Path,
    content: str,
    *,
    model: str,
) -> int:
    """Chunk, embed, and store a single file. Returns chunk count.

    Uses a savepoint so that if embedding fails, the old chunks are
    preserved and the content_hash is reset for retry on next run.
    """
    chunks = chunk_markdown(content)
    if not chunks:
        return 0

    embed_texts = [
        (_embed_prefix(c.context) + "\n" + c.content) if c.context else c.content
        for c in chunks
    ]

    conn.execute("SAVEPOINT reindex_file")
    try:
        delete_chunks_for_file(conn, file_id)
        chunk_embeddings = _batched_embed(embed_texts, model=model)
        if len(chunk_embeddings) != len(chunks):
            raise ValueError(
                f"Embedding count mismatch for {filepath}: "
                f"got {len(chunk_embeddings)} embeddings for {len(chunks)} chunks"
            )
        insert_chunks(conn, file_id, chunks, chunk_embeddings)
        conn.execute("RELEASE reindex_file")
        return len(chunks)
    except Exception:
        conn.execute("ROLLBACK TO reindex_file")
        conn.execute("RELEASE reindex_file")
        conn.execute(
            "UPDATE files SET content_hash = '' WHERE id = ?",
            (file_id,),
        )
        conn.commit()
        raise


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


def _discover_vault_files(vault_dir: "Path") -> list[tuple["Path", str]]:
    """Return (path, memory_type) pairs for all indexable markdown files in the vault."""
    found = []
    for p in sorted(vault_dir.rglob("*.md")):
        if any(part.startswith(".") for part in p.relative_to(vault_dir).parts):
            continue  # skip .obsidian, .trash, etc.
        found.append((p, "note"))
    return found


def _index_files(
    conn: sqlite3.Connection,
    collection_id: int,
    files: list[tuple[Path, str]],
    stats: dict,
    *,
    model: str,
) -> None:
    """Index a list of files, updating stats in place."""
    discovered_paths = {str(fp) for fp, _ in files}
    indexed_paths = get_indexed_paths(conn, collection_id)
    for path, file_id in indexed_paths.items():
        if path not in discovered_paths:
            delete_file(conn, file_id)
            stats["deleted"] += 1

    for filepath, memory_type in files:
        try:
            content = filepath.read_text(encoding="utf-8", errors="replace")
            content_hash = _content_hash(content)
            stat = filepath.stat()
        except OSError as e:
            print(f"  [warning] skipping {filepath}: {e}", file=_stderr)
            continue

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

        try:
            chunk_count = _index_file(conn, file_id, filepath, content, model=model)
            stats["chunks"] += chunk_count
            stats["indexed"] += 1
        except Exception as e:
            print(f"  [warning] indexing failed for {filepath}: {e}", file=_stderr)


def build_notes_index(*, model: str = DEFAULT_EMBEDDING_MODEL) -> dict:
    """Index all personal vault notes into sqlite-vec. Returns stats."""
    stats = {"indexed": 0, "skipped": 0, "chunks": 0, "deleted": 0}

    vault_dir = _notes_dir()
    if vault_dir is None:
        return stats

    db_path = _notes_db_path()
    if db_path is None:
        return stats

    cached_dim, model_changed = _prepare_db(model, db_path=db_path)
    dim = cached_dim if cached_dim is not None else embedding_dimensions(model)

    conn = init_db(dim=dim, db_path=db_path)
    if conn is None:
        raise RuntimeError("Failed to initialize notes index database")

    try:
        collection_id = ensure_collection(conn, name="notes")

        if model_changed:
            conn.execute(
                "UPDATE files SET content_hash = '' WHERE collection_id = ?",
                (collection_id,),
            )
            conn.commit()

        stored_model = _get_metadata(conn, "embedding_model")
        if stored_model != model:
            _set_metadata(conn, "embedding_model", model)
            conn.commit()

        files = _discover_vault_files(vault_dir)
        _index_files(conn, collection_id, files, stats, model=model)
    finally:
        conn.close()

    return stats


def build_index(*, model: str = DEFAULT_EMBEDDING_MODEL) -> dict:
    """Index all memory files into sqlite-vec. Returns stats."""
    stats = {"indexed": 0, "skipped": 0, "chunks": 0, "deleted": 0}

    memory_dir = _memory_dir()
    if memory_dir is None:
        return stats

    cached_dim, model_changed = _prepare_db(model)
    dim = cached_dim if cached_dim is not None else embedding_dimensions(model)

    conn = init_db(dim=dim)
    if conn is None:
        raise RuntimeError("Failed to initialize index database")

    try:
        collection_id = ensure_collection(conn)

        if model_changed:
            conn.execute(
                "UPDATE files SET content_hash = '' WHERE collection_id = ?",
                (collection_id,),
            )
            conn.commit()

        stored_model = _get_metadata(conn, "embedding_model")
        if stored_model != model:
            _set_metadata(conn, "embedding_model", model)
            conn.commit()

        files = _discover_files(memory_dir)
        _index_files(conn, collection_id, files, stats, model=model)
    finally:
        conn.close()

    return stats
