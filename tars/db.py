"""sqlite-vec database for chunk embeddings."""

import re
import sqlite3
import struct
from pathlib import Path

import sqlite_vec

from tars.memory import _memory_dir


def _serialize_f32(vector: list[float]) -> bytes:
    """Pack a float list into raw little-endian f32 bytes for sqlite-vec."""
    return struct.pack(f"<{len(vector)}f", *vector)


def _db_path() -> Path | None:
    d = _memory_dir()
    if d is None:
        return None
    return d / "tars.db"


def _connect(db_file: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_file))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS collections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    collection_id INTEGER NOT NULL REFERENCES collections(id),
    path TEXT NOT NULL,
    title TEXT,
    media_type TEXT NOT NULL DEFAULT 'text/markdown',
    memory_type TEXT,
    content_hash TEXT NOT NULL,
    mtime REAL NOT NULL,
    size INTEGER NOT NULL,
    UNIQUE(collection_id, path)
);

CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

_VEC_TABLE_SQL = """\
CREATE VIRTUAL TABLE vec_chunks USING vec0(
    embedding float[{dim}],
    +file_id INTEGER,
    +chunk_sequence INTEGER,
    +content_hash TEXT,
    +start_line INTEGER,
    +end_line INTEGER,
    +content TEXT
);
"""


def _vec_table_exists(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='vec_chunks'"
    ).fetchone()
    return row is not None


def _get_metadata(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute(
        "SELECT value FROM metadata WHERE key = ?",
        (key,),
    ).fetchone()
    return row["value"] if row else None


def _set_metadata(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        """\
        INSERT INTO metadata (key, value)
        VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value""",
        (key, value),
    )


def _get_vec_dim_from_schema(conn: sqlite3.Connection) -> int | None:
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='vec_chunks'"
    ).fetchone()
    if not row or not row["sql"]:
        return None
    match = re.search(r"embedding\s+float\[(\d+)\]", row["sql"])
    if not match:
        return None
    return int(match.group(1))


def init_db(*, dim: int) -> sqlite3.Connection | None:
    """Create or open the database, ensuring schema exists.

    Returns None if TARS_MEMORY_DIR is not configured.
    """
    p = _db_path()
    if p is None:
        return None
    conn = _connect(p)
    conn.executescript(_SCHEMA_SQL)
    if _vec_table_exists(conn):
        stored_dim = _get_metadata(conn, "vec_dim")
        schema_dim = _get_vec_dim_from_schema(conn)
        if stored_dim is not None and schema_dim is not None:
            if int(stored_dim) != schema_dim:
                raise ValueError(
                    f"Vector dimension metadata mismatch: stored {stored_dim}, schema {schema_dim}."
                )
        actual_dim = int(stored_dim) if stored_dim is not None else schema_dim
        if actual_dim is None:
            raise ValueError(
                "Could not determine vec_chunks dimension; delete the database or migrate."
            )
        if stored_dim is None:
            _set_metadata(conn, "vec_dim", str(actual_dim))
            conn.commit()
        if actual_dim != dim:
            raise ValueError(
                f"Vector dimension mismatch: stored {actual_dim}, requested {dim}."
            )
    else:
        conn.execute(_VEC_TABLE_SQL.format(dim=dim))
        _set_metadata(conn, "vec_dim", str(dim))
        conn.commit()
    return conn


def ensure_collection(conn: sqlite3.Connection, name: str = "tars_memory") -> int:
    """Get or create a collection by name. Returns the collection id."""
    row = conn.execute(
        "SELECT id FROM collections WHERE name = ?", (name,)
    ).fetchone()
    if row:
        return row["id"]
    cur = conn.execute("INSERT INTO collections (name) VALUES (?)", (name,))
    conn.commit()
    return cur.lastrowid


def get_file_by_path(
    conn: sqlite3.Connection, collection_id: int, path: str
) -> dict | None:
    """Look up a file record by collection and path."""
    row = conn.execute(
        "SELECT * FROM files WHERE collection_id = ? AND path = ?",
        (collection_id, path),
    ).fetchone()
    return dict(row) if row else None


def upsert_file(
    conn: sqlite3.Connection,
    *,
    collection_id: int,
    path: str,
    title: str | None = None,
    media_type: str = "text/markdown",
    memory_type: str | None = None,
    content_hash: str,
    mtime: float,
    size: int,
) -> tuple[int, bool]:
    """Insert or update a file record.

    Returns (file_id, changed). Skips update if content_hash is unchanged.
    """
    existing = get_file_by_path(conn, collection_id, path)
    if existing is not None:
        if existing["content_hash"] == content_hash:
            return existing["id"], False
        conn.execute(
            """\
            UPDATE files
            SET title = ?, media_type = ?, memory_type = ?,
                content_hash = ?, mtime = ?, size = ?
            WHERE id = ?""",
            (title, media_type, memory_type, content_hash, mtime, size, existing["id"]),
        )
        conn.commit()
        return existing["id"], True

    cur = conn.execute(
        """\
        INSERT INTO files (collection_id, path, title, media_type, memory_type,
                           content_hash, mtime, size)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (collection_id, path, title, media_type, memory_type, content_hash, mtime, size),
    )
    conn.commit()
    return cur.lastrowid, True


def delete_chunks_for_file(conn: sqlite3.Connection, file_id: int) -> None:
    """Remove all chunks for a file before re-indexing."""
    conn.execute("DELETE FROM vec_chunks WHERE file_id = ?", (file_id,))
    conn.commit()


def insert_chunks(
    conn: sqlite3.Connection,
    file_id: int,
    chunks: list,
    embeddings: list[list[float]],
) -> None:
    """Bulk insert chunks with their embeddings."""
    count = min(len(chunks), len(embeddings))
    for i in range(count):
        chunk = chunks[i]
        conn.execute(
            """\
            INSERT INTO vec_chunks (embedding, file_id, chunk_sequence,
                                    content_hash, start_line, end_line, content)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                _serialize_f32(embeddings[i]),
                file_id,
                chunk.sequence,
                chunk.content_hash,
                chunk.start_line,
                chunk.end_line,
                chunk.content,
            ),
        )
    conn.commit()
