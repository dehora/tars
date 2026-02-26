"""Hybrid search: FTS5 keyword + sqlite-vec KNN, fused with RRF."""

import json
from dataclasses import dataclass

from tars.db import _connect, _db_path, _fts_table_exists, _get_metadata, _serialize_f32, _vec_table_exists
from tars.embeddings import DEFAULT_EMBEDDING_MODEL, embed


@dataclass(frozen=True, slots=True)
class SearchResult:
    """A single search result from hybrid search."""

    content: str
    score: float
    file_path: str
    file_title: str | None
    memory_type: str | None
    start_line: int
    end_line: int
    chunk_rowid: int


def _sanitize_fts_query(query: str) -> str:
    """Convert user query into a safe FTS5 query string."""
    tokens = query.split()
    safe = ['"' + t.replace('"', '""') + '"' for t in tokens if t.strip()]
    return " ".join(safe)


def search_vec(conn, query_embedding: list[float], *, limit: int = 20) -> list[int]:
    """Vector KNN search. Returns chunk rowids in distance order."""
    rows = conn.execute(
        "SELECT rowid, distance FROM vec_chunks "
        "WHERE embedding MATCH ? AND k = ?",
        (_serialize_f32(query_embedding), limit),
    ).fetchall()
    return [r["rowid"] for r in rows]


def search_fts(conn, query: str, *, limit: int = 20) -> list[int]:
    """FTS5 BM25 search. Returns chunk rowids in relevance order."""
    if not _fts_table_exists(conn):
        return []
    safe_query = _sanitize_fts_query(query)
    if not safe_query:
        return []
    rows = conn.execute(
        "SELECT rowid, rank FROM chunks_fts "
        "WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?",
        (safe_query, limit),
    ).fetchall()
    return [r["rowid"] for r in rows]


def _reciprocal_rank_fusion(
    *ranked_lists: list[int],
    k: int = 60,
) -> list[tuple[int, float]]:
    """Combine ranked lists using RRF. Returns [(rowid, score)] sorted by score desc."""
    scores: dict[int, float] = {}
    n_lists = len(ranked_lists)
    for rlist in ranked_lists:
        for rank, rowid in enumerate(rlist, start=1):
            if rowid not in scores:
                scores[rowid] = 0.0
            scores[rowid] += 1.0 / (k + rank)

    max_score = n_lists / (k + 1) if n_lists > 0 else 1.0
    normalized = [(rid, s / max_score) for rid, s in scores.items()]
    normalized.sort(key=lambda x: x[1], reverse=True)
    return normalized


def search(
    query: str,
    *,
    model: str = DEFAULT_EMBEDDING_MODEL,
    limit: int = 10,
    min_score: float = 0.0,
    mode: str = "hybrid",
    db_path=None,
) -> list[SearchResult]:
    """Search memory chunks. Returns results sorted by relevance."""
    p = db_path if db_path is not None else _db_path()
    if p is None or not p.exists():
        return []

    conn = _connect(p)
    try:
        if not _vec_table_exists(conn):
            return []

        vec_rowids: list[int] = []
        fts_rowids: list[int] = []

        if mode in ("hybrid", "vec"):
            # Use the model the index was built with, not the caller's default
            stored_model = _get_metadata(conn, "embedding_model")
            vec_model = stored_model if stored_model else model
            query_vec = embed(query, model=vec_model)[0]
            vec_rowids = search_vec(conn, query_vec, limit=limit * 2)

        if mode in ("hybrid", "fts"):
            fts_rowids = search_fts(conn, query, limit=limit * 2)

        if mode == "hybrid":
            fused = _reciprocal_rank_fusion(vec_rowids, fts_rowids)
        elif mode == "vec":
            fused = _reciprocal_rank_fusion(vec_rowids)
        else:
            fused = _reciprocal_rank_fusion(fts_rowids)

        fused = [(rid, score) for rid, score in fused if score >= min_score]
        fused = fused[:limit]

        if not fused:
            return []

        results = []
        for rowid, score in fused:
            row = conn.execute(
                "SELECT vc.content, vc.file_id, vc.start_line, vc.end_line, "
                "f.path, f.title, f.memory_type "
                "FROM vec_chunks vc "
                "JOIN files f ON f.id = vc.file_id "
                "WHERE vc.rowid = ?",
                (rowid,),
            ).fetchone()
            if row is None:
                continue
            results.append(SearchResult(
                content=row["content"],
                score=score,
                file_path=row["path"],
                file_title=row["title"],
                memory_type=row["memory_type"],
                start_line=row["start_line"],
                end_line=row["end_line"],
                chunk_rowid=rowid,
            ))

        return results
    finally:
        conn.close()


def search_notes(query: str, **kwargs) -> list[SearchResult]:
    """Search the personal notes vault index."""
    from tars.db import _notes_db_path
    return search(query, db_path=_notes_db_path(), **kwargs)


def _run_notes_search_tool(name: str, args: dict) -> str:
    """Handle notes_search tool calls."""
    query = args.get("query", "")
    limit = args.get("limit", 5)
    if not query:
        return json.dumps({"error": "query is required"})
    results = search_notes(query, limit=limit)
    if not results:
        return json.dumps({"results": [], "message": "No matching notes found."})
    return json.dumps({
        "results": [
            {
                "content": r.content,
                "score": round(r.score, 3),
                "file": r.file_title or r.file_path,
                "type": r.memory_type,
                "lines": f"{r.start_line}-{r.end_line}",
            }
            for r in results
        ]
    })


def _run_search_tool(name: str, args: dict) -> str:
    """Handle memory_search tool calls."""
    query = args.get("query", "")
    limit = args.get("limit", 5)
    if not query:
        return json.dumps({"error": "query is required"})
    results = search(query, limit=limit)
    if not results:
        return json.dumps({"results": [], "message": "No matching memories found."})
    return json.dumps({
        "results": [
            {
                "content": r.content,
                "score": round(r.score, 3),
                "file": r.file_title or r.file_path,
                "type": r.memory_type,
                "lines": f"{r.start_line}-{r.end_line}",
            }
            for r in results
        ]
    })
