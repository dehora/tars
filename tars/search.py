"""Hybrid search: FTS5 keyword + sqlite-vec KNN, fused with RRF."""

import json
from dataclasses import dataclass

from tars.db import _connect, _db_path, _fts_table_exists, _get_metadata, _serialize_f32, _vec_table_exists
from tars.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    _DEFAULT_QUERY_INSTRUCT,
    _supports_instruct,
    embed,
)
from tars import rewriter


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
    file_id: int = 0
    chunk_sequence: int = 0


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


def _fetch_window_chunks(conn, file_id: int, seq_lo: int, seq_hi: int) -> list[dict]:
    """Fetch chunks for file_id with chunk_sequence in [seq_lo, seq_hi]."""
    rows = conn.execute(
        "SELECT content, chunk_sequence, start_line, end_line "
        "FROM vec_chunks WHERE file_id = ? "
        "AND chunk_sequence BETWEEN ? AND ? "
        "ORDER BY chunk_sequence",
        (file_id, seq_lo, seq_hi),
    ).fetchall()
    return [dict(r) for r in rows]


def _merge_intervals(
    intervals: list[tuple[int, int, float]],
) -> list[tuple[int, int, float]]:
    """Merge overlapping/adjacent (seq_lo, seq_hi, score) intervals."""
    if not intervals:
        return []
    sorted_ivs = sorted(intervals, key=lambda c: c[0])
    merged = [sorted_ivs[0]]
    for seq_lo, seq_hi, score in sorted_ivs[1:]:
        prev_lo, prev_hi, prev_score = merged[-1]
        if seq_lo <= prev_hi + 1:
            merged[-1] = (prev_lo, max(prev_hi, seq_hi), max(prev_score, score))
        else:
            merged.append((seq_lo, seq_hi, score))
    return merged


def _expand_windows(
    raw: list[tuple[SearchResult, int, int]],
    window: int,
    conn,
) -> list[SearchResult]:
    """Expand search results to include neighboring chunks."""
    from collections import defaultdict

    by_file: dict[int, list[tuple[SearchResult, int]]] = defaultdict(list)
    file_meta: dict[int, tuple[str, str | None, str | None]] = {}

    for result, file_id, seq in raw:
        by_file[file_id].append((result, seq))
        if file_id not in file_meta:
            file_meta[file_id] = (result.file_path, result.file_title, result.memory_type)

    expanded = []
    for file_id, entries in by_file.items():
        candidates = [
            (max(0, seq - window), seq + window, result.score)
            for result, seq in entries
        ]
        merged = _merge_intervals(candidates)

        path, title, mtype = file_meta[file_id]
        for seq_lo, seq_hi, best_score in merged:
            chunks = _fetch_window_chunks(conn, file_id, seq_lo, seq_hi)
            if not chunks:
                continue
            content = "".join(c["content"] for c in chunks)
            expanded.append(SearchResult(
                content=content,
                score=best_score,
                file_path=path,
                file_title=title,
                memory_type=mtype,
                start_line=chunks[0]["start_line"],
                end_line=chunks[-1]["end_line"],
                chunk_rowid=0,
                file_id=file_id,
                chunk_sequence=seq_lo,
            ))

    expanded.sort(key=lambda r: r.score, reverse=True)
    return expanded


def expand_results(
    results: list[SearchResult],
    *,
    window: int = 1,
    db_path=None,
) -> list[SearchResult]:
    """Expand search results to include neighboring chunks.

    Opens its own DB connection. Groups results by file_id, merges
    overlapping window intervals, and fetches the expanded chunks.
    """
    from collections import defaultdict

    if not results or window < 1:
        return list(results)

    p = db_path if db_path is not None else _db_path()
    if p is None or not p.exists():
        return list(results)

    conn = _connect(p)
    try:
        by_file: dict[int, list[SearchResult]] = defaultdict(list)
        file_meta: dict[int, tuple[str, str | None, str | None]] = {}

        for r in results:
            by_file[r.file_id].append(r)
            if r.file_id not in file_meta:
                file_meta[r.file_id] = (r.file_path, r.file_title, r.memory_type)

        expanded = []
        for file_id, file_results in by_file.items():
            candidates = [
                (max(0, r.chunk_sequence - window), r.chunk_sequence + window, r.score)
                for r in file_results
            ]
            merged = _merge_intervals(candidates)

            path, title, mtype = file_meta[file_id]
            for seq_lo, seq_hi, best_score in merged:
                chunks = _fetch_window_chunks(conn, file_id, seq_lo, seq_hi)
                if not chunks:
                    continue
                content = "".join(c["content"] for c in chunks)
                expanded.append(SearchResult(
                    content=content,
                    score=best_score,
                    file_path=path,
                    file_title=title,
                    memory_type=mtype,
                    start_line=chunks[0]["start_line"],
                    end_line=chunks[-1]["end_line"],
                    chunk_rowid=0,
                    file_id=file_id,
                    chunk_sequence=seq_lo,
                ))

        expanded.sort(key=lambda r: r.score, reverse=True)
        return expanded
    finally:
        conn.close()


def _apply_char_cap(results: list[SearchResult], max_chars: int) -> list[SearchResult]:
    """Greedy cap: accumulate results by score until char budget is exceeded."""
    capped = []
    total = 0
    for r in results:
        if total + len(r.content) > max_chars and capped:
            break
        capped.append(r)
        total += len(r.content)
    return capped


def search(
    query: str,
    *,
    model: str = DEFAULT_EMBEDDING_MODEL,
    limit: int = 10,
    min_score: float = 0.0,
    mode: str = "hybrid",
    db_path=None,
    window: int = 0,
    max_context_chars: int = 0,
) -> list[SearchResult]:
    """Search memory chunks. Returns results sorted by relevance."""
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 10
    limit = min(max(limit, 1), 100)
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
            stored_model = _get_metadata(conn, "embedding_model")
            vec_model = stored_model if stored_model else model
            instruct = _DEFAULT_QUERY_INSTRUCT if _supports_instruct(vec_model) else None
            query_vec = embed(query, model=vec_model, instruct=instruct)[0]
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

        _raw: list[tuple[SearchResult, int, int]] = []
        for rowid, score in fused:
            row = conn.execute(
                "SELECT vc.content, vc.file_id, vc.chunk_sequence, "
                "vc.start_line, vc.end_line, "
                "f.path, f.title, f.memory_type "
                "FROM vec_chunks vc "
                "JOIN files f ON f.id = vc.file_id "
                "WHERE vc.rowid = ?",
                (rowid,),
            ).fetchone()
            if row is None:
                continue
            result = SearchResult(
                content=row["content"],
                score=score,
                file_path=row["path"],
                file_title=row["title"],
                memory_type=row["memory_type"],
                start_line=row["start_line"],
                end_line=row["end_line"],
                chunk_rowid=rowid,
                file_id=row["file_id"],
                chunk_sequence=row["chunk_sequence"],
            )
            _raw.append((result, row["file_id"], row["chunk_sequence"]))

        if window > 0:
            results = _expand_windows(_raw, window, conn)
        else:
            results = [r for r, _, _ in _raw]

        if max_context_chars > 0:
            results = _apply_char_cap(results, max_context_chars)

        return results
    finally:
        conn.close()


def search_expanded(
    query: str,
    *,
    model: str = DEFAULT_EMBEDDING_MODEL,
    limit: int = 10,
    min_score: float = 0.0,
    mode: str = "hybrid",
    db_path=None,
    window: int = 0,
    max_context_chars: int = 0,
) -> list[SearchResult]:
    """Multi-query search with query rewriting and optional HyDE.

    Generates keyword-dense rewrites of the query (and a hypothetical
    document for longer queries), runs each through vec+fts, and fuses
    all ranked lists via RRF.
    """

    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 10
    limit = min(max(limit, 1), 100)
    p = db_path if db_path is not None else _db_path()
    if p is None or not p.exists():
        return []

    conn = _connect(p)
    try:
        if not _vec_table_exists(conn):
            return []

        stored_model = _get_metadata(conn, "embedding_model")
        vec_model = stored_model if stored_model else model
        instruct = _DEFAULT_QUERY_INSTRUCT if _supports_instruct(vec_model) else None

        queries = rewriter.expand_queries(query)
        hyde_text = rewriter.generate_hyde(query)

        ranked_lists: list[list[int]] = []
        oversample = limit * 2

        for q in queries:
            if mode in ("hybrid", "vec"):
                q_vec = embed(q, model=vec_model, instruct=instruct)[0]
                ranked_lists.append(search_vec(conn, q_vec, limit=oversample))
            if mode in ("hybrid", "fts"):
                ranked_lists.append(search_fts(conn, q, limit=oversample))

        if hyde_text and mode in ("hybrid", "vec"):
            hyde_vec = embed(hyde_text, model=vec_model)[0]
            ranked_lists.append(search_vec(conn, hyde_vec, limit=oversample))

        fused = _reciprocal_rank_fusion(*ranked_lists)
        fused = [(rid, score) for rid, score in fused if score >= min_score]
        fused = fused[:limit]

        if not fused:
            return []

        _raw: list[tuple[SearchResult, int, int]] = []
        for rowid, score in fused:
            row = conn.execute(
                "SELECT vc.content, vc.file_id, vc.chunk_sequence, "
                "vc.start_line, vc.end_line, "
                "f.path, f.title, f.memory_type "
                "FROM vec_chunks vc "
                "JOIN files f ON f.id = vc.file_id "
                "WHERE vc.rowid = ?",
                (rowid,),
            ).fetchone()
            if row is None:
                continue
            result = SearchResult(
                content=row["content"],
                score=score,
                file_path=row["path"],
                file_title=row["title"],
                memory_type=row["memory_type"],
                start_line=row["start_line"],
                end_line=row["end_line"],
                chunk_rowid=rowid,
                file_id=row["file_id"],
                chunk_sequence=row["chunk_sequence"],
            )
            _raw.append((result, row["file_id"], row["chunk_sequence"]))

        if window > 0:
            results = _expand_windows(_raw, window, conn)
        else:
            results = [r for r, _, _ in _raw]

        if max_context_chars > 0:
            results = _apply_char_cap(results, max_context_chars)

        return results
    finally:
        conn.close()


def search_notes(query: str, **kwargs) -> list[SearchResult]:
    """Search the personal notes vault index."""
    from tars.db import _notes_db_path
    return search(query, db_path=_notes_db_path(), **kwargs)


def _safe_int(val, default: int) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def _run_notes_search_tool(name: str, args: dict) -> str:
    """Handle notes_search tool calls."""
    query = args.get("query", "")
    limit = args.get("limit", 5)
    if not query:
        return json.dumps({"error": "query is required"})
    window = min(max(_safe_int(args.get("window", 2), 2), 0), 5)
    results = search_notes(query, limit=limit, window=window, max_context_chars=12000)
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
    window = min(max(_safe_int(args.get("window", 1), 1), 0), 5)
    results = search(query, limit=limit, window=window, max_context_chars=12000)
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
