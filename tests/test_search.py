import json
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.modules.setdefault("anthropic", mock.Mock())
sys.modules.setdefault("dotenv", mock.Mock(load_dotenv=lambda: None))
sys.modules.setdefault("ollama", mock.MagicMock())

try:
    import sqlite_vec

    _HAS_SQLITE_VEC = True
except ImportError:
    _HAS_SQLITE_VEC = False

from tars import db, embeddings
from tars.chunker import Chunk
from tars.search import (
    SearchResult,
    _reciprocal_rank_fusion,
    _run_notes_search_tool,
    _run_search_tool,
    _sanitize_fts_query,
    search,
    search_fts,
    search_notes,
    search_vec,
)

_DIM = 4


def _fake_embed(**kwargs):
    texts = kwargs.get("input", [])
    return {"embeddings": [[0.1] * _DIM for _ in texts]}


def _setup_db_with_chunks(tmpdir):
    """Create a DB with two files and their chunks for testing."""
    conn = db.init_db(dim=_DIM)
    cid = db.ensure_collection(conn)

    fid1, _ = db.upsert_file(
        conn,
        collection_id=cid,
        path="/memory/Memory.md",
        title="Memory",
        memory_type="semantic",
        content_hash="aaa",
        mtime=1.0,
        size=100,
    )
    chunks1 = [
        Chunk(content="My dog's name is Perry", sequence=0, start_line=1, end_line=1, content_hash="c1"),
        Chunk(content="I like Python and Rust", sequence=1, start_line=2, end_line=2, content_hash="c2"),
    ]
    emb1 = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
    db.insert_chunks(conn, fid1, chunks1, emb1)

    fid2, _ = db.upsert_file(
        conn,
        collection_id=cid,
        path="/memory/sessions/log.md",
        title="log",
        memory_type="episodic",
        content_hash="bbb",
        mtime=2.0,
        size=200,
    )
    chunks2 = [
        Chunk(content="Discussed Perry the dog and walks", sequence=0, start_line=1, end_line=1, content_hash="c3"),
    ]
    emb2 = [[0.15, 0.25, 0.35, 0.45]]
    db.insert_chunks(conn, fid2, chunks2, emb2)

    return conn, cid, fid1, fid2


class SanitizeFtsQueryTests(unittest.TestCase):
    def test_simple_query(self) -> None:
        self.assertEqual(_sanitize_fts_query("hello world"), '"hello" "world"')

    def test_empty_query(self) -> None:
        self.assertEqual(_sanitize_fts_query(""), "")

    def test_special_characters(self) -> None:
        result = _sanitize_fts_query('foo:bar (baz) "quoted"')
        self.assertIn('"foo:bar"', result)
        self.assertIn('"(baz)"', result)

    def test_whitespace_only(self) -> None:
        self.assertEqual(_sanitize_fts_query("   "), "")

    def test_quotes_in_token(self) -> None:
        result = _sanitize_fts_query('say "hello"')
        # Internal quotes should be escaped
        self.assertIn('""hello""', result)


class RRFTests(unittest.TestCase):
    def test_single_list(self) -> None:
        result = _reciprocal_rank_fusion([10, 20, 30])
        self.assertEqual(len(result), 3)
        # First item should have highest score
        self.assertEqual(result[0][0], 10)
        self.assertGreater(result[0][1], result[1][1])

    def test_two_lists_overlapping(self) -> None:
        list1 = [10, 20, 30]
        list2 = [20, 10, 40]
        result = _reciprocal_rank_fusion(list1, list2)
        scores = {rid: score for rid, score in result}
        # Items in both lists should score higher than items in one
        self.assertGreater(scores[10], scores[30])
        self.assertGreater(scores[20], scores[30])

    def test_scores_normalized_0_to_1(self) -> None:
        result = _reciprocal_rank_fusion([1, 2], [1, 3])
        for _, score in result:
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_perfect_overlap_at_rank_1(self) -> None:
        result = _reciprocal_rank_fusion([42], [42])
        self.assertAlmostEqual(result[0][1], 1.0)

    def test_empty_lists(self) -> None:
        result = _reciprocal_rank_fusion([], [])
        self.assertEqual(result, [])


@unittest.skipUnless(_HAS_SQLITE_VEC, "sqlite-vec not installed")
class SearchVecTests(unittest.TestCase):
    def test_returns_rowids_in_distance_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {"TARS_MEMORY_DIR": tmpdir}, clear=True):
                conn, *_ = _setup_db_with_chunks(tmpdir)
                # Query close to first embedding [0.1, 0.2, 0.3, 0.4]
                rowids = search_vec(conn, [0.1, 0.2, 0.3, 0.4], limit=10)
                self.assertGreater(len(rowids), 0)
                conn.close()


@unittest.skipUnless(_HAS_SQLITE_VEC, "sqlite-vec not installed")
class SearchFtsTests(unittest.TestCase):
    def test_keyword_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {"TARS_MEMORY_DIR": tmpdir}, clear=True):
                conn, *_ = _setup_db_with_chunks(tmpdir)
                rowids = search_fts(conn, "Perry dog", limit=10)
                self.assertGreater(len(rowids), 0)
                conn.close()

    def test_no_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {"TARS_MEMORY_DIR": tmpdir}, clear=True):
                conn, *_ = _setup_db_with_chunks(tmpdir)
                rowids = search_fts(conn, "xyzzyplugh", limit=10)
                self.assertEqual(rowids, [])
                conn.close()

    def test_empty_query(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {"TARS_MEMORY_DIR": tmpdir}, clear=True):
                conn, *_ = _setup_db_with_chunks(tmpdir)
                rowids = search_fts(conn, "", limit=10)
                self.assertEqual(rowids, [])
                conn.close()


@unittest.skipUnless(_HAS_SQLITE_VEC, "sqlite-vec not installed")
class SearchHybridTests(unittest.TestCase):
    def setUp(self) -> None:
        self._patcher = mock.patch.object(embeddings, "ollama")
        self._mock_ollama = self._patcher.start()
        self._mock_ollama.embed.side_effect = _fake_embed

    def tearDown(self) -> None:
        self._patcher.stop()

    def test_hybrid_returns_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {"TARS_MEMORY_DIR": tmpdir}, clear=True):
                conn, *_ = _setup_db_with_chunks(tmpdir)
                conn.close()
                results = search("Perry dog", model="test-model", limit=5)
                self.assertGreater(len(results), 0)
                self.assertIsInstance(results[0], SearchResult)
                self.assertGreater(results[0].score, 0.0)

    def test_vec_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {"TARS_MEMORY_DIR": tmpdir}, clear=True):
                conn, *_ = _setup_db_with_chunks(tmpdir)
                conn.close()
                results = search("anything", model="test-model", mode="vec")
                self.assertGreater(len(results), 0)

    def test_fts_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {"TARS_MEMORY_DIR": tmpdir}, clear=True):
                conn, *_ = _setup_db_with_chunks(tmpdir)
                conn.close()
                results = search("Python Rust", model="test-model", mode="fts")
                self.assertGreater(len(results), 0)

    def test_min_score_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {"TARS_MEMORY_DIR": tmpdir}, clear=True):
                conn, *_ = _setup_db_with_chunks(tmpdir)
                conn.close()
                all_results = search("Perry", model="test-model", min_score=0.0)
                filtered = search("Perry", model="test-model", min_score=1.01)
                # Score > 1.0 is impossible, so everything should be filtered
                self.assertEqual(len(filtered), 0)
                self.assertGreater(len(all_results), 0)

    def test_empty_db(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {"TARS_MEMORY_DIR": tmpdir}, clear=True):
                db.init_db(dim=_DIM).close()
                results = search("anything", model="test-model")
                self.assertEqual(results, [])

    def test_no_memory_dir(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            os.environ.pop("TARS_MEMORY_DIR", None)
            results = search("anything", model="test-model")
            self.assertEqual(results, [])

    def test_result_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {"TARS_MEMORY_DIR": tmpdir}, clear=True):
                conn, *_ = _setup_db_with_chunks(tmpdir)
                conn.close()
                results = search("Perry", model="test-model", limit=1)
                self.assertGreater(len(results), 0)
                r = results[0]
                self.assertIsNotNone(r.file_path)
                self.assertIsNotNone(r.memory_type)
                self.assertGreater(r.start_line, 0)
                self.assertGreater(r.end_line, 0)


@unittest.skipUnless(_HAS_SQLITE_VEC, "sqlite-vec not installed")
class FtsSyncTests(unittest.TestCase):
    def test_fts_cleaned_on_chunk_delete(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {"TARS_MEMORY_DIR": tmpdir}, clear=True):
                conn, cid, fid1, fid2 = _setup_db_with_chunks(tmpdir)
                # Verify FTS has entries
                fts_count = conn.execute(
                    "SELECT count(*) as cnt FROM chunks_fts"
                ).fetchone()["cnt"]
                self.assertEqual(fts_count, 3)

                db.delete_chunks_for_file(conn, fid1)
                fts_count = conn.execute(
                    "SELECT count(*) as cnt FROM chunks_fts"
                ).fetchone()["cnt"]
                self.assertEqual(fts_count, 1)  # only fid2's chunk remains
                conn.close()

    def test_fts_cleaned_on_file_delete(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {"TARS_MEMORY_DIR": tmpdir}, clear=True):
                conn, cid, fid1, fid2 = _setup_db_with_chunks(tmpdir)
                db.delete_file(conn, fid2)
                fts_count = conn.execute(
                    "SELECT count(*) as cnt FROM chunks_fts"
                ).fetchone()["cnt"]
                self.assertEqual(fts_count, 2)  # only fid1's chunks remain
                conn.close()


@unittest.skipUnless(_HAS_SQLITE_VEC, "sqlite-vec not installed")
class FtsBackfillTests(unittest.TestCase):
    def test_ensure_fts_backfills(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {"TARS_MEMORY_DIR": tmpdir}, clear=True):
                conn = db.init_db(dim=_DIM)
                # Drop FTS table to simulate pre-FTS database
                conn.execute("DROP TABLE chunks_fts")
                conn.commit()
                self.assertFalse(db._fts_table_exists(conn))

                # Insert a chunk without FTS
                cid = db.ensure_collection(conn)
                fid, _ = db.upsert_file(
                    conn,
                    collection_id=cid,
                    path="/test.md",
                    content_hash="abc",
                    mtime=1.0,
                    size=10,
                )
                conn.execute(
                    "INSERT INTO vec_chunks (embedding, file_id, chunk_sequence, "
                    "content_hash, start_line, end_line, content) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (db._serialize_f32([0.1] * _DIM), fid, 0, "h1", 1, 1, "hello world"),
                )
                conn.commit()

                # Backfill
                db._ensure_fts(conn)
                self.assertTrue(db._fts_table_exists(conn))
                fts_count = conn.execute(
                    "SELECT count(*) as cnt FROM chunks_fts"
                ).fetchone()["cnt"]
                self.assertEqual(fts_count, 1)

                # Verify FTS content is searchable
                rowids = search_fts(conn, "hello", limit=5)
                self.assertEqual(len(rowids), 1)
                conn.close()


@unittest.skipUnless(_HAS_SQLITE_VEC, "sqlite-vec not installed")
class SearchToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self._patcher = mock.patch.object(embeddings, "ollama")
        self._mock_ollama = self._patcher.start()
        self._mock_ollama.embed.side_effect = _fake_embed

    def tearDown(self) -> None:
        self._patcher.stop()

    def test_returns_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {"TARS_MEMORY_DIR": tmpdir}, clear=True):
                conn, *_ = _setup_db_with_chunks(tmpdir)
                conn.close()
                result = json.loads(_run_search_tool("memory_search", {"query": "Perry"}))
                self.assertIn("results", result)
                self.assertGreater(len(result["results"]), 0)
                r = result["results"][0]
                self.assertIn("content", r)
                self.assertIn("score", r)
                self.assertIn("file", r)
                self.assertIn("type", r)

    def test_empty_query_returns_error(self) -> None:
        result = json.loads(_run_search_tool("memory_search", {"query": ""}))
        self.assertIn("error", result)

    def test_no_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {"TARS_MEMORY_DIR": tmpdir}, clear=True):
                db.init_db(dim=_DIM).close()
                result = json.loads(_run_search_tool("memory_search", {"query": "anything"}))
                self.assertEqual(result["results"], [])


@unittest.skipUnless(_HAS_SQLITE_VEC, "sqlite-vec not installed")
class SearchWithDbPathTests(unittest.TestCase):
    def setUp(self) -> None:
        self._patcher = mock.patch.object(embeddings, "ollama")
        self._mock_ollama = self._patcher.start()
        self._mock_ollama.embed.side_effect = _fake_embed

    def tearDown(self) -> None:
        self._patcher.stop()

    def test_explicit_db_path_opens_that_db(self) -> None:
        with tempfile.TemporaryDirectory() as td1, tempfile.TemporaryDirectory() as td2:
            # Set up tars memory DB in td1
            with mock.patch.dict(os.environ, {"TARS_MEMORY_DIR": td1}, clear=True):
                conn, *_ = _setup_db_with_chunks(td1)
                conn.close()

            # Set up a second DB in td2 with different content
            from pathlib import Path
            from tars.db import init_db, ensure_collection, upsert_file, insert_chunks
            alt_db = Path(td2) / "notes.db"
            with mock.patch.dict(os.environ, {"TARS_MEMORY_DIR": td2}, clear=True):
                conn2 = init_db(dim=_DIM, db_path=alt_db)
                cid = ensure_collection(conn2, name="notes")
                fid, _ = upsert_file(
                    conn2,
                    collection_id=cid,
                    path="/notes/routing.md",
                    title="routing",
                    memory_type="note",
                    content_hash="nnn",
                    mtime=1.0,
                    size=50,
                )
                insert_chunks(
                    conn2, fid,
                    [Chunk(content="routing algorithms explained", sequence=0,
                           start_line=1, end_line=1, content_hash="n1")],
                    [[0.9, 0.8, 0.7, 0.6]],
                )
                conn2.close()

            results = search("routing", model="test-model", db_path=alt_db)
            self.assertGreater(len(results), 0)
            self.assertEqual(results[0].memory_type, "note")

    def test_search_notes_no_notes_dir(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            os.environ.pop("TARS_NOTES_DIR", None)
            results = search_notes("anything", model="test-model")
            self.assertEqual(results, [])

    def test_search_notes_finds_indexed_content(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            from pathlib import Path
            from tars.db import init_db, ensure_collection, upsert_file, insert_chunks
            notes_db = Path(td) / "notes.db"
            with mock.patch.dict(os.environ, {"TARS_NOTES_DIR": td, "TARS_MEMORY_DIR": td}, clear=True):
                conn = init_db(dim=_DIM, db_path=notes_db)
                cid = ensure_collection(conn, name="notes")
                fid, _ = upsert_file(
                    conn,
                    collection_id=cid,
                    path=str(Path(td) / "ideas.md"),
                    title="ideas",
                    memory_type="note",
                    content_hash="qqq",
                    mtime=1.0,
                    size=80,
                )
                insert_chunks(
                    conn, fid,
                    [Chunk(content="quantum computing basics", sequence=0,
                           start_line=1, end_line=1, content_hash="q1")],
                    [[0.3, 0.4, 0.5, 0.6]],
                )
                conn.close()

                results = search_notes("quantum computing", model="test-model")
                self.assertGreater(len(results), 0)
                self.assertEqual(results[0].file_title, "ideas")


class NotesSearchToolTests(unittest.TestCase):
    def test_returns_results(self) -> None:
        fake_results = [
            SearchResult(
                content="recipe for pasta",
                score=0.85,
                file_path="/notes/cooking.md",
                file_title="cooking",
                memory_type="note",
                start_line=1,
                end_line=3,
                chunk_rowid=1,
            )
        ]
        with mock.patch("tars.search.search_notes", return_value=fake_results):
            result = json.loads(_run_notes_search_tool("notes_search", {"query": "pasta"}))
            self.assertIn("results", result)
            self.assertEqual(len(result["results"]), 1)
            self.assertEqual(result["results"][0]["content"], "recipe for pasta")
            self.assertEqual(result["results"][0]["score"], 0.85)
            self.assertEqual(result["results"][0]["file"], "cooking")

    def test_no_results(self) -> None:
        with mock.patch("tars.search.search_notes", return_value=[]):
            result = json.loads(_run_notes_search_tool("notes_search", {"query": "nonexistent"}))
            self.assertEqual(result["results"], [])
            self.assertIn("message", result)

    def test_empty_query(self) -> None:
        result = json.loads(_run_notes_search_tool("notes_search", {"query": ""}))
        self.assertIn("error", result)

    def test_limit_passed(self) -> None:
        with mock.patch("tars.search.search_notes", return_value=[]) as mock_search:
            _run_notes_search_tool("notes_search", {"query": "test", "limit": 3})
            mock_search.assert_called_once_with("test", limit=3)


if __name__ == "__main__":
    unittest.main()
