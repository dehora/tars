import os
import struct
import sys
import tempfile
import unittest
from unittest import mock

sys.modules.setdefault("anthropic", mock.Mock())
sys.modules.setdefault("ollama", mock.Mock())
sys.modules.setdefault("dotenv", mock.Mock(load_dotenv=lambda: None))

try:
    import sqlite_vec
    _HAS_SQLITE_VEC = True
except ImportError:
    _HAS_SQLITE_VEC = False

from tars.chunker import Chunk
from tars import db


@unittest.skipUnless(_HAS_SQLITE_VEC, "sqlite-vec not installed")
class DbPathTests(unittest.TestCase):
    def test_db_path_returns_none_without_config(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(db._db_path())

    def test_db_path_returns_path_with_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {"TARS_MEMORY_DIR": tmpdir}, clear=True):
                p = db._db_path()
                self.assertIsNotNone(p)
                self.assertEqual(p.name, "tars.db")


@unittest.skipUnless(_HAS_SQLITE_VEC, "sqlite-vec not installed")
class InitDbTests(unittest.TestCase):
    def test_returns_none_without_memory_dir(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(db.init_db(dim=4))

    def test_creates_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {"TARS_MEMORY_DIR": tmpdir}, clear=True):
                conn = db.init_db(dim=4)
                self.assertIsNotNone(conn)
                # Check tables exist
                tables = {
                    row["name"]
                    for row in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    ).fetchall()
                }
                self.assertIn("collections", tables)
                self.assertIn("files", tables)
                self.assertIn("vec_chunks", tables)
                self.assertIn("metadata", tables)
                conn.close()

    def test_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {"TARS_MEMORY_DIR": tmpdir}, clear=True):
                conn1 = db.init_db(dim=4)
                conn1.close()
                conn2 = db.init_db(dim=4)
                self.assertIsNotNone(conn2)
                conn2.close()

    def test_stores_vec_dim(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {"TARS_MEMORY_DIR": tmpdir}, clear=True):
                conn = db.init_db(dim=4)
                row = conn.execute(
                    "SELECT value FROM metadata WHERE key = 'vec_dim'"
                ).fetchone()
                self.assertIsNotNone(row)
                self.assertEqual(row["value"], "4")
                conn.close()

    def test_stores_distance_metric(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {"TARS_MEMORY_DIR": tmpdir}, clear=True):
                conn = db.init_db(dim=4)
                row = conn.execute(
                    "SELECT value FROM metadata WHERE key = 'distance_metric'"
                ).fetchone()
                self.assertIsNotNone(row)
                self.assertEqual(row["value"], "cosine")
                conn.close()

    def test_uses_cosine_distance_in_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {"TARS_MEMORY_DIR": tmpdir}, clear=True):
                conn = db.init_db(dim=4)
                row = conn.execute(
                    "SELECT sql FROM sqlite_master WHERE type='table' AND name='vec_chunks'"
                ).fetchone()
                self.assertIn("distance_metric=cosine", row["sql"])
                conn.close()

    def test_dim_mismatch_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {"TARS_MEMORY_DIR": tmpdir}, clear=True):
                conn = db.init_db(dim=4)
                conn.close()
                with self.assertRaises(ValueError):
                    db.init_db(dim=5)


@unittest.skipUnless(_HAS_SQLITE_VEC, "sqlite-vec not installed")
class PrepareDbTests(unittest.TestCase):
    def test_missing_model_drops_vec_and_forces_rebuild(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {"TARS_MEMORY_DIR": tmpdir}, clear=True):
                conn = db.init_db(dim=4)
                # DB has vec_chunks and vec_dim but no embedding_model
                self.assertTrue(db._vec_table_exists(conn))
                conn.close()

                cached_dim, model_changed = db._prepare_db("test-model")

                self.assertIsNone(cached_dim)
                self.assertTrue(model_changed)

                path = db._db_path()
                conn = db._connect(path)
                try:
                    self.assertFalse(db._vec_table_exists(conn))
                    row = conn.execute(
                        "SELECT value FROM metadata WHERE key = 'vec_dim'"
                    ).fetchone()
                    self.assertIsNone(row)
                finally:
                    conn.close()

    def test_metric_change_forces_rebuild(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {"TARS_MEMORY_DIR": tmpdir}, clear=True):
                conn = db.init_db(dim=4)
                db._set_metadata(conn, "embedding_model", "test-model")
                db._set_metadata(conn, "distance_metric", "l2")
                conn.commit()
                conn.close()

                cached_dim, model_changed = db._prepare_db("test-model")

                self.assertIsNone(cached_dim)
                self.assertTrue(model_changed)

                path = db._db_path()
                conn = db._connect(path)
                try:
                    self.assertFalse(db._vec_table_exists(conn))
                finally:
                    conn.close()

    def test_corrupt_dim_falls_back(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {"TARS_MEMORY_DIR": tmpdir}, clear=True):
                conn = db.init_db(dim=4)
                db._set_metadata(conn, "embedding_model", "test-model")
                db._set_metadata(conn, "vec_dim", "not-a-number")
                conn.commit()
                conn.close()

                cached_dim, model_changed = db._prepare_db("test-model")

                self.assertIsNone(cached_dim)
                self.assertFalse(model_changed)

                path = db._db_path()
                self.assertIsNotNone(path)
                conn = db._connect(path)
                try:
                    row = conn.execute(
                        "SELECT value FROM metadata WHERE key = 'vec_dim'"
                    ).fetchone()
                    self.assertIsNone(row)
                finally:
                    conn.close()


@unittest.skipUnless(_HAS_SQLITE_VEC, "sqlite-vec not installed")
class CollectionTests(unittest.TestCase):
    def test_create_and_get(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {"TARS_MEMORY_DIR": tmpdir}, clear=True):
                conn = db.init_db(dim=4)
                cid1 = db.ensure_collection(conn)
                cid2 = db.ensure_collection(conn)
                self.assertEqual(cid1, cid2)
                conn.close()

    def test_custom_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {"TARS_MEMORY_DIR": tmpdir}, clear=True):
                conn = db.init_db(dim=4)
                cid = db.ensure_collection(conn, "custom")
                self.assertIsNotNone(cid)
                conn.close()


@unittest.skipUnless(_HAS_SQLITE_VEC, "sqlite-vec not installed")
class FileUpsertTests(unittest.TestCase):
    def _setup_db(self, tmpdir: str) -> tuple:
        conn = db.init_db(dim=4)
        cid = db.ensure_collection(conn)
        return conn, cid

    def test_new_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {"TARS_MEMORY_DIR": tmpdir}, clear=True):
                conn, cid = self._setup_db(tmpdir)
                fid, changed = db.upsert_file(
                    conn, collection_id=cid, path="/test.md",
                    content_hash="abc", mtime=1.0, size=100,
                )
                self.assertTrue(changed)
                self.assertIsNotNone(fid)
                conn.close()

    def test_unchanged_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {"TARS_MEMORY_DIR": tmpdir}, clear=True):
                conn, cid = self._setup_db(tmpdir)
                fid1, _ = db.upsert_file(
                    conn, collection_id=cid, path="/test.md",
                    content_hash="abc", mtime=1.0, size=100,
                )
                fid2, changed = db.upsert_file(
                    conn, collection_id=cid, path="/test.md",
                    content_hash="abc", mtime=1.0, size=100,
                )
                self.assertEqual(fid1, fid2)
                self.assertFalse(changed)
                conn.close()

    def test_changed_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {"TARS_MEMORY_DIR": tmpdir}, clear=True):
                conn, cid = self._setup_db(tmpdir)
                fid1, _ = db.upsert_file(
                    conn, collection_id=cid, path="/test.md",
                    content_hash="abc", mtime=1.0, size=100,
                )
                fid2, changed = db.upsert_file(
                    conn, collection_id=cid, path="/test.md",
                    content_hash="def", mtime=2.0, size=200,
                )
                self.assertEqual(fid1, fid2)
                self.assertTrue(changed)
                conn.close()

    def test_get_file_by_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {"TARS_MEMORY_DIR": tmpdir}, clear=True):
                conn, cid = self._setup_db(tmpdir)
                db.upsert_file(
                    conn, collection_id=cid, path="/test.md",
                    title="Test", content_hash="abc", mtime=1.0, size=100,
                )
                row = db.get_file_by_path(conn, cid, "/test.md")
                self.assertIsNotNone(row)
                self.assertEqual(row["title"], "Test")
                self.assertIsNone(db.get_file_by_path(conn, cid, "/nope.md"))
                conn.close()


@unittest.skipUnless(_HAS_SQLITE_VEC, "sqlite-vec not installed")
class ChunkTests(unittest.TestCase):
    def test_insert_and_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {"TARS_MEMORY_DIR": tmpdir}, clear=True):
                conn = db.init_db(dim=4)
                cid = db.ensure_collection(conn)
                fid, _ = db.upsert_file(
                    conn, collection_id=cid, path="/test.md",
                    content_hash="abc", mtime=1.0, size=100,
                )
                chunks = [
                    Chunk(content="hello", sequence=0, start_line=1, end_line=1, content_hash="h1"),
                    Chunk(content="world", sequence=1, start_line=2, end_line=2, content_hash="h2"),
                ]
                embeddings = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
                db.insert_chunks(conn, fid, chunks, embeddings)
                count = conn.execute(
                    "SELECT count(*) as cnt FROM vec_chunks WHERE file_id = ?", (fid,)
                ).fetchone()["cnt"]
                self.assertEqual(count, 2)
                conn.close()

    def test_delete_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {"TARS_MEMORY_DIR": tmpdir}, clear=True):
                conn = db.init_db(dim=4)
                cid = db.ensure_collection(conn)
                fid, _ = db.upsert_file(
                    conn, collection_id=cid, path="/test.md",
                    content_hash="abc", mtime=1.0, size=100,
                )
                chunks = [
                    Chunk(content="hello", sequence=0, start_line=1, end_line=1, content_hash="h1"),
                ]
                embeddings = [[0.1, 0.2, 0.3, 0.4]]
                db.insert_chunks(conn, fid, chunks, embeddings)
                db.delete_chunks_for_file(conn, fid)
                count = conn.execute(
                    "SELECT count(*) as cnt FROM vec_chunks WHERE file_id = ?", (fid,)
                ).fetchone()["cnt"]
                self.assertEqual(count, 0)
                conn.close()

    def test_min_length_safety(self) -> None:
        """More chunks than embeddings — only insert as many as we have embeddings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {"TARS_MEMORY_DIR": tmpdir}, clear=True):
                conn = db.init_db(dim=4)
                cid = db.ensure_collection(conn)
                fid, _ = db.upsert_file(
                    conn, collection_id=cid, path="/test.md",
                    content_hash="abc", mtime=1.0, size=100,
                )
                chunks = [
                    Chunk(content="a", sequence=0, start_line=1, end_line=1, content_hash="h1"),
                    Chunk(content="b", sequence=1, start_line=2, end_line=2, content_hash="h2"),
                    Chunk(content="c", sequence=2, start_line=3, end_line=3, content_hash="h3"),
                ]
                embeddings = [[0.1, 0.2, 0.3, 0.4]]  # only 1
                db.insert_chunks(conn, fid, chunks, embeddings)
                count = conn.execute(
                    "SELECT count(*) as cnt FROM vec_chunks WHERE file_id = ?", (fid,)
                ).fetchone()["cnt"]
                self.assertEqual(count, 1)
                conn.close()


@unittest.skipUnless(_HAS_SQLITE_VEC, "sqlite-vec not installed")
class DbStatsTests(unittest.TestCase):
    def test_returns_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {"TARS_MEMORY_DIR": tmpdir}, clear=True):
                conn = db.init_db(dim=4)
                db._set_metadata(conn, "embedding_model", "test-model")
                cid = db.ensure_collection(conn)
                db.upsert_file(
                    conn, collection_id=cid, path="/test.md",
                    content_hash="abc", mtime=1.0, size=100,
                )
                conn.close()
                stats = db.db_stats()
                self.assertEqual(stats["files"], 1)
                self.assertEqual(stats["embedding_model"], "test-model")
                self.assertEqual(stats["embedding_dim"], "4")
                self.assertIn("db_size_mb", stats)
                self.assertIn("chunks", stats)

    def test_no_database(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            stats = db.db_stats()
            self.assertIn("error", stats)


@unittest.skipUnless(_HAS_SQLITE_VEC, "sqlite-vec not installed")
class FileLinksTests(unittest.TestCase):
    def _setup(self, tmpdir):
        conn = db.init_db(dim=4)
        cid = db.ensure_collection(conn)
        fid1, _ = db.upsert_file(
            conn, collection_id=cid, path="/a.md", title="Alpha",
            content_hash="a1", mtime=1.0, size=10,
        )
        fid2, _ = db.upsert_file(
            conn, collection_id=cid, path="/b.md", title="Beta",
            content_hash="b1", mtime=1.0, size=10,
        )
        fid3, _ = db.upsert_file(
            conn, collection_id=cid, path="/c.md", title="Gamma",
            content_hash="c1", mtime=1.0, size=10,
        )
        return conn, cid, fid1, fid2, fid3

    def test_upsert_file_links(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {"TARS_MEMORY_DIR": tmpdir}, clear=True):
                conn, cid, fid1, fid2, fid3 = self._setup(tmpdir)
                db.upsert_file_links(conn, fid1, ["Beta", "Gamma"])
                conn.commit()
                rows = conn.execute(
                    "SELECT target_title, target_file_id FROM file_links "
                    "WHERE source_file_id = ? ORDER BY target_title",
                    (fid1,),
                ).fetchall()
                self.assertEqual(len(rows), 2)
                self.assertEqual(rows[0]["target_title"], "Beta")
                self.assertEqual(rows[0]["target_file_id"], fid2)
                self.assertEqual(rows[1]["target_title"], "Gamma")
                self.assertEqual(rows[1]["target_file_id"], fid3)
                conn.close()

    def test_upsert_replaces_old_links(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {"TARS_MEMORY_DIR": tmpdir}, clear=True):
                conn, cid, fid1, fid2, fid3 = self._setup(tmpdir)
                db.upsert_file_links(conn, fid1, ["Beta", "Gamma"])
                conn.commit()
                db.upsert_file_links(conn, fid1, ["Beta"])
                conn.commit()
                count = conn.execute(
                    "SELECT COUNT(*) as cnt FROM file_links WHERE source_file_id = ?",
                    (fid1,),
                ).fetchone()["cnt"]
                self.assertEqual(count, 1)
                conn.close()

    def test_unresolved_link(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {"TARS_MEMORY_DIR": tmpdir}, clear=True):
                conn, cid, fid1, fid2, fid3 = self._setup(tmpdir)
                db.upsert_file_links(conn, fid1, ["Nonexistent"])
                conn.commit()
                row = conn.execute(
                    "SELECT target_file_id FROM file_links WHERE source_file_id = ?",
                    (fid1,),
                ).fetchone()
                self.assertIsNone(row["target_file_id"])
                conn.close()

    def test_resolve_file_links(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {"TARS_MEMORY_DIR": tmpdir}, clear=True):
                conn, cid, fid1, fid2, fid3 = self._setup(tmpdir)
                db.upsert_file_links(conn, fid1, ["NewNote"])
                conn.commit()
                row = conn.execute(
                    "SELECT target_file_id FROM file_links "
                    "WHERE source_file_id = ? AND target_title = 'NewNote'",
                    (fid1,),
                ).fetchone()
                self.assertIsNone(row["target_file_id"])
                fid_new, _ = db.upsert_file(
                    conn, collection_id=cid, path="/new.md", title="NewNote",
                    content_hash="n1", mtime=1.0, size=10,
                )
                db.resolve_file_links(conn, cid)
                conn.commit()
                row = conn.execute(
                    "SELECT target_file_id FROM file_links "
                    "WHERE source_file_id = ? AND target_title = 'NewNote'",
                    (fid1,),
                ).fetchone()
                self.assertEqual(row["target_file_id"], fid_new)
                conn.close()

    def test_get_linked_file_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {"TARS_MEMORY_DIR": tmpdir}, clear=True):
                conn, cid, fid1, fid2, fid3 = self._setup(tmpdir)
                db.upsert_file_links(conn, fid1, ["Beta"])
                conn.commit()
                linked = db.get_linked_file_ids(conn, {fid1})
                self.assertIn(fid2, linked)
                self.assertNotIn(fid1, linked)
                # Bidirectional: querying fid2 should find fid1
                linked_back = db.get_linked_file_ids(conn, {fid2})
                self.assertIn(fid1, linked_back)
                conn.close()

    def test_get_linked_file_ids_empty(self) -> None:
        result = db.get_linked_file_ids(mock.Mock(), set())
        self.assertEqual(result, set())

    def test_delete_file_cascades_links(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {"TARS_MEMORY_DIR": tmpdir}, clear=True):
                conn, cid, fid1, fid2, fid3 = self._setup(tmpdir)
                # Insert dummy chunks so delete_file doesn't fail on vec_chunks
                chunks = [Chunk(content="x", sequence=0, start_line=1, end_line=1, content_hash="x1")]
                embs = [[0.1, 0.2, 0.3, 0.4]]
                db.insert_chunks(conn, fid1, chunks, embs)
                conn.commit()
                db.upsert_file_links(conn, fid1, ["Beta"])
                conn.commit()
                count_before = conn.execute(
                    "SELECT COUNT(*) as cnt FROM file_links WHERE source_file_id = ?",
                    (fid1,),
                ).fetchone()["cnt"]
                self.assertEqual(count_before, 1)
                db.delete_file(conn, fid1)
                count_after = conn.execute(
                    "SELECT COUNT(*) as cnt FROM file_links WHERE source_file_id = ?",
                    (fid1,),
                ).fetchone()["cnt"]
                self.assertEqual(count_after, 0)
                conn.close()


@unittest.skipUnless(_HAS_SQLITE_VEC, "sqlite-vec not installed")
class SerializeTests(unittest.TestCase):
    def test_serialize_f32(self) -> None:
        vec = [1.0, 2.0, 3.0]
        raw = db._serialize_f32(vec)
        self.assertEqual(len(raw), 12)  # 3 * 4 bytes
        unpacked = struct.unpack("<3f", raw)
        self.assertAlmostEqual(unpacked[0], 1.0)
        self.assertAlmostEqual(unpacked[1], 2.0)
        self.assertAlmostEqual(unpacked[2], 3.0)


if __name__ == "__main__":
    unittest.main()
