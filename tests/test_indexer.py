import os
import sys
import tempfile
import unittest
from pathlib import Path
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
from tars.indexer import (
    _batched_embed,
    _discover_files,
    _discover_vault_files,
    _embed_prefix,
    _extract_wikilinks,
    build_index,
    build_notes_index,
)

_DIM = 4


def _fake_embed(**kwargs):
    texts = kwargs.get("input", [])
    return {"embeddings": [[0.1] * _DIM for _ in texts]}


class ExtractWikilinksTests(unittest.TestCase):
    def test_basic(self) -> None:
        self.assertEqual(_extract_wikilinks("See [[Note]]"), ["Note"])

    def test_alias(self) -> None:
        self.assertEqual(_extract_wikilinks("See [[Note|Alias]]"), ["Note"])

    def test_heading(self) -> None:
        self.assertEqual(_extract_wikilinks("See [[Note#Section]]"), ["Note"])

    def test_dedup(self) -> None:
        result = _extract_wikilinks("[[Note]] and [[Note]] again")
        self.assertEqual(result, ["Note"])

    def test_multiple(self) -> None:
        result = _extract_wikilinks("[[Alpha]] then [[Beta]]")
        self.assertEqual(result, ["Alpha", "Beta"])

    def test_no_links(self) -> None:
        self.assertEqual(_extract_wikilinks("plain text"), [])

    def test_alias_and_heading_combined(self) -> None:
        result = _extract_wikilinks("[[Note#Section|Display]]")
        self.assertEqual(result, ["Note"])

    def test_skips_images(self) -> None:
        content = "![[photo.png]] and [[Note]] and ![[clip.mp4]]"
        self.assertEqual(_extract_wikilinks(content), ["Note"])

    def test_skips_pdf(self) -> None:
        self.assertEqual(_extract_wikilinks("[[doc.pdf]]"), [])

    def test_empty_content(self) -> None:
        self.assertEqual(_extract_wikilinks(""), [])


class DiscoverFilesTests(unittest.TestCase):
    def test_finds_memory_and_session_files(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "Memory.md").write_text("# Memory", encoding="utf-8")
            (d / "Procedural.md").write_text("# Proc", encoding="utf-8")
            sessions = d / "sessions"
            sessions.mkdir()
            (sessions / "2025-01-01.md").write_text("session 1", encoding="utf-8")
            (sessions / "2025-01-02.md").write_text("session 2", encoding="utf-8")
            (sessions / "notes.txt").write_text("not markdown", encoding="utf-8")

            result = _discover_files(d)
            paths = [str(p) for p, _ in result]
            types = [t for _, t in result]

            self.assertIn(str(d / "Memory.md"), paths)
            self.assertIn(str(d / "Procedural.md"), paths)
            self.assertIn("semantic", types)
            self.assertIn("procedural", types)
            self.assertEqual(types.count("episodic"), 2)
            # .txt file should not be included
            self.assertNotIn(str(sessions / "notes.txt"), paths)

    def test_empty_dir(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            result = _discover_files(Path(td))
            self.assertEqual(result, [])


class BuildIndexTests(unittest.TestCase):
    def setUp(self) -> None:
        # Patch the ollama module reference inside tars.embeddings directly
        self._patcher = mock.patch.object(embeddings, "ollama")
        self._mock_ollama = self._patcher.start()
        self._mock_ollama.embed.side_effect = _fake_embed

    def tearDown(self) -> None:
        self._patcher.stop()

    def test_no_memory_dir(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            os.environ.pop("TARS_MEMORY_DIR", None)
            stats = build_index()
        self.assertEqual(stats, {"indexed": 0, "skipped": 0, "chunks": 0, "deleted": 0})

    def test_indexes_files(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "Memory.md").write_text("# Semantic\n\nSome facts.\n", encoding="utf-8")
            sessions = d / "sessions"
            sessions.mkdir()
            (sessions / "log.md").write_text("# Session\n\nA session.\n", encoding="utf-8")

            with mock.patch.dict(os.environ, {"TARS_MEMORY_DIR": td}):
                stats = build_index(model="test-model")

            self.assertEqual(stats["indexed"], 2)
            self.assertEqual(stats["skipped"], 0)
            self.assertGreater(stats["chunks"], 0)

    def test_incremental_skip(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "Memory.md").write_text("# Memory\n\nFacts.\n", encoding="utf-8")

            with mock.patch.dict(os.environ, {"TARS_MEMORY_DIR": td}):
                stats1 = build_index(model="test-model")
                stats2 = build_index(model="test-model")

            self.assertEqual(stats1["indexed"], 1)
            self.assertEqual(stats2["indexed"], 0)
            self.assertEqual(stats2["skipped"], 1)

    def test_reindex_on_change(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "Memory.md").write_text("# Memory\n\nVersion 1.\n", encoding="utf-8")

            with mock.patch.dict(os.environ, {"TARS_MEMORY_DIR": td}):
                stats1 = build_index(model="test-model")
                self.assertEqual(stats1["indexed"], 1)

                # Change file content
                (d / "Memory.md").write_text("# Memory\n\nVersion 2 updated.\n", encoding="utf-8")
                stats2 = build_index(model="test-model")

            self.assertEqual(stats2["indexed"], 1)
            self.assertEqual(stats2["skipped"], 0)


    def test_reindex_on_model_change(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "Memory.md").write_text("# Memory\n\nFacts.\n", encoding="utf-8")

            with mock.patch.dict(os.environ, {"TARS_MEMORY_DIR": td}):
                stats1 = build_index(model="test-model")
                self.assertEqual(stats1["indexed"], 1)
                self.assertEqual(stats1["skipped"], 0)

                # Same content, different model → should reindex
                stats2 = build_index(model="different-model")
                self.assertEqual(stats2["indexed"], 1)
                self.assertEqual(stats2["skipped"], 0)

    def test_deleted_file_removed_from_index(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "Memory.md").write_text("# Memory\n\nFacts.\n", encoding="utf-8")
            sessions = d / "sessions"
            sessions.mkdir()
            (sessions / "log.md").write_text("# Session\n\nA session.\n", encoding="utf-8")

            with mock.patch.dict(os.environ, {"TARS_MEMORY_DIR": td}):
                stats1 = build_index(model="test-model")
                self.assertEqual(stats1["indexed"], 2)

                # Delete the session file
                (sessions / "log.md").unlink()
                stats2 = build_index(model="test-model")

            self.assertEqual(stats2["deleted"], 1)
            self.assertEqual(stats2["skipped"], 1)  # Memory.md unchanged


    def test_embedding_mismatch_leaves_file_reindexable(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "Memory.md").write_text("# Memory\n\nFacts.\n", encoding="utf-8")

            with mock.patch.dict(os.environ, {"TARS_MEMORY_DIR": td}):
                build_index(model="test-model")

                (d / "Memory.md").write_text(
                    "# Memory\n\nUpdated facts.\n\n## Section\n\nMore content.\n",
                    encoding="utf-8",
                )
                # Bad embeddings — _index_files catches the error per-file
                with mock.patch("tars.indexer.embed", return_value=[]):
                    stats_bad = build_index(model="test-model")
                self.assertEqual(stats_bad["indexed"], 0)

                # Restore good embeddings — file should be reindexed, not skipped
                stats = build_index(model="test-model")

            self.assertEqual(stats["indexed"], 1)
            self.assertEqual(stats["skipped"], 0)


class DiscoverVaultFilesTests(unittest.TestCase):
    def test_finds_markdown_files_recursively(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "note1.md").write_text("# Note 1", encoding="utf-8")
            sub = d / "subfolder"
            sub.mkdir()
            (sub / "note2.md").write_text("# Note 2", encoding="utf-8")
            (sub / "image.png").write_text("not text", encoding="utf-8")

            result = _discover_vault_files(d)
            paths = [str(p) for p, _ in result]
            types = [t for _, t in result]

            self.assertIn(str(d / "note1.md"), paths)
            self.assertIn(str(sub / "note2.md"), paths)
            self.assertNotIn(str(sub / "image.png"), paths)
            self.assertTrue(all(t == "note" for t in types))

    def test_skips_hidden_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "visible.md").write_text("# Visible", encoding="utf-8")
            hidden = d / ".obsidian"
            hidden.mkdir()
            (hidden / "config.md").write_text("# Config", encoding="utf-8")
            trash = d / ".trash"
            trash.mkdir()
            (trash / "deleted.md").write_text("# Deleted", encoding="utf-8")

            result = _discover_vault_files(d)
            paths = [str(p) for p, _ in result]

            self.assertIn(str(d / "visible.md"), paths)
            self.assertNotIn(str(hidden / "config.md"), paths)
            self.assertNotIn(str(trash / "deleted.md"), paths)

    def test_empty_vault(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            result = _discover_vault_files(Path(td))
            self.assertEqual(result, [])


class BuildNotesIndexTests(unittest.TestCase):
    def setUp(self) -> None:
        self._patcher = mock.patch.object(embeddings, "ollama")
        self._mock_ollama = self._patcher.start()
        self._mock_ollama.embed.side_effect = _fake_embed

    def tearDown(self) -> None:
        self._patcher.stop()

    def test_no_notes_dir(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            os.environ.pop("TARS_NOTES_DIR", None)
            stats = build_notes_index()
        self.assertEqual(stats, {"indexed": 0, "skipped": 0, "chunks": 0, "deleted": 0})

    def test_creates_notes_db_in_notes_dir(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "note.md").write_text("# My Note\n\nSome content.\n", encoding="utf-8")
            with mock.patch.dict(os.environ, {"TARS_NOTES_DIR": td}):
                stats = build_notes_index(model="test-model")
            self.assertTrue((d / "notes.db").exists())
            self.assertEqual(stats["indexed"], 1)
            self.assertGreater(stats["chunks"], 0)

    def test_uses_notes_collection(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "note.md").write_text("# Note\n\nContent.\n", encoding="utf-8")
            with mock.patch.dict(os.environ, {"TARS_NOTES_DIR": td}):
                build_notes_index(model="test-model")
            import sqlite3
            conn = sqlite3.connect(str(d / "notes.db"))
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT name FROM collections").fetchone()
            conn.close()
            self.assertEqual(row["name"], "notes")

    def test_incremental_skip(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "note.md").write_text("# Note\n\nFacts.\n", encoding="utf-8")
            with mock.patch.dict(os.environ, {"TARS_NOTES_DIR": td}):
                stats1 = build_notes_index(model="test-model")
                stats2 = build_notes_index(model="test-model")
            self.assertEqual(stats1["indexed"], 1)
            self.assertEqual(stats2["indexed"], 0)
            self.assertEqual(stats2["skipped"], 1)


class ContextInEmbeddingTests(unittest.TestCase):
    def setUp(self) -> None:
        self._patcher = mock.patch.object(embeddings, "ollama")
        self._mock_ollama = self._patcher.start()
        self._mock_ollama.embed.side_effect = _fake_embed

    def tearDown(self) -> None:
        self._patcher.stop()

    def test_heading_context_prepended_in_embed_input(self) -> None:
        content = "# Recipes\n\n## Main Dishes\n\n" + ("Pasta with sauce and garlic bread.\n" * 200)
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "Memory.md").write_text(content, encoding="utf-8")
            with mock.patch.dict(os.environ, {"TARS_MEMORY_DIR": td}):
                with mock.patch("tars.indexer.embed", side_effect=lambda texts, **kw: [[0.1] * _DIM for _ in texts]) as mock_embed:
                    build_index(model="test-model")
            self.assertTrue(mock_embed.called)
            texts = mock_embed.call_args[0][0]
            context_texts = [t for t in texts if t.startswith("Recipes > Main Dishes")]
            self.assertGreater(len(context_texts), 0,
                               "At least one embed input should have heading context prepended")


class EmbedPrefixTests(unittest.TestCase):
    def test_empty_context(self) -> None:
        self.assertEqual(_embed_prefix(""), "")

    def test_short_context_unchanged(self) -> None:
        self.assertEqual(_embed_prefix("Alpha > Beta"), "Alpha > Beta")

    def test_caps_at_three_levels(self) -> None:
        result = _embed_prefix("L1 > L2 > L3 > L4 > L5")
        self.assertEqual(result, "L1 > L2 > L3")

    def test_truncates_long_breadcrumb(self) -> None:
        long = " > ".join(f"Heading{i:03d}" for i in range(3))
        # Should not exceed _MAX_CONTEXT_CHARS (120)
        result = _embed_prefix(long)
        self.assertLessEqual(len(result), 120)

    def test_truncation_preserves_whole_segments(self) -> None:
        ctx = "A" * 50 + " > " + "B" * 50 + " > " + "C" * 50
        result = _embed_prefix(ctx)
        self.assertNotIn(" > C", result)
        self.assertIn(" > ", result)


class BatchedEmbedTests(unittest.TestCase):
    def test_single_batch(self) -> None:
        texts = ["hello", "world"]
        with mock.patch("tars.indexer.embed", return_value=[[0.1] * _DIM, [0.2] * _DIM]) as m:
            result = _batched_embed(texts, model="test")
        self.assertEqual(len(result), 2)
        m.assert_called_once()

    def test_multiple_batches(self) -> None:
        texts = [f"text{i}" for i in range(100)]
        call_count = 0

        def fake_embed(batch, **kw):
            nonlocal call_count
            call_count += 1
            return [[0.1] * _DIM for _ in batch]

        with mock.patch("tars.indexer.embed", side_effect=fake_embed):
            result = _batched_embed(texts, model="test")
        self.assertEqual(len(result), 100)
        self.assertEqual(call_count, 2)  # 64 + 36

    def test_retry_on_transient_failure(self) -> None:
        attempt = {"n": 0}

        def flaky_embed(batch, **kw):
            attempt["n"] += 1
            if attempt["n"] == 1:
                raise ConnectionError("transient")
            return [[0.1] * _DIM for _ in batch]

        with mock.patch("tars.indexer.embed", side_effect=flaky_embed):
            with mock.patch("tars.indexer.time.sleep"):
                result = _batched_embed(["hello"], model="test")
        self.assertEqual(len(result), 1)
        self.assertEqual(attempt["n"], 2)

    def test_raises_after_max_retries(self) -> None:
        with mock.patch("tars.indexer.embed", side_effect=ConnectionError("down")):
            with mock.patch("tars.indexer.time.sleep"):
                with self.assertRaises(ConnectionError):
                    _batched_embed(["hello"], model="test")

    def test_count_mismatch_raises(self) -> None:
        with mock.patch("tars.indexer.embed", return_value=[[0.1] * _DIM]):
            with self.assertRaises(ValueError):
                _batched_embed(["a", "b"], model="test")


class SavepointAtomicityTests(unittest.TestCase):
    def setUp(self) -> None:
        self._patcher = mock.patch.object(embeddings, "ollama")
        self._mock_ollama = self._patcher.start()
        self._mock_ollama.embed.side_effect = _fake_embed

    def tearDown(self) -> None:
        self._patcher.stop()

    def test_embed_failure_preserves_old_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "Memory.md").write_text(
                "# Memory\n\nOriginal content here.\n", encoding="utf-8"
            )

            with mock.patch.dict(os.environ, {"TARS_MEMORY_DIR": td}):
                stats1 = build_index(model="test-model")
                self.assertEqual(stats1["indexed"], 1)

                (d / "Memory.md").write_text(
                    "# Memory\n\nChanged content.\n", encoding="utf-8"
                )

                # Fail embedding — _index_files catches per-file
                with mock.patch("tars.indexer.embed", side_effect=RuntimeError("embed service down")):
                    stats2 = build_index(model="test-model")
                self.assertEqual(stats2["indexed"], 0)

                # Restore embed and retry — file should be reindexable (content_hash reset)
                stats3 = build_index(model="test-model")
                self.assertEqual(stats3["indexed"], 1)

    def test_failed_file_does_not_block_others(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "Memory.md").write_text("# Memory\n\nGood file.\n", encoding="utf-8")
            sessions = d / "sessions"
            sessions.mkdir()
            (sessions / "bad.md").write_text("# Bad\n\nWill fail.\n", encoding="utf-8")

            call_count = {"n": 0}

            def fail_for_bad(texts, **kw):
                call_count["n"] += 1
                if any("Will fail" in t for t in texts):
                    raise RuntimeError("boom")
                return [[0.1] * _DIM for _ in texts]

            with mock.patch.dict(os.environ, {"TARS_MEMORY_DIR": td}):
                with mock.patch("tars.indexer.embed", side_effect=fail_for_bad):
                    stats = build_index(model="test-model")

            self.assertEqual(stats["indexed"], 1)


@unittest.skipUnless(_HAS_SQLITE_VEC, "sqlite-vec not installed")
class ZeroChunkCleanupTests(unittest.TestCase):
    def setUp(self) -> None:
        self._patcher = mock.patch.object(embeddings, "ollama")
        self._mock_ollama = self._patcher.start()
        self._mock_ollama.embed.side_effect = _fake_embed

    def tearDown(self) -> None:
        self._patcher.stop()

    def test_emptied_file_cleans_stale_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "Memory.md").write_text(
                "# Memory\n\nSome real content here.\n", encoding="utf-8"
            )

            with mock.patch.dict(os.environ, {"TARS_MEMORY_DIR": td}):
                stats1 = build_index(model="test-model")
                self.assertEqual(stats1["indexed"], 1)
                self.assertGreater(stats1["chunks"], 0)

                # Empty the file — chunker produces zero chunks
                (d / "Memory.md").write_text("", encoding="utf-8")
                stats2 = build_index(model="test-model")

                self.assertEqual(stats2["indexed"], 1)
                self.assertEqual(stats2["chunks"], 0)

                # Verify no stale chunks remain in DB
                conn = db._connect(d / "tars.db")
                count = conn.execute(
                    "SELECT count(*) as cnt FROM vec_chunks"
                ).fetchone()["cnt"]
                conn.close()
                self.assertEqual(count, 0)


class StartupIndexTests(unittest.TestCase):
    def test_swallows_exceptions(self) -> None:
        from tars.cli import _startup_index

        with mock.patch("tars.cli.build_index", side_effect=RuntimeError("boom")):
            _startup_index()  # should not raise

    def test_prints_exception_type(self) -> None:
        from tars.cli import _startup_index

        with mock.patch("tars.cli.build_index", side_effect=ValueError("bad dim")):
            with mock.patch("builtins.print") as mock_print:
                _startup_index()
            mock_print.assert_called_once()
            msg = mock_print.call_args[0][0]
            self.assertIn("ValueError", msg)
            self.assertIn("bad dim", msg)


if __name__ == "__main__":
    unittest.main()
