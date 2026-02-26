import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.modules.setdefault("anthropic", mock.Mock())
sys.modules.setdefault("dotenv", mock.Mock(load_dotenv=lambda: None))
sys.modules.setdefault("ollama", mock.MagicMock())

from tars import embeddings
from tars.indexer import _discover_files, _discover_vault_files, build_index, build_notes_index

_DIM = 4


def _fake_embed(**kwargs):
    texts = kwargs.get("input", [])
    return {"embeddings": [[0.1] * _DIM for _ in texts]}


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
                # First run succeeds normally
                build_index(model="test-model")

                # Change content to trigger reindex, but with bad embeddings
                (d / "Memory.md").write_text(
                    "# Memory\n\nUpdated facts.\n\n## Section\n\nMore content.\n",
                    encoding="utf-8",
                )
                # Patch embed at the indexer level so embedding_dimensions still works
                with mock.patch("tars.indexer.embed", return_value=[]):
                    with self.assertRaises(ValueError):
                        build_index(model="test-model")

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
