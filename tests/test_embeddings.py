import sys
import unittest
from unittest import mock

sys.modules.setdefault("anthropic", mock.Mock())
sys.modules.setdefault("dotenv", mock.Mock(load_dotenv=lambda: None))

# Mock ollama at module level before importing embeddings
_mock_ollama = mock.MagicMock()
sys.modules["ollama"] = _mock_ollama

from tars import embeddings

# Rebind in case another test file already imported tars.embeddings with a
# different ollama mock (test load order issue — see MEMORY.md).
embeddings.ollama = _mock_ollama


class EmbedTests(unittest.TestCase):
    def setUp(self) -> None:
        _mock_ollama.reset_mock()

    def test_single_string(self) -> None:
        _mock_ollama.embed.return_value = {"embeddings": [[0.1, 0.2, 0.3]]}
        result = embeddings.embed("hello")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], [0.1, 0.2, 0.3])
        _mock_ollama.embed.assert_called_once_with(
            model=embeddings.DEFAULT_EMBEDDING_MODEL, input=["hello"]
        )

    def test_batch(self) -> None:
        _mock_ollama.embed.return_value = {
            "embeddings": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        }
        result = embeddings.embed(["a", "b", "c"])
        self.assertEqual(len(result), 3)

    def test_custom_model(self) -> None:
        _mock_ollama.embed.return_value = {"embeddings": [[1.0]]}
        embeddings.embed("hi", model="custom-model")
        _mock_ollama.embed.assert_called_once_with(
            model="custom-model", input=["hi"]
        )

    def test_min_length_safety(self) -> None:
        # API returns fewer embeddings than inputs
        _mock_ollama.embed.return_value = {"embeddings": [[0.1]]}
        result = embeddings.embed(["a", "b", "c"])
        self.assertEqual(len(result), 1)

    def test_empty_embeddings(self) -> None:
        _mock_ollama.embed.return_value = {"embeddings": []}
        result = embeddings.embed("hello")
        self.assertEqual(result, [])


class InstructPrefixTests(unittest.TestCase):
    def setUp(self) -> None:
        _mock_ollama.reset_mock()

    def test_instruct_wraps_single_text(self) -> None:
        _mock_ollama.embed.return_value = {"embeddings": [[0.1, 0.2]]}
        embeddings.embed("hello", instruct="find relevant docs")
        call_args = _mock_ollama.embed.call_args
        self.assertEqual(
            call_args[1]["input"],
            ["Instruct: find relevant docs\nQuery:hello"],
        )

    def test_instruct_wraps_batch(self) -> None:
        _mock_ollama.embed.return_value = {"embeddings": [[0.1], [0.2]]}
        embeddings.embed(["a", "b"], instruct="search task")
        call_args = _mock_ollama.embed.call_args
        self.assertEqual(
            call_args[1]["input"],
            ["Instruct: search task\nQuery:a", "Instruct: search task\nQuery:b"],
        )

    def test_no_instruct_passes_raw(self) -> None:
        _mock_ollama.embed.return_value = {"embeddings": [[0.1]]}
        embeddings.embed("hello")
        call_args = _mock_ollama.embed.call_args
        self.assertEqual(call_args[1]["input"], ["hello"])

    def test_none_instruct_passes_raw(self) -> None:
        _mock_ollama.embed.return_value = {"embeddings": [[0.1]]}
        embeddings.embed("hello", instruct=None)
        call_args = _mock_ollama.embed.call_args
        self.assertEqual(call_args[1]["input"], ["hello"])


class SupportsInstructTests(unittest.TestCase):
    def test_qwen3_embedding_supported(self) -> None:
        self.assertTrue(embeddings._supports_instruct("qwen3-embedding:0.6b"))
        self.assertTrue(embeddings._supports_instruct("qwen3-embedding:8b"))

    def test_other_models_not_supported(self) -> None:
        self.assertFalse(embeddings._supports_instruct("nomic-embed-text"))
        self.assertFalse(embeddings._supports_instruct("test-model"))
        self.assertFalse(embeddings._supports_instruct("mxbai-embed-large"))


class EmbeddingDimensionsTests(unittest.TestCase):
    def setUp(self) -> None:
        _mock_ollama.reset_mock()

    def test_dimension_probe(self) -> None:
        _mock_ollama.embed.return_value = {"embeddings": [[0.0] * 1024]}
        dim = embeddings.embedding_dimensions()
        self.assertEqual(dim, 1024)

    def test_no_embeddings_raises(self) -> None:
        _mock_ollama.embed.return_value = {"embeddings": []}
        with self.assertRaises(RuntimeError):
            embeddings.embedding_dimensions()


class EmbeddingModelEnvTests(unittest.TestCase):
    def test_default_when_unset(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True):
            import importlib
            importlib.reload(embeddings)
        self.assertEqual(embeddings.DEFAULT_EMBEDDING_MODEL, "qwen3-embedding:8b")
        embeddings.ollama = _mock_ollama

    def test_default_when_empty(self) -> None:
        with mock.patch.dict("os.environ", {"TARS_MODEL_EMBEDDING": ""}, clear=True):
            import importlib
            importlib.reload(embeddings)
        self.assertEqual(embeddings.DEFAULT_EMBEDDING_MODEL, "qwen3-embedding:8b")
        embeddings.ollama = _mock_ollama

    def test_default_when_whitespace(self) -> None:
        with mock.patch.dict("os.environ", {"TARS_MODEL_EMBEDDING": "  "}, clear=True):
            import importlib
            importlib.reload(embeddings)
        self.assertEqual(embeddings.DEFAULT_EMBEDDING_MODEL, "qwen3-embedding:8b")
        embeddings.ollama = _mock_ollama

    def test_explicit_override(self) -> None:
        with mock.patch.dict("os.environ", {"TARS_MODEL_EMBEDDING": "nomic-embed-text"}, clear=True):
            import importlib
            importlib.reload(embeddings)
        self.assertEqual(embeddings.DEFAULT_EMBEDDING_MODEL, "nomic-embed-text")
        embeddings.ollama = _mock_ollama


if __name__ == "__main__":
    unittest.main()
