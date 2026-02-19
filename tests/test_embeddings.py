import sys
import unittest
from unittest import mock

sys.modules.setdefault("anthropic", mock.Mock())
sys.modules.setdefault("dotenv", mock.Mock(load_dotenv=lambda: None))

# Mock ollama at module level before importing embeddings
_mock_ollama = mock.MagicMock()
sys.modules["ollama"] = _mock_ollama

from tars import embeddings


class EmbedTests(unittest.TestCase):
    def setUp(self) -> None:
        _mock_ollama.reset_mock()

    def test_single_string(self) -> None:
        _mock_ollama.embed.return_value = {"embeddings": [[0.1, 0.2, 0.3]]}
        result = embeddings.embed("hello")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], [0.1, 0.2, 0.3])
        _mock_ollama.embed.assert_called_once_with(
            model="qwen3-embedding:0.6b", input=["hello"]
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


if __name__ == "__main__":
    unittest.main()
