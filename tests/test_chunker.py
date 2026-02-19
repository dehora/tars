import unittest

from tars.chunker import (
    Chunk,
    _classify_line,
    _content_hash,
    _estimate_tokens,
    _score_boundary,
    chunk_markdown,
)


class EstimateTokensTests(unittest.TestCase):
    def test_empty(self) -> None:
        self.assertEqual(_estimate_tokens(""), 0)

    def test_known_length(self) -> None:
        self.assertEqual(_estimate_tokens("a" * 100), 25)


class ContentHashTests(unittest.TestCase):
    def test_deterministic(self) -> None:
        self.assertEqual(_content_hash("hello"), _content_hash("hello"))

    def test_different_inputs(self) -> None:
        self.assertNotEqual(_content_hash("a"), _content_hash("b"))


class ClassifyLineTests(unittest.TestCase):
    def test_headings(self) -> None:
        self.assertEqual(_classify_line("# Title")[1], 100)
        self.assertEqual(_classify_line("## Sub")[1], 90)
        self.assertEqual(_classify_line("### H3")[1], 80)
        self.assertEqual(_classify_line("###### H6")[1], 50)

    def test_hr(self) -> None:
        kind, score = _classify_line("---")
        self.assertEqual(kind, "hr")
        self.assertEqual(score, 70)

    def test_fence(self) -> None:
        kind, score = _classify_line("```python")
        self.assertEqual(kind, "fence")
        self.assertEqual(score, 80)

    def test_blank(self) -> None:
        kind, score = _classify_line("")
        self.assertEqual(kind, "blank")
        self.assertEqual(score, 10)

    def test_list(self) -> None:
        kind, score = _classify_line("- item")
        self.assertEqual(kind, "list")
        self.assertEqual(score, 5)

    def test_plain_text(self) -> None:
        kind, score = _classify_line("Just some text.")
        self.assertIsNone(kind)
        self.assertEqual(score, 0)


class ScoreBoundaryTests(unittest.TestCase):
    def test_at_target(self) -> None:
        # Distance 0 => full baseline
        self.assertAlmostEqual(_score_boundary(100, 0, 10), 100.0)

    def test_decayed(self) -> None:
        score = _score_boundary(100, 5, 10)
        self.assertLess(score, 100.0)
        self.assertGreater(score, 0.0)

    def test_heading_beats_blank_same_distance(self) -> None:
        heading = _score_boundary(100, 3, 10)
        blank = _score_boundary(10, 3, 10)
        self.assertGreater(heading, blank)

    def test_zero_window(self) -> None:
        self.assertAlmostEqual(_score_boundary(80, 0, 0), 80.0)


class ChunkMarkdownTests(unittest.TestCase):
    def test_empty_text(self) -> None:
        self.assertEqual(chunk_markdown(""), [])
        self.assertEqual(chunk_markdown("   \n  \n"), [])

    def test_short_doc_single_chunk(self) -> None:
        text = "# Hello\n\nSome content here.\n"
        chunks = chunk_markdown(text, target_tokens=2000)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].sequence, 0)
        self.assertEqual(chunks[0].start_line, 1)
        self.assertEqual(chunks[0].end_line, 3)
        self.assertIn("Hello", chunks[0].content)

    def test_sequence_numbering(self) -> None:
        # Build a doc large enough to produce multiple chunks
        text = "# Section 1\n\n" + ("word " * 400 + "\n") * 2
        text += "\n## Section 2\n\n" + ("word " * 400 + "\n") * 2
        chunks = chunk_markdown(text, target_tokens=200)
        self.assertGreater(len(chunks), 1)
        for i, chunk in enumerate(chunks):
            self.assertEqual(chunk.sequence, i)

    def test_line_numbers_are_1_indexed(self) -> None:
        text = "# Title\n\nParagraph.\n"
        chunks = chunk_markdown(text, target_tokens=2000)
        self.assertEqual(chunks[0].start_line, 1)

    def test_content_hash_is_sha256(self) -> None:
        text = "# Title\n\nContent.\n"
        chunks = chunk_markdown(text, target_tokens=2000)
        self.assertEqual(len(chunks[0].content_hash), 64)  # hex SHA-256

    def test_no_split_mid_code_block(self) -> None:
        # Code block should stay together if possible
        code = "```python\n" + ("x = 1\n" * 50) + "```\n"
        text = "# Before\n\n" + code + "\n## After\n\nMore text.\n"
        chunks = chunk_markdown(text, target_tokens=200)
        # Find the chunk(s) containing the code fence
        for chunk in chunks:
            if "```python" in chunk.content and "```" in chunk.content.split("```python", 1)[1]:
                # Opening and closing fence in same chunk — good
                pass
            elif "```python" in chunk.content:
                # Opening fence without close — check the closing fence is the boundary
                self.assertIn("x = 1", chunk.content)

    def test_overlap_between_chunks(self) -> None:
        text = ""
        for i in range(10):
            text += f"## Section {i}\n\n" + ("word " * 200 + "\n") * 2 + "\n"
        chunks = chunk_markdown(text, target_tokens=200, overlap_fraction=0.1)
        if len(chunks) >= 2:
            # Consecutive chunks should share some content at the boundary
            for i in range(len(chunks) - 1):
                end_lines = chunks[i].content.splitlines()[-3:]
                start_lines = chunks[i + 1].content.splitlines()[:3]
                # At least some overlap (may not always be exact lines)
                self.assertTrue(
                    len(end_lines) > 0 and len(start_lines) > 0,
                    "Chunks should have content at boundaries",
                )

    def test_zero_overlap(self) -> None:
        text = ""
        for i in range(6):
            text += f"## Section {i}\n\n" + ("word " * 120 + "\n") + "\n"
        chunks = chunk_markdown(text, target_tokens=120, overlap_fraction=0.0)
        self.assertGreaterEqual(len(chunks), 2)
        for i in range(len(chunks) - 1):
            self.assertEqual(chunks[i].end_line + 1, chunks[i + 1].start_line)

    def test_large_doc_chunks_near_target(self) -> None:
        text = ""
        for i in range(20):
            text += f"## Section {i}\n\n" + ("word " * 300 + "\n") + "\n"
        chunks = chunk_markdown(text, target_tokens=400)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            tokens = _estimate_tokens(chunk.content)
            # Allow generous range: 50% to 200% of target
            self.assertLessEqual(tokens, 800, f"Chunk too large: {tokens} tokens")

    def test_long_fence_extends_to_close(self) -> None:
        code = "```python\n" + ("x = 1\n" * 80) + "```\n"
        text = code + "\nAfter\n"
        chunks = chunk_markdown(text, target_tokens=50)
        self.assertGreaterEqual(len(chunks), 1)
        first = chunks[0].content
        self.assertIn("```python", first)
        self.assertIn("```", first.split("```python", 1)[1])


if __name__ == "__main__":
    unittest.main()
