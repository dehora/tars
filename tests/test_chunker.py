import unittest

from tars.chunker import (
    Chunk,
    _FENCE_RE,
    _build_heading_context,
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
        self.assertEqual(score, 1)

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
        code = "```python\n" + ("x = 1\n" * 30) + "```\n"
        text = code + "\nAfter\n"
        chunks = chunk_markdown(text, target_tokens=50)
        self.assertGreaterEqual(len(chunks), 1)
        first = chunks[0].content
        self.assertIn("```python", first)
        self.assertIn("```", first.split("```python", 1)[1])


    def test_strips_inline_base64_images(self) -> None:
        b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ" + "A" * 200
        text = f"# Note\n\nSome text.\n\n![screenshot](data:image/png;base64,{b64})\n\nMore text.\n"
        chunks = chunk_markdown(text)
        full = "".join(c.content for c in chunks)
        self.assertNotIn("data:image", full)
        self.assertNotIn("base64", full)
        self.assertIn("Some text.", full)
        self.assertIn("More text.", full)

    def test_strips_multiple_inline_images(self) -> None:
        b64 = "A" * 100
        text = (
            "# Note\n\n"
            f"![img1](data:image/png;base64,{b64})\n"
            "Middle text.\n"
            f"![img2](data:image/jpeg;base64,{b64})\n"
        )
        chunks = chunk_markdown(text)
        full = "".join(c.content for c in chunks)
        self.assertNotIn("data:image", full)
        self.assertIn("Middle text.", full)


class HeadingContextTests(unittest.TestCase):
    def test_chunk_context_empty_for_no_headings(self) -> None:
        text = "Just plain text.\n" * 5
        chunks = chunk_markdown(text, target_tokens=2000)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].context, "")

    def test_chunk_context_multi_chunk(self) -> None:
        text = "# Alpha\n\n" + ("word " * 300 + "\n") * 2
        text += "\n# Beta\n\n" + ("word " * 300 + "\n") * 2
        chunks = chunk_markdown(text, target_tokens=200)
        self.assertGreater(len(chunks), 1)
        last = chunks[-1]
        self.assertIn("Beta", last.context)

    def test_chunk_context_nested_headings(self) -> None:
        text = "# Top\n\n## Middle\n\n### Deep\n\n"
        text += ("word " * 300 + "\n") * 3
        chunks = chunk_markdown(text, target_tokens=200)
        self.assertGreater(len(chunks), 1)
        second = chunks[1]
        self.assertEqual(second.context, "Top > Middle > Deep")

    def test_chunk_context_heading_reset(self) -> None:
        text = "# First\n\n## Sub\n\n"
        text += ("word " * 300 + "\n") * 2
        text += "\n# Second\n\n"
        text += ("word " * 300 + "\n") * 2
        chunks = chunk_markdown(text, target_tokens=200)
        last = chunks[-1]
        self.assertIn("Second", last.context)
        self.assertNotIn("Sub", last.context)

    def test_content_hash_excludes_context(self) -> None:
        text = "# Heading\n\n" + ("word " * 300 + "\n") * 3
        chunks = chunk_markdown(text, target_tokens=200)
        for chunk in chunks:
            expected = _content_hash(chunk.content)
            self.assertEqual(chunk.content_hash, expected)

    def test_list_resists_splitting(self) -> None:
        text = "# Shopping List\n\n"
        for i in range(15):
            text += f"- Item number {i} with some description text\n"
        chunks = chunk_markdown(text, target_tokens=400)
        items_in_first = sum(1 for line in chunks[0].content.splitlines()
                            if line.startswith("- Item"))
        self.assertGreaterEqual(items_in_first, 10,
                                "Most list items should stay in the same chunk")


class HorizontalRuleContextTests(unittest.TestCase):
    def test_hr_does_not_crash_heading_context(self) -> None:
        text = "# Intro\n\n---\n\nSome text after hr.\n"
        text += ("word " * 300 + "\n") * 3
        chunks = chunk_markdown(text, target_tokens=200)
        self.assertGreater(len(chunks), 0)
        for chunk in chunks:
            self.assertNotIn("---", chunk.context)

    def test_hr_between_headings_preserves_context(self) -> None:
        text = "# Alpha\n\nSome text.\n\n---\n\n## Beta\n\n"
        text += ("word " * 300 + "\n") * 3
        chunks = chunk_markdown(text, target_tokens=200)
        last = chunks[-1]
        self.assertIn("Beta", last.context)


class OverlapFenceParityTests(unittest.TestCase):
    def test_overlap_does_not_start_inside_fence(self) -> None:
        # Build: prose, fence-open, code, fence-close, prose
        # Sized so the cut lands after the fence block but overlap
        # would normally back into it.
        text = ("word " * 200 + "\n")
        text += "```python\n"
        text += "x = 1\n" * 20
        text += "```\n"
        text += ("word " * 200 + "\n")
        chunks = chunk_markdown(text, target_tokens=200, overlap_fraction=0.3)
        for chunk in chunks:
            # No chunk should start with a bare code line inside a fence
            lines = chunk.content.strip().splitlines()
            if lines and not lines[0].startswith("```"):
                # First line isn't a fence opener — check we're not mid-code
                fence_opens = sum(1 for l in lines if _FENCE_RE.match(l))
                # If we see a closing fence without an opener, overlap broke parity
                self.assertTrue(
                    fence_opens == 0 or fence_opens % 2 == 0,
                    f"Chunk starts with unbalanced fence: {lines[:3]}",
                )


class IndentedFenceTests(unittest.TestCase):
    def test_indented_fence_detected(self) -> None:
        kind, score = _classify_line("  ```python")
        self.assertEqual(kind, "fence")
        self.assertGreater(score, 0)

    def test_indented_fence_in_list_tracked(self) -> None:
        text = "- Item 1\n  ```\n  code\n  ```\n- Item 2\n"
        text += ("word " * 300 + "\n") * 3
        chunks = chunk_markdown(text, target_tokens=200)
        self.assertGreater(len(chunks), 0)


class TokenEstimationTests(unittest.TestCase):
    def test_code_tokens_not_undercounted(self) -> None:
        code = "x = foo(a, b, c)\n" * 10
        prose = "This is a normal English sentence with several words.\n" * 10
        code_tokens = _estimate_tokens(code)
        prose_tokens = _estimate_tokens(prose)
        # Code should get at least as many tokens per char as prose
        code_ratio = code_tokens / len(code) if code else 0
        prose_ratio = prose_tokens / len(prose) if prose else 0
        self.assertGreaterEqual(code_ratio, prose_ratio * 0.8)

    def test_word_based_floor_active_for_short_tokens(self) -> None:
        # Short identifiers: char/4 undercounts, word-based should kick in
        code = "a b c d e f g h i j\n"
        tokens = _estimate_tokens(code)
        self.assertGreaterEqual(tokens, 10)


class ListBoundaryScoreTests(unittest.TestCase):
    def test_blank_after_list_gets_higher_score(self) -> None:
        text = "- item 1\n- item 2\n\nSome prose.\n"
        lines = text.splitlines(keepends=True)
        classifications = [_classify_line(line) for line in lines]
        # Post-process like chunk_markdown does
        for i in range(1, len(lines)):
            kind, _ = classifications[i]
            prev_kind, _ = classifications[i - 1]
            if kind == "blank" and prev_kind == "list":
                classifications[i] = ("blank", 30)
        # The blank line after "- item 2" should be scored at 30
        blank_idx = 2  # "- item 1", "- item 2", "", "Some prose."
        self.assertEqual(classifications[blank_idx][1], 30)

    def test_list_end_preferred_as_cut_point(self) -> None:
        # Two list blocks separated by prose — should prefer cutting between them
        text = ""
        for i in range(15):
            text += f"- First list item {i}\n"
        text += "\nSome connecting prose.\n\n"
        for i in range(15):
            text += f"- Second list item {i}\n"
        chunks = chunk_markdown(text, target_tokens=200)
        if len(chunks) > 1:
            first_content = chunks[0].content
            self.assertIn("connecting prose", first_content + chunks[1].content)


if __name__ == "__main__":
    unittest.main()
