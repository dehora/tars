"""Markdown-aware chunking with boundary scoring."""

import hashlib
import re
from dataclasses import dataclass

_HEADING_RE = re.compile(r"^(#{1,6})\s")
_HR_RE = re.compile(r"^(-{3,}|\*{3,}|_{3,})\s*$")
_FENCE_RE = re.compile(r"^(`{3,}|~{3,})")
_LIST_RE = re.compile(r"^(\s*[-*+]|\s*\d+\.)\s")
_DATA_IMG_RE = re.compile(r"!\[[^\]]*\]\(data:[^)]+\)")

_BOUNDARY_SCORES = {
    "h1": 100, "h2": 90, "h3": 80, "h4": 70, "h5": 60, "h6": 50,
    "fence": 80, "hr": 70, "blank": 10, "list": 5,
}


@dataclass(frozen=True, slots=True)
class Chunk:
    content: str
    sequence: int
    start_line: int  # 1-indexed
    end_line: int    # 1-indexed, inclusive
    content_hash: str


def _estimate_tokens(text: str) -> int:
    return len(text) // 4


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _classify_line(line: str) -> tuple[str | None, int]:
    """Return (boundary_type, baseline_score) for a line."""
    m = _HEADING_RE.match(line)
    if m:
        level = len(m.group(1))
        kind = f"h{level}"
        return kind, _BOUNDARY_SCORES[kind]
    if _HR_RE.match(line):
        return "hr", _BOUNDARY_SCORES["hr"]
    if _FENCE_RE.match(line):
        return "fence", _BOUNDARY_SCORES["fence"]
    if line.strip() == "":
        return "blank", _BOUNDARY_SCORES["blank"]
    if _LIST_RE.match(line):
        return "list", _BOUNDARY_SCORES["list"]
    return None, 0


def _score_boundary(baseline: int, distance: int, window: int) -> float:
    """Score a boundary candidate. Higher = better cut point."""
    if window <= 0:
        return float(baseline)
    ratio = distance / window
    return baseline * (1.0 - ratio * ratio)


def chunk_markdown(
    text: str,
    *,
    target_tokens: int = 800,
    overlap_fraction: float = 0.1,
) -> list[Chunk]:
    if not text or not text.strip():
        return []

    text = _DATA_IMG_RE.sub("", text)
    lines = text.splitlines(keepends=True)
    total_lines = len(lines)

    # Pre-classify every line
    classifications = [_classify_line(line) for line in lines]

    chunks: list[Chunk] = []
    pos = 0  # current line index
    seq = 0

    while pos < total_lines:
        # Accumulate tokens, tracking code fence state
        in_fence = False
        tokens = 0
        window_start = None  # line index where search window opens (70% of target)
        end = pos  # exclusive end of current chunk

        for i in range(pos, total_lines):
            line_tokens = _estimate_tokens(lines[i])
            tokens += line_tokens
            end = i + 1

            kind, _ = classifications[i]
            if kind == "fence":
                in_fence = not in_fence

            if window_start is None and tokens >= target_tokens * 0.7:
                window_start = i

            if tokens >= target_tokens * 1.3:
                break

        # If we consumed everything, just take it
        if end >= total_lines:
            chunk_text = "".join(lines[pos:total_lines])
            if chunk_text.strip():
                chunks.append(Chunk(
                    content=chunk_text,
                    sequence=seq,
                    start_line=pos + 1,
                    end_line=total_lines,
                    content_hash=_content_hash(chunk_text),
                ))
            break

        # Find best boundary in the search window
        if window_start is None:
            window_start = pos

        best_score = -1.0
        best_idx = end  # default: hard cut at end

        # Distance reference: ideal cut is at target_tokens worth of content
        target_line = None
        running = 0
        for i in range(pos, end):
            running += _estimate_tokens(lines[i])
            if target_line is None and running >= target_tokens:
                target_line = i
                break
        if target_line is None:
            target_line = end - 1

        window_size = end - window_start

        # Track fence state up to window_start to know if we're inside a code block
        fence_state = False
        for i in range(pos, window_start):
            if classifications[i][0] == "fence":
                fence_state = not fence_state

        for i in range(window_start, end):
            kind, baseline = classifications[i]
            if kind == "fence":
                fence_state = not fence_state

            if baseline == 0:
                continue

            # Don't split inside a code fence (but allow splitting ON a fence boundary)
            if fence_state and kind != "fence":
                continue

            distance = abs(i - target_line)
            score = _score_boundary(baseline, distance, window_size)
            if score > best_score:
                best_score = score
                best_idx = i  # cut BEFORE this line (this line starts next chunk)

        # If we are inside a fence and found no safe boundary, extend to fence close (with a cap)
        if best_score < 0 and in_fence:
            max_tokens = int(target_tokens * 3.0)
            tokens_ext = tokens
            extend_end = end
            fence_state_ext = in_fence
            while extend_end < total_lines and tokens_ext < max_tokens:
                kind, _ = classifications[extend_end]
                tokens_ext += _estimate_tokens(lines[extend_end])
                extend_end += 1
                if kind == "fence":
                    fence_state_ext = not fence_state_ext
                    if not fence_state_ext:
                        break
            if extend_end > end:
                best_idx = extend_end

        # If best_idx == pos, we'd make an empty chunk; push forward
        if best_idx <= pos:
            best_idx = end

        chunk_text = "".join(lines[pos:best_idx])
        if chunk_text.strip():
            chunks.append(Chunk(
                content=chunk_text,
                sequence=seq,
                start_line=pos + 1,
                end_line=best_idx,
                content_hash=_content_hash(chunk_text),
            ))
            seq += 1

        # Next chunk starts with overlap
        overlap_lines = max(0, int((best_idx - pos) * overlap_fraction))
        next_pos = best_idx - overlap_lines
        # Snap to a line boundary (already line-aligned)
        # Don't go backwards
        if next_pos <= pos:
            next_pos = best_idx
        pos = next_pos

    return chunks
