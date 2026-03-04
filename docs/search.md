# Search Pipeline

End-to-end hybrid search: markdown-aware chunking, ollama embeddings, sqlite-vec KNN + FTS5 keyword, fused with Reciprocal Rank Fusion. Query rewriting and HyDE for expanded retrieval. Two-pass context packing for automatic search on first message.

## Overview

```
[markdown files] → chunker → embeddings → sqlite-vec (vec_chunks) + FTS5 (chunks_fts)
                                                    ↓
[user query] → rewriter (expand + HyDE) → vec KNN + FTS5 BM25 → RRF fusion → results
                                                    ↓
                              two-pass context packing → system prompt injection
```

## Indexing

`indexer.py` orchestrates the full pipeline:

1. Discover files — named memory files (`Memory.md`, `Procedural.md`, etc.) plus `sessions/*.md`
2. Diff against indexed paths — delete removed files, skip unchanged (via `content_hash`)
3. For each changed file: chunk → embed → store inside a savepoint

Incremental by default — only re-indexes files whose content hash changed. A separate `build_notes_index()` handles the personal vault (`TARS_NOTES_DIR`) with its own database (`notes.db`).

### Failure handling

Each file is indexed inside a SQLite savepoint. On embedding failure, the savepoint rolls back (preserving old chunks), `content_hash` is reset to `''` so the file reindexes next run, then the error re-raises.

Batch embedding uses `_embed_batch_size = 64` with 3 retries and exponential backoff (0.5s, 1s).

## Chunking

`chunker.py` splits markdown into chunks with awareness of document structure.

**Parameters**: `target_tokens=400`, `overlap_fraction=0.1`

### Boundary scoring

Lines are classified by type with baseline scores:

| Type | Score |
|------|-------|
| h1-h6 | 100-50 (decreasing by level) |
| Code fence | 80 |
| Horizontal rule | 70 |
| Blank line | 10 (30 at list boundaries) |
| List item | 1 |

Score decays quadratically with distance from the ideal cut point: `baseline * (1 - (distance/window)^2)`. The search window opens at 70% of target tokens and closes at 130%.

### Code fence safety

The chunker tracks fence state and refuses to split inside a fenced block except on the delimiter line itself. If no safe boundary is found, it extends up to 3x target tokens to find the closing fence.

### Overlap

10% of the chunk's line count is repeated at the start of the next chunk. If the overlap region contains an unclosed fence, the overlap is pushed past the fence's last line.

### Heading context

Each chunk carries a `context` field — a breadcrumb like `H1 > H2 > H3` built from all headings before the chunk's start line. This is prepended to the embedding text at index time (capped to 3 levels, 120 chars).

### Token estimation

`max(len(text) // 4, int(len(text.split()) * 1.3))` — same formula used in both `chunker.py` and `core.py`.

## Embedding

`embeddings.py` wraps `ollama.embed()`.

**Default model**: `TARS_MODEL_EMBEDDING` env var, fallback `qwen3-embedding:8b`.

### Instruction asymmetry

Models that support it (currently `qwen3-embedding` prefix) use asymmetric query/document encoding:

- **Query embeddings**: wrapped as `Instruct: {task}\nQuery:{text}` where the task is `"Given a search query, retrieve relevant passages that answer the query"`
- **Document embeddings**: no instruction prefix (plain text)
- **HyDE embeddings**: no instruction prefix (treated as pseudo-documents)

This asymmetry improves retrieval quality for instruction-aware models.

### Dimension probing

`embedding_dimensions()` embeds a probe string and returns `len(vecs[0])`. Used at index creation to size the sqlite-vec table.

## Storage

`db.py` manages the SQLite schema with sqlite-vec and FTS5 extensions.

### Schema

| Table | Type | Purpose |
|-------|------|---------|
| `collections` | regular | Named groups (`tars_memory`, `notes`) |
| `files` | regular | File metadata, `content_hash` for incremental indexing |
| `metadata` | regular | Stores `embedding_model`, `vec_dim`, `distance_metric` |
| `vec_chunks` | sqlite-vec virtual | `embedding float[N] distance_metric=cosine`, plus `file_id`, `chunk_sequence`, `content_hash`, `start_line`, `end_line`, `content` |
| `chunks_fts` | FTS5 virtual | `tokenize='porter unicode61'`, shares rowids with `vec_chunks` |

Pragmas: `journal_mode=WAL`, `foreign_keys=ON`.

### Model change detection

`_prepare_db()` checks the stored `embedding_model` and `distance_metric` against the current config. On mismatch, it drops `vec_chunks` and `chunks_fts`, resets dimension metadata, and sets `content_hash = ''` on all files to force a full reindex.

## Query-time search

`search.py` provides three search modes.

### Modes

| Mode | Sources | Use case |
|------|---------|----------|
| `hybrid` | vec KNN + FTS5, fused with RRF | Default — best overall quality |
| `vec` | sqlite-vec cosine KNN only | Semantic similarity |
| `fts` | FTS5 BM25 only | Exact keyword matching |

Both vec and FTS oversample at `limit * 2` before fusion.

### FTS query sanitization

User queries are tokenized and each token is wrapped in double quotes to prevent FTS5 syntax injection: `"token1" "token2"`.

### Reciprocal Rank Fusion (RRF)

Combines ranked lists with `k=60`: `score(doc) = sum(1 / (k + rank))` across all lists.

Scores are normalized against the theoretical maximum `n_lists / (k + 1)` to produce values in `[0, 1]`.

### Windowed retrieval

When `window > 0`, results are expanded to include neighboring chunks:

1. Group results by `file_id`
2. Compute intervals `[seq - window, seq + window]` for each hit
3. Merge overlapping intervals
4. Fetch the full chunk range and concatenate content in sequence order

This provides surrounding context for matched chunks.

### Character budget

`_apply_char_cap()` greedily accumulates results in score order until the character budget would be exceeded.

## Query Rewriting and HyDE

`rewriter.py` generates expanded queries using a local model (`TARS_MODEL_RETRIEVAL`, default `gemma3:4b`).

### Query expansion

`expand_queries(query)` asks the retrieval model for 2-4 keyword-dense one-line rewrites. Always returns the original query as the first element, capped at `_MAX_REWRITES = 4` generated alternatives (5 total).

### HyDE (Hypothetical Document Embedding)

`generate_hyde(query)` asks the model to write 3-5 bullet points that a relevant document might contain. Gated by a minimum word count (`_MIN_HYDE_WORDS = 5`) — short queries skip HyDE.

The HyDE text is embedded without the instruction prefix (treated as a pseudo-document, not a query) and used as an additional vec search pass.

### Security

Both functions wrap the user query in `<untrusted-user-query>` tags with closing-tag escape (`</` → `&lt;/`) to prevent tag breakout in the prompt template.

## Expanded search

`search_expanded()` combines rewriting and HyDE with standard search:

1. Generate keyword rewrites via `expand_queries()`
2. Generate HyDE text via `generate_hyde()`
3. Run each rewrite through both vec and FTS (producing multiple ranked lists)
4. Run the HyDE embedding through vec (no instruction prefix)
5. Fuse all ranked lists in a single RRF pass

This surfaces results that a single-query search might miss, especially for conversational or vague queries.

## Two-pass context packing

`core.py` `_search_relevant_context()` assembles search context for the system prompt on the first message of a conversation.

### Constants

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `_MAX_SEARCH_CONTEXT_TOKENS` | 3000 | Total token budget |
| `_TOP_N_CANDIDATES` | 20 | Initial search limit |
| `_AUTO_SEARCH_MIN_SCORE` | 0.25 | Minimum score for baseline results |
| `_EXPANSION_SCORE_THRESHOLD` | 0.30 | Below this top score, try expanded search |
| `_EXPAND_WINDOW` | 1 | Neighboring chunks to include |
| `_ANCHOR_BUDGET_RATIO_MIN` | 0.3 | Min % of budget for anchors |
| `_ANCHOR_BUDGET_RATIO_MAX` | 0.7 | Max % of budget for anchors |

### Algorithm

**Step 0 — Baseline search**: run standard `search()` with `limit=20, min_score=0.25, window=0`.

**Step 0b — Expansion fallback**: if the baseline is weak (empty or top score < 0.30), try `search_expanded()`. If it surfaces new chunk IDs not in the baseline, replace the anchor set.

**Step 1 — Pass 1 (anchor packing)**: dedupe to best-per-file for breadth. Compute an adaptive anchor budget based on score distribution:

- `dominance = (top_score - second_score) / top_score`
- `ratio = 0.7 - dominance * 0.4` (clamped to 0.3-0.7)
- Focused queries (high dominance) get a lower anchor ratio, reserving more budget for depth
- Broad queries (low dominance) get a higher anchor ratio for breadth

Greedily pack results in score order under the anchor budget.

**Step 2 — Pass 2 (windowed expansion)**: expand packed anchors with `window=1` to include neighboring chunks. For each anchor, include the expanded version only if the additional cost fits within the remaining budget.

The result is injected into the system prompt as `<memory-search-context>`.
