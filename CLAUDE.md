# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Tars is a Python project for AI assistant experiments. Uses uv for package management with Python 3.14.

## Commands

- **Run**: `uv run main.py`
- **Add dependency**: `uv add <package>`
- **Sync environment**: `uv sync`

## Coding guidelines

- Never use `or` for numeric defaults — `0` and `0.0` are falsy. Use explicit `None` checks instead.
- Never assume parallel arrays from external APIs are equal length — use `min(len(...))` across all arrays.
- Wrap untrusted user data in tagged blocks with a preface when injecting into prompts — never concatenate raw content into system prompts.
- Use specific patterns for placeholder comments (e.g. `<!-- tars:memory -->`) — broad regex like `<!--.*?-->` will eat legitimate comments.

# Development

## tars assistant ideas

- an agent frontend/porcelain called 'tars'
- tars has cli, webpage and whatsapp channels
- tars also has state somewhere on the path i can ask it to store
- messages to tars routes to an AI — configurable
- AI works with tools/task/mcp etc in the background

A few integration ideas:

  1. route reminders and todos from tars to todoist: eg "add eggs to groceries, remind me to go to shop at 3pm"1.
  2. Route regular chat to tars and use the AI to handle them
  3. Have tars remember things about me, eg "my dog's name is Perry" that persist across sessions and channels


## System design idea

```                     
                               + ---> [calls other apis via mcp/tools/skils]
                             /
[whatsapp/web/cli] --> [AI] +
                           / \
        [replies] <----- +    + ---- [access to mcp/tools/skils]
                              |
                              + ---- [state: assistant memory]  
```

## Technology ideas

Here's what I see:

### State and Memory

#### Three considerations
  - what to remember?
  - where does it go?
  - when to remember?

#### Four kinds of memory
  - Semantic. Preferences, facts and stable things that are broadly true over time, especially about the user.
  - Procedural. Operational information, rules, idioms and and approaches.
  - Episodic:Context. Recent context based on time (today, yesterday), these are rollups.
  - Episodic:Sessions. Captures conversations in session, without the technical details/commands. Folders of timestamped logs.  

- Can use markdown files. Don't need a vectordb/database for storage
  - Multiple files in a hierarchy not one big file (cf. context windows).
  - A Memory.md file. Mostly semantic memory and lists of other files, just a few hundred lines. 
  - The agent can load the recent Episodic:Context files.
  - The agent can load the recent Episodic:Sessions that are relevant to this session.

- Could host in an obsidian vault called tars. This creates a dep on Obsidian but that's ok since it'll be just used as a holder and for syncing, since Obsidian works on markdown.

#### Memory handling
  - Memory.md is always loaded and given to tars. The agent always has it present. To manage the context window is one reason to keep this light.
  - The Episodic:Contexts, today.md, yesterday.md, etc are loaded in by tars to remind itself of recent activity.
  - The Episodic:Sessions:
    - Snapshot count based compaction near the context limit to move memory information from the session and working desk to a session log, saving information before it gets flushed away. 
    - Snapshot at the end of a session before closing
  - Remember at the user's request ('remember this'). The agent can decide where to remember.

#### Session logging TODOs
  - Compaction summaries should be incremental — pass the previous summary to `_summarize_session` so the model only covers new material since the last compaction, avoiding redundancy across sections


#### Semantic Memory Guidelines

Suggested by Claude:

```
# Core
- Name: John
- Location: Dublin, IE
- Pets: Luna (cat), Max (dog)

# Work
- Role: Senior engineer at Acme
- Stack: Python, TypeScript, AWS

# Preferences
- Editor: Neovim
- Style: concise responses, no fluff
- Task mgmt: Todoist, Obsidian for notes

# Active projects
- Learning Claude CLI/agents
- Home automation (Home Assistant)
```

**Memory guidelines**

- Keep memory.md under 100 lines / ~1000 tokens
- Use terse bullet points, not prose
- Only store durable facts, not transient state
- When adding: check for duplicates/conflicts first
- When removing: confirm with user before deleting

Rough targets for a persistent memory file:

```
| Approach | Size | Tokens (~) |
|----------|------|------------|
| Minimal | 20-50 lines | 200-500 |
| Comfortable | 50-150 lines | 500-1500 |
| Upper bound | 200-300 lines | 2000-3000 |
```

**Claude's Recommendation**: 

Aim for ~50-100 lines / ~500-1000 tokens. This gives you room for meaningful context without meaningfully impacting the available context for actual work. Claude's context window is large (200k tokens for Sonnet/Opus), so even 1-2k tokens is <1% — but memory gets loaded every turn, so smaller is better for:

- Faster processing
- Lower cost (if API-based)
- Less noise competing with the actual task

Optional: add a brief schema hint at the top

```
<!-- Format: flat bullets under category headers, ~100 lines max -->
```


#### Searching
  - aside from plain text storage, we can let the agent search memory by giving it a tool like [sqlite vec](https://github.com/asg017/sqlite-vec)

  - The search model would be
    - hybrid: keyword search and semantic search
    - keyword: grep plus bm25s would be enough for search and ranking results. 
    - semantic: based on embeddings, using something like open ai's [textembedding 3.small](https://developers.openai.com/api/docs/models/text-embedding-3-small) or a [local ollama embdedding model](https://ollama.com/search?c=embedding) to create embeddings of the memory and the query. Configurable with the default being ollama  (cost and latency management).
    - TODO: select some embedding/ranking models, Qwen3-Reranker, Embeddinggemma, etc
    - Want both because semantic search isn't great for exact/literal match.
    - Can use a fusion technique to combine them:
      - weighted score, eg keyword 30% and vector 70%
      - reciprocal ranked fusion (RRF) that combines the ranks. Preference is RRF.
    - Aways search first before calling a model (cost and latency management)

**Search/Query Flow**

Flow:

```
- State the Query
  - Embeding Model Expansion. 
      - run a BM25 for fulltext grepping
      - run a vector embedding search for semantics
  - Combine using RRF
  - Return top N 
```

**Score Ranking**

| Score | Rank |
|-------|---------|
| 0.75 - 1.0 | Relevant |
| 0.5 - 0.75 | Kinda relevant |
| 0.25 - 0.5 | Maybe relevant |
| 0.0 - 0.25 | Not relevant |


#### Search Embedding Schema
  - a files table and a chunks table and a collections table. One to many from collections to files to chunks.
  - collections table to name the file set:
    - columns could be: 
      - id
      - name
    we'll have just one for now, 'tars_memory' (but see 'User search, not just agents' below for why)
  - files table to point at the memory files
    - columns could be: 
        - collection_id,
        - path,
        - title, 
        - media_type, 
        - memory_type, 
        - content_hash, 
        - mtime, 
        - size, 
        etc. to enable incremental syncing.
  - Chunks table to hold the files' embeddings
    - columns could be: 
      - file_id, 
      - chunk_model,
      - chunk_sequence, 
      - content_hash, # similar to git
      - start_line, 
      - end_line, 
      - embedding, 
      - updated_at  

#### Chunking files at meaningful boundaries

Instead of chunking at hard token boundaries, we can provide a baseline score from 0-100 to steer the chunker to decompose at 'meaningful' markdown boundaries to preserve structure and semantics (also to reduce chunk scanning for vectors). For example (we can make the baseline scores configurable later):

| Pattern | Score | Description |
|---------|-------|-------------|
| `# Heading` | 100 | H1 section |
| `## Heading` | 90 | H2 subsection |
| `### Heading` | 80 | H3 |
| `#### Heading` | 70 | H4 |
| `##### Heading` | 60 | H5 |
| `###### Heading` | 50 | H6 |
| `---` / `***` | 70 | Horizontal rule |
| ` ``` ` | 80 | Code block boundary |
| Blank line | 10 | Paragraph boundary |
| `- item` / `1. item` | 5 | List item |

So: 

- Scan whole document to get candidate chunk points
- When approaching the token target, search ahead before the cutoff
- Score each candidate chunk point: `finalScore = baseline_score * (1 - (distance/window)^2)`
- Chunk at the winning (highest score) point
- Have a configurable overlap between chunks, say 10% default

For inline images, we'll skip over them for now and come back to it, if we see the agent trying to save images to memory (obsidian enables this). It might need another table for media.

**Embedding Flow**

```
- Get Document
  - Get its details
  - Chunk Doc (~800 tokens or so). 
      - hash, sequence, line positions, etc
  - Embed Chunks
    - run an embedding using an Ollama
  - Save Chunks
     - write into sqlite 
```



#### User search, not just agents

This generalises so we can add commands to let the user search memory as well or add it as a skill to claude. Say:

```
/search: keyword and semantic search with fusion
/sgrep: full text search using BM25
/svec: vector search using embeddings
```

We can also add other content for searching not just memory.  But that's for later. Just agent support for now.

### AIs:

- ollama and openai/claude direct

### Frontends:

- for cli: `uv tool install tars`
- for whatspp: WhiskeySockets/Baileys using a group chat
- for web: node/web

### Deploy targets:

- Local Macbook for development

- RPi using git checkout / uv tool install for the home