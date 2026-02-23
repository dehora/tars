# AGENTS.md

This file provides guidance to coding agents.

## Commands

- always use uv for running and managing python code.

- **Run**: `uv run main.py`
- **Add dependency**: `uv add <package>`
- **Sync environment**: `uv sync`

## Coding guidelines

- Never use `or` for numeric defaults — `0` and `0.0` are falsy. Use explicit `None` checks instead.

- Never assume parallel arrays from external APIs are equal length — use `min(len(...))` across all arrays.

- Wrap untrusted user data in tagged blocks with a preface when injecting into prompts — never concatenate raw content into system prompts.

- Use specific patterns for placeholder comments (e.g. `<!-- tars:memory -->`) — broad regex like `<!--.*?-->` will eat legitimate comments.

- Always use `encoding="utf-8", errors="replace"` on file I/O for user-managed files — memory files live in an obsidian vault and can be edited externally.

## Git Commits

Use present tense verbs in commit messages (e.g., "Adds emacs", "Ignores license", "Renames gitconfig template").

Add this by default to commits that include Codex contributions:
`Co-Authored-By: OpenAI Codex <codex@openai.com>`
