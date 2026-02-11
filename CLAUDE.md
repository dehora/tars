# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Tars is a Python project for AI assistant experiments. Uses uv for package management with Python 3.14.

## Commands

- **Run**: `uv run main.py`
- **Add dependency**: `uv add <package>`
- **Sync environment**: `uv sync`

## Development

### system design idea

```                     
                               + ---> [calls other apis via mcp/tools/skils]
                             /
[whatsapp/web/cli] --> [AI] +
                           / \
        [replies] <----- +    + ---- [access to mcp/tools/skils]
                              |
                              + ---- [state: assistant memory]  
```

### Technology ideas

#### AIs:

- ollama and openai/claude direct

#### Frontends:

- for cli: `uv tool install nanobot-ai`
- for whatspp: WhiskeySockets/Baileys using a group chat
- for web: node/web

#### State and Memory:

- flat text or .md file, can manage it as a gist.

#### Deploy targets:

- Local Macbook for development

- RPi using git checkout / uv tool install for the home