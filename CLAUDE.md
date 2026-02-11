# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Tars is a Python project for AI assistant experiments. Uses uv for package management with Python 3.14.

## Commands

- **Run**: `uv run main.py`
- **Add dependency**: `uv add <package>`
- **Sync environment**: `uv sync`

## Development

### tars assistant ideas

- an agent frontend/porcelain called 'tars'
- tars has cli, webpage and whatsapp channels
- tars also has state somewhere on the path i can ask it to store
- messages to tars routes to an AI — configurable
- AI works with tools/task/mcp etc in the background

A few integration ideas:

  1. route reminders and todos from tars to todoist: eg "add eggs to groceries, remind me to go to shop at 3pm"1.
  2. Route regular chat to tars and use the AI to handle them
  3. Have tars remember things about me, eg "my dog's name is Perry" that persist across sessions and channels


### System design idea

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