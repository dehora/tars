"""Post-conversation fact extraction for daily memory buffer."""

import json
import os
import re

from tars.core import chat
from tars.sessions import _escape_prompt_text

_MAX_FACTS = 5
_MIN_USER_MESSAGES = 3

_EXTRACTION_PROMPT = """\
Extract concrete, reusable facts from this conversation that would be valuable \
to remember in future sessions. Return a JSON array of short fact strings.

Include:
- User preferences and personal details
- Project names, tools, technologies mentioned
- Corrections or clarifications the user made
- Decisions reached during the conversation
- Important context that saves re-explaining

Exclude:
- Pleasantries and greetings
- Obvious context already visible in the conversation
- Speculation or uncertain information
- Tool call payloads and raw data
- Meta-observations about the conversation itself

Return ONLY a JSON array of strings, e.g. ["fact one", "fact two"]. \
Return [] if nothing worth remembering."""


def _auto_extract_enabled() -> bool:
    return os.environ.get("TARS_AUTO_EXTRACT", "true").lower() not in ("false", "0", "no")


def _parse_json_list(raw: str) -> list[str]:
    cleaned = raw.strip()
    # Strip markdown code fences
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        cleaned = cleaned.rsplit("```", 1)[0].strip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            return [str(item) for item in parsed if item]
        return []
    except (json.JSONDecodeError, ValueError):
        pass
    # Fallback: find [...] anywhere in response
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, list):
                return [str(item) for item in parsed if item]
        except (json.JSONDecodeError, ValueError):
            pass
    return []


def _format_messages(messages: list[dict]) -> str:
    lines = []
    for m in messages:
        role = m.get("role", "")
        if role in ("system", "tool"):
            continue
        content = m.get("content")
        if content is None:
            continue
        if not isinstance(content, str):
            continue
        lines.append(f"{role}: {_escape_prompt_text(content)}")
    return "\n".join(lines)


def extract_facts(messages: list[dict], provider: str, model: str) -> list[str]:
    if not _auto_extract_enabled():
        return []
    user_count = sum(1 for m in messages if m.get("role") == "user")
    if user_count < _MIN_USER_MESSAGES:
        return []
    if not messages:
        return []
    try:
        convo_text = _format_messages(messages)
        prompt = (
            f"{_EXTRACTION_PROMPT}\n\n"
            f"<conversation>\n{convo_text}\n</conversation>"
        )
        raw = chat([{"role": "user", "content": prompt}], provider, model, use_tools=False)
        facts = _parse_json_list(raw)
        return facts[:_MAX_FACTS]
    except Exception:
        return []
