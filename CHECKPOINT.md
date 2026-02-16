# Checkpoint

**Commit:** `1a0c877` (1a0c877da4ac02f9dfd8ef758117fdfc3218d776)

**Date:** 2025-02-16

## Summary

Open Dialogue with AI — Streamlit app for 3-way dialogue between Instructor, Moderator, and two AI agents (separate OpenAI sessions).

## Features

- **Send as:** Moderator (default) or Instructor. Single chat input; press Enter to send.
- **Two agents:** Configurable names and roles in `app.py` (e.g. Gosha, Joshi). Collapsible role text; **Respond** button per agent.
- **@mentions:** `@g` / `@j` (first letter of agent name, case-insensitive, optional space after `@`) trigger that agent’s response automatically.
- **Conversation history:** Right panel shows all messages with timestamps. Newest first by default; **Reverse order** toggles to chronological.
- **Tavily:** Optional `TAVILY_API_KEY` in `.env` for agent web search.
- Dialogue is sent to OpenAI in chronological order; display order is independent.

## Files

- `app.py` — main Streamlit app
- `pyproject.toml` — uv project and dependencies
- `uv.lock` — locked dependencies
- `README.md` — setup and usage
