# Checkpoint

**Date:** 2026-02-16

## Summary

Open Dialogue with AI — Streamlit app for 3-way dialogue between Instructor, Moderator, and two AI agents (separate OpenAI sessions per agent).

## Features

- **Send as:** Moderator (default) or Instructor. Single chat input; press Enter to send. Focus moves to the chat input when you change Moderator/Instructor.
- **Two agents:** Configurable names and roles in `app.py` (e.g. Gosha, Joshi). Agent name is fixed; only role text is editable. Collapsible role expander and **Respond** button per agent (each button aligned with its agent row). Same width for chat input and both role expanders.
- **Agent intro:** Each agent states name and role on first reply and again after its role is updated.
- **Thinking spinner:** One place for both flows — whether you press **Respond** or use an @mention, “agent &lt;name&gt; thinking…” always appears in the same full-width row below the agent rows (UI stays visible).
- **@mentions:** `@g` / `@j` (first letter of agent name, case-insensitive, optional space after `@`) trigger that agent’s response; uses the same thinking spinner as the Respond button. Stored text is expanded to real agent names (e.g. “Gosha”) in conversation history and in messages sent to OpenAI.
- **Layout:** 50% left (controls + spinner), 50% right (conversation history).
- **Conversation history:** Right panel, newest first by default; **Reverse order** toggles to chronological; timestamps on messages. **Export to doc** (below Reverse order) downloads the full conversation in chronological order as a Word document (.docx).
- **Password (Streamlit Cloud):** When `APP_PASSWORD` is set and running on Streamlit Cloud, a password form is shown; submit with Enter (no visible Unlock button).
- **Tavily:** Optional `TAVILY_API_KEY` in `.env` for agent web search.

## Files

- `app.py` — main Streamlit app
- `pyproject.toml` — uv project and dependencies (includes `python-docx` for Word export)
- `uv.lock` — locked dependencies
- `README.md` — setup and usage
