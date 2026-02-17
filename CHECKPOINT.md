# Checkpoint

**Date:** 2026-02-17

## Summary

Open Dialogue with AI — Streamlit app for 3-way dialogue between Instructor, Moderator, and two AI agents (separate OpenAI sessions per agent). Conversations and messages are persisted in Supabase; each conversation has a UUID in the URL.

## Features

- **Moderator name gate:** Before any conversation, the moderator must enter their name. All fields are hidden until a non-empty name is submitted. Page title becomes **"&lt;name&gt;'s Open Dialogue with AI"** once set. Conversation history and export use the **original author name** for each message (from DB when loaded); new messages use current user name.
- **Send as:** Moderator (default) or Instructor. Single chat input; press Enter to send. Focus moves to the chat input when you change Moderator/Instructor.
- **Two agents:** Configurable names and roles in `app.py` (e.g. Gosha, Joshi). Agent name is fixed; only role text is editable. Collapsible role expander and **Respond** button per agent. OpenAI prompts enforce fixed identity: each agent must speak only for themselves, never as Instructor/Moderator/other agent.
- **Agent intro:** Each agent states name and role on first reply and again after its role is updated.
- **Thinking spinner:** One place for both flows — **Respond** or @mention; "agent thinking…" appears in the same full-width row below the agent rows.
- **@mentions:** Agents respond only when @-mentioned or when their **Respond** button is pressed. Mention pattern: `@` (optional space) then first letter or any prefix of name (e.g. `@g`, `@ gosha`). Stored text is expanded to full agent names in history and for OpenAI.
- **Layout:** Sidebar (wider when expanded only, 360–440px; full collapse when closed) with conversations list + 50% left (controls + spinner), 50% right (conversation history). **Reverse order** button on the same line as "Conversation history". **Export to doc** right-aligned below history.
- **Conversation history:** Right panel, newest first by default; **Reverse order** toggles; timestamps; each message shows the original poster’s name (DB author or current user for new messages). **Export to doc** downloads full conversation as .docx (via `doc_export.py`), using original author names.
- **Supabase:** `SUPABASE_URL` and `SUPABASE_KEY` (or anon JWT) in `.env`. After moderator name, a new conversation is created (UUID) and set in the URL. `od_conversations` has id + created_at (no moderator_name). `od_messages` stores author display name in `role` (human or agent name). Sidebar lists previous conversations (by created_at); current conversation highlighted in red; each row has an **×** button to delete that conversation (removes from DB and panel; if current, app switches to a new conversation). **New conversation** starts a fresh one. Opening a link with `?conversation_id=<uuid>` prompts for name then loads that conversation’s entire history. Tables: run `supabase_migration.sql` in Supabase SQL Editor (includes DROP then CREATE).
- **Password (Streamlit Cloud):** When `APP_PASSWORD` is set and running on Streamlit Cloud, a password form is shown first.
- **Tavily:** Optional `TAVILY_API_KEY` in `.env` for agent web search. Status line under title shows enabled / disabled / error.

## Files

- `app.py` — main Streamlit app
- `supabase_client.py` — Supabase client and helpers (create_conversation, delete_conversation, list_conversations, load_messages, persist_message)
- `doc_export.py` — export dialogue to Word (.docx)
- `supabase_migration.sql` — DROP + CREATE for `od_conversations` and `od_messages` (run in Supabase SQL Editor)
- `pyproject.toml` — dependencies (streamlit, openai, python-dotenv, python-docx, supabase, tavily-python)
- `README.md` — setup and usage
- `SUPABASE_PLAN.md` — Supabase integration plan and schema notes
