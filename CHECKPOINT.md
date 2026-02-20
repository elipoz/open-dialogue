# Checkpoint

**Date:** 2026-02-19

## Summary

Open Dialogue with AI — Streamlit app for N-way dialogue between multiple Moderators, and two AI agents (separate OpenAI sessions per agent). Conversations and messages are persisted in Supabase; each conversation has a UUID in the URL.

**Live app:** https://open-dialogue-with-ai.streamlit.app/

## Features

- **Login gate:** Single combined screen: **"To start a dialogue - state your name"**; name field (max 15 chars), then app password (only when not running locally, i.e. Streamlit Cloud with `APP_PASSWORD`), then **Submit**. All fields hidden until name (and password if required) are accepted. Page title becomes **"&lt;name&gt;'s Open Dialogue with AI"** once set. Conversation history and export use the **original author name** for each message (from DB when loaded); new messages use current user name.
- **Send as:** Moderator (default) or Instructor. Single chat input; press Enter to send. Focus moves to the chat input when you change Moderator/Instructor.
- **Two agents:** Configurable names and roles in `app.py` (e.g. Gosha, Joshi). Agent name is fixed; only role text is editable. Collapsible role expander and **Respond** button per agent. OpenAI prompts enforce fixed identity: each agent must speak only for themselves, never as Instructor/Moderator/other agent.
- **Agent intro:** Each agent states name and role on first reply and again after its role is updated.
- **Thinking spinner:** One place for both flows — **Respond** or @mention; "agent thinking…" appears in the same full-width row below the agent rows.
- **@mentions:** Agents respond only when @-mentioned or when their **Respond** button is pressed. Mention pattern: `@` (optional space) then first letter or any prefix of name (e.g. `@g`, `@ gosha`). Stored text is expanded to full agent names in history and for OpenAI.
- **Layout:** Sidebar (wider when expanded only, 360–440px; full collapse when closed) with conversations list + 50% left (controls + spinner), 50% right (conversation history). **Reverse order** button on the same line as "Conversation history". **Export to doc** right-aligned below history.
- **Conversation history:** Right panel, newest first by default; **Reverse order** toggles; timestamps; each message shows the original poster’s name (DB author or current user for new messages). **Export to doc** downloads full conversation as .docx (via `doc_export.py`), using original author names.
- **Supabase:** `SUPABASE_URL` and `SUPABASE_KEY` (or anon JWT) in `.env`. New conversation is created **only** when the user clicks **New conversation** (no auto-creation on load). `od_conversations` has id + created_at (no moderator_name). `od_messages` stores author display name in `role` (human or agent name). Sidebar lists previous conversations (by created_at); current conversation highlighted in red; each row has an **×** button to delete (removes from DB and panel; if current, app switches to a new conversation). Opening a link with `?conversation_id=<uuid>` prompts for name then loads that conversation’s entire history. Tables: run `supabase_migration.sql` in Supabase SQL Editor (includes DROP then CREATE).
- **Multi-user sync:** Multiple users share the same conversation. **2s fragment** (right column) polls and reloads dialogue from DB so history stays in sync; **10s fragment** (sidebar) refreshes the conversation list. Conversation/query state preserved when URL param is missing (no accidental reset); sidebar only clears current when list is non-empty and current id missing. After every `persist_message` (human message, agent reply, role update) we call `_reload_dialogue_from_db()` so the UI always shows the canonical DB state. New conversation is prepended to the list when created. `conversation_exists()`; opening or polling a deleted conversation clears state and reloads. **Dedupe** by conversation id; if current conversation is missing from the list (deleted by another user), selection is cleared. Session-state reset in `_clear_conversation_state()`. Agent intro flags (`agent1_needs_intro`, `agent2_needs_intro`) are synced from dialogue when loading from DB (`_sync_agent_intro_state_from_dialogue()`) so agents don’t re-introduce if they already have messages in the conversation.
- **Timestamps in PST:** All UI timestamps in **America/Los_Angeles (PST)**. `_format_in_pst()` handles both datetime and ISO strings from Supabase; new conversation label uses `_now_pst()`.
- **OpenAI context:** Conversation sent to OpenAI as **one [user] message**: full transcript with **"At &lt;timestamp&gt; &lt;role&gt; said: &lt;message&gt;"** (chronological, including this agent's past replies) then **[Reply now only as &lt;name&gt;.]**. No per-turn user/assistant; who said what is clear from the transcript. Timestamp is message `created_at` from DB; role uses actual names. Agent system prompt instructs: reply with message content only—do not echo "At … said:" or your name as a label.
- **Agent replies:** If the model echoes "At &lt;timestamp&gt; &lt;name&gt; said:" we strip it (`_strip_at_timestamp_said_prefix`); leading "Name: " is also stripped (`_strip_agent_name_prefix`) so the UI label is not duplicated.
- **Refactoring:** Agent thinking flow in `_run_agent_thinking_if_set(agent_key, agent_name)`; agent role row (expander + form + Respond button) in `_render_agent_role_row(agent_key, agent_name, agent_role, role_col, button_col)`.
- **Password (Streamlit Cloud):** When `APP_PASSWORD` is set and running on Streamlit Cloud, the login screen includes name + password + Submit (password omitted when running locally).
- **Tavily:** Optional `TAVILY_API_KEY` in `.env` for agent web search. Status line under title shows enabled / disabled / error (only after moderator name is set; not on the name-entry page).
- **Model:** OpenAI model **gpt-5-mini** (no temperature param; model uses default). Agent prompt instructs taking a different perspective from the other agent when relevant.
- **Request / response log:** In sidebar at bottom; latest request and response after each Respond or @mention. Two collapsible subsections (Request, Response) inside the main expander; widget keys include log ts+agent so content refreshes when a new agent reply is logged (avoids stale Streamlit session state). Log written in `_run_agent_thinking_if_set` after `call_openai_for_agent` returns.
- **Timestamps in DB:** Messages stored with UTC `created_at` so all users have consistent timestamps regardless of app host.

## Files

- `app.py` — main Streamlit app
- `supabase_client.py` — Supabase client and helpers (conversation_exists, create_conversation, delete_conversation, list_conversations, load_messages, persist_message)
- `doc_export.py` — export dialogue to Word (.docx)
- `supabase_migration.sql` — DROP + CREATE for `od_conversations` and `od_messages` (run in Supabase SQL Editor)
- `pyproject.toml` — dependencies (streamlit, openai, python-dotenv, python-docx, supabase, tavily-python)
- `README.md` — setup and usage
- `SUPABASE_PLAN.md` — Supabase integration plan and schema notes
