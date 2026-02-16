# Open Dialogue with AI

A Streamlit app for 3-way dialogue between a human moderator, an instructor, and two AI agents. Each agent uses a separate OpenAI session. Agents can use Tavily for web search when configured.

## Setup

Uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
cd /Users/elip/ai/open-dialogue
uv sync
```

Create a `.env` file with:

```
OPENAI_API_KEY=sk-...
```

Optional (for agent web search):

```
TAVILY_API_KEY=tvly-...
```

**Password protection (when not running locally):** If you set `APP_PASSWORD` in `.env`, the app shows a password screen before the main UI—unless `RUNNING_LOCALLY=1` is set (e.g. for local dev). So locally you can set `RUNNING_LOCALLY=1` to skip the gate; in production set only `APP_PASSWORD`.

## Run

```bash
uv run streamlit run app.py
```

Optional: custom port and bind address:

```bash
uv run streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

## App layout

- **Left (~60%)**
  - **Send as** — Radio: **Moderator** (default) or **Instructor**. Type in the chat input and press **Enter** to send (no Send button).
  - **Instructor** — Messages that instruct the agents what to do and when to respond.
  - **Moderator** — Messages that participate in the 3-way conversation with the two agents.
  - **Agent 1 & 2** — Collapsible role text (edit `AGENT_1_ROLE`, `AGENT_2_ROLE` and names `AGENT_1_NAME`, `AGENT_2_NAME` in `app.py`). **Respond** button to generate that agent’s reply.
- **Right (~40%)** — **Full dialogue**: all messages in order, with timestamps. Newest first by default; **Reverse order** toggles to chronological.

## @mentions

In a message, mention an agent by the first letter of their name with `@` to trigger an automatic response:

- `@g` or `@ g` → first agent (e.g. Gosha)
- `@j` or `@ j` → second agent (e.g. Joshi)

Matching is case-insensitive. Optional space after `@` is allowed.

## Technical notes

- Dialogue is sent to OpenAI in **chronological order**; the “newest first” view is display-only.
- Each agent is called with its own role and the full dialogue (with speaker labels). No shared session state between agents.
- With `TAVILY_API_KEY` set, agents can call a `web_search` tool during response generation.
