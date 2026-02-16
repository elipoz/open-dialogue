"""
Open Dialogue with AI — 3-way discussion between a human moderator and two AI agents.

The human instructor instructs the agents on what to do and when to respond.
The human moderator chats with both agents in a shared dialogue.
Agents can use Tavily to search the web when they need up-to-date information.
"""

import json
import os
import re
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def _is_streamlit_cloud() -> bool:
    """True when running on Streamlit Cloud; skip password when running locally (not cloud)."""
    return os.path.exists("/mount/src") or os.environ.get("STREAMLIT_SHARING_MODE") == "true"


def _require_password() -> bool:
    """True when on Streamlit Cloud and APP_PASSWORD is set (skip when running locally)."""
    return _is_streamlit_cloud() and bool(os.environ.get("APP_PASSWORD"))


# Optional Tavily client for web search (requires TAVILY_API_KEY in .env)
_tavily_client = None

def _get_tavily_client():
    global _tavily_client
    if _tavily_client is None and os.environ.get("TAVILY_API_KEY"):
        try:
            from tavily import TavilyClient
            _tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
        except Exception:
            pass
    return _tavily_client

# OpenAI tool definition for web search
SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for current information. Use this when you need facts, recent events, or anything you are unsure about.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query to run"},
            },
            "required": ["query"],
        },
    },
}

# -----------------------------------------------------------------------------
# Agent names and roles (populate manually)
# -----------------------------------------------------------------------------
AGENT_1_NAME = "Gosha"  # Display name (fixed); role text is editable in UI
AGENT_1_ROLE = """I am a professional therapist specializing in open dialogue."""
AGENT_1_ROLE_AND_NAME = f"You are {AGENT_1_NAME}. {AGENT_1_ROLE}"

AGENT_2_NAME = "Joshi"
AGENT_2_ROLE = """I am a philosopher specializing on Dostoevsky and Bakhtin."""
AGENT_2_ROLE_AND_NAME = f"You are {AGENT_2_NAME}. {AGENT_2_ROLE}"


def init_session_state():
    if "dialogue" not in st.session_state:
        st.session_state.dialogue = []  # list of (party, content, timestamp)
    if "send_as_radio" not in st.session_state:
        st.session_state.send_as_radio = "Moderator"
    if "dialogue_newest_first" not in st.session_state:
        st.session_state.dialogue_newest_first = True  # True = newest first, False = chronological
    # Store only role text (no name); name stays fixed from constants
    if "agent1_role" not in st.session_state:
        st.session_state.agent1_role = AGENT_1_ROLE
    if "agent2_role" not in st.session_state:
        st.session_state.agent2_role = AGENT_2_ROLE
    if "agent1_needs_intro" not in st.session_state:
        st.session_state.agent1_needs_intro = False  # True after role update until agent posts
    if "agent2_needs_intro" not in st.session_state:
        st.session_state.agent2_needs_intro = False


def _get_agent_role(agent_key: str) -> str:
    """Return full system prompt for agent: always 'You are {name}. {role}' so name stays fixed."""
    if agent_key == "agent1":
        role_text = st.session_state.get("agent1_role", AGENT_1_ROLE)
        return f"You are {AGENT_1_NAME}. {role_text}"
    role_text = st.session_state.get("agent2_role", AGENT_2_ROLE)
    return f"You are {AGENT_2_NAME}. {role_text}"


def _get_agent_role_text_only(agent_key: str) -> str:
    """Return only the role text for the agent (no name), for use in first-message intro."""
    if agent_key == "agent1":
        return st.session_state.get("agent1_role", AGENT_1_ROLE)
    return st.session_state.get("agent2_role", AGENT_2_ROLE)


def _speaker_label(party: str) -> str:
    """Return display label for a party in the dialogue."""
    labels = {"instructor": "Instructor", "moderator": "Moderator", "agent1": AGENT_1_NAME, "agent2": AGENT_2_NAME}
    return labels.get(party, party)


def _mentioned_agents(text: str) -> list[str]:
    """Return list of agent keys triggered by @first_letter (e.g. @g or @ g -> Gosha). Case-insensitive; allows optional space after @."""
    mentioned = []
    lower = text.lower()
    if AGENT_1_NAME:
        c1 = AGENT_1_NAME[0].lower()
        if re.search(rf"@\s*{re.escape(c1)}", lower):
            mentioned.append("agent1")
    if AGENT_2_NAME:
        c2 = AGENT_2_NAME[0].lower()
        if re.search(rf"@\s*{re.escape(c2)}", lower):
            mentioned.append("agent2")
    return mentioned


def _agent_has_spoken(speaker: str) -> bool:
    """True if this agent has already posted at least one message in the dialogue."""
    return any(entry[0] == speaker for entry in st.session_state.dialogue)


def _render_agent_respond_button(agent_key: str, name: str, spin_col, btn_col) -> None:
    """Render reserved spinner slot and Respond button; on click show spinner in slot and call API. Same implementation for both agents."""
    with spin_col:
        pass  # Reserved for spinner when this agent is thinking
    with btn_col:
        if st.button("Respond", key=f"gen_{agent_key}"):
            with spin_col:
                with st.spinner(f"{name} thinking…"):
                    reply = call_openai_for_agent(_get_agent_role(agent_key), agent_key)
            st.session_state.dialogue.append((agent_key, reply, datetime.now()))
            if agent_key == "agent1":
                st.session_state.agent1_needs_intro = False
            elif agent_key == "agent2":
                st.session_state.agent2_needs_intro = False
            st.rerun()


def build_messages_for_agent(role_prompt: str, speaker: str, role_text_only: str | None = None) -> list:
    """Build OpenAI messages from the full dialogue in chronological order (oldest first). Each message is attributed so the agent has full context."""
    tools_instruction = ""
    if _get_tavily_client():
        tools_instruction = "\n\nYou have access to a web_search tool. Use it when you need to look up current information, facts, or events. After receiving search results, use them to inform your response."
    intro_required = (
        not _agent_has_spoken(speaker)
        or (speaker == "agent1" and st.session_state.get("agent1_needs_intro", False))
        or (speaker == "agent2" and st.session_state.get("agent2_needs_intro", False))
    )
    if intro_required:
        intro = (
            "\n\nYou must begin this message by clearly introducing yourself: state your name, then state your role. "
            "Your role is: " + (role_text_only or "as defined above") + " "
            "Include both your name and this role in your opening sentence or two, then respond to the discussion."
        )
        tools_instruction += intro
    messages = [{"role": "system", "content": role_prompt + tools_instruction}]
    # dialogue is always stored chronologically; send to API in that order (party, content, optional timestamp)
    for entry in st.session_state.dialogue:
        party, content = entry[0], entry[1]
        label = _speaker_label(party)
        attributed = f"{label}: {content}"
        if party == speaker:
            messages.append({"role": "assistant", "content": attributed})
        else:
            messages.append({"role": "user", "content": attributed})
    return messages


def _run_tavily_search(query: str) -> str:
    """Run a Tavily search and return a summary string for the model."""
    client = _get_tavily_client()
    if not client:
        return "Web search is not available (TAVILY_API_KEY not set)."
    try:
        response = client.search(query=query, max_results=5, search_depth="basic")
        results = response.get("results", [])
        if not results:
            return "No results found for that query."
        parts = []
        for i, r in enumerate(results[:5], 1):
            title = r.get("title", "")
            url = r.get("url", "")
            content = r.get("content", "")
            parts.append(f"[{i}] {title}\nURL: {url}\n{content}")
        return "\n\n".join(parts)
    except Exception as e:
        return f"Search failed: {e}"


def call_openai_for_agent(role_prompt: str, speaker: str) -> str:
    """Call OpenAI chat completion for the given agent (separate session per agent; no shared state). Agents can use web_search tool if Tavily is configured."""
    role_text_only = _get_agent_role_text_only(speaker)
    messages = build_messages_for_agent(role_prompt, speaker, role_text_only=role_text_only)
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))  # new client per call; each agent has its own independent request
    tools = [SEARCH_TOOL] if _get_tavily_client() else None
    max_tool_rounds = 5
    for _ in range(max_tool_rounds):
        kwargs = {"model": "gpt-4o-mini", "messages": messages}
        if tools:
            kwargs["tools"] = tools
        response = client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        msg = choice.message
        if not getattr(msg, "tool_calls", None):
            return (msg.content or "").strip()
        # Append assistant message with tool_calls
        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ],
        })
        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments) if tc.function.arguments else {}
            if name == "web_search":
                result = _run_tavily_search(args.get("query", ""))
            else:
                result = f"Unknown tool: {name}"
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
    # After tool rounds, get a final text response (no more tools)
    final = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    return (final.choices[0].message.content or "").strip()


def main():
    st.set_page_config(page_title="Open Dialogue with AI", page_icon="💬", layout="wide")

    # Password gate (only when APP_PASSWORD is set and not running locally)
    if _require_password():
        if "authenticated" not in st.session_state:
            st.session_state.authenticated = False
        if not st.session_state.authenticated:
            st.title("Open Dialogue with AI")
            st.caption("Enter the app password to continue.")
            pwd = st.text_input("Password", type="password", key="password_input", label_visibility="collapsed", placeholder="Password")
            if st.button("Unlock"):
                if pwd and pwd == os.environ.get("APP_PASSWORD"):
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Incorrect password.")
            st.stop()

    st.title("Open Dialogue with AI")
    st.caption("Human moderator + two AI agents in a 3-way conversation.")

    init_session_state()

    left_col, right_col = st.columns([6, 4])  # 60% left, 40% Conversation history

    with left_col:
        # Human message: choose role, type and press Enter (no Send button)
        st.radio(
            "Send as",
            options=["Moderator", "Instructor"],
            horizontal=True,
            key="send_as_radio",
            label_visibility="collapsed",
        )
        human_prompt = st.chat_input("Type a message and press Enter to send")

        # Agent rows first (fixed layout) so thinking spinner below doesn't shift them
        # [expander | spinner slot | button] — spinner slot reserves space so button never moves
        a1_col1, a1_spin, a1_col2 = st.columns([3, 1, 1])
        with a1_col1:
            with st.expander(f"{AGENT_1_NAME} role", expanded=False):
                with st.form("agent1_role_form"):
                    st.text_area("Role", value=st.session_state.agent1_role, height=120, key="agent1_role_input", label_visibility="collapsed")
                    if st.form_submit_button("Update role"):
                        new_role = st.session_state.get("agent1_role_input", st.session_state.agent1_role)
                        st.session_state.agent1_role = new_role
                        st.session_state.agent1_needs_intro = True  # introduce again on next response
                        st.session_state.dialogue.append(("instructor", f"Updated {AGENT_1_NAME}'s role:\n\n{new_role}", datetime.now()))
                        st.rerun()
        _render_agent_respond_button("agent1", AGENT_1_NAME, a1_spin, a1_col2)

        a2_col1, a2_spin, a2_col2 = st.columns([3, 1, 1])
        with a2_col1:
            with st.expander(f"{AGENT_2_NAME} role", expanded=False):
                with st.form("agent2_role_form"):
                    st.text_area("Role", value=st.session_state.agent2_role, height=120, key="agent2_role_input", label_visibility="collapsed")
                    if st.form_submit_button("Update role"):
                        new_role = st.session_state.get("agent2_role_input", st.session_state.agent2_role)
                        st.session_state.agent2_role = new_role
                        st.session_state.agent2_needs_intro = True  # introduce again on next response
                        st.session_state.dialogue.append(("instructor", f"Updated {AGENT_2_NAME}'s role:\n\n{new_role}", datetime.now()))
                        st.rerun()
        _render_agent_respond_button("agent2", AGENT_2_NAME, a2_spin, a2_col2)

        # @mention spinners: show below agent rows so layout above doesn't shift
        if human_prompt:
            role = "moderator" if st.session_state.get("send_as_radio") == "Moderator" else "instructor"
            text = human_prompt.strip()
            if text:
                st.session_state.dialogue.append((role, text, datetime.now()))
                for agent_key in _mentioned_agents(text):
                    _c1, _c2 = st.columns([4, 1])
                    with _c2:
                        with st.spinner(f"{_speaker_label(agent_key)} thinking…"):
                            reply = call_openai_for_agent(_get_agent_role(agent_key), agent_key)
                    st.session_state.dialogue.append((agent_key, reply, datetime.now()))
                    if agent_key == "agent1":
                        st.session_state.agent1_needs_intro = False
                    elif agent_key == "agent2":
                        st.session_state.agent2_needs_intro = False
                st.rerun()

    with right_col:
        st.subheader("Conversation history")
        SPEAKER_LABELS = {
            "instructor": "Instructor",
            "moderator": "Moderator",
            "agent1": AGENT_1_NAME,
            "agent2": AGENT_2_NAME,
        }
        if not st.session_state.dialogue:
            st.caption("No messages yet. Send an instructor or moderator message, or generate an agent response.")
        messages = reversed(st.session_state.dialogue) if st.session_state.dialogue_newest_first else st.session_state.dialogue
        for entry in messages:
            party, content = entry[0], entry[1]
            ts = entry[2] if len(entry) >= 3 else None
            label = SPEAKER_LABELS.get(party, party)
            is_human = party in ("instructor", "moderator")
            with st.chat_message("user" if is_human else "assistant"):
                st.markdown(f"**{label}:**")
                st.markdown(content)
                if ts:
                    st.caption(ts.strftime("%Y-%m-%d %H:%M"))
        if st.session_state.dialogue:
            if st.button("Reverse order", key="dialogue_order_btn"):
                st.session_state.dialogue_newest_first = not st.session_state.dialogue_newest_first
                st.rerun()


if __name__ == "__main__":
    main()
