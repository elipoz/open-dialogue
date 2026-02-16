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
AGENT_1_NAME = "Gosha"  # Display name for first agent
AGENT_2_NAME = "Joshi"  # Display name for second agent

AGENT_1_ROLE = f"""You are {AGENT_1_NAME}. You are a professional therapist specializing in open dialogue."""

AGENT_2_ROLE = f"""You are {AGENT_2_NAME}. You are a philosopher specializing on Dostoevsky and Bakhtin."""


def init_session_state():
    if "dialogue" not in st.session_state:
        st.session_state.dialogue = []  # list of (party, content, timestamp)
    if "send_as_radio" not in st.session_state:
        st.session_state.send_as_radio = "Moderator"
    if "dialogue_newest_first" not in st.session_state:
        st.session_state.dialogue_newest_first = True  # True = newest first, False = chronological


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


def build_messages_for_agent(role_prompt: str, speaker: str) -> list:
    """Build OpenAI messages from the full dialogue in chronological order (oldest first). Each message is attributed so the agent has full context."""
    tools_instruction = ""
    if _get_tavily_client():
        tools_instruction = "\n\nYou have access to a web_search tool. Use it when you need to look up current information, facts, or events. After receiving search results, use them to inform your response."
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
    messages = build_messages_for_agent(role_prompt, speaker)
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
    st.title("Open Dialogue with AI")
    st.caption("Human moderator + two AI agents in a 3-way conversation.")

    init_session_state()

    left_col, right_col = st.columns([6, 4])  # 60% left, 40% Full dialogue

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
        if human_prompt:
            role = "moderator" if st.session_state.get("send_as_radio") == "Moderator" else "instructor"
            text = human_prompt.strip()
            if text:
                st.session_state.dialogue.append((role, text, datetime.now()))
                for agent_key in _mentioned_agents(text):
                    # Use same [4,1] columns so spinner appears to the right, aligned with Respond buttons
                    _c1, _c2 = st.columns([4, 1])
                    with _c2:
                        with st.spinner(f"{_speaker_label(agent_key)} thinking…"):
                            agent_role = AGENT_1_ROLE if agent_key == "agent1" else AGENT_2_ROLE
                            reply = call_openai_for_agent(agent_role, agent_key)
                    st.session_state.dialogue.append((agent_key, reply, datetime.now()))
                st.rerun()

        # Agent 1 role + Respond button (one line; [4,1] aligns with Send button column)
        a1_col1, a1_col2 = st.columns([4, 1])
        with a1_col1:
            with st.expander(f"{AGENT_1_NAME} role", expanded=False):
                st.text_area("Role", value=AGENT_1_ROLE, height=120, disabled=True, key="role1_display", label_visibility="collapsed")
        with a1_col2:
            if st.button("Respond", key="gen_a1"):
                with st.spinner(f"{AGENT_1_NAME} thinking…"):
                    reply = call_openai_for_agent(AGENT_1_ROLE, "agent1")
                st.session_state.dialogue.append(("agent1", reply, datetime.now()))
                st.rerun()

        # Agent 2 role + Respond button (one line; [4,1] aligns with Send button column)
        a2_col1, a2_col2 = st.columns([4, 1])
        with a2_col1:
            with st.expander(f"{AGENT_2_NAME} role", expanded=False):
                st.text_area("Role", value=AGENT_2_ROLE, height=120, disabled=True, key="role2_display", label_visibility="collapsed")
        with a2_col2:
            if st.button("Respond", key="gen_a2"):
                with st.spinner(f"{AGENT_2_NAME} thinking…"):
                    reply = call_openai_for_agent(AGENT_2_ROLE, "agent2")
                st.session_state.dialogue.append(("agent2", reply, datetime.now()))
                st.rerun()

    with right_col:
        st.subheader("Full dialogue")
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
