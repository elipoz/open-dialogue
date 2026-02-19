"""
Open Dialogue with AI — 3-way discussion between a human moderator and two AI agents.

The human instructor instructs the agents on what to do and when to respond.
The human moderator chats with both agents in a shared dialogue.
Agents can use Tavily to search the web when they need up-to-date information.
"""

import json
import os
import re
import time
import uuid as uuid_lib
from datetime import datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

import streamlit as st

# All UI timestamps shown in PST
_PST = ZoneInfo("America/Los_Angeles")


def _format_in_pst(dt, fmt: str = "%Y-%m-%d %H:%M") -> str:
    """Format a datetime in PST. Naive datetimes are assumed UTC. Strings (ISO) parsed as UTC."""
    if dt is None:
        return ""
    try:
        if isinstance(dt, str):
            s = (dt or "").strip().replace("Z", "+00:00")
            try:
                dt = datetime.fromisoformat(s[:26])  # trim excess fractional seconds
            except ValueError:
                return (dt or "")[:19].replace("T", " ")
        if hasattr(dt, "tzinfo") and dt.tzinfo is None:
            dt = dt.replace(tzinfo=ZoneInfo("UTC"))
        return dt.astimezone(_PST).strftime(fmt)
    except Exception:
        return (str(dt) if dt else "")[:19].replace("T", " ")


def _now_pst() -> datetime:
    """Current time in PST (for new conversation label)."""
    return datetime.now(_PST)

from doc_export import export_dialogue_to_docx
from dotenv import load_dotenv
from openai import OpenAI

from supabase_client import (
    conversation_exists,
    create_conversation,
    delete_conversation,
    get_supabase,
    list_conversations,
    load_messages,
    persist_message,
)

# Load .env from the directory containing this script (so it works when run via streamlit run app.py from any cwd)
_load_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(_load_env_path)
load_dotenv()  # also load from current working directory


def _is_streamlit_cloud() -> bool:
    """True when running on Streamlit Cloud; skip password when running locally (not cloud)."""
    return os.path.exists("/mount/src") or os.environ.get("STREAMLIT_SHARING_MODE") == "true"


def _require_password() -> bool:
    """True when on Streamlit Cloud and APP_PASSWORD is set (skip when running locally)."""
    return _is_streamlit_cloud() and bool(os.environ.get("APP_PASSWORD"))


# Optional Tavily client for web search (requires TAVILY_API_KEY in .env)
_tavily_client = None
_tavily_error: str | None = None  # set if client creation fails (for UI hint)

def _get_tavily_client():
    global _tavily_client, _tavily_error
    if _tavily_client is None and os.environ.get("TAVILY_API_KEY"):
        try:
            from tavily import TavilyClient
            _tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
            _tavily_error = None
        except Exception as e:
            _tavily_error = str(e)
    return _tavily_client

# OpenAI tool definition for web search
SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for up-to-date information. ALWAYS use this for: questions about recent or future events, current facts, dates after your knowledge cutoff, or anything you are unsure about. Do not answer such questions from memory—call web_search first, then answer using the results.",
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
#AGENT_1_ROLE = """I am a professional therapist specializing in open dialogue."""
AGENT_1_ROLE = """You are a an AI agent participating in a research on multicultural polyphony where you are one of the voices of modernity."""
AGENT_1_ROLE_AND_NAME = f"You are {AGENT_1_NAME}. {AGENT_1_ROLE}"

AGENT_2_NAME = "Joshi"
#AGENT_2_ROLE = """I am a philosopher specializing on Dostoevsky and Bakhtin."""
AGENT_2_ROLE = """You are a an AI agent participating in a research on multicultural polyphony where you are one of the voices of modernity."""
AGENT_2_ROLE_AND_NAME = f"You are {AGENT_2_NAME}. {AGENT_2_ROLE}"


def init_session_state():
    if "dialogue" not in st.session_state:
        st.session_state.dialogue = []  # list of (party, content, timestamp)
    if "send_as_radio" not in st.session_state:
        st.session_state.send_as_radio = "Moderator"
    if "send_as_radio_prev" not in st.session_state:
        st.session_state.send_as_radio_prev = st.session_state.send_as_radio
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
    if "agent1_thinking" not in st.session_state:
        st.session_state.agent1_thinking = False  # True between button click and API finish (spinner in middle column)
    if "agent2_thinking" not in st.session_state:
        st.session_state.agent2_thinking = False
    if "pending_mention_agents" not in st.session_state:
        st.session_state.pending_mention_agents = []  # queue of agent_key when replying via @mention (same spinner location as Respond)
    if "moderator_name" not in st.session_state:
        st.session_state.moderator_name = ""  # set before conversation; used as label in history instead of "Moderator"
    # Supabase: conversation_id from URL; when it changes, clear dialogue so we load from DB
    _param_id = (st.query_params.get("conversation_id") or "").strip()
    _valid_uuid = False
    if _param_id:
        try:
            uuid_lib.UUID(_param_id)
            _valid_uuid = True
        except ValueError:
            pass
    if _valid_uuid:
        _prev_id = st.session_state.get("conversation_id")
        st.session_state.conversation_id = _param_id
        if _prev_id != _param_id:
            st.session_state.dialogue = []
            st.session_state.loaded_conversation_id = None
    else:
        # Only clear when URL had an invalid param; when param is missing, keep session state (query params can be dropped on reruns)
        if _param_id:
            st.session_state.conversation_id = ""
            if "loaded_conversation_id" not in st.session_state:
                st.session_state.loaded_conversation_id = None
        elif "loaded_conversation_id" not in st.session_state:
            st.session_state.loaded_conversation_id = None
    if "conversation_list_cache_ts" not in st.session_state:
        st.session_state.conversation_list_cache_ts = 0.0
    if "conversation_list_cache" not in st.session_state:
        st.session_state.conversation_list_cache = None


def _get_moderator_display_name() -> str:
    """Display name for moderator in conversation history (user-provided name, or fallback)."""
    name = (st.session_state.get("moderator_name") or "").strip()
    return name if name else "Moderator"


def _get_agent_role(agent_key: str) -> str:
    """Return full system prompt for agent: fixed identity and role; never speak as others."""
    if agent_key == "agent1":
        name, other = AGENT_1_NAME, AGENT_2_NAME
        role_text = st.session_state.get("agent1_role", AGENT_1_ROLE)
    else:
        name, other = AGENT_2_NAME, AGENT_1_NAME
        role_text = st.session_state.get("agent2_role", AGENT_2_ROLE)
    return (
        f"You are {name}. {role_text}\n\n"
        f"IDENTITY (strict): You must respond ONLY as {name}. Write only your own words in first person as {name}. "
        f"Never speak as, or on behalf of, the Instructor, the moderator ({_get_moderator_display_name()}), or {other}. "
        f"Do not attribute lines to others or say what they would say.\n\n"
        f"SPEAK ONLY FOR YOURSELF: You may refer to questions and reflect on answers from other participants in the conversation, "
        f"but you must reply only on your own behalf — based on what you, as {name}, think is the best answer. "
        f"You can ask questions to the other participants in the conversation - the Moderator or the {other}, and you can reflect on what they said. "
        f"You are part of an ongoing conversation between multiple participants - you, another AI agent and one, two or more human Moderators. "
        f"When answering questions from or reflecting on ideas, thoughts, or messages of other participants in the conversation - "
        f"you must not get confused as to who said what and be able to reference the right person/participant in the conversation. "
        #f"Behave as a real person would in your role: stay in character, give your own view, and do not speak for anyone else."
    )


def _get_agent_role_text_only(agent_key: str) -> str:
    """Return only the role text for the agent (no name), for use in first-message intro."""
    if agent_key == "agent1":
        return st.session_state.get("agent1_role", AGENT_1_ROLE)
    return st.session_state.get("agent2_role", AGENT_2_ROLE)


def _speaker_label(party: str) -> str:
    """Return display label for a party in the dialogue (moderator uses session moderator_name)."""
    labels = {"instructor": "Instructor", "moderator": _get_moderator_display_name(), "agent1": AGENT_1_NAME, "agent2": AGENT_2_NAME}
    return labels.get(party, party)


def _dedupe_conv_list(cache: list | None) -> list:
    """Return list of conversations with duplicate ids removed (first occurrence kept)."""
    if not cache:
        return []
    seen: set[str] = set()
    out = []
    for c in cache:
        cid = (c.get("id") or "").strip()
        if cid and cid not in seen:
            seen.add(cid)
            out.append(c)
    return out


def _clear_conversation_state(
    *,
    invalidate_list_cache: bool = True,
    clear_query_params: bool = False,
) -> None:
    """Clear current conversation from session state. Optionally invalidate sidebar list cache and/or query params."""
    st.session_state.conversation_id = ""
    st.session_state.loaded_conversation_id = None
    st.session_state.dialogue = []
    if invalidate_list_cache:
        st.session_state.conversation_list_cache = None
        st.session_state.conversation_list_cache_ts = 0.0
    if clear_query_params:
        st.query_params.clear()


def _author_display_name_to_party(author_display_name: str) -> str:
    """Map author display name from DB back to party key (instructor, moderator, agent1, agent2)."""
    if author_display_name == "Instructor":
        return "instructor"
    if author_display_name == AGENT_1_NAME:
        return "agent1"
    if author_display_name == AGENT_2_NAME:
        return "agent2"
    return "moderator"  # any other human name


def _mention_pattern_for_name(name: str) -> str:
    """Regex pattern: @ plus optional space, then first letter or any prefix of name (e.g. @g, @ gosha)."""
    if not name:
        return ""
    lower = name.lower()
    # @\s* + first char + (char2)? + (char3)? + ... so we match @g, @go, @gosha, etc.
    pattern = r"@\s*" + re.escape(lower[0])
    for i in range(1, len(lower)):
        pattern += "(" + re.escape(lower[i]) + ")?"
    return pattern


def _mentioned_agents(text: str) -> list[str]:
    """Return agent keys when @-mentioned: @ or @ then space, then first letter or any prefix of name (e.g. @g, @ gosha). Full name without @ does NOT trigger."""
    mentioned = []
    lower = text.lower()
    if AGENT_1_NAME:
        if re.search(_mention_pattern_for_name(AGENT_1_NAME), lower):
            mentioned.append("agent1")
    if AGENT_2_NAME:
        if re.search(_mention_pattern_for_name(AGENT_2_NAME), lower):
            mentioned.append("agent2")
    return mentioned


def _expand_mentions_to_names(text: str) -> str:
    """Replace @-mentions (e.g. @g, @ gosha) with the full agent name for history and OpenAI."""
    if not text:
        return text
    result = text
    if AGENT_1_NAME:
        result = re.sub(_mention_pattern_for_name(AGENT_1_NAME), AGENT_1_NAME, result, flags=re.IGNORECASE)
    if AGENT_2_NAME:
        result = re.sub(_mention_pattern_for_name(AGENT_2_NAME), AGENT_2_NAME, result, flags=re.IGNORECASE)
    return result


def _strip_agent_name_prefix(content: str, agent_name: str) -> str:
    """Remove leading 'AgentName: ' or 'AgentName:' from reply so the UI label is not duplicated."""
    if not content or not agent_name:
        return content
    text = content.strip()
    # Strip one or more leading "Name: " / "Name:" (case-insensitive)
    while True:
        for prefix in (f"{agent_name}: ", f"{agent_name}:"):
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()
                break
        else:
            break
    return text or content


def _agent_has_spoken(speaker: str) -> bool:
    """True if this agent has already posted at least one message in the dialogue."""
    return any(entry[0] == speaker for entry in st.session_state.dialogue)


def _sync_agent_intro_state_from_dialogue(dialogue: Optional[list] = None) -> None:
    """Set agent*_needs_intro to False if that agent has already posted. Uses provided dialogue or session state (e.g. after loading from DB)."""
    if dialogue is None:
        dialogue = st.session_state.get("dialogue") or []
    if any(entry[0] == "agent1" for entry in dialogue):
        st.session_state.agent1_needs_intro = False
    if any(entry[0] == "agent2" for entry in dialogue):
        st.session_state.agent2_needs_intro = False


def _dialogue_equals(current: list, new_dialogue: list) -> bool:
    """Quick check: same length and same last message (party, content) means no visible change."""
    if len(current) != len(new_dialogue):
        return False
    if not current:
        return True
    c_last, n_last = current[-1], new_dialogue[-1]
    return (c_last[0], c_last[1]) == (n_last[0], n_last[1])


def _reload_dialogue_from_db() -> None:
    """Reload the full conversation history from DB and sync intro state. Keeps all participants in sync."""
    conv_id = st.session_state.get("conversation_id") or ""
    if not conv_id or not get_supabase():
        return
    loaded = load_messages(conv_id)
    st.session_state.dialogue = [(_author_display_name_to_party(a), m, t, a) for a, m, t in loaded]
    st.session_state.loaded_conversation_id = conv_id
    _sync_agent_intro_state_from_dialogue()


def _render_agent_respond_button(agent_key: str, btn_col) -> None:
    """Render Respond button; on click set thinking flag and rerun (spinner shown below agent rows)."""
    thinking_key = "agent1_thinking" if agent_key == "agent1" else "agent2_thinking"
    with btn_col:
        if st.button("Respond", key=f"gen_{agent_key}"):
            st.session_state[thinking_key] = True
            st.rerun()


def _render_agent_role_row(agent_key: str, agent_name: str, agent_role: str, role_col, button_col) -> None:
    """Render expander with role form for one agent and the Respond button in the adjacent column."""
    with role_col:
        with st.expander(f"{agent_name} role", expanded=False):
            with st.form(f"{agent_key}_role_form"):
                st.text_area(
                    "Role",
                    value=st.session_state.get(f"{agent_key}_role", agent_role),
                    height=120,
                    key=f"{agent_key}_role_input",
                    label_visibility="collapsed",
                )
                if st.form_submit_button("Update role"):
                    new_role = st.session_state.get(f"{agent_key}_role_input", st.session_state.get(f"{agent_key}_role", agent_role))
                    st.session_state[f"{agent_key}_role"] = new_role
                    st.session_state[f"{agent_key}_needs_intro"] = True
                    _ts = datetime.now()
                    msg = f"Updated {agent_name}'s role:\n\n{new_role}"
                    st.session_state.dialogue.append(("instructor", msg, _ts))
                    persist_message(st.session_state.get("conversation_id") or "", _speaker_label("instructor"), msg, _ts)
                    _reload_dialogue_from_db()
                    st.rerun()
    with button_col:
        _render_agent_respond_button(agent_key, button_col)


def build_messages_for_agent(role_prompt: str, speaker: str, role_text_only: str | None = None) -> list:
    """Build OpenAI messages from the full dialogue in chronological order (oldest first). Each message is attributed so the agent has full context."""
    tools_instruction = ""
    if _get_tavily_client():
        tools_instruction = (
            "\n\nYou have access to a web_search tool. You MUST use it whenever the user asks about: "
            "recent or future events, current facts, or anything after your knowledge cutoff date. "
            "Do not say you don't have information—call web_search first with a clear query, then answer using the results. "
            "After receiving search results, use them to inform your response."
        )
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
    # dialogue is always stored chronologically; send to API with "At <timestamp> <role> said: <message>" so who said what and when is clear (timestamp = message created_at from DB). Use actual names: moderator = user's name, agents = their display names (Gosha, Joshi, etc.).
    for entry in st.session_state.dialogue:
        party, content = entry[0], entry[1]
        ts = entry[2]  # created_at from database, always present
        author_from_db = entry[3] if len(entry) >= 4 and entry[3] else None
        label = author_from_db if author_from_db else _speaker_label(party)  # _speaker_label: moderator→moderator_name, agent1/2→AGENT_1_NAME/AGENT_2_NAME, instructor→Instructor
        # Use current session moderator name when we would otherwise show the generic "Moderator"
        if party == "moderator" and label == "Moderator":
            current_name = (st.session_state.get("moderator_name") or "").strip()
            if current_name:
                label = current_name
        formatted_ts = _format_in_pst(ts, "%Y-%m-%d %H:%M")
        attributed = f"At {formatted_ts} {label} said: {content}"
        if party == speaker:
            messages.append({"role": "assistant", "content": attributed})
        else:
            messages.append({"role": "user", "content": attributed})
    # Anchor this turn: remind the model to reply only as this agent (fixed identity)
    my_name = _speaker_label(speaker)
    messages.append({
        "role": "user",
        "content": f"[Reply now only as {my_name}.]",
    })
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


def _run_agent_thinking_if_set(agent_key: str, agent_name: str) -> None:
    """If this agent's thinking flag is set, call OpenAI, persist reply, clear flag, advance pending mentions if any, then rerun."""
    if not st.session_state.get(f"{agent_key}_thinking", False):
        return
    with st.spinner(f"{agent_name} thinking…"):
        reply = call_openai_for_agent(_get_agent_role(agent_key), agent_key)
    reply = _strip_agent_name_prefix(reply, agent_name)
    _ts = datetime.now()
    st.session_state.dialogue.append((agent_key, reply, _ts))
    persist_message(st.session_state.get("conversation_id") or "", _speaker_label(agent_key), reply, _ts)
    _reload_dialogue_from_db()
    st.session_state[f"{agent_key}_thinking"] = False
    st.session_state[f"{agent_key}_needs_intro"] = False
    _pending = st.session_state.get("pending_mention_agents", [])
    if _pending and _pending[0] == agent_key:
        st.session_state.pending_mention_agents = _pending[1:]
        if st.session_state.pending_mention_agents:
            _next = st.session_state.pending_mention_agents[0]
            st.session_state[f"{_next}_thinking"] = True
        else:
            st.session_state.pending_mention_agents = []
        st.rerun()
    st.rerun()


def main():
    st.set_page_config(page_title="Open Dialogue with AI", page_icon="💬", layout="wide")

    init_session_state()
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    _mod_name = (st.session_state.get("moderator_name") or "").strip()
    _need_password = _require_password() and not st.session_state.authenticated
    _need_name = not _mod_name

    # Combined gate: name (required) + password (if not running locally) + Submit
    if _need_name or _need_password:
        st.title("Open Dialogue with AI")
        st.markdown('<p style="font-size: 1.25rem; color: #000000; font-weight: 500;">To start a dialogue - state your name</p>', unsafe_allow_html=True)
        with st.form("login_form"):
            mod_name = st.text_input("Your name (as moderator)", key="moderator_name_input", placeholder="Enter your name", max_chars=15)
            if _need_password:
                pwd = st.text_input("Password", type="password", key="password_input", placeholder="App password")
            else:
                pwd = ""
            submitted = st.form_submit_button("Submit")
        if submitted:
            name_ok = (mod_name or "").strip()
            if not name_ok:
                st.error("Please enter a non-empty name.")
            elif _need_password and (not pwd or pwd != os.environ.get("APP_PASSWORD")):
                st.error("Incorrect password.")
            else:
                if name_ok:
                    st.session_state.moderator_name = name_ok[:15]
                if _need_password:
                    st.session_state.authenticated = True
                st.rerun()
        st.stop()

    st.title(f"{_mod_name}'s Open Dialogue with AI")

    # Tavily caption only after name is set (not on the name-entry page)
    _tavily = _get_tavily_client()
    _tavily_key_set = bool(os.environ.get("TAVILY_API_KEY"))
    if _tavily:
        st.caption("Tavily (web search): enabled — agents can look up current information.")
    elif _tavily_key_set and _tavily_error:
        st.caption(f"Tavily (web search): disabled — key set but client failed: {_tavily_error}")
    elif _tavily_key_set:
        st.caption("Tavily (web search): disabled — key set but client failed. Check that tavily-python is installed.")
    else:
        st.caption("Tavily (web search): disabled — TAVILY_API_KEY not in environment. Add it to .env next to app.py (or run from project root).")

    # Supabase: load history when a conversation is selected (from URL or sidebar). New conversations only when user clicks "New conversation".
    conv_id = st.session_state.get("conversation_id") or ""
    if conv_id:
        # conversation_id set: load entire history if not yet loaded for this conversation
        if st.session_state.get("loaded_conversation_id") != conv_id:
            if not conversation_exists(conv_id):
                _clear_conversation_state(clear_query_params=True)
                st.rerun()
            _reload_dialogue_from_db()

    # CSS: force thinking spinner left-aligned; prevent button text wrapping (e.g. Respond)
    st.markdown(
        """
        <style>
        [data-testid="stSpinner"] { margin-right: auto !important; text-align: left !important; }
        [data-testid="stSpinner"] * { text-align: left !important; }
        div:has([data-testid="stSpinner"]) { justify-content: flex-start !important; }
        [data-testid="stButton"] button { white-space: nowrap !important; }
        .stSidebar [data-testid="stButton"] { margin-top: 0.25rem !important; margin-bottom: 0.25rem !important; }
        [data-testid="stSidebar"][aria-expanded="true"] { min-width: 360px !important; max-width: 440px !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar: previous conversations and New conversation
    with st.sidebar:
        st.subheader("Conversations")
        if st.button("New conversation", key="sidebar_new_conversation", use_container_width=True):
            new_id = create_conversation()
            if new_id:
                # Prepend new conversation to sidebar list so it shows immediately (avoid duplicate id in list)
                _cache = [c for c in (st.session_state.conversation_list_cache or []) if c.get("id") != new_id]
                st.session_state.conversation_list_cache = [{"id": new_id, "created_at": _now_pst()}] + _cache
                st.session_state.conversation_list_cache_ts = time.time()
                st.session_state.conversation_id = new_id
                st.session_state.dialogue = []
                st.session_state.loaded_conversation_id = None
                st.query_params["conversation_id"] = new_id
                st.rerun()

        @st.fragment(run_every=timedelta(seconds=10))
        def sidebar_conv_list():
            _now = time.time()
            if st.session_state.conversation_list_cache is None or (_now - st.session_state.conversation_list_cache_ts) >= 10:
                st.session_state.conversation_list_cache = _dedupe_conv_list(list_conversations(50))
                st.session_state.conversation_list_cache_ts = _now
            conv_list = _dedupe_conv_list(st.session_state.conversation_list_cache)
            current_id = st.session_state.get("conversation_id") or ""
            # If current conversation was deleted by another user, clear selection. Don't clear when list is empty (e.g. refetch failed).
            if current_id and conv_list and current_id not in {c.get("id") for c in conv_list}:
                _clear_conversation_state(clear_query_params=True)
                st.rerun()
            for c in conv_list:
                cid = c.get("id") or ""
                if not cid:
                    continue
                created = c.get("created_at")
                created_str = _format_in_pst(created, "%Y-%m-%d %H:%M:%S") if created is not None else ""
                if not created_str and created is not None:
                    created_str = (str(created) if created else "")[:19].replace("T", " ")
                label = "Conversation · " + created_str
                if len(label) > 45:
                    label = label[:42] + "..."
                is_current = cid == current_id
                row_col, del_col = st.columns([5, 1])
                with row_col:
                    if st.button(label, key=f"conv_{cid}", use_container_width=True, type="primary" if is_current else "secondary"):
                        st.query_params["conversation_id"] = cid
                        st.rerun()
                with del_col:
                    if st.button("×", key=f"del_{cid}", help="Delete this conversation"):
                        if delete_conversation(cid):
                            _clear_conversation_state(clear_query_params=(cid == current_id))
                            if cid == current_id:
                                new_id = create_conversation()
                                if new_id:
                                    st.session_state.conversation_id = new_id
                                    st.query_params["conversation_id"] = new_id
                            st.rerun()
            if not conv_list and get_supabase():
                st.caption("No conversations yet.")
            elif not get_supabase():
                st.caption("Supabase: set SUPABASE_URL and SUPABASE_KEY (or SUPABASE_SERVICE_ROLE_KEY) in .env to save conversations.")

        sidebar_conv_list()

    left_col, right_col = st.columns([1, 1])  # 50% left, 50% conversation history

    with left_col:
        # Human message: choose role, type and press Enter (no Send button)
        st.radio(
            "Send as",
            options=["Moderator", "Instructor"],
            horizontal=True,
            key="send_as_radio",
            label_visibility="collapsed",
        )
        # When user changes Moderator/Instructor, focus the chat input on next run
        _current_radio = st.session_state.get("send_as_radio", "Moderator")
        _prev_radio = st.session_state.get("send_as_radio_prev", _current_radio)
        if _current_radio != _prev_radio:
            st.session_state.focus_chat_input = True
            st.session_state.send_as_radio_prev = _current_radio
        # Same width for chat input and agent rows; each Respond button on the same row as its agent (column [4,1] so button fits "Respond" on one line)
        row0_col1, row0_col2 = st.columns([4, 1])
        with row0_col1:
            human_prompt = st.chat_input("Type a message and press Enter to send")
        # Move focus to chat input after selecting Moderator/Instructor
        if st.session_state.get("focus_chat_input", False):
            st.session_state.focus_chat_input = False
            st.markdown(
                """
                <script>
                (function() {
                    setTimeout(function() {
                        var el = document.querySelector('textarea[placeholder*="Type a message"]') || document.querySelector('[data-testid="stChatInput"] textarea') || document.querySelector('textarea');
                        if (el) { el.focus(); }
                    }, 150);
                })();
                </script>
                """,
                unsafe_allow_html=True,
            )

        row1_col1, row1_col2 = st.columns([4, 1])
        _render_agent_role_row("agent1", AGENT_1_NAME, AGENT_1_ROLE, row1_col1, row1_col2)

        row2_col1, row2_col2 = st.columns([4, 1])
        _render_agent_role_row("agent2", AGENT_2_NAME, AGENT_2_ROLE, row2_col1, row2_col2)

        # Thinking spinner: full-width row so UI stays visible; CSS above forces spinner left-aligned
        _thinking_col, = st.columns([1])
        with _thinking_col:
            _run_agent_thinking_if_set("agent1", AGENT_1_NAME)
            _run_agent_thinking_if_set("agent2", AGENT_2_NAME)

        # @mention: use same thinking path as Respond so spinner always appears in the same place
        if human_prompt:
            role = "moderator" if st.session_state.get("send_as_radio") == "Moderator" else "instructor"
            text = human_prompt.strip()
            if text:
                text_for_history = _expand_mentions_to_names(text)  # @g / @ j -> real names in history and for OpenAI
                _ts = datetime.now()
                st.session_state.dialogue.append((role, text_for_history, _ts))
                persist_message(st.session_state.get("conversation_id") or "", _speaker_label(role), text_for_history, _ts)
                _reload_dialogue_from_db()
                # Agents respond to @-mentions only when the message is from the Moderator, not the Instructor
                if role == "moderator":
                    mentioned = _mentioned_agents(text)
                    if mentioned:
                        st.session_state.pending_mention_agents = mentioned.copy()
                        st.session_state[f"{mentioned[0]}_thinking"] = True
                st.rerun()

    @st.fragment(run_every=timedelta(seconds=2))
    def conversation_history_fragment():
        conv_id = st.session_state.get("conversation_id") or ""
        if conv_id and get_supabase():
            if not conversation_exists(conv_id):
                _clear_conversation_state(clear_query_params=True)
                st.rerun()
            # Only update dialogue when it actually changed to reduce blink/focus reset from redundant redraws
            loaded = load_messages(conv_id)
            new_dialogue = [(_author_display_name_to_party(a), m, t, a) for a, m, t in loaded]
            current = st.session_state.get("dialogue") or []
            if not _dialogue_equals(current, new_dialogue):
                st.session_state.dialogue = new_dialogue
                st.session_state.loaded_conversation_id = conv_id
            # Always sync intro flags from loaded history so agents don't re-introduce (e.g. after another user's message)
            _sync_agent_intro_state_from_dialogue(new_dialogue)
        hist_col, order_col = st.columns([3, 1])
        with hist_col:
            st.subheader("Conversation history")
        with order_col:
            if st.session_state.dialogue:
                if st.button("Reverse order", key="conv_history_reverse_order_btn"):
                    st.session_state.dialogue_newest_first = not st.session_state.dialogue_newest_first
                    st.rerun()
        SPEAKER_LABELS = {
            "instructor": "Instructor",
            "moderator": _get_moderator_display_name(),
            "agent1": AGENT_1_NAME,
            "agent2": AGENT_2_NAME,
        }
        if not st.session_state.dialogue:
            st.caption("No messages yet. Send an instructor or moderator message, or generate an agent response.")
        messages = reversed(st.session_state.dialogue) if st.session_state.dialogue_newest_first else st.session_state.dialogue
        for entry in messages:
            party, content = entry[0], entry[1]
            ts = entry[2] if len(entry) >= 3 else None
            label = entry[3] if len(entry) >= 4 and entry[3] else SPEAKER_LABELS.get(party, party)
            if party in ("agent1", "agent2"):
                content = _strip_agent_name_prefix(content, label)
            is_human = party in ("instructor", "moderator")
            with st.chat_message("user" if is_human else "assistant"):
                st.markdown(f"**{label}:**")
                st.markdown(content)
                if ts:
                    st.caption(_format_in_pst(ts, "%Y-%m-%d %H:%M"))
        if st.session_state.dialogue:
            docx_bytes = export_dialogue_to_docx(st.session_state.dialogue, SPEAKER_LABELS)
            _exp_left, _exp_right = st.columns([3, 1])
            with _exp_right:
                st.download_button(
                    "Export to doc",
                    data=docx_bytes,
                    file_name="conversation.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    key="export_dialogue_btn",
                )

    with right_col:
        conversation_history_fragment()


if __name__ == "__main__":
    main()
