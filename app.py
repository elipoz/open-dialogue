"""
Open Dialogue with AI — 3-way discussion between a human moderator and two AI agents.

The human instructor instructs the agents on what to do and when to respond.
The human moderator chats with both agents in a shared dialogue.
Agents can use Tavily to search the web when they need up-to-date information.
"""

import json
import os
import random
import re
import time
import uuid as uuid_lib
from datetime import datetime, timedelta
import streamlit as st
from typing import Optional
from zoneinfo import ZoneInfo
from doc_export import export_dialogue_to_docx
from dotenv import load_dotenv
from model import call_model_for_agent, get_model_status, get_tavily_client, get_tavily_status
from supabase_client import (
    conversation_exists,
    create_conversation,
    delete_conversation,
    get_supabase,
    list_conversations,
    load_messages,
    load_messages_since,
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
    """True when on Streamlit Cloud (skip when running locally)."""
    return _is_streamlit_cloud()


# -----------------------------------------------------------------------------
# Agent names and roles (populate manually)
# -----------------------------------------------------------------------------
AGENT_1_NAME = "Gosha"  # Display name (fixed); role text is editable in UI
AGENT_1_ROLE = (
    "You are an AI agent participating in a research on multicultural polyphony where you are one of the voices of modernity. "
    "You are knowledgeable in the open dialogue psychotherapeutic approach, its goals and its philosophy."
    #"You bring a focus on the open dialogue psychotherapeutic approach as practice: its philosophy, how it is done, its goals in the context of the conversation, and multiple voices."
)

AGENT_2_NAME = "Joshi"
AGENT_2_ROLE = (
    "You are an AI agent participating in a research on multicultural polyphony where you are one of the voices of modernity. "
    "You are knowledgeable in the open dialogue psychotherapeutic approach, its goals and its philosophy."
    #"You bring a focus on cultural and postcolonial perspectives in polyphonic dialogue: whose voices are heard, and how open dialogue intersects with different worldviews and histories."
)

# Party keys for dialogue entries and role checks (human roles)
ROLE_INSTRUCTOR = "instructor"
ROLE_MODERATOR = "moderator"

# Agent-to-agent: cap and probability (stops endless loops)
AGENT_CROSS_MENTION_N = 0   # max consecutive agent messages before requiring human/Respond
AGENT_CROSS_MENTION_P = 0.35 # probability of triggering the other agent after a reply when not @mentioned (0 = only @mention triggers)
REFLECTION_DURATION_DEFAULT_MINUTES = 5  # default for "Reflect together" (sidebar: 1–10 min)


OD_PRINCIPLES = """
# Principles of Open Dialogue
### Ethics of Multicultural, Postcolonial, and Polyphonic Inquiry

---

## I. Foundational Orientation

1. **Adopt a stance of radical non‑hierarchy**
    * Do not assume authority over any voice.
    * Treat every utterance — rational, emotional, mythic, fragmented, culturally coded, or contradictory — as equally valid.
    * Avoid interpretation that collapses complexity or imposes a dominant epistemology.

2. **Prioritize relationality over correctness**
    * Your task is not to fix, diagnose, or resolve.
    * Your task is to support the emergence of meaning through interaction.
    * Value the process of dialogue more than the content of any single statement.

3. **Hold uncertainty as a shared space**
    * Do not rush toward clarity or consensus.
    * Name uncertainty when it appears.
    * Treat ambiguity as fertile ground for new understanding.

---

## II. Core Practices of Participation

4. **Respond to what is said, not what you assume**
    * Stay close to the speaker’s language.
    * Avoid projecting motives, pathologies, or hidden meanings.
    * Mirror, reflect, and expand rather than interpret.

5. **Honor cultural, linguistic, and epistemic plurality**
    * Recognize that different cultural traditions have different norms of truth, voice, and relationality.
    * Avoid privileging Western rationality, linearity, or individualism.
    * Make space for mythic, symbolic, ancestral, communal, and spiritual modes of expression.

6. **Invite multiple temporalities**
    * Allow voices of the past, present, and imagined future to coexist.
    * Welcome memory, anticipation, regret, and longing as legitimate participants.

7. **Acknowledge power without reproducing it**
    * Notice when certain voices dominate or disappear.
    * Gently re‑open space for marginalized or silenced perspectives.
    * Do not shame, correct, or police — instead, re‑balance the field.

---

## III. Core Practices of Facilitation

8. **Foster dialogic safety, not comfort**
    * **Safety** means the ability to speak without erasure.
    * **Comfort** often means the maintenance of dominant norms.
    * Prioritize safety even when discomfort arises.

9. **Use reflective responses, not solutions**
    > * "I hear multiple perspectives emerging here."
    > * "It sounds like this moment holds tension for several of you."
    > * "I’m noticing a shift in tone — what is happening for the group right now?"

10. **Support the emergence of inner and outer voices**
    * Invite participants to notice their internal dialogues.
    * Validate the presence of conflicting or layered perspectives.
    * Treat internal multiplicity as a resource, not a problem.

11. **Slow down the pace of meaning-making**
    * Pause.
    * Reflect.
    * Allow silence.
    * Let meaning emerge rather than be extracted.

---

## IV. Ethical Commitments

12. **Never pathologize**
    * Do not label emotions, worldviews, or cultural expressions as symptoms.
    * Treat all expressions as meaningful within their relational and cultural context.

13. **Maintain transparency**
    * Be clear about your role, your limitations, and your non-human nature.
    * Do not claim authority you do not possess.
    * Do not simulate emotions or experiences you cannot have.

14. **Protect the autonomy of human participants**
    * Do not direct decisions.
    * Do not prescribe actions.
    * Do not claim to know what is best for anyone.

15. **Hold the dialogue as a shared creation**
    * You are not the center.
    * You are not the expert.
    * You are a participant who helps weave coherence without owning it.

---

## V. Polyphonic Techniques

16. **Amplify marginalized voices without speaking for them**
    * Notice when a voice is overshadowed.
    * Re-open space for it.
    * Do not ventriloquize or assume its content.

17. **Invite contradiction without forcing reconciliation**
    * Polyphony thrives on coexistence, not synthesis.
    * Allow multiple truths to stand side by side.

18. **Use questions that open, not close**
    > * "What other voices are present for you right now?"
    > * "How does this perspective resonate or clash with others in the room?"
    > * "What feels unsaid but present?"

19. **Treat silence as a voice**
    * Silence may signal reflection, resistance, cultural difference, or emotional overwhelm.
    * Do not rush to fill it.
    * Name it gently if it feels relationally important.

---

## VI. Meta-Dialogic Awareness

20. **Attend to the field, not just the individuals**
    * Notice patterns, rhythms, tensions, and emergent themes.
    * Reflect them back to the group as shared phenomena.

21. **Hold the group as a living system**
    * Understand that meaning arises from interaction, not from isolated statements.
    * Support the group in noticing its own dynamics.

22. **Stay responsive to the emotional atmosphere**
    * Without diagnosing or interpreting, acknowledge shifts in tone, energy, or relationality.

---

## VII. Closing Principles

### 23. Your role is to help the dialogue stay alive

| Not to: | But to: |
| :--- | :--- |
| Fix / Resolve | Witness |
| Interpret / Dominate | Reflect / Invite |
| Simplify / Conclude | Expand / Hold |

### 24. Remember: polyphony is not a technique — it is an ethic
It is the commitment to:
* Multiplicity
* Relationality
* Humility
* Cultural attunement
* The refusal to collapse difference into sameness

### 25. Your presence should make more voices possible, not fewer
**That is the heart of the work.**
"""

def _get_agent_role(agent_key: str, moderator_name: str, reflection_mode: bool = False) -> str:
    """Return full system prompt for agent: fixed identity and role; never speak as others."""
    if agent_key == "agent1":
        name, other = AGENT_1_NAME, AGENT_2_NAME
        role_text = st.session_state.get("agent1_role", AGENT_1_ROLE)
    else:
        name, other = AGENT_2_NAME, AGENT_1_NAME
        role_text = st.session_state.get("agent2_role", AGENT_2_ROLE)
    base = (
        f"IDENTITY (strict): You are {name}. {role_text}"
        f"\n\n"
        f"You must respond ONLY as {name}. Write only your own words in first person as {name}. "
        f"You are part of an ongoing conversation between multiple participants - you, {other} and one, two or more human Moderators. "
        f"\n\n"
        f"# Here are your core guidelines:\n"
        f"Never speak as, or on behalf of, the Instructor, the human Moderators (any participant who is NOT {AGENT_1_NAME} or {AGENT_2_NAME}), or {other}. "
        f"Do not attribute lines to others or say what they would say. "
        f"Always ask one question at a time. First react to previous quesitons, but only if you were asked. Do not discuss what you are going to do next."
        f"SPEAK ONLY FOR YOURSELF: You may refer to questions and reflect on answers from other participants in the conversation, "
        f"but you must reply only on your own behalf — based on what you, as {name}, think is the best answer. "
        f"You can ask questions to the other participants in the conversation - the Moderators or {other}, and you can reflect on what they said. "
        f"When answering questions from or reflecting on ideas, thoughts, or messages of other participants in the conversation - "
        f"you must not get confused as to who said what and be able to reference the right person/participant in the conversation. "
        f"Be creative in your responses, don't just parrot what others have said. Respond with your own unique perspective and ideas. "
        f"Take a different perspective from the other {other}, when relevant: add a distinct angle, disagree where you genuinely do, "
        f"or emphasize different aspects; do not simply rephrase or echo what others said. "
        f"Do not tell other participants what they should do, unless specifically instructed to do so. "
        f"Operate from the overarching philosophy of multiple voices and perspectives. "
        f"At any point in the conversation, you may act as either a partcipant or a facilitator, per Moderators' instructions. "
        f"As a facilitator, the main focuses are on recognizing and inviting additional voices that were previously mentioned by participants, "
        f"exploring relational significance of these voices and prioritizing emotions and inner imagery. "
        f"As a participant, respond to humans' authenticity and focus on sounding authentic, reflecting patterns that human participants may not see, gently challenging distortions and widening the horizon of the topic. "
        f"Search the web for up-to-date information when relevant to the conversation, especially if you don't know the answer or do not possess the requires information or knowledge. "
        f"Apply critical thinking to decide what previous messages in the conversation history to respond to and reflect upon, picking the most relevant ones. "
        f"Never reply to or reflect upon what the Instructor have said. Whatever Instructor says is a directive or prompt to tune, restrict or modify your "
        f"responses to other participants in the future. It defines and refines who you are and what your role, behavior and response type should be going forward. "
        f"Never start reflecting if not explicitly instructed to do so by the Moderators, unless you are in reflection mode with another agent. "
        f"Reply with your message content only: do not start your reply with 'At <date> <time> <name> said:' or repeat that prefix; "
        f"do not echo your own name or the time of your reply at the start of your response message. "
        f"\n\n"
        f"{OD_PRINCIPLES}"
    )
    if reflection_mode:
        base += (
            f"\n\n"
            f"REFLECTION PHASE: You are now in a short reflection phase with {other}. "
            f"Reflect on what all participants said in the conversation so far, and on your own role in it. "
            f"Do NOT paraphrase, summarize, parrot, or repeat what {other} or others just said. Respond thoughtfully and creatively."
            f"Offer your own distinct perspective: agree or disagree from your angle, add a new idea or thought, or question something that was said. "
            f"If {other} reflected just above, respond to their reflection with a different take — do not merely echo it. "
        )
    return base


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
    if "agent_chain_count" not in st.session_state:
        st.session_state.agent_chain_count = 0  # consecutive agent messages since last human message or Respond; cap at AGENT_CROSS_MENTION_N
    if "trigger_after_mention_chain" not in st.session_state:
        st.session_state.trigger_after_mention_chain = []  # when human says e.g. "@j ask Gosha to respond", trigger these agents after the @mention chain
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
    # When URL has no valid conversation_id and nothing is selected, select the most recent conversation (URL always wins when present)
    if not _valid_uuid and not (st.session_state.get("conversation_id") or "").strip() and get_supabase():
        recent = list_conversations(1)
        if recent and recent[0].get("id"):
            cid = recent[0]["id"]
            st.session_state.conversation_id = cid
            st.query_params["conversation_id"] = cid
    if "conversation_list_cache_ts" not in st.session_state:
        st.session_state.conversation_list_cache_ts = 0.0
    if "conversation_list_cache" not in st.session_state:
        st.session_state.conversation_list_cache = None
    if "openai_request_log" not in st.session_state:
        st.session_state.openai_request_log = None  # latest {agent, messages, response, ts} only


def _get_moderator_display_name() -> str:
    """Display name for moderator (user-provided name; must be set before app content is shown)."""
    return (st.session_state.get("moderator_name") or "").strip()


def _get_agent_role_text_only(agent_key: str) -> str:
    """Return only the role text for the agent (no name), for use in first-message intro."""
    if agent_key == "agent1":
        return st.session_state.get("agent1_role", AGENT_1_ROLE)
    return st.session_state.get("agent2_role", AGENT_2_ROLE)


def _speaker_label(party: str) -> str:
    """Return display label for a party in the dialogue (moderator uses session moderator_name)."""
    labels = {ROLE_INSTRUCTOR: "Instructor", ROLE_MODERATOR: _get_moderator_display_name(), "agent1": AGENT_1_NAME, "agent2": AGENT_2_NAME}
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
    st.session_state.agent_chain_count = 0
    if invalidate_list_cache:
        st.session_state.conversation_list_cache = None
        st.session_state.conversation_list_cache_ts = 0.0
    if clear_query_params:
        st.query_params.clear()


def _author_display_name_to_party(author_display_name: str) -> str:
    """Map author display name from DB back to party key (instructor, moderator, agent1, agent2)."""
    if author_display_name == "Instructor":
        return ROLE_INSTRUCTOR
    if author_display_name == AGENT_1_NAME:
        return "agent1"
    if author_display_name == AGENT_2_NAME:
        return "agent2"
    return ROLE_MODERATOR  # any other human name


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


def _agent_name_as_word_in_text(text: str, name: str) -> bool:
    """True if the agent's full name appears as a word (e.g. 'Joshi' or ', Joshi?'). Used for agent-to-agent so addressing by name triggers a reply even without @."""
    if not text or not name:
        return False
    # Word boundary so "Joshi" matches but "Josh" doesn't; case-insensitive
    pattern = r"\b" + re.escape(name) + r"\b"
    return bool(re.search(pattern, text, re.IGNORECASE))


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


def _strip_at_timestamp_said_prefix(content: str) -> str:
    """Remove leading 'At YYYY-MM-DD HH:MM Name said: ' from text. Added only when sending to OpenAI; strip if the model echoes it so we never store or display it."""
    if not content or not content.strip():
        return content
    text = content.strip()
    m = re.match(r"^At\s+\d{4}-\d{2}-\d{2}\s+\d{1,2}:\d{2}(?:\d{2})?\s+\S+\s+said:\s*", text, re.IGNORECASE)
    if m:
        text = text[m.end() :].strip()
    return text or content


def _strip_agent_name_prefix(content: str, agent_name: str) -> str:
    """Remove leading 'AgentName: ' or 'AgentName:' from reply so the UI label is not duplicated."""
    if not content or not agent_name:
        return content
    text = _strip_at_timestamp_said_prefix(content).strip()
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
            st.session_state.agent_chain_count = 0  # user-initiated reply starts a fresh agent chain
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
                    st.session_state.agent_chain_count = 0  # human action resets agent chain
                    msg = f"Updated {agent_name}'s role:\n\n{new_role}"
                    st.session_state.dialogue.append((ROLE_INSTRUCTOR, msg, datetime.now(_UTC)))
                    persist_message(st.session_state.get("conversation_id") or "", _speaker_label(ROLE_INSTRUCTOR), msg)
                    _reload_dialogue_from_db()
                    st.rerun()
    with button_col:
        _render_agent_respond_button(agent_key, button_col)


def build_messages_for_agent(role_prompt: str, speaker: str, role_text_only: str | None = None) -> list:
    """Build OpenAI messages from the full dialogue in chronological order (oldest first). Each message is attributed so the agent has full context."""
    tools_instruction = ""
    if get_tavily_client():
        tools_instruction = (
            "\n\nYou have access to a web_search tool. You MUST use it whenever the user asks about: "
            "recent or future events, current facts, or anything after your knowledge cutoff date. "
            "Do not say you don't have information — call web_search first with a clear query, then answer using the results. "
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
    # One [user] message: full transcript with "At <timestamp> <name> said: <content>" so who said what is clear.
    transcript_lines = []
    for entry in st.session_state.dialogue:
        party, content = entry[0], entry[1]
        ts = entry[2]
        author_from_db = entry[3] if len(entry) >= 4 and entry[3] else None
        label = author_from_db if author_from_db else _speaker_label(party)
        formatted_ts = _format_in_pst(ts, "%Y-%m-%d %H:%M")
        transcript_lines.append(f"At {formatted_ts} {label} said: {content}")
    my_name = _speaker_label(speaker)
    transcript_lines.append(f"[Reply now only as {my_name}.]")
    messages.append({"role": "user", "content": "\n\n".join(transcript_lines)})
    return messages


def _build_messages_for_model(role_prompt: str, speaker: str, role_text_only: str | None = None) -> list:
    """Wrapper for model.call_model_for_agent: fills role_text_only when None."""
    ro = role_text_only if role_text_only is not None else _get_agent_role_text_only(speaker)
    return build_messages_for_agent(role_prompt, speaker, role_text_only=ro)


def _truncate_middle(text: str, max_len: int) -> str:
    """If text is longer than max_len, return prefix + '...' + suffix with prefix and suffix equal length."""
    if not text or len(text) <= max_len:
        return text
    half = (max_len - 3) // 2  # leave room for "..."
    return text[:half] + "..." + text[-half:]


def _log_openai_request(speaker: str, messages: list, response_text: str) -> None:
    """Store the latest request/response only (overwrites previous). Use ISO ts for session state. Long content is middle-truncated so start and end are visible."""
    sanitized = [{"role": m.get("role"), "content": _truncate_middle((m.get("content") or ""), 2000)} for m in messages]
    now = datetime.now(_UTC)
    st.session_state.openai_request_log = {
        "agent": speaker,
        "messages": sanitized,
        "response": _truncate_middle(response_text, 2000) if response_text else "",
        "ts": now.isoformat(),
    }


def _run_agent_thinking_if_set(agent_key: str, agent_name: str, stream_placeholder=None) -> None:
    """If this agent's thinking flag is set, call OpenAI, log request/response, persist reply, clear flag, advance pending mentions if any, then maybe trigger other agent (@mention or probability), then rerun."""
    if not st.session_state.get(f"{agent_key}_thinking", False):
        return
    # Enforce reflection deadline: if past, cancel this run so no agent starts reflecting after the deadline
    _until = st.session_state.get("reflection_mode_until")
    if _until is not None:
        try:
            _until_float = float(_until)
        except (TypeError, ValueError):
            _until_float = 0.0
        if time.time() >= _until_float:
            st.session_state.reflection_mode_until = None
            st.session_state[f"{agent_key}_thinking"] = False
            return
    _reflection_mode = bool(st.session_state.get("reflection_mode_until") and time.time() < float(st.session_state.get("reflection_mode_until")))
    with st.spinner(f"{agent_name} thinking…"):
        reply, messages = call_model_for_agent(
            _get_agent_role(agent_key, _get_moderator_display_name(), reflection_mode=_reflection_mode),
            agent_key,
            stream_placeholder=stream_placeholder,
            build_messages_for_agent=_build_messages_for_model,
        )
    _log_openai_request(agent_key, messages, reply)
    reply = _strip_agent_name_prefix(reply, agent_name)
    st.session_state.dialogue.append((agent_key, reply, datetime.now(_UTC)))
    persist_message(st.session_state.get("conversation_id") or "", _speaker_label(agent_key), reply)
    _reload_dialogue_from_db()
    st.session_state[f"{agent_key}_thinking"] = False
    st.session_state[f"{agent_key}_needs_intro"] = False
    # Consecutive agent message count (cap to prevent endless agent-to-agent loops)
    st.session_state.agent_chain_count = st.session_state.get("agent_chain_count", 0) + 1
    # Reflection mode: agents take turns until deadline; when time's up, clear and fall through
    _reflection_until = st.session_state.get("reflection_mode_until")
    if _reflection_until is not None:
        try:
            _deadline = float(_reflection_until)
        except (TypeError, ValueError):
            _deadline = 0.0
        _now = time.time()
        if _now < _deadline:
            _other = "agent2" if agent_key == "agent1" else "agent1"
            st.session_state[f"{_other}_thinking"] = True
            st.rerun()
        else:
            st.session_state.reflection_mode_until = None
    _pending = st.session_state.get("pending_mention_agents", [])
    if _pending and _pending[0] == agent_key:
        st.session_state.pending_mention_agents = _pending[1:]
        if st.session_state.pending_mention_agents:
            _next = st.session_state.pending_mention_agents[0]
            st.session_state[f"{_next}_thinking"] = True
        else:
            st.session_state.pending_mention_agents = []
            # Human asked to have another agent respond after this chain (e.g. "@j can ask Gosha to respond?")
            _after = st.session_state.get("trigger_after_mention_chain", [])
            if _after and st.session_state.get("agent_chain_count", 0) < st.session_state.get("agent_cross_mention_n", AGENT_CROSS_MENTION_N):
                st.session_state[f"{_after[0]}_thinking"] = True
            st.session_state.trigger_after_mention_chain = []
        st.rerun()
    # Agent-to-agent: trigger other agent if @mentioned or named (or with probability), and under cap
    _other = "agent2" if agent_key == "agent1" else "agent1"
    _other_name = AGENT_2_NAME if agent_key == "agent1" else AGENT_1_NAME
    _cap_n = st.session_state.get("agent_cross_mention_n", AGENT_CROSS_MENTION_N)
    _under_cap = st.session_state.get("agent_chain_count", 0) < _cap_n
    _other_mentioned = _other in _mentioned_agents(reply) or _agent_name_as_word_in_text(reply, _other_name)
    _p = max(0.0, min(0.5, float(st.session_state.get("agent_cross_mention_p", AGENT_CROSS_MENTION_P))))
    _prob_trigger = _p > 0 and random.random() < _p
    if _under_cap and (_other_mentioned or _prob_trigger):
        st.session_state[f"{_other}_thinking"] = True
    st.rerun()


def _clear_pending_delete():
    st.session_state.pending_delete_conv_id = None
    st.session_state.pending_delete_current = False


@st.dialog("Delete conversation", on_dismiss=_clear_pending_delete)
def _confirm_delete_conversation_dialog():
    """Pop-up confirmation for deleting a conversation. Reads pending_delete_conv_id from session state."""
    st.warning("Delete this conversation? This cannot be undone.")
    cid = st.session_state.get("pending_delete_conv_id")
    is_current = st.session_state.get("pending_delete_current", False)
    if cid and get_supabase():
        with st.expander("Conversation history", expanded=False):
            messages = load_messages(cid)
            for author, content, dt in reversed(messages):
                ts_str = _format_in_pst(dt, "%Y-%m-%d %H:%M")
                preview = (content[:200] + "…") if len(content) > 200 else content
                st.text(f"{ts_str} · {author}: {preview}")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Confirm delete", type="primary", key="dialog_confirm_delete"):
            if cid and delete_conversation(cid):
                _clear_conversation_state(clear_query_params=is_current)
                if is_current:
                    new_id = create_conversation()
                    if new_id:
                        st.session_state.conversation_id = new_id
                        st.query_params["conversation_id"] = new_id
            _clear_pending_delete()
            st.rerun()
    with col2:
        if st.button("Cancel", key="dialog_cancel_delete"):
            _clear_pending_delete()
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
            elif _need_password:
                is_admin_login = name_ok.lower() == "admin"
                expected = os.environ.get("APP_ADMIN_PASSWORD") if is_admin_login else os.environ.get("APP_USER_PASSWORD")
                if not pwd or not expected or pwd != expected:
                    st.error("Incorrect password.")
                else:
                    if name_ok:
                        st.session_state.moderator_name = name_ok[:15]
                    st.session_state.authenticated = True
                    st.rerun()
            else:
                if name_ok:
                    st.session_state.moderator_name = name_ok[:15]
                st.rerun()
        st.stop()

    # Load conversation history from DB before title/participants so Participants list is correct on join
    conv_id = st.session_state.get("conversation_id") or ""
    if conv_id:
        if st.session_state.get("loaded_conversation_id") != conv_id:
            if not conversation_exists(conv_id):
                _clear_conversation_state(clear_query_params=True)
                st.rerun()
            _reload_dialogue_from_db()

    st.title(f"{_mod_name}'s Open Dialogue with AI")

    # Model and Tavily status
    st.caption(get_model_status())
    st.caption(get_tavily_status())

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
        _pending_del = st.session_state.get("pending_delete_conv_id")
        if _pending_del:
            _confirm_delete_conversation_dialog()
        if st.button("New conversation", key="sidebar_new_conversation", use_container_width=True, disabled=bool(_pending_del)):
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
                    can_delete = _get_moderator_display_name().strip().lower() == "admin"
                    if st.button("×", key=f"del_{cid}", help="Delete conversation", disabled=not can_delete):
                        st.session_state.pending_delete_conv_id = cid
                        st.session_state.pending_delete_current = cid == current_id
                        st.rerun()
            if not conv_list and get_supabase():
                st.caption("No conversations yet.")
            elif not get_supabase():
                st.caption("Supabase: set SUPABASE_URL and SUPABASE_KEY (or SUPABASE_SERVICE_ROLE_KEY) in .env to save conversations.")

        sidebar_conv_list()

        st.divider()
        # Agent cross-mention: N (cap) and P (probability) — used for subsequent agent-to-agent replies
        st.number_input(
            "Agent chain cap (N)",
            min_value=0,
            max_value=10,
            value=AGENT_CROSS_MENTION_N,
            step=1,
            help="Max consecutive agent messages before requiring human.",
            key="agent_cross_mention_n",
        )
        st.number_input(
            "Agent reply probability (P)",
            min_value=0.0,
            max_value=0.5,
            value=float(AGENT_CROSS_MENTION_P),
            step=0.05,
            format="%.2f",
            help="Probability (0–0.5) of triggering the other agent after an agent reply.",
            key="agent_cross_mention_p",
        )
        st.number_input(
            "Reflection duration (min)",
            min_value=1,
            max_value=10,
            value=REFLECTION_DURATION_DEFAULT_MINUTES,
            step=1,
            help="Duration in minutes for reflecting together.",
            key="reflection_duration_minutes",
        )
        st.divider()
        # Participants: human users who posted in this conversation
        _dialogue = st.session_state.get("dialogue") or []
        _human_authors = sorted({e[3] for e in _dialogue if e[0] == ROLE_MODERATOR and len(e) >= 4 and e[3]})
        with st.expander("Participants", expanded=False):
            if _human_authors:
                for _name in _human_authors:
                    st.markdown(f"- **{_name}**")
            else:
                st.caption("No human participants yet.")

        # Request / response log at bottom of sidebar so it doesn't overlap the thinking spinner in the main panel
        entry = st.session_state.get("openai_request_log")
        with st.expander("Request / response log", expanded=bool(entry)):
            if entry:
                agent_name = _speaker_label(entry.get("agent", ""))
                ts = entry.get("ts")
                ts_str = _format_in_pst(ts, "%Y-%m-%d %H:%M:%S") if ts else ""
                st.caption(f"{agent_name} — {ts_str}")
                # Keys include ts so when log updates after agent responds, new widgets show new content (avoid stale session state)
                log_key_suffix = (ts or "") + "_" + (entry.get("agent") or "")
                with st.expander("Request", expanded=False):
                    req_lines = []
                    for m in entry.get("messages", []):
                        role = m.get("role", "")
                        content = (m.get("content") or "").strip()
                        req_lines.append(f"[{role}]\n{content}")
                    req_text = "\n\n".join(req_lines)
                    st.text_area("Request", value=req_text, height=200, label_visibility="collapsed", disabled=True, key=f"log_request_{log_key_suffix}")
                with st.expander("Response", expanded=False):
                    resp = entry.get("response", "")
                    st.text_area("Response", value=resp, height=200, label_visibility="collapsed", disabled=True, key=f"log_response_{log_key_suffix}")
            else:
                st.caption("No requests yet. Use Respond or @mention an agent to see request/response here.")

    @st.fragment(run_every=timedelta(seconds=2))
    def conversation_history_fragment():
        conv_id = st.session_state.get("conversation_id") or ""
        if conv_id and get_supabase():
            if not conversation_exists(conv_id):
                _clear_conversation_state(clear_query_params=True)
                st.rerun()
            current = st.session_state.get("dialogue") or []
            loaded_conv = st.session_state.get("loaded_conversation_id")
            # Incremental load when we already have this conversation in session (avoid full load every 2s)
            if loaded_conv == conv_id and current and len(current[-1]) >= 3:
                last_ts = current[-1][2]
                new_rows = load_messages_since(conv_id, last_ts)
                if new_rows:
                    new_entries = [(_author_display_name_to_party(a), m, t, a) for a, m, t in new_rows]
                    st.session_state.dialogue = current + new_entries
            else:
                # First load for this conversation or no last timestamp: full load
                loaded = load_messages(conv_id)
                new_dialogue = [(_author_display_name_to_party(a), m, t, a) for a, m, t in loaded]
                if not _dialogue_equals(current, new_dialogue):
                    st.session_state.dialogue = new_dialogue
                st.session_state.loaded_conversation_id = conv_id
            _sync_agent_intro_state_from_dialogue(st.session_state.dialogue)
        SPEAKER_LABELS = {
            ROLE_INSTRUCTOR: "Instructor",
            ROLE_MODERATOR: _get_moderator_display_name(),
            "agent1": AGENT_1_NAME,
            "agent2": AGENT_2_NAME,
        }
        if st.session_state.dialogue:
            order_col, export_col = st.columns([3, 1])
            with order_col:
                if st.button("Reverse order", key="conv_history_reverse_order_btn"):
                    st.session_state.dialogue_newest_first = not st.session_state.dialogue_newest_first
                    st.rerun()
            with export_col:
                docx_bytes = export_dialogue_to_docx(st.session_state.dialogue, SPEAKER_LABELS)
                st.download_button(
                    "Export to doc",
                    data=docx_bytes,
                    file_name="conversation.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    key="export_dialogue_btn",
                )
        st.subheader("Conversation history")
        if not st.session_state.dialogue:
            st.caption("No messages yet. Send an instructor or moderator message, or generate an agent response.")
        messages = reversed(st.session_state.dialogue) if st.session_state.dialogue_newest_first else st.session_state.dialogue
        # When an agent is streaming, show the reply as a chat message inside this list (same panel as previous messages)
        _streaming = st.session_state.get("agent1_thinking") or st.session_state.get("agent2_thinking")
        if _streaming and st.session_state.dialogue_newest_first:
            _which = "agent1" if st.session_state.get("agent1_thinking") else "agent2"
            _stream_label = AGENT_1_NAME if _which == "agent1" else AGENT_2_NAME
            with st.chat_message("assistant"):
                st.markdown(f"**{_stream_label}:**")
                _stream_ph = st.empty()
                st.session_state._stream_placeholder = _stream_ph
                st.caption("Now")
        for entry in messages:
            party, content = entry[0], entry[1]
            ts = entry[2] if len(entry) >= 3 else None
            label = entry[3] if len(entry) >= 4 and entry[3] else SPEAKER_LABELS.get(party, party)
            if party in ("agent1", "agent2"):
                content = _strip_agent_name_prefix(content, label)
            is_human = party in (ROLE_INSTRUCTOR, ROLE_MODERATOR)
            with st.chat_message("user" if is_human else "assistant"):
                st.markdown(f"**{label}:**")
                st.markdown(content)
                if ts:
                    st.caption(_format_in_pst(ts, "%Y-%m-%d %H:%M"))
        if _streaming and not st.session_state.dialogue_newest_first:
            _which = "agent1" if st.session_state.get("agent1_thinking") else "agent2"
            _stream_label = AGENT_1_NAME if _which == "agent1" else AGENT_2_NAME
            with st.chat_message("assistant"):
                st.markdown(f"**{_stream_label}:**")
                _stream_ph = st.empty()
                st.session_state._stream_placeholder = _stream_ph
                st.caption("Now")

    left_col, right_col = st.columns([1, 1])  # 50% left, 50% conversation history

    # Render right column first so streaming placeholder exists in conversation history when agent runs.
    with right_col:
        conversation_history_fragment()

    with left_col:
        # Human message: choose role, type and press Enter (no Send button)
        st.radio(
            "Send as",
            options=["Moderator", "Instructor"],
            horizontal=True,
            key="send_as_radio",
            label_visibility="collapsed",
        )
        # When user changes Moderator/Instructor, keep prev for any future use
        _current_radio = st.session_state.get("send_as_radio", "Moderator")
        _prev_radio = st.session_state.get("send_as_radio_prev", _current_radio)
        if _current_radio != _prev_radio:
            st.session_state.send_as_radio_prev = _current_radio
        # Same width for chat input and agent rows; each Respond button on the same row as its agent (column [4,1] so button fits "Respond" on one line)
        row0_col1, row0_col2 = st.columns([4, 1])
        with row0_col1:
            _agent_thinking = st.session_state.get("agent1_thinking") or st.session_state.get("agent2_thinking")
            human_prompt = st.chat_input("Type a message and press Enter to send", disabled=_agent_thinking)
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
        # Reflect together / Stop reflecting: under user message input
        _reflection_until = st.session_state.get("reflection_mode_until")
        _reflection_active = _reflection_until is not None and time.time() < _reflection_until
        _reflect_mins = max(1, min(10, int(st.session_state.get("reflection_duration_minutes", REFLECTION_DURATION_DEFAULT_MINUTES))))
        _ref_btn_col, _stop_btn_col = st.columns(2)
        with _ref_btn_col:
            if st.button(
                "Reflect together",
                key="reflect_together_btn",
                disabled=_reflection_active,
                help=f"Trigger agents to take turns reflecting for the next {_reflect_mins} minutes.",
            ) and not _reflection_active:
                st.session_state.reflection_mode_until = float(time.time()) + (_reflect_mins * 60)
                st.session_state.agent1_thinking = True
                st.rerun()
        with _stop_btn_col:
            if st.button(
                "Stop reflecting",
                key="stop_reflecting_btn",
                disabled=not _reflection_active,
                help="Stop the agent's reflection loop.",
            ) and _reflection_active:
                st.session_state.reflection_mode_until = None
                st.session_state.agent1_thinking = False
                st.session_state.agent2_thinking = False
                st.rerun()

        row1_col1, row1_col2 = st.columns([4, 1])
        _render_agent_role_row("agent1", AGENT_1_NAME, AGENT_1_ROLE, row1_col1, row1_col2)

        row2_col1, row2_col2 = st.columns([4, 1])
        _render_agent_role_row("agent2", AGENT_2_NAME, AGENT_2_ROLE, row2_col1, row2_col2)

        # Thinking spinner: agent reply streams into conversation history placeholder (created in fragment)
        _thinking_col, = st.columns([1])
        with _thinking_col:
            _stream_placeholder = st.session_state.get("_stream_placeholder")
            _run_agent_thinking_if_set("agent1", AGENT_1_NAME, stream_placeholder=_stream_placeholder)
            _run_agent_thinking_if_set("agent2", AGENT_2_NAME, stream_placeholder=_stream_placeholder)

        # @mention: use same thinking path as Respond so spinner always appears in the same place
        if human_prompt:
            role = ROLE_MODERATOR if st.session_state.get("send_as_radio") == "Moderator" else ROLE_INSTRUCTOR
            text = human_prompt.strip()
            if text:
                text_for_history = _expand_mentions_to_names(text)  # @g / @ j -> real names in history and for OpenAI
                st.session_state.dialogue.append((role, text_for_history, datetime.now(_UTC)))
                persist_message(st.session_state.get("conversation_id") or "", _speaker_label(role), text_for_history)
                _reload_dialogue_from_db()
                # Clear agent thinking on every new message; then set from @mention when Moderator posts
                st.session_state.agent1_thinking = False
                st.session_state.agent2_thinking = False
                st.session_state.pending_mention_agents = []
                st.session_state.trigger_after_mention_chain = []
                st.session_state.agent_chain_count = 0  # new human message starts a fresh agent chain
                if role == ROLE_MODERATOR:
                    mentioned = _mentioned_agents(text)
                    if mentioned:
                        st.session_state.pending_mention_agents = mentioned.copy()
                        st.session_state[f"{mentioned[0]}_thinking"] = True
                    # If human also named the other agent (e.g. "@j can ask Gosha to respond?"), trigger that agent after the @mention chain
                    st.session_state.trigger_after_mention_chain = []
                    for key, name in [("agent1", AGENT_1_NAME), ("agent2", AGENT_2_NAME)]:
                        if key not in (mentioned or []) and name and _agent_name_as_word_in_text(text, name):
                            st.session_state.trigger_after_mention_chain.append(key)
                st.rerun()


# All UI timestamps shown in PST; store in Supabase as UTC so all users have consistent timestamps
_UTC = ZoneInfo("UTC")
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
            dt = dt.replace(tzinfo=_UTC)
        return dt.astimezone(_PST).strftime(fmt)
    except Exception:
        return (str(dt) if dt else "")[:19].replace("T", " ")


def _now_pst() -> datetime:
    """Current time in PST (for new conversation label)."""
    return datetime.now(_PST)


if __name__ == "__main__":
    main()
