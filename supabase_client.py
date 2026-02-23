"""
Supabase persistence for Open Dialogue: conversations and messages.
Requires SUPABASE_URL and either SUPABASE_KEY or SUPABASE_SERVICE_ROLE_KEY in the environment (e.g. from .env).
"""

import os
import uuid as uuid_lib
from datetime import datetime
from zoneinfo import ZoneInfo

_UTC = ZoneInfo("UTC")

_supabase_client = None


def get_supabase():
    """Return Supabase client or None if URL/key not set."""
    global _supabase_client
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if _supabase_client is None and url and key:
        try:
            from supabase import create_client
            _supabase_client = create_client(url, key)
        except Exception:
            _supabase_client = False  # mark as attempted
    return _supabase_client if _supabase_client else None


def create_conversation() -> str:
    """Insert a new row in od_conversations and return its id (UUID string). DB sets created_at via default now()."""
    sb = get_supabase()
    if not sb:
        return ""
    conv_id = str(uuid_lib.uuid4())
    try:
        sb.table("od_conversations").insert({"id": conv_id}).execute()
        return conv_id
    except Exception:
        return ""


def list_conversations(limit: int = 50) -> list[dict]:
    """Return list of {id, created_at} sorted by created_at desc."""
    sb = get_supabase()
    if not sb:
        return []
    try:
        r = sb.table("od_conversations").select("id, created_at").order("created_at", desc=True).limit(limit).execute()
        return [{"id": row["id"], "created_at": row["created_at"]} for row in (r.data or [])]
    except Exception:
        return []


def _sanitize_timestamp(ts: object) -> datetime:
    """Normalize a DB created_at value to timezone-aware UTC datetime. Handles datetime, ISO string, or None."""
    if ts is None:
        return datetime.now(_UTC)
    try:
        if hasattr(ts, "isoformat"):
            dt = ts
        else:
            dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        if getattr(dt, "tzinfo", None) is None:
            dt = dt.replace(tzinfo=_UTC)
        else:
            dt = dt.astimezone(_UTC)
        return dt
    except Exception:
        return datetime.now(_UTC)


def _parse_message_row(row: dict) -> tuple:
    """Parse a single od_messages row to (role, message, dt)."""
    dt = _sanitize_timestamp(row.get("created_at"))
    return (row.get("role", ""), row.get("message", ""), dt)


def load_messages(conversation_id: str) -> list[tuple]:
    """Return list of (author_display_name, message, datetime) in chronological order. role column stores the display name of who posted."""
    sb = get_supabase()
    if not sb:
        return []
    try:
        r = sb.table("od_messages").select("created_at, role, message").eq("conversation_id", conversation_id).order("created_at").execute()
        return [_parse_message_row(row) for row in (r.data or [])]
    except Exception:
        return []


def load_messages_since(conversation_id: str, after_created_at: datetime) -> list[tuple]:
    """Return list of (author_display_name, message, datetime) for messages with created_at > after_created_at, in chronological order.
    Use for incremental reload; after_created_at should be UTC."""
    sb = get_supabase()
    if not sb or not conversation_id:
        return []
    try:
        ts_str = after_created_at.astimezone(_UTC).isoformat()
        r = (
            sb.table("od_messages")
            .select("created_at, role, message")
            .eq("conversation_id", conversation_id)
            .gt("created_at", ts_str)
            .order("created_at")
            .execute()
        )
        return [_parse_message_row(row) for row in (r.data or [])]
    except Exception:
        return []


def conversation_exists(conversation_id: str) -> bool:
    """Return True if the conversation exists in od_conversations."""
    sb = get_supabase()
    if not sb or not conversation_id:
        return False
    try:
        r = sb.table("od_conversations").select("id").eq("id", conversation_id).limit(1).execute()
        return bool(r.data and len(r.data) > 0)
    except Exception:
        return False


def delete_conversation(conversation_id: str) -> bool:
    """Delete a conversation and its messages (messages cascade). Returns True if deleted."""
    sb = get_supabase()
    if not sb or not conversation_id:
        return False
    try:
        sb.table("od_conversations").delete().eq("id", conversation_id).execute()
        return True
    except Exception:
        return False


def persist_message(conversation_id: str, author_display_name: str, message: str) -> None:
    """Insert one message into od_messages. role column = author_display_name (human or agent name).
       DB sets created_at via default now(). No-op if Supabase unavailable."""
    sb = get_supabase()
    if not sb or not conversation_id:
        return
    try:
        payload = {"conversation_id": conversation_id, "role": author_display_name, "message": message}
        # Omit created_at so Postgres DEFAULT now() is used (DB server time); avoids wrong time when app host clock is stale
        sb.table("od_messages").insert(payload).execute()
    except Exception:
        pass
