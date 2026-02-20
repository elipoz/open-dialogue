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
    """Insert a new row in od_conversations and return its id (UUID string)."""
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


def load_messages(conversation_id: str) -> list[tuple]:
    """Return list of (author_display_name, message, datetime) in chronological order. role column stores the display name of who posted."""
    sb = get_supabase()
    if not sb:
        return []
    try:
        r = sb.table("od_messages").select("created_at, role, message").eq("conversation_id", conversation_id).order("created_at").execute()
        out = []
        for row in (r.data or []):
            ts = row.get("created_at")
            if ts:
                try:
                    if hasattr(ts, "isoformat"):
                        dt = ts
                    else:
                        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                except Exception:
                    dt = datetime.now()
            else:
                dt = datetime.now()
            out.append((row.get("role", ""), row.get("message", ""), dt))
        return out
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


def persist_message(conversation_id: str, author_display_name: str, message: str, created_at: datetime | None = None) -> None:
    """Insert one message into od_messages. role column = author_display_name (human or agent name).
       The created_at is stored in UTC so all users have consistent timestamps.
       No-op if Supabase unavailable."""
    sb = get_supabase()
    if not sb or not conversation_id:
        return
    try:
        payload = {"conversation_id": conversation_id, "role": author_display_name, "message": message}
        if created_at is not None:
            # Always store in UTC: normalize so Postgres timestamptz is unambiguous (avoids mixed UTC/PST from different app hosts)
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=_UTC)
            else:
                created_at = created_at.astimezone(_UTC)
            payload["created_at"] = created_at.isoformat()
        sb.table("od_messages").insert(payload).execute()
    except Exception:
        pass
