"""Identifiers. Every id that crosses the API or names a file uses one safe format."""

import re
import secrets

# Letters, digits, '_' and '-', 1-64 chars, not starting with a separator. Safe as a file name and
# in a URL path, so a client-supplied run id can never point outside traces/.
ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$"
_ID_RE = re.compile(ID_PATTERN)


def new_id(prefix: str) -> str:
    """A fresh random id such as ``task_3f9c2a71b0d4e815``."""
    return f"{prefix}_{secrets.token_hex(8)}"


def is_valid_id(value: str) -> bool:
    return bool(_ID_RE.fullmatch(value))


def thread_id_for(user_id: str, session_id: str) -> str:
    """The LangGraph thread of one user's session. The user is part of the thread's identity, so
    the same session id from another user is another thread."""
    if not (is_valid_id(user_id) and is_valid_id(session_id)):
        raise ValueError(f"invalid user or session id: {user_id!r}, {session_id!r}")
    return f"{user_id}:{session_id}"


def session_of(thread_id: str) -> str:
    return thread_id.split(":", 1)[1] if ":" in thread_id else thread_id
