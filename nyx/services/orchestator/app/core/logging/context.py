# ==========================================================================================
# Author: Pablo González García.
# Created: 16/12/2025
# Last edited: 16/12/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from typing import Dict
from contextvars import ContextVar


# ==============================
# VARIABLES
# ==============================

_session_id_ctx: ContextVar[str|None] = ContextVar("session_id", default=None)
_request_id_ctx: ContextVar[str|None] = ContextVar("request_id", default=None)
_user_id_ctx: ContextVar[str|None] = ContextVar("user_id", default=None)


# ==============================
# FUNCTIONS
# ==============================

def set_log_context(
    *,
    session_id:str|None = None,
    request_id:str|None = None,
    user_id:str|None = None
) -> None:
    """
    Set logging context for the current asyncio task. Context is
    automatically isolated per task.

    Args:
        session_id (str|None): Optional session identifier.
        request_id (str|None): Optional request or correlation identifier.
        user_id (str|None): Optional user identifier.
    """
    # Sets the properties.
    if session_id is not None: _session_id_ctx.set(session_id)
    if request_id is not None: _request_id_ctx.set(request_id)
    if user_id is not None: _user_id_ctx.set(user_id)

def reset_log_context() -> None:
    """
    Reset all logging context variables.
    """
    _session_id_ctx.set(None)
    _request_id_ctx.set(None)
    _user_id_ctx.set(None)

def _get_log_context() -> Dict[str, str|None]:
    """
    Retrieve current logging context.

    Returns:
        Dictionary containing session_id, request_id, user_id.
    """
    return {
        "session_id":_session_id_ctx.get(),
        "request_id":_request_id_ctx.get(),
        "user_id":_user_id_ctx.get()
    }