
# ==========================================================================================
# Author: Pablo González García.
# Created: 16/12/2025
# Last edited: 16/12/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Internal:
from fastapi import Request
from core.logging import context as ctx


# ==============================
# FUNCTIONS
# ==============================

async def logging_middleware(request:Request, call_next):
    """
    Middleware to set logging context for each HTTP request.

    This middleware extracts session, request, and user identifiers
    from HTTP headers and sets them in the async logging context.
    The context is automatically isolated per request/task, allowing
    concurrent requests to maintain separate logging information.

    Args:
        request (Request): Incoming FastAPI request.
        call_next (Callable): The next request handler in the middleware chain.

    Returns:
        Response: The HTTP response returned by the downstream handler.
    """
    # Gets request context.
    ctx.set_log_context(
        session_id=request.headers.get("X-Session-ID"),
        request_id=request.headers.get("X-Request-ID"),
        user_id=request.headers.get("X-User-ID")
    )
    
    # Try-Except to manage errors.
    try:
        # Gets response.
        return await call_next(request)

    # Executes finally.
    finally:
        # Reset context.
        ctx.reset_log_context()
        