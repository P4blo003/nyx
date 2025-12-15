# ==========================================================================================
# Author: Pablo González García.
# Created: 11/12/2025
# Last edited: 11/12/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
import signal
import asyncio
from contextlib import asynccontextmanager
# External:
from fastapi import FastAPI
# Internal:
from api.ws.routes.chat import router as chat_router
from api.ws.dependencies import GLOBAL_EVENT_BUS as global_event_bus
from core.config import logger


# ==============================
# FUNCTIONS
# ==============================

@asynccontextmanager
async def lifespan(app:FastAPI):
    """
    Lifespan context manager for startup and shutdown events.

    Yields:
        None.
    """
    # Gets current loop.
    loop:asyncio.AbstractEventLoop = asyncio.get_running_loop()
    # Gets original handler.
    orig_handler = signal.getsignal(signalnum=signal.SIGINT)

    # Add event bus to state.
    app.state.global_event_bus = global_event_bus

    # Function to handle Ctrl+C.
    def handle_sigint(signum:int, frame) -> None:
        # Notify close.
        loop.create_task(app.state.global_event_bus.publish("app.close"))

        # Checks if there is an original signal handler.
        if callable(orig_handler):
            # Calls the handler.
            orig_handler(signum, frame)

    # Add handler.
    signal.signal(signalnum=signal.SIGINT, handler=handle_sigint)

    # Returns.
    yield


# ==============================
# CONSTANTS
# ==============================

# Creates the app.
app:FastAPI = FastAPI(
    title="Nyx-Assistant",
    lifespan=lifespan
)

# Includes the routes.
app.include_router(chat_router)