# ==========================================================================================
# Author: Pablo González García.
# Created: 11/12/2025
# Last edited: 16/12/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
import signal
import asyncio
import logging
from contextlib import asynccontextmanager
# External:
from fastapi import FastAPI
# Internal:
from core.logging.facade import Log
from core.logging.handler import StandardLogHandler
from api import dependencies
from api.routes.chat import router as chat_router


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
    # Initialize dependencies.
    dependencies.setup_dependencies()
    # Add event bus to state.
    app.state.global_event_bus = dependencies.GLOBAL_EVENT_BUS

    # Gets current loop.
    loop:asyncio.AbstractEventLoop = asyncio.get_running_loop()
    # Gets original handler.
    orig_handler = signal.getsignal(signalnum=signal.SIGINT)

    # Function to handle Ctrl+C.
    def handle_sigint(signum:int, frame) -> None:
        # Prints information.
        logging.debug("Ctrl+C detected. Shutting down the server ...")

        # Notify close to active sessions.
        loop.create_task(app.state.global_event_bus.publish("app.close"))

        # Close loggers synchronously.
        Log.shutdown()

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