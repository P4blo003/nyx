# ==========================================================================================
# Author: Pablo González García.
# Created: 13/12/2025
# Last edited: 13/12/2025
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
from api.middleware.logging import logging_middleware
from api.dependencies import cleanup
from core.logging.facade import Log
from core.logging.handler import StandardLogHandler



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
    # Initialize the loggers.
    Log.init(
        handlers=[StandardLogHandler(logger_name="app")],
        queue_size=10,
        num_workers=2
    )

    # Gets current loop.
    loop:asyncio.AbstractEventLoop = asyncio.get_running_loop()
    # Gets original handler.
    orig_handler = signal.getsignal(signalnum=signal.SIGINT)


    # Function to handle Ctrl+C.
    def handle_sigint(signum:int, frame) -> None:
        # Prints information.
        logging.debug("Ctrl+C detected. Shuting down the server ...")

        # Cleanup dependencies.
        loop.call_soon_threadsafe(
            lambda: asyncio.create_task(cleanup())
        )
        
        # Close loggers.
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
    title="Rag-Service",
    lifespan=lifespan
)

# Includes middleware.
app.middleware("http")(logging_middleware)