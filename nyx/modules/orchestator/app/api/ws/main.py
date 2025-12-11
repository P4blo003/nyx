# ==========================================================================================
# Author: Pablo González García.
# Created: 11/12/2025
# Last edited: 11/12/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from contextlib import asynccontextmanager
# External:
from fastapi import FastAPI
# Internal:
from api.ws.routes.chat import router as chat_router


# ==============================
# FUNCTIONS
# ==============================

@asynccontextmanager
async def lifespan(app:FastAPI):
    """
    Lifespan context manaer for startup and shutdown events.

    Yields:
        None.
    """
    # Prints information.
    print(f"Starting application ...")

    # Returns.
    yield

    # Prints information.
    print(f"App shutdown ...")

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