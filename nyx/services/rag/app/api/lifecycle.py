# ==========================================================================================
# Author: Pablo González García.
# Created: 24/12/2025
# Last edited: 24/12/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
import os
from contextlib import asynccontextmanager
from typing import Dict
# External:
from fastapi import FastAPI
# Internal:
from transport.connection.vector_store.qdrant import AsyncQdrantConnection


# ==============================
# FUNCTIONS
# ==============================

@asynccontextmanager
async def lifespan(app:FastAPI):
    """
    FastAPI lifespan. Manage initialization and shutdown.
    """

    # ---- Initialization ---- #

    # Initializes vector store connection.
    app.state.vector_store_connection = AsyncQdrantConnection(
        host=os.environ.get("QDRANT_HOST", "localhost"),
        port=int(os.environ.get("QDRANT_PORT", "6333"))
    )


    # ---- Starting ---- #

    # Starts vector store connection.
    await app.state.vector_store_connection.connect()

    yield


    # ---- Shutdown ---- #

    # Close vector store connection.
    await app.state.vector_store_connection.close()