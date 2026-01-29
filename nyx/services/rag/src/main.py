# ==========================================================================================
# Author: Pablo González García.
# Created: 28/01/2026
# Last edited: 29/01/2026
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from contextlib import asynccontextmanager

# External:
from fastapi import FastAPI

# Internal:
from infrastructure.vector_store.client.interfaces import IAsyncClient
from infrastructure.vector_store.client.qdrant import AsyncQdrantClient
from interfaces.api.v1.api import api_router


# ==============================
# FUNCTIONS
# ==============================

@asynccontextmanager
async def lifespan(app:FastAPI):
    """
    
    """

    # ---- Initialization ---- #

    # Initializes vector_store client.
    vs_client:IAsyncClient = AsyncQdrantClient(
        url="qdrant",
        http_port=6333,
        grpc_port=6334,
        api_key="change-me",
        prefer_grpc=True
    )

    # Add common instances to application state.
    app.state.vs_client = vs_client
    
    yield

    # Close vector_store connection.
    await vs_client.close()


# ==============================
# MAIN
# ==============================

# Initializes FastAPI application and include v1 routes.
app:FastAPI = FastAPI(
    lifespan=lifespan
)

app.include_router(
    router=api_router,
    prefix="/api/v1"
)