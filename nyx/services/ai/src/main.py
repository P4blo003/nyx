# ==========================================================================================
# Author: Pablo González García.
# Created: 23/01/2026
# Last edited: 27/01/2026
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from contextlib import asynccontextmanager
from typing import Dict

# External:
from fastapi import FastAPI

# Internal:
from application.cache.base import ICache
from application.cache.in_memory import InMemoryCache
from application.services.cache_service import CacheService
from domain.models.triton.model import CachedTritonModel
from infrastructure.triton.config import Config as TritonConfig
from infrastructure.triton.client import builder as TritonClientBuilder
from infrastructure.triton.client.interfaces import IAsyncClient, IClientManager
from infrastructure.triton.client.manager import TritonAsyncClientManager
from interfaces.api.v1.api import api_router
from shared.utilities import yaml
from shared.utilities import log


# ==============================
# FUNCTIONS
# ==============================

@asynccontextmanager
async def lifespan(app:FastAPI):
    """
    """

    # ---- Initialization ---- #

    # Initializes application logger.
    log.setup()

    # Loads application configuration.
    triton_config:TritonConfig = TritonConfig(**yaml.load_data(path="config/triton_config.yaml"))

    # Initializes application instances. This instances are shared between all
    # application clients.
    clients:Dict[str, IAsyncClient] = await TritonClientBuilder.AsyncClientBuilder.build(
        config=triton_config,
        client_class='grpc'
    )
    client_manager:IClientManager[IAsyncClient] = TritonAsyncClientManager(clients=clients)
    model_cache:ICache[CachedTritonModel] = InMemoryCache()
    cache_service:CacheService = CacheService(
        cache=model_cache,
        client_manager=client_manager
    )

    # Add common instances to application state.
    app.state.client_manager = client_manager
    app.state.cache_service = cache_service

    # Starts all instances.
    await client_manager.start()
    cache_service.start()

    yield


    # ---- Shutdown ---- #

    # Gracefully shutdown for all instances.
    await cache_service.stop()
    await client_manager.stop()


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