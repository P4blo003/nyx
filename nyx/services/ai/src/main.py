# ==========================================================================================
# Author: Pablo González García.
# Created: 23/01/2026
# Last edited: 27/01/2026
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Internal:
from contextlib import asynccontextmanager

# External:
from fastapi import FastAPI

# Internal:
from shared.utilities import yaml
from shared.utilities import log
from application.cache import ICache, InMemoryCache
from application.services.cache_updater import TritonCacheUpdater
from infrastructure.triton.client import ITritonClientManager, TritonClientManager
from infrastructure.triton.config import TritonConfig
from infrastructure.models.model import CachedTritonModel
from interfaces.api.v1.api import api_router


# ==============================
# FUNCTIONS
# ==============================

@asynccontextmanager
async def lifespan(app:FastAPI):
    """
    """

    # Setup application logging.
    log.setup()

    # Load configurations.
    triton_config:TritonConfig = TritonConfig(**yaml.load_data(path="config/triton_config.yaml"))

    # Initializes clients manager.
    client_manager:ITritonClientManager = TritonClientManager(triton_config=triton_config)
    await client_manager.startup()
    # Initializes model cache.
    model_cache:ICache[CachedTritonModel] = InMemoryCache()

    # Load Triton Cache updater.
    triton_cache_updater:TritonCacheUpdater = TritonCacheUpdater(
        cache=model_cache,
        context=client_manager,
        interval=10_000
    )
    triton_cache_updater.start()

    # Save the instances in application state.
    app.state.client_manager=client_manager
    app.state.model_cache = model_cache
    app.state.triton_cache_updater = triton_cache_updater

    yield

    # Clos cache updater.
    await triton_cache_updater.stop()
    # Close active connections.
    await client_manager.shutdown()


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