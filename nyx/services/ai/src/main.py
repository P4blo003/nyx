# ==========================================================================================
# Author: Pablo González García.
# Created: 30/01/2026
# Last edited: 30/01/2026
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from contextlib import asynccontextmanager

# External:
from fastapi import FastAPI

# Internal:
from infrastructure.triton.config import TritonConfig
from infrastructure.triton.client import builder
from infrastructure.triton.client.service import AsyncClientService
from interfaces.api.v1.router import router
from shared.utilities import logging, yaml


# ==============================
# FUNCTIONS
# ==============================

@asynccontextmanager
async def lifespan(app:FastAPI):
    """
    
    """

    # ---- Initialization ---- #

    # Setup logging.
    logging.setup()

    # Load configuration.
    triton_cfg:TritonConfig = TritonConfig(**yaml.load(path="config/triton_config.yaml"))

    # Context services.
    client_service:AsyncClientService = AsyncClientService(clients=builder.build(config=triton_cfg, client_class="grpc"))

    # Add to application context.
    app.state.client_service = client_service

    # ---- Startup ---- #

    await client_service.start()

    yield


    # ---- Shutdown ---- #

    await client_service.stop()


# ==============================
# MAIN
# ==============================

try:
    
    # Initializes FastAPI application.
    app:FastAPI = FastAPI(
        title="AI Service",
        description="AI Service to manage Triton Inference Servers.",
        version="0.0.1",
        debug=True,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan
    )

    # Add application routes.
    app.include_router(
        router=router,
        prefix="/api"
    )

# If an error occurs.
except Exception as ex:

    raise