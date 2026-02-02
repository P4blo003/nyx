# ==========================================================================================
# Author: Pablo González García.
# Created: 23/01/2026
# Last edited: 27/01/2026
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
from infrastructure.cache.redis_cache import RedisCache


# ==============================
# FUNCTIONS
# ==============================

@asynccontextmanager
async def lifespan(app:FastAPI):
    """
    
    """
    
    # ---- Startup ---- #

    model_cache:RedisCache = RedisCache(
        host=os.environ.get("REDIS_HOST", "localhost"),
        port=int(os.environ.get("REDIS_PORT", "6379")),
        decode_responses=os.environ.get("REDIS_DECODE_RESPONSES", "True").lower() == "True".lower()
    )

    # Add classes to application context.
    app.state.model_cache = model_cache

    yield


    # ---- Shutdown ---- #

    await model_cache.close()


# ==============================
# MAIN
# ==============================

# Initializes FastAPI application and include v1 routes.
app:FastAPI = FastAPI(
    lifespan=lifespan
)
