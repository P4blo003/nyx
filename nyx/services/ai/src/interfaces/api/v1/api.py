# ==========================================================================================
# Author: Pablo González García.
# Created: 23/01/2026
# Last edited: 23/01/2026
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# External:
from fastapi import APIRouter

# Internal:
from interfaces.api.v1.routes.models import router as models_router
from interfaces.api.v1.routes.inference import router as inference_router


# ==============================
# MAIN
# ==============================

# Initializes the api router and includes all application endpoints.
api_router:APIRouter = APIRouter()
api_router.include_router(router=models_router)
api_router.include_router(router=inference_router)