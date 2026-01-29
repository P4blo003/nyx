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
from interfaces.api.v1.routes.documents import router as documents_router


# ==============================
# MAIN
# ==============================

# Initializes the api router and includes all application endpoints.
api_router:APIRouter = APIRouter()
api_router.include_router(router=documents_router)