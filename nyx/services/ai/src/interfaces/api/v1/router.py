# ==========================================================================================
# Author: Pablo González García.
# Created: 30/01/2026
# Last edited: 30/01/2026
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# External:
from fastapi import APIRouter

# Internal:
from interfaces.api.v1.routes import inference, models


# ==============================
# MAIN
# ==============================

try:
    # Initializes the router.
    router:APIRouter = APIRouter(prefix="/v1")

    # Add routes.
    router.include_router(router=models.router)
    router.include_router(router=inference.router)

# If an error occurs.
except Exception as ex:

    raise