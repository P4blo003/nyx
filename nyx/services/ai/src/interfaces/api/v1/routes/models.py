# ==========================================================================================
# Author: Pablo González García.
# Created: 23/01/2026
# Last edited: 30/01/2026
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
import logging
from typing import List

# External:
from fastapi import APIRouter, HTTPException
from fastapi import status

# Internal:
from interfaces.api.v1.schemas.models_response import ModelSummary
from application.services.model_service import ModelService


# ==============================
# MAIN
# ==============================

try:
    # Initializes router.
    router:APIRouter = APIRouter(prefix="/models")
    # Initializes log.
    log:logging.Logger = logging.Logger(name="ModelsRouter")

# If an error occurs.
except Exception as ex:

    raise


# ==============================
# ENDPOINTS
# ==============================

router.get(
    path="/",
    status_code=status.HTTP_200_OK
)
async def get_models() -> List[ModelSummary]:
    """
    
    """

    try:
        
        # Awaits to retrieve models.
        return await ModelService().get_models()

    # If it's an expected error.
    except HTTPException: raise

    # If an error occurs.
    except Exception as ex:

        # Prints the error.
        log.error(f"Unable to list models: {ex}")
        # Returns status code to client.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error. Unable to list models."
        )