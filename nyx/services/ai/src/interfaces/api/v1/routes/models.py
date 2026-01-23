# ==========================================================================================
# Author: Pablo González García.
# Created: 23/01/2026
# Last edited: 23/01/2026
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from typing import List
from logging import Logger
from logging import getLogger

# External:
from fastapi import APIRouter, status
from fastapi import Depends
from fastapi import HTTPException

# Internal:
from application.services.model_service import ModelService
from domain.models.model import TritonModel
from interfaces.api.v1.dependencies.injection import get_model_service


# ==============================
# MAIN
# ==============================

router:APIRouter = APIRouter(prefix="/models")
log:Logger = getLogger(__name__)


# ==============================
# ENDPOINTS
# ==============================

@router.get(
    path="/",
    status_code=status.HTTP_200_OK
)
async def get_models(
    service:ModelService = Depends(get_model_service)
) -> List[TritonModel]:
    """
    
    """

    try:
        
        # Await for service response.
        return await service.get_models()

    # If an error occurs.
    except Exception as ex:

        # Prints the error.
        log.error(f"Unable to list Triton models: {ex}")
        # Returns status 500 to client.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to list Triton models."
        )