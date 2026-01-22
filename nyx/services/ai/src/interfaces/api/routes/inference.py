# ==========================================================================================
# Author: Pablo González García.
# Created: 20/01/2025
# Last edited: 21/01/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
import logging
from logging import Logger
from typing import List

# External:
from fastapi import APIRouter, Depends, status
from fastapi import HTTPException

# Internal:
from interfaces.api.dependencies import get_inference_service
from interfaces.api.schemas.inference_request import InferenceRequest
from application.services.inference_service import IInferenceService


# ==============================
# VARIABLES
# ==============================

router:APIRouter = APIRouter(prefix="/inference")
logger:Logger = logging.getLogger(__name__)


# ==============================
# ENDPOINTS
# ============================== 

@router.post(
    path="/{task}",
    status_code=status.HTTP_200_OK
)
async def infer(
    task:str,
    request:InferenceRequest,
    service:IInferenceService = Depends(get_inference_service)
):
    """
    """

    try:

        # Make inference.
        return await service.make_infer(texts=request.texts)

    # If an error occurs.
    except Exception as ex:

        # Prints the error.
        logger.error(f"Unable to make inference '{task}': {ex}")
        # Returns 500 to client.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unable to make inference '{task}'.")