# ==========================================================================================
# Author: Pablo González García.
# Created: 27/01/2026
# Last edited: 27/01/2026
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
from application.services.inference_service import InferenceService
from interfaces.api.v1.dependencies.injection import get_inference_service
from interfaces.api.v1.models.inference_request import InferenceRequest
from interfaces.api.v1.models.inference_response import InferenceResponse


# ==============================
# MAIN
# ==============================

router:APIRouter = APIRouter(prefix="/inference")
log:Logger = getLogger(__name__)


# ==============================
# ENDPOINTS
# ==============================

@router.post(
    path="/{task}",
    status_code=status.HTTP_200_OK
)
async def make_inference(
    task:str,
    req:InferenceRequest,
    service:InferenceService = Depends(get_inference_service),
) -> InferenceResponse:
    """
    """

    try:
    
        # Await for service response.
        return await service.make_infer(inputs=req.inputs)

    # If an error occurs.
    except Exception as ex:

        # Prints the error.
        log.error(f"Unable to make inference for task '{task}': {ex}")

        # Returns status 500 to client.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unable to make inference for task '{task}'."
        )

@router.post(
    path="/{task}/stream",
    status_code=status.HTTP_200_OK
)
async def make_inference_stream(
    task:str,
    service:InferenceService = Depends(get_inference_service)
) -> None:
    """
    """
    
    try:
    
        # Await for service response.
        print("")

    # If an error occurs.
    except Exception as ex:

        # Prints the error.
        log.error(f"Unable make inference (stream) for task '{task}: {ex}")

        # Returns status 500 to client.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unable make inference (stream) for task '{task}."
        )