# ==========================================================================================
# Author: Pablo González García.
# Created: 20/01/2025
# Last edited: 20/01/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from logging import Logger
from logging import getLogger

# External:
from fastapi import APIRouter, HTTPException
from fastapi import status

# Internal:
from application.services.inference import InferenceManager
from interfaces.api.schemas.inference_request import InferenceRequest


# ==============================
# ENDPOINTS
# ==============================

router:APIRouter = APIRouter(prefix="/inference")
logger:Logger = getLogger(__name__)

@router.post(
    path="/",
    status_code=status.HTTP_200_OK
)
async def inference(
    request:InferenceRequest
) -> None:
    """
    """
    
    try:
        
        pass

    # If an unexpected error occurs.
    except Exception as ex:
        
        # Prints the error.
        logger.error(f"Unable to make '{request.task}' inference: {ex}")
        # Returns error to client.
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Unable to make '{request.task}' inference'.")

@router.post(
    path="/stream",
    status_code=status.HTTP_200_OK
)
async def inference_stream(
    request:InferenceRequest
) -> None:
    """
    """
    
    try:
        
        pass

    # If an unexpected error occurs.
    except Exception as ex:
        
        # Prints the error.
        logger.error(f"Unable to make '{request.task}' stream-inference: {ex}")
        # Returns error to client.
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Unable to make '{request.task}' stream-inference.")