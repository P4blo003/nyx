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
from interfaces.api.schemas.model_request import LoadModelRequest, UnloadModelRequest


# ==============================
# ENDPOINTS
# ==============================

router:APIRouter = APIRouter(prefix="/models")
logger:Logger = getLogger(__name__)

@router.post(
    path="/load",
    status_code=status.HTTP_200_OK
)
async def load_model(
    request:LoadModelRequest
) -> None:
    """
    Load a model corresponding to a specific inference task into memory.

    This endpoint triggers the InferenceManager to load the model associated
    with the given task. The mode will be loaded on its assigned Triton server.

    Args:
        request (LoadModelRequest): Contains the task type to load.

    Raises:
        HTTPException: Returns 500 if there are unhandled errors.
    """

    try:
        
        # Await to load the model.
        await InferenceManager.get().load_model_async(task_type=request.task)

    # If an unexpected error occurs.
    except Exception as ex:
        
        # Prints the error.
        logger.error(f"Unable to load model for task '{request.task}': {ex}")
        # Returns error to client.
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Unable to load model for task '{request.task}'.")
    
@router.post(
    path="/unload",
    status_code=status.HTTP_200_OK
)
async def unload_model(
    request:UnloadModelRequest
) -> None:
    """
    Unload a model corresponding to a specific inference task into memory.

    This endpoint triggers the InferenceManager to unload the model associated
    with the given task. The mode will be unloaded on its assigned Triton server.

    Args:
        request (UnloadModelRequest): Contains the task type to unload.

    Raises:
        HTTPException: Returns 500 if there are unhandled errors.
    """

    try:

        # Await to unload the model.
        await InferenceManager.get().unload_model_async(task_type=request.task)


    # If an unexpected error occurs.
    except Exception as ex:
        
        # Prints the error.
        logger.error(f"Unable to unload model for task '{request.task}': {ex}")
        # Returns error to client.
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Unable to unload model for task'{request.task}'.")