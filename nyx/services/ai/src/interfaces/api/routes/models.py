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
from typing import List, Optional

# External:
from fastapi import APIRouter, Depends, status
from fastapi import HTTPException

# Internal:
from interfaces.api.dependencies import get_model_service
from application.services.model_service import IModelService
from domain.models.triton_model import TritonModel


# ==============================
# VARIABLES
# ==============================

router:APIRouter = APIRouter(prefix="/models")
logger:Logger = logging.getLogger(__name__)


# ==============================
# ENDPOINTS
# ==============================

@router.get(
    path="/",
    status_code=status.HTTP_200_OK,
    summary="List all available models",
    description="Returns all Triton models with metadata and state"
)
async def get_models(
    service:IModelService = Depends(get_model_service)
) -> List[TritonModel]:
    """
    """

    try:

        # Returns available models.
        return await service.get_models()

    # If an error occurs.
    except Exception as ex:

        # Prints the error.
        logger.error(f"Unable to list models: {ex}")
        # Returns 500 to client.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unable to list models.")
    
@router.get(
    path="/{model_name}",
    status_code=status.HTTP_200_OK,
    summary="List all available models",
    description="Returns all Triton models with metadata and state"
)
async def get_model_data(
    model_name:str,
    service:IModelService = Depends(get_model_service)
) -> Optional[TritonModel]:
    """
    """

    try:

        # Returns available models.
        return await service.get_model_data(model_name=model_name)

    # If an error occurs.
    except Exception as ex:

        # Prints the error.
        logger.error(f"Unable to list models: {ex}")
        # Returns 500 to client.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unable to list models.")
    
@router.post(
    path="/{model_name}/load",
    status_code=status.HTTP_200_OK,
    summary="Load the specified model.",
    description="Returns all Triton models with metadata and state"
)
async def load_model(
    model_name:str,
    service:IModelService = Depends(get_model_service)
) -> None:
    """
    """

    try:

        # Load the model.
        await service.load_model(model_name=model_name)

    # If an error occurs.
    except Exception as ex:

        # Prints the error.
        logger.error(f"Unable to load model '{model_name}': {ex}")
        # Returns 500 to client.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unable to load model '{model_name}'.")
    
@router.post(
    path="/{model_name}/unload",
    status_code=status.HTTP_200_OK,
    summary="Unload the specified model.",
    description="Returns all Triton models with metadata and state"
)
async def unload_model(
    model_name:str,
    service:IModelService = Depends(get_model_service)
) -> None:
    """
    """

    try:

        # Load the model.
        await service.unload_model(model_name=model_name)

    # If an error occurs.
    except Exception as ex:

        # Prints the error.
        logger.error(f"Unable to unload model '{model_name}': {ex}")
        # Returns 500 to client.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unable to unload model '{model_name}'.")