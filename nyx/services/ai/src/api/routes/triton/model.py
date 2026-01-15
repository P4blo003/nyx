# ==========================================================================================
# Author: Pablo González García.
# Created: 15/01/2025
# Last edited: 15/01/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
import logging
from typing import List

# External:
from fastapi import APIRouter, status
from fastapi import Request
from fastapi import HTTPException
from tritonclient.grpc import InferenceServerClient

# Internal:
from dto.model import TritonModel


# ==============================
# CONSTANTS
# ==============================

_router:APIRouter = APIRouter(
    prefix="/models"
)
_logger:logging.Logger = logging.getLogger(__name__)


# ==============================
# ROUTES
# ==============================

@_router.get(
    "/",
    response_model=List[TritonModel],
    status_code=status.HTTP_200_OK
)
async def get_models(req:Request):
    """
    List all models available in all Triton Servers.

    Args:
        req (Request): FastAPI request object with app state.

    Returns:
        List[TritonModel]: List of all Triton models across servers.

    Raises:
        HTTPException: 500 if unable to retrieve models from Triton Servers.
    """

    # Response to return.
    response:List[TritonModel] = []

    # Try-Except to manage errors.
    try:

        # Iterate over all clients. 
        for key, client in req.app.state.triton_clients.items():

            # Try-Except to manage errors.
            try:

                # Gets model repository. 
                repo_index = client.get_model_repository_index(as_json=True)
                # Checks if there isn't data.
                if repo_index is None: continue

                # Iterate over model list.
                for model in repo_index.get("models", []):

                    # Append de model information to the response. 
                    response.append(TritonModel(
                        server=key,
                        name=model.get("name", None),
                        version=model.get("version", None),
                        state=model.get("state", None) 
                    ))

            # If an error occurs. 
            except Exception as ex: 
                
                # Prints error.
                _logger.error(f"Error fetching models from server '{key}': {ex}")

        
        return response
    
    # If it's an HTTPException.
    except HTTPException: raise

    # If an error occurs. 
    except Exception as ex: 
        
        # Prints error.
        # Prints error.
        _logger.error(f"Internal error while listing models: {ex}")
        # Raises HTTP error.
        raise HTTPException(status_code=500, detail="Internal server error while getting available models.")