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
from dto.model import TritonModel, ModelMetadata


# ==============================
# CONSTANTS
# ==============================

_router:APIRouter = APIRouter(
    prefix="/servers"
)
_logger:logging.Logger = logging.getLogger(__name__)


# ==============================
# ROUTES
# ==============================

# ---- GET ---- #

@_router.get(
    path="/",
    response_model=List[str],
    status_code=status.HTTP_200_OK
)
async def get_servers(req:Request):
    """
    List all available Triton servers.

    Args:
        req (Request): FastAPI request object with app state.
    
    Returns:
        List[TritonModel]: List of all Triton Server names.
    
    Raises:
        HTTPException: 500 if unable to retrieve servers from Triton Server.
    """

    # Try-Except to manage errors.
    try:
        
        # Returns all servers key.
        return list(req.app.state.triton_clients.keys())

    # If it's an HTTPException.
    except HTTPException: raise

    # If an error occurs.
    except Exception as ex:
        
        # Prints error.
        _logger.error(f"Internal error while listing servers: {ex}")
        # Raises HTTP error.
        raise HTTPException(status_code=500, detail="Internal server error while getting available servers.")

@_router.get(
    path="/{server}/models",
    response_model=List[TritonModel],
    status_code=status.HTTP_200_OK
)
async def get_server_models(
    req:Request,
    server:str
):
    """
    List all models for a specific Triton Server.

    Args:
        req (Request): FastAPI request object with app state.
        server (str): Name of the Triton Server.

    Returns:
        List[TritonModel]: List of models for the given server.

    Raises:
        HTTPException: 404 if server client doesn't exist.
        HTTPException: 500 if unable to retrieve models.
    """

    # Response to return.
    response:List[TritonModel] = []
    
    # Try-Except to manage errors.
    try:

        # Gets server client.
        client:InferenceServerClient|None = req.app.state.triton_clients.get(server, None)

        # Checks if the client is not available.
        if client is None: raise HTTPException(status_code=404, detail=f"No client found for server {server}")

        # Gets model repository.
        repo_index = client.get_model_repository_index(as_json=True)

        # Checks if there isn't data.
        if repo_index is None: return response
        
        # Gets given models.
        model_list:List = repo_index.get("models", [])

        # Iterate over model list.
        for model in model_list:

            # Append de model information to the response.
            response.append(TritonModel(
                server=server,
                name=model.get("name", None),
                version=model.get("version", None),
                state=model.get("state", None)
            ))

        return response

    # If it's an HTTPException.
    except HTTPException: raise

    # If an error occurs.
    except Exception as ex:

        # Prints error.
        _logger.error(f"Internal error while listing models from server '{server}': {ex}")
        # Raises HTTP error.
        raise HTTPException(status_code=500, detail=f"Internal server error while getting available models on server {server}")
    
@_router.get(
    path="/{server}/models/{model_name}"
)
async def get_server_model(
    req:Request,
    server:str,
    model_name:str
):
    """
    Retrieve detailed information about a specific model deployed on a given
    Triton Server.

    Args:
        req (Request): FastAPI request object with app state.
        server (str): Name of the Triton Server.
        model_name (str): Name of the model.

    Returns:
        TritonModel: Information about the model.

    Raises:
        HTTPException: 404 if server client doesn't exist.
        HTTPException: 500 if unable to load the model.
    """
    # Try-Except to manage errors.
    try:
        
        # Gets server client.
        client:InferenceServerClient|None = req.app.state.triton_clients.get(server, None)

        # Check if the client is not available.
        if client is None: raise HTTPException(status_code=404, detail=f"No client found for server {server}")

        # Get model repository. 
        repo_index = client.get_model_repository_index(as_json=True)

        # Checks if the repository is empty.
        if repo_index is None: raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Server {server} doesn't contains any model.")

        # Iterate over models in the repository.
        for model in repo_index.get("models", []):

            # Gets the model name.
            name:str|None = model.get("name", None)

            # Checks for the requested model.
            if name == model_name:
                
                # Gets the model state.
                state:str|None = model.get("state", None)

                # Ensure the model is running.
                if state != "READY": raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Model {model_name} at {server} is not running")

                # Retrieve model metadata from Triton.
                metadata = client.get_model_metadata(model_name=model_name, as_json=True)

                # Checks if the metadata is valid.
                if metadata is None or not isinstance(metadata, dict): raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Invalid metadata returned by Triton")

                # Return model metadata. 
                return TritonModel(
                    server=server,
                    name=model.get("name", None),
                    version=model.get("version", None),
                    state=model.get("state", None),
                    metadata=ModelMetadata(**metadata)
                )
            
        # Model not found.
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Model '{model_name}' not found on server '{server}'")

    # If it's an HTTPException.
    except HTTPException: raise

    # If an error occurs.
    except Exception as ex:

        # Prints the error.
        _logger.error(f"Error loading model {model_name} on server {server}: {ex}")
        # Raises HTTP error.
        raise HTTPException(status_code=500, detail=f"Internal server error while getting {model_name} on server {server}")


# ---- PUT ---- #

@_router.put(
    "/{server}/models/{model_name}/load",
    status_code=status.HTTP_200_OK
)
async def load_server_model(
    req:Request,
    server:str,
    model_name:str
):
    """
    Load a specific model on a given Triton Server.

    Args:
        req (Request): FastAPI request object with app state.
        server (str): Name of the Triton Server.
        model_name (str): Name of the model to load.

    Raises:
        HTTPException: 404 if server client doesn't exist.
        HTTPException: 500 if unable to load the model.
    """

    # Try-Except to manage errors.
    try:
        
        # Gets server client.
        client:InferenceServerClient|None = req.app.state.triton_clients.get(server, None)

        # Checks if the client is not available.
        if client is None: raise HTTPException(status_code=404, detail=f"No client found for server {server}")

        # Load the model.
        client.load_model(model_name=model_name)

    # If it's an HTTPException.
    except HTTPException: raise

    # If an error occurs.
    except Exception as ex:

        # Prints the error.
        _logger.error(f"Error loading model {model_name} on server {server}: {ex}")
        # Raises HTTP error.
        raise HTTPException(status_code=500, detail=f"Internal server error while loading {model_name}")

@_router.put(
    "/{server}/models/{model_name}/unload",
    status_code=status.HTTP_200_OK)
async def unload_server_model(
    req:Request,
    server:str,
    model_name:str
):
    """
    Unload a specific model on a given Triton Server.

    Args:
        req (Request): FastAPI request object with app state.
        server (str): Name of the Triton Server.
        model_name (str): Name of the model to unload.

    Raises:
        HTTPException: 404 if server client doesn't exist.
        HTTPException: 500 if unable to load the model.
    """

    # Try-Except to manage errors.
    try:
        
        # Gets server client.
        client:InferenceServerClient|None = req.app.state.triton_clients.get(server, None)

        # Checks if the client is not available.
        if client is None: raise HTTPException(status_code=404, detail=f"No client found for server {server}")

        # Load the model.
        client.unload_model(model_name=model_name)

    # If it's an HTTPException.
    except HTTPException: raise
    
    # If an error occurs.
    except Exception as ex:

        # Prints the error.
        _logger.error(f"Error unloading model {model_name} on server {server}: {ex}")
        # Raises HTTP error.
        raise HTTPException(status_code=500, detail=f"Internal server error while unloading {model_name}")