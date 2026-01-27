# ==========================================================================================
# Author: Pablo González García.
# Created: 23/01/2026
# Last edited: 27/01/2026
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from typing import Optional, Dict, List

# Internal:
from application.cache import ICache
from infrastructure.triton.client import ITritonClientManager
from infrastructure.triton.sdk import TritonSdk
from infrastructure.models.model import CachedTritonModel
from interfaces.api.v1.models.models_responses import ModelSummary


# ==============================
# CLASSES
# ==============================

class ModelService:
    """
    """

    # ---- Default ---- #

    def __init__(
        self,
        triton_client_manager:ITritonClientManager,
        model_cache:ICache[CachedTritonModel]
    ) -> None:
        """
        Initializes the service.
        """
        
        # Initializes the class properties.
        self._triton_client_manager:ITritonClientManager = triton_client_manager
        self._model_cache:ICache[CachedTritonModel] = model_cache

        self._triton_sdk:TritonSdk = TritonSdk()


    # ---- Default ---- #

    async def get_models(self) -> List[ModelSummary]:
        """
        Retrieves all Triton models for external clients.

        This method first attempts to load models from the internal cache.
        If the cache is empty, it should query Triton Inference Servers
        and populate the cache.

        Returns:
            List[TritonModel]: A list of TritonModel instances containing model
                information.
        """

        # Load models from cache.
        models:Dict[str, CachedTritonModel] = await self._model_cache.get_all() or {}
        # If models is None, it could be for this reasons:
        #   1. The Triton Inference Server is empty.
        #   2. The cache does not load models from Triton.
        # In any case, to prevent this issue, if models is `None`, the program
        # should retrieve models from Triton Inference Servers.
        if not models:
        
            # TODO: Update cache.

            pass
        
        # Load models from cache again.
        models = await self._model_cache.get_all() or {}
        
        return [ModelSummary(
            name=name,
            version=model.model.version,
            server=model.server
        ) for name, model in models.items()]
    
    async def load_model(
        self,
        model_name:str
    ) -> None:
        """
        
        """

        # Get model from cache.
        model:Optional[CachedTritonModel] = await self._model_cache.get(model_name)
        # Checks if the model si None.
        if model is None: raise ValueError(f"Unable to find model '{model_name}' in cache.")
        
        # Gets server assigned to the model.
        client = self._triton_client_manager.get_client(model.server)
        # Check it the server assigned to the model is None.
        if client is None: raise ValueError(f"Unable to find the assigned client for server '{model.server}' of model '{model_name}'.")
        
        # Awaits to load model.
        await self._triton_sdk.load_model(
            client=client,
            model_name=model_name
        )

    async def unload_model(
        self,
        model_name:str
    ) -> None:
        """
        
        """

        # Get model from cache.
        model:Optional[CachedTritonModel] = await self._model_cache.get(model_name)
        # Checks if the model si None.
        if model is None: raise ValueError(f"Unable to find model '{model_name}' in cache.")
        
        # Gets server assigned to the model.
        client = self._triton_client_manager.get_client(model.server)
        # Check it the server assigned to the model is None.
        if client is None: raise ValueError(f"Unable to find the assigned client for server '{model.server}' of model '{model_name}'.")
        
        # Awaits to load model.
        await self._triton_sdk.unload_model(
            client=client,
            model_name=model_name
        )