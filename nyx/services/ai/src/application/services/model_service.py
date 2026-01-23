# ==========================================================================================
# Author: Pablo González García.
# Created: 23/01/2026
# Last edited: 23/01/2026
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
import asyncio
from typing import Dict, List

# Internal:
from application.cache import ICache
from domain.models.model import TritonModel, CachedTritonModel
from infrastructure.triton.client import ITritonClientManager
from infrastructure.triton.sdk import TritonSdk


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
        model_cache:ICache
    ) -> None:
        """
        Initializes the service.
        """
        
        # Initializes the class properties.
        self._triton_client_manager:ITritonClientManager = triton_client_manager
        self._model_cache:ICache = model_cache

        self._triton_sdk:TritonSdk = TritonSdk()


    # ---- Default ---- #

    async def get_models(self) -> List[TritonModel]:
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
        models:Dict[str, CachedTritonModel] = self._model_cache.get_all() or {}
        # If models is None, it could be for this reasons:
        #   1. The Triton Inference Server is empty.
        #   2. The cache does not load models from Triton.
        # In any case, to prevent this issue, if models is `None`, the program
        # should retrieve models from Triton Inference Servers.
        if not models:
            
            tasks = [self._triton_sdk.get_models(client=client)
                    for client in self._triton_client_manager.get_clients().values()]
            # Await for all tasks results.
            results:List[Dict[str, List[TritonModel]]] = await asyncio.gather(*tasks)
            
            # Get models from updated cache.
            models = self._model_cache.get_all() or {}

        print(models)

        return [
            TritonModel(
                name=model.name,
                version=model.version,
                metadata=model.metadata,
                config=model.config
            )
            for __, model in models.items()]