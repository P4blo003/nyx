# ==========================================================================================
# Author: Pablo González García.
# Created: 27/01/2026
# Last edited: 27/01/2026
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from typing import Optional, List

# External:
import numpy as np

# Internal:
from application.services.cache_service import CacheService
from domain.models.triton.model import CachedTritonModel
from infrastructure.triton.client.interfaces import IClientManager, IAsyncClient
from infrastructure.triton.sdk import SDK as TritonSdk
from interfaces.api.v1.models.inference_request import InferenceInput
from interfaces.api.v1.models.inference_response import InferenceResponse, InferenceOutput


# ==============================
# CLASSES
# ==============================

class InferenceService:
    """
    Application service responsible for executing inference requests against
    Triton Inference Server models.
    """

    # ---- Default ---- #

    def __init__(
        self,
        client_manager:IClientManager,
        cache_service:CacheService
    ) -> None:
        """
        Initializes the service.

        Args:
            client_manager (IClientManager): Manager responsible for providing Triton clients.
            cache_service (CacheService): Service that maintains the cache of available
                Triton models.
        """
        
        # Initializes the class properties.
        self._client_manager:IClientManager = client_manager
        self._cache_service:CacheService = cache_service

        self._triton_sdk:TritonSdk = TritonSdk()

    
    # ---- Methods ---- #

    async def make_infer(
        self,
        inputs:List[InferenceInput]
    ) -> InferenceResponse:
        """
        Execute inference request using a cached Triton model.

        Args:
            inputs (List[InferenceInput]): List of inference inputs containing the data
                to be processed by the model.

        Returns:
            response (InferenceResponse): Response containing embeddings associated with each input.
        """

        # Retrieve the model metadata from cache.
        model:Optional[CachedTritonModel] = await self._cache_service.get_cache().get("bge_m3_ensemble") or None
        if model is None: raise ValueError(f"Unable to find model in cache.")

        # Resolve the Triton client for the model's server.
        client:Optional[IAsyncClient] = self._client_manager.get_client(key=model.server)
        if client is None: raise ValueError(f"Unable to find client for server '{model.server}'")

        # Execute the inference request through the Triton SDK.
        embeddings:Optional[np.ndarray] = await self._triton_sdk.make_infer(
            client=client,
            model_name=model.model.name,
            inputs=inputs
        )
        if embeddings is None: raise ValueError(f"Unable to make inference. Calculated embeddings are None for {len(inputs)} texts.")

        # Map embeddings to API response format.
        return InferenceResponse(
            embeddings=[InferenceOutput(
                id=inputs[index].id,
                embedding=embedding
            ) for index, embedding in enumerate(embeddings.tolist())]
        )