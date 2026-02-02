# ==========================================================================================
# Author: Pablo González García.
# Created: 23/01/2026
# Last edited: 02/02/2026
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from typing import List

# Internal:
from infrastructure.triton.client.service import AsyncClientService
from interfaces.api.v1.schemas.models_response import ModelSummary


# ==============================
# CLASSES
# ==============================

class ModelService:
    """
    Application service responsible for managing Triton model lifecycle and exposing
    model information to external clients.
    """

    # ---- Default ---- #

    def __init__(
        self,
        client_service:AsyncClientService
    ) -> None:
        """
        Initializes the service.
        """

        # Initializes the class properties.
        self._client_service:AsyncClientService = client_service


    # ---- Methods ---- #

    async def get_models(self) -> List[ModelSummary]:
        """
        Retrieves all Triton models for external clients.

        This method first attempts to load models from the cache.
        If the cache is empty, it should query Triton Inference Servers
        and populate the cache.

        Returns:
            List[ModelSummary]: A list of ModelSummary instances containing
                model information.
        """

        return []