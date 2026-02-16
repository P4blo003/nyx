# ==========================================================================================
# Author: Pablo González García.
# Created: 20/01/2026
# Last edited: 03/02/2026
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from abc import ABC
from abc import abstractmethod
from typing import List

# Internal:
from domain.ports.client import ISyncClient, IAsyncClient


# ==============================
# INTERFACES
# ==============================

class AIServiceSyncClient(ISyncClient, ABC):
    """
    """

    pass

class AIServiceAsyncClient(IAsyncClient, ABC):
    """
    Abstract client for AI Service inference operations.
    Defines the business contract independent of transport protocol.
    """

    # ---- Methods ---- #

    @abstractmethod
    async def make_infer(self, task:str, texts:List[str]) -> List[List[float]]:
        """
        Sends a batch of texts for inference and returns embeddings.
        """
        ...