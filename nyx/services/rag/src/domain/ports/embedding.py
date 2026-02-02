# ==========================================================================================
# Author: Pablo González García.
# Created: 02/02/2026
# Last edited: 02/02/2026
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from abc import ABC
from abc import abstractmethod
from typing import List


# ==============================
# INTERFACES
# ==============================

class IEmbeddingService(ABC):
    """
    
    """

    # ---- Methods ---- #

    @abstractmethod
    async def get_embeddings(self, texts:List[str]) -> List[List[float]]:
        """
        
        """
        pass