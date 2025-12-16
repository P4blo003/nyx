# ==========================================================================================
# Author: Pablo González García.
# Created: 16/12/2025
# Last edited: 16/12/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from abc import ABC
from abc import abstractmethod
from typing import Dict, Any


# ==============================
# INTERFACES
# ==============================

class IRagHandler(ABC):
    """
    Interface for RAG handlers.
    """
    # ---- Methods ---- #

    @abstractmethod
    async def process(
        self,
        query:str
    ) -> Dict[str, Any]:
        """
        Process query and return results.

        Args:
            query (str): Query to process.

        Returns:
            Dict[str, Any]: Results.
        """
        pass