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
from typing import BinaryIO

# Internal:
from domain.entities.document import DocumentEmbedding


# ==============================
# INTERFACES
# ==============================

class IVectorStore(ABC):
    """
    
    """

    # ---- Methods ---- #

    @abstractmethod
    async def upsert_vectors(self, documents:List[DocumentEmbedding]) -> None:
        """
        """
        pass