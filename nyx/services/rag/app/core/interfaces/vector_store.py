# ==========================================================================================
# Author: Pablo González García.
# Created: 17/12/2025
# Last edited: 17/12/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from abc import ABC
from abc import abstractmethod


# ==============================
# INTERFACES
# ==============================

class IVectorStore(ABC):
    """
    Interface for vector store instances.

    Defines the contract for vector database implementations.
    """
    # ---- Methods ---- #

    @abstractmethod
    async def upload(
        self
    ) -> None:
        """
        Insert or update vectorized documents.
        """
        pass

    @abstractmethod
    async def delete(
        self,
        id:str
    ) -> None:
        """
        Delete document by id.
        """
        pass

    @abstractmethod
    async def get(
        self,
        id:str
    ) -> None:
        """
        Retrieve a document by id.
        """