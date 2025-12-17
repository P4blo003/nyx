# ==========================================================================================
# Author: Pablo González García.
# Created: 17/12/2025
# Last edited: 17/12/2025
# ==========================================================================================

# ==============================
# IMPORTS
# ==============================

# External:
from qdrant_client import AsyncQdrantClient
# Internal:
from core.interfaces.vector_database import IVectorStore


# ==============================
# CLASSES
# ==============================

class QdrantVectorStore(IVectorStore):
    """
    Qdrant implementation of IVectorStore.
    """
    # ---- Default ---- #

    def __init__(
        self,
        client:AsyncQdrantClient,
        collection:str
    ) -> None:
        """
        Initialize the qdrant vector store.
        """
        # Initialize the properties.
        self._client:AsyncQdrantClient = client
        self._collection:str = collection


    # ---- Methods ---- #