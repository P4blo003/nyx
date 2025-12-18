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
from typing import List
# External:
import numpy as np
from dto.models.vector_store import VectorDocument
from dto.models.retrieval import RetrievalResult


# ==============================
# INTERFACES
# ==============================

class IVectorStore(ABC):
    """
    Interface for vector database operations.

    Abstracts storage and retrieval of embed chunks, allowing for
    swapping vector DB implementations without affecting business logic.
    """
    # ---- Methods ---- #

    @abstractmethod
    async def upsert(
        self,
        vector_doc:VectorDocument
    ) -> None:
        """
        Store an embedded chunk in the store.

        Args:
            vector_doc (VectorDocument): Document containing chunk data and embedding.
        """
        pass
    
    @abstractmethod
    async def flush(
        self,
        vector_docs:List[VectorDocument]
    ) -> None:
        """
        Store an array of embedded chunks in Qdrant vector store.

        Args:
            vector_docs (List[VectorDocument]): List of documents containing chunk data and embedding.
        """
        pass

    @abstractmethod
    async def search(
        self,
        query_embedding:np.typing.NDArray[np.float32],
        limit:int,
    ) -> List:
        """
        Search index using a query embedding vector.

        Args:
            query_embedding (np.typing.NDArray[np.float32]): Embedding vector for the query.
            limit (int): Max number of results.

        Returns:
            List[RetrievalResult]: Lis of retrieval results ranked by similarity.
        """
        pass

class IVectorStoreController(ABC):
    """
    
    """
    # ---- Methods ---- #

    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the controller.

        This method should subscribe to relevant events on the
        EventBus.
        """
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """
        Cleanup controller resources.

        Called during system shutdown.
        """
        pass

    @abstractmethod
    async def add_document(
        self,
        doc:VectorDocument
    ) -> None:
        """
        Add a document to the vector store, with optional batching.

        Args:
            doc (VectorDocument): Document to add.
        """
        pass

    @abstractmethod
    async def add_documents(
        self,
        docs:List[VectorDocument]
    ) -> None:
        """
        Add multiple documents to the vector store.

        Args:
            docs (List[VectorDocument]): List of documents to add.
        """
        pass

    @abstractmethod
    async def search(
        self,
        query_embedding:np.typing.NDArray[np.float32],
        limit:int,
    ) -> List[RetrievalResult]:
        """
        Search the vector store using a query embedding.

        Args:
            query_embedding (np.typing.NDArray[np.float32]): Embedding vector for the query.
            limit (int): Max number of results.

        Returns:
            List[RetrievalResult]: List of retrieval results ranked by similarity.
        """
        pass