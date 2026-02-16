# ==========================================================================================
# Author: Pablo González García.
# Created: 16/02/2026
# Last edited: 16/02/2026
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from abc import ABC
from abc import abstractmethod
from typing import Any, Dict, List, Optional

# Internal:
from domain.ports.client import IAsyncClient


# ==============================
# TYPES
# ==============================

Payload = Dict[str, Any]


# ==============================
# DATA CLASSES
# ==============================

class VectorPoint:
    """
    Represents a single point to be stored in the vector database.
    """

    __slots__ = ("id", "vector", "payload")

    def __init__(self, id:str, vector:List[float], payload:Payload) -> None:
        self.id:str = id
        self.vector:List[float] = vector
        self.payload:Payload = payload


class ScoredPoint:
    """
    Represents a search result with its similarity score.
    """

    __slots__ = ("id", "score", "payload")

    def __init__(self, id:str, score:float, payload:Payload) -> None:
        self.id:str = id
        self.score:float = score
        self.payload:Payload = payload


# ==============================
# INTERFACES
# ==============================

class QdrantAsyncClient(IAsyncClient, ABC):
    """
    Abstract client for Qdrant vector database operations.
    Defines the business contract independent of the underlying SDK.
    """

    # ---- Collections ---- #

    @abstractmethod
    async def ensure_collection(self, collection:str, vector_size:int) -> None:
        """
        Creates the collection if it does not already exist.
        """
        ...

    @abstractmethod
    async def delete_collection(self, collection:str) -> None:
        """
        Deletes an entire collection and all its points.
        """
        ...

    # ---- Points ---- #

    @abstractmethod
    async def upsert(self, collection:str, points:List[VectorPoint]) -> None:
        """
        Inserts or updates a batch of vector points in a collection.
        """
        ...

    @abstractmethod
    async def delete_by_filter(self, collection:str, filter:Payload) -> None:
        """
        Deletes all points matching the given payload filter.
        """
        ...

    # ---- Search ---- #

    @abstractmethod
    async def search(
        self,
        collection:str,
        vector:List[float],
        limit:int,
        score_threshold:Optional[float] = None,
        filter:Optional[Payload] = None
    ) -> List[ScoredPoint]:
        """
        Performs a vector similarity search and returns scored results.
        """
        ...
