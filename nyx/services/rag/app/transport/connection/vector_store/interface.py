# ==========================================================================================
# Author: Pablo González García.
# Created: 24/12/2025
# Last edited: 24/12/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from abc import ABC
from abc import abstractmethod
from typing import List, Dict, Any


# ==============================
# INTERFACES
# ==============================

class IVectorStoreConnection(ABC):
    """
    Interface for vector store connections.
    """

    # ---- Methods ---- #

    @abstractmethod
    async def connect(self) -> None:
        """
        Connects to the vector store.
        """
        raise NotImplementedError
    
    @abstractmethod
    async def close(self) -> None:
        """
        Closes connection with the vector store.
        """
        raise NotImplementedError
    
    @abstractmethod
    async def collection_exist(
        self,
        collection_name:str
    ) -> bool:
        """
        Checks if the collection exists.

        Args:
            collection_name (str): Collection to check.

        Returns:
            bool: True if the collection exist, False otherwise.
        """
        raise NotImplementedError
    
    @abstractmethod
    async def create_collection(
        self,
        collection_name:str,
        vector_dim:int,
        distance:str
    ) -> None:
        """
        Creates a new collection in the vector store.

        Args:
            collection_name (str): Name of the collection to create.
            vector_dim (int): Size of the vectors.
            distance (str): Distance applied for similarity.
        """
        raise NotImplementedError
    
    @abstractmethod
    async def delete_collection(
        self,
        collection_name:str
    ) -> None:
        """
        Delete a collection from the vector store.
        
        Args:
            collection_name (str): Name of the collection to delete.
        """
        raise NotImplementedError
    
    @abstractmethod
    async def upsert(
        self,
        collection_name:str,
        vectors:List[Dict[str, Any]]
    ) -> None:
        """
        Insert or update vector int the collection from the vector store.

        Args:
            collection_name (str): Name of the collection to insert.
            vectors (List[Dict[str,Any]]): List with data to insert. 
        """
        raise NotImplementedError
    
    @abstractmethod
    async def delete(
        self,
        collection_name:str,
        ids:List[str]
    ) -> None:
        """
        Delete vectors from the collection in the vector store.

        Args:
            collection_name (str): Name of the collection.
            ids (List[str]): List with ids to delete.
        """
        raise NotImplementedError