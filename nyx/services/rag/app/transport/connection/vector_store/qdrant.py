# ==========================================================================================
# Author: Pablo González García.
# Created: 24/12/2025
# Last edited: 24/12/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
import os
from typing import List, Dict, Any
# External:
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.models import PointStruct, PointIdsList
# Internal:
from errors.vector_store import CollectionExist, CollectionNotExist
from transport.connection.vector_store.interface import IVectorStoreConnection


# ==============================
# ENUMS
# ==============================

# References between str and Qdrant distances.
QDRANT_DISTANCES:Dict[str, Distance] = {
    'cosine': Distance.COSINE,
    'dot' : Distance.DOT,
    'euclid': Distance.EUCLID,
    'manhattan': Distance.MANHATTAN
}

# ==============================
# CLASSES
# ==============================

class AsyncQdrantConnection(IVectorStoreConnection):
    """
    Qdrant vector store asynchronous connection.
    """

    # ---- Default ---- #

    def __init__(
        self,
        host:str,
        port:int,
        timeout:int = 1
    ) -> None:
        """
        Initialize Qdrant connection.
        """

        # Initializes properties.
        self._host:str = host
        self._port:int = port
        self._timeout:int = timeout
        
        self._connection:AsyncQdrantClient|None = None


    # ---- Methods ----- #

    async def connect(self) -> None:
        """
        Connects to the Qdrant vector store.
        """
        
        # Checks if the connection is already initialized.
        if self._connection is not None: raise ValueError("Qdrant connection is already initialized.")

        # Initializes the connection.
        self._connection = AsyncQdrantClient(
            url=f"https://{self._host}:{self._port}",
            api_key=os.environ.get("QDRANT_API_KEY", ""),
            verify=False
        )

        # Try-Except to manage errors.
        try:

            # Makes basic call to test if the connection is Ok.
            await self._connection.get_collections()

        # If an unexpected error occurs.
        except Exception as ex:

            # Raises the error.
            raise ConnectionError(f"Unable to connect with Qdrant vector store: {ex}")

    async def close(self) -> None:
        """
        Close connection with the Qdrant vector store.
        """
        
        # Checks if the connection is not initialized.
        if self._connection is None: raise ValueError("Qdrant connection is not initialized.")

        # Close the connection.
        await self._connection.close()
        self._connection = None

    async def collection_exist(
        self, 
        collection_name: str
    ) -> bool:
        """
        Checks if the collection exists.

        Args:
            collection_name (str): Collection to check.

        Returns:
            bool: True if the collection exist, False otherwise.
        """
        # Checks if the connection is not initialized.
        if self._connection is None: raise ValueError("Qdrant connection is not initialized.")
        
        # Checks if exist.
        return await self._connection.collection_exists(collection_name=collection_name)
    
    async def create_collection(
        self,
        collection_name: str,
        vector_dim: int,
        distance: str
    ) -> None:
        """
        Creates a new collection in Qdrant the vector store.

        Args:
            collection_name (str): Name of the collection to create.
            vector_dim (int): Size of the vectors.
            distance (str): Distance applied for similarity.
        """

        # Checks if the connection is not initialized.
        if self._connection is None: raise ValueError("Qdrant connection is not initialized.")

        # Checks if the collection already exist.
        if await self.collection_exist(collection_name=collection_name): raise CollectionExist(f"Collection {collection_name} already exists.")

        # Create a new collection.
        await self._connection.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_dim,
                distance=QDRANT_DISTANCES.get(distance, Distance.COSINE)
            )
        )

    async def delete_collection(
        self,
        collection_name: str
    ) -> None:
        """
        Delete a collection from the vector store.
        
        Args:
            collection_name (str): Name of the collection to delete.
        """

        # Checks if the connection is not initialized.
        if self._connection is None: raise ValueError("Qdrant connection is not initialized.")

        # Checks if the collection doesn't exist.
        if not await self.collection_exist(collection_name=collection_name): raise CollectionNotExist(f"Collection {collection_name} doesn't exists.")

        # Delete the collection.
        await self._connection.delete_collection(collection_name=collection_name)

    async def upsert(
        self,
        collection_name: str,
        vectors: List[Dict[str, Any]]
    ) -> None:
        """
        Insert or update vector int the collection from the Qdrant vector store.

        Args:
            collection_name (str): Name of the collection to insert.
            vectors (List[Dict[str,Any]]): List with data to insert. 
        """
        
        # Checks if the connection is not initialized.
        if self._connection is None: raise ValueError("Qdrant connection is not initialized.")

        # Checks if the collection doesn't exist.
        if not await self.collection_exist(collection_name=collection_name): raise CollectionNotExist(f"Collection {collection_name} doesn't exists.")

        # Generates PointStruct array to upsert all in one.
        points:List[PointStruct] = [PointStruct(
            id = element["id"],
            vector= element["vector"],
            payload= element.get("payload", None)
        ) for element in vectors]

        # Upsert the array of points.
        await self._connection.upsert(
            collection_name=collection_name,
            points=points
        )

    async def delete(
        self,
        collection_name: str,
        ids: List[str]
    ) -> None:
        """
        Delete vectors from the collection in the vector store.

        Args:
            collection_name (str): Name of the collection.
            ids (List[str]): List with ids to delete.
        """

        # Checks if the connection is not initialized.
        if self._connection is None: raise ValueError("Qdrant connection is not initialized.")

        # Checks if the collection doesn't exist.
        if not await self.collection_exist(collection_name=collection_name): raise CollectionNotExist(f"Collection {collection_name} doesn't exists.")

        # Delete the elements from the Qdrant collection.
        await self._connection.delete(
            collection_name=collection_name,
            points_selector=PointIdsList(
                points=[id for id in ids]
            )
        )