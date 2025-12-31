# ==========================================================================================
# Author: Pablo González García.
# Created: 24/12/2025
# Last edited: 31/12/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
import os
import asyncio
from typing import List, Dict, Any
# External:
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import CollectionsResponse
from qdrant_client.models import Distance, VectorParams
from qdrant_client.models import PointStruct, PointIdsList
# Internal:
from transport.connection.vector_store.interface import IVectorStoreConnection
from dto.models.collection import VectorInfo, CollectionInfo


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

    # ---- Constructor ---- #

    def __init__(
        self,
        host:str,
        port:int,
        timeout:int = 1,
        max_concurrent_calls_per_request:int = 10
    ) -> None:
        """
        Initialize Qdrant connection.

        Args:
            host (str): Qdrant host.
            port (int): Qdrant port.
            timeout (int, optional): Qdrant timeout.
            max_concurrent_calls_per_request (int): Max concurrent calls per request.
        """

        # Initializes class properties.
        self._host = host
        self._port = port
        self._timeout:int = timeout
        self._max_concurrent_calls_per_request:int = max_concurrent_calls_per_request
        
        self._conn:AsyncQdrantClient|None = None

    
    # ---- Methods ---- #

    async def _fetch_collection(
        self,
        name:str,
        semaphore:asyncio.Semaphore
    ) -> CollectionInfo|None:
        """
        Fetches collection information.
        
        Args:
            name (str): Name of the collection.
            semaphore (asyncio.Semaphore): Semaphore to limit concurrent calls.

        Returns:
            CollectionInfo|None: Collection information or None if not found.
        """
        
        # Checks if the connection is not initialized.
        if self._conn is None: raise ValueError("Qdrant connection is not initialized.")

        # Use semaphore to limit concurrent calls.
        async with semaphore:

            # Try-Except to manage errors.
            try:

                # Gets collection information.
                info = await self._conn.get_collection(name)
                # Gets vector information.
                vector = info.config.params.vectors

                # Checks if vector information is None.
                if vector is None: return None

                # If vector is a dict (multiple vector fields).
                if isinstance(vector, dict):
                    # Creates vector info list.
                    vector_info:List[VectorInfo] = [
                        VectorInfo(
                            name=name,
                            dim=v.size,
                            distance=v.distance
                        ) for name, v in vector.items()
                    ]
                
                # If vector is a VectorParams (single vector field).
                else:
                    vector_info:List[VectorInfo] = [
                        VectorInfo(
                            name="default",
                            dim=vector.size,
                            distance=vector.distance
                        )
                    ]

                return CollectionInfo(
                    name=name,
                    vector_info=vector_info
                )

            # If an unexpected error occurs.
            except Exception as ex:

                return None

    async def connect(self) -> None:
        """
        Connects to the Qdrant vector store.
        """
        
        # Checks if the connection is already initialized.
        if self._conn is not None: raise ValueError("Qdrant connection is already initialized.")

        # Initializes the connection.
        self._conn = AsyncQdrantClient(
            url=f"https://{self._host}:{self._port}",
            api_key=os.environ.get("QDRANT_API_KEY", ""),
            verify=False
        )

        # Try-Except to manage errors.
        try:

            # Makes basic call to test if the connection is Ok.
            await self._conn.get_collections()

        # If an unexpected error occurs.
        except Exception as ex:

            # Raises the error.
            raise ConnectionError(f"Unable to connect with Qdrant vector store: {ex}")
    
    async def close(self) -> None:
        """
        Close connection with the Qdrant vector store.
        """
        
        # Checks if the connection is not initialized.
        if self._conn is None: raise ValueError("Qdrant connection is not initialized.")

        # Close the connection.
        await self._conn.close()
        self._conn = None

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
        if self._conn is None: raise ValueError("Qdrant connection is not initialized.")
        
        # Checks if exist.
        return await self._conn.collection_exists(collection_name=collection_name)
    
    async def get_collections(self) -> List[CollectionInfo]:
        """
        Gets all collections from the vector store.

        Returns:
            List[CollectionInfo]: List with all collection information.
        """

        # Checks if the connection is not initialized.
        if self._conn is None: raise ValueError("Qdrant connection is not initialized.")

        # Gets collections.
        collections: CollectionsResponse = await self._conn.get_collections()

        # Variable to manage calls per request.
        semaphore:asyncio.Semaphore = asyncio.Semaphore(self._max_concurrent_calls_per_request) 

        # Generate tasks to get collection information.
        tasks:List = [self._fetch_collection(name=c.name, semaphore=semaphore) for c in collections.collections]
        # Awaits to get all results.
        results = await asyncio.gather(*tasks)

        return [r for r in results if r is not None]

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
        if self._conn is None: raise ValueError("Qdrant connection is not initialized.")

        # Checks if the collection already exist.
        if await self.collection_exist(collection_name=collection_name): raise RuntimeError(f"Collection {collection_name} already exists.")

        # Create a new collection.
        await self._conn.create_collection(
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
        if self._conn is None: raise ValueError("Qdrant connection is not initialized.")

        # Checks if the collection doesn't exist.
        if not await self.collection_exist(collection_name=collection_name): raise RuntimeError(f"Collection {collection_name} doesn't exists.")

        # Delete the collection.
        await self._conn.delete_collection(collection_name=collection_name)

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
        if self._conn is None: raise ValueError("Qdrant connection is not initialized.")

        # Checks if the collection doesn't exist.
        if not await self.collection_exist(collection_name=collection_name): raise RuntimeError(f"Collection {collection_name} doesn't exists.")

        # Generates PointStruct array to upsert all in one.
        points:List[PointStruct] = [PointStruct(
            id = element["id"],
            vector= element["vector"],
            payload= element.get("payload", None)
        ) for element in vectors]

        # Upsert the array of points.
        await self._conn.upsert(
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
        if self._conn is None: raise ValueError("Qdrant connection is not initialized.")

        # Checks if the collection doesn't exist.
        if not await self.collection_exist(collection_name=collection_name): raise RuntimeError(f"Collection {collection_name} doesn't exists.")

        # Delete the elements from the Qdrant collection.
        await self._conn.delete(
            collection_name=collection_name,
            points_selector=PointIdsList(
                points=[id for id in ids]
            )
        )