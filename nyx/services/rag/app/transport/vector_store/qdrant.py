# ==========================================================================================
# Author: Pablo González García.
# Created: 17/12/2025
# Last edited: 17/12/2025
# ==========================================================================================

# ==============================
# IMPORTS
# ==============================

# External:
import asyncio
from typing import Any, Dict, List
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct, ScoredPoint
# External:
import numpy as np
# Internal:
from core.interfaces.vector_store import IVectorStore
from dto.models.retrieval import RetrievalResult
from dto.models.vector_store import VectorDocument


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

    async def upsert(
        self,
        vector_doc: VectorDocument
    ) -> None:
        """
        Store an embedded chunk in Qdrant vector store.

        Args:
            vector_doc (VectorDocument): Document containing chunk data and embedding.
        """
        # Creates the payload.
        payload:Dict[str, Any] = {
            "document_id": vector_doc.document_id,
            "content": vector_doc.content,
            **vector_doc.metadata
        }

        # Creates the point for Qdrant vector store.
        point:PointStruct = PointStruct(
            id=vector_doc.chunk_id,
            vector=vector_doc.embedding.tolist(),
            payload=payload
        )

        # Index into the collection.
        await self._client.upsert(
            collection_name=self._collection,
            points=[point]
        )

    async def flush(
        self,
        vector_docs: List[VectorDocument]
    ) -> None:
        """
        Store an array of embedded chunks in Qdrant vector store.

        Args:
            vector_docs (List[VectorDocument]): List of documents containing chunk data and embedding.
        """
        # Variable to hold the points.
        points:List[PointStruct] = []

        # Iterates over the array.
        for vector_doc in vector_docs:
            # Creates the payload and add to the array.
            payload:Dict[str, Any] = {
                "document_id": vector_doc.document_id,
                "content": vector_doc.content,
                **vector_doc.metadata
            }
            points.append(PointStruct(
                id=vector_doc.chunk_id,
                vector=vector_doc.embedding.tolist(),
                payload=payload
            ))
        
        # Index into the collection.
        await self._client.upsert(
            collection_name=self._collection,
            points=points
        )

    async def search(
        self, 
        query_embedding: np.typing.NDArray[np.float32], 
        limit: int
    ) -> List[RetrievalResult]:
        """
        Search index using a query embedding vector.

        Args:
            query_embedding (np.typing.NDArray[np.float32]): Embedding vector for the query.
            limit (int): Max number of results.

        Returns:
            List[RetrievalResult]: Lis of retrieval results ranked by similarity.
        """
        # Gets the results from Qdrant.
        response = await self._client.query_points(
            collection_name=self._collection,
            query_embedding=query_embedding.tolist(),
            limit=limit
        )

        # Variable to hold the results.
        results:List[RetrievalResult] = []
        # Parse the results.
        for result in response.points:  
            
            # If there is no payload, skip.
            if result.payload is None: continue

            # Add the result.      
            results.append(RetrievalResult(
                chunk_id=str(result.id),
                doc_id=str(result.payload.get("document_id", None)),
                content=str(result.payload.get("content", None)),
                score=result.score if result.score is not None else 0.0,
                metadata={k:v for k, v in result.payload.items() if k not in ("document_id", "content")}
            ))
        
        return results