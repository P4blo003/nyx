# ==========================================================================================
# Author: Pablo González García.
# Created: 16/02/2026
# Last edited: 16/02/2026
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from typing import List, Optional

# External:
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Condition,
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    ScoredPoint as QdrantScoredPoint
)

# Internal:
from infrastructure.qdrant.client.base import (
    QdrantAsyncClient,
    VectorPoint,
    ScoredPoint,
    Payload
)


# ==============================
# CLASSES
# ==============================

class QdrantGrpcAsyncClient(QdrantAsyncClient):
    """
    Qdrant client implementation using the gRPC transport.
    """

    # ---- Default ---- #

    def __init__(
        self,
        host:str,
        port:int,
        grpc_port:int = 6334
    ) -> None:
        """
        Initializes the Qdrant async client.
        """

        # Initializes the class properties.
        self._host:str = host
        self._port:int = port
        self._grpc_port:int = grpc_port

        self._client:Optional[AsyncQdrantClient] = None


    # ---- Methods ---- #

    def get_server_url(self) -> str:
        """
        """

        return f"{self._host}:{self._port}"

    async def connect(self) -> None:
        """
        """

        self._client = AsyncQdrantClient(
            host=self._host,
            port=self._port,
            grpc_port=self._grpc_port,
            prefer_grpc=True
        )

    async def disconnect(self) -> None:
        """
        """

        if self._client:
            await self._client.close()


    # ---- Collections ---- #

    async def ensure_collection(self, collection:str, vector_size:int) -> None:
        """
        """

        if self._client is None:
            raise RuntimeError("Client not connected. Call connect() first.")

        if not await self._client.collection_exists(collection_name=collection):
            await self._client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )

    async def delete_collection(self, collection:str) -> None:
        """
        """

        if self._client is None:
            raise RuntimeError("Client not connected. Call connect() first.")

        await self._client.delete_collection(collection_name=collection)


    # ---- Points ---- #

    async def upsert(self, collection:str, points:List[VectorPoint]) -> None:
        """
        """

        if self._client is None:
            raise RuntimeError("Client not connected. Call connect() first.")

        await self._client.upsert(
            collection_name=collection,
            points=[
                PointStruct(
                    id=point.id,
                    vector=point.vector,
                    payload=point.payload
                )
                for point in points
            ]
        )

    async def delete_by_filter(self, collection:str, filter:Payload) -> None:
        """
        """

        if self._client is None:
            raise RuntimeError("Client not connected. Call connect() first.")

        conditions:List[Condition] = [
            FieldCondition(key=key, match=MatchValue(value=value))
            for key, value in filter.items()
        ]

        await self._client.delete(
            collection_name=collection,
            points_selector=Filter(must=conditions)
        )


    # ---- Search ---- #

    async def search(
        self,
        collection:str,
        vector:List[float],
        limit:int,
        score_threshold:Optional[float] = None,
        filter:Optional[Payload] = None
    ) -> List[ScoredPoint]:
        """
        """

        if self._client is None:
            raise RuntimeError("Client not connected. Call connect() first.")

        query_filter:Optional[Filter] = None
        if filter:
            conditions:List[Condition] = [
                FieldCondition(key=key, match=MatchValue(value=value))
                for key, value in filter.items()
            ]
            query_filter = Filter(must=conditions)

        response = await self._client.query_points(
            collection_name=collection,
            query=vector,
            limit=limit,
            score_threshold=score_threshold,
            query_filter=query_filter
        )
        results:List[QdrantScoredPoint] = response.points

        return [
            ScoredPoint(
                id=str(hit.id),
                score=hit.score,
                payload=hit.payload or {}
            )
            for hit in results
        ]
