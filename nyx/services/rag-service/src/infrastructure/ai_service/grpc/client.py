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
import grpc.aio

# Internal:
from infrastructure.ai_service.client.base import AIServiceAsyncClient
from infrastructure.ai_service.grpc.generated import ai_service_pb2 as pb2
from infrastructure.ai_service.grpc.generated import ai_service_pb2_grpc as pb2_grpc


# ==============================
# CLASSES
# ==============================

class AIServiceGrpcAsyncClient(AIServiceAsyncClient):
    """
    gRPC implementation of the AI Service client.
    """

    # ---- Default ---- #

    def __init__(
        self,
        host:str,
        port:int
    ) -> None:
        """
        Initializes the gRPC async client.
        """

        # Initializes the class properties.
        self._host:str = host
        self._port:int = port

        self._channel:Optional[grpc.aio.Channel] = None
        self._stub:Optional[pb2_grpc.AIServiceStub] = None


    # ---- Methods ---- #

    def get_server_url(self) -> str:
        """
        """

        return f"{self._host}:{self._port}"

    async def connect(self) -> None:
        """
        """

        self._channel = grpc.aio.insecure_channel(self.get_server_url())
        self._stub = pb2_grpc.AIServiceStub(channel=self._channel)

    async def disconnect(self) -> None:
        """
        """

        if self._channel:
            await self._channel.close()

    async def make_infer(self, task:str, texts:List[str]) -> List[List[float]]:
        """
        """

        if self._stub is None:
            raise RuntimeError("Client not connected. Call connect() first.")

        request:pb2.InferenceRequest = pb2.InferenceRequest(
            task=task,
            text_batch=pb2.TextBatch(content=texts)
        )

        response:pb2.InferenceResponse = await self._stub.make_infer(request)

        return [list(vector.values) for vector in response.embedding_batch.vectors]