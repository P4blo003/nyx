# ==========================================================================================
# Author: Pablo González García.
# Created: 16/02/2026
# Last edited: 16/02/2026
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
import logging
from typing import Dict, List

# External:
from grpc import aio

# Internal:
from application.services.worker_service import WorkerService
from infrastructure.ai_service.client.base import AIServiceAsyncClient
from infrastructure.grpc.interceptor import RequestLogInterceptor
import infrastructure.grpc.generated.rag_service_pb2_grpc as pb2_grpc
from infrastructure.grpc.service import RagServiceServicer
from infrastructure.qdrant.client.base import QdrantAsyncClient


# ==============================
# CLASSES
# ==============================

class GrpcServer:
    """
    
    """

    # ---- Default ---- #

    def __init__(
        self,
        worker_service:WorkerService,
        ai_client:AIServiceAsyncClient,
        qdrant_client:QdrantAsyncClient,
        host:str = "[::]",
        port:int = 8001,
    ) -> None:
        """
        Initializes the server.
        """

        # Initializes the class properties.
        self._host:str = host
        self._port:int = port
        self._url:str = f"{self._host}:{self._port}"

        self._log:logging.Logger = logging.getLogger("GrpcServer")

        self._interceptors:List[aio.ServerInterceptor] = [
            RequestLogInterceptor(log=self._log)
        ]

        self._server:aio.Server = aio.server(interceptors=self._interceptors)
        self._server.add_insecure_port(address=self._url)

        self._worker_service:WorkerService = worker_service
        self._ai_client:AIServiceAsyncClient = ai_client
        self._qdrant_client:QdrantAsyncClient = qdrant_client
        pb2_grpc.add_RAGServiceServicer_to_server(
            servicer=RagServiceServicer(
                worker_service=self._worker_service,
                ai_client=self._ai_client,
                qdrant_client=self._qdrant_client
            ),
            server=self._server
        )


    # ---- Methods ---- #

    async def start(self) -> None:
        """

        """

        await self._ai_client.connect()
        self._log.info(f"Connected to AI Service at {self._ai_client.get_server_url()}")

        await self._qdrant_client.connect()
        self._log.info(f"Connected to Qdrant at {self._qdrant_client.get_server_url()}")

        await self._worker_service.startup()

        await self._server.start()

        self._log.info(f"Running server at {self._url}")

        await self._server.wait_for_termination()


    async def stop(self, grace:int=5) -> None:
        """

        """

        await self._server.stop(grace=grace)

        self._log.info(f"Server stopped")

        await self._worker_service.shutdown()
        await self._ai_client.disconnect()
        await self._qdrant_client.disconnect()