# ==========================================================================================
# Author: Pablo González García.
# Created: 03/02/2026
# Last edited: 03/02/2026
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
from domain.ports.client import IAsyncClientService
import infrastructure.grpc.generated.ai_service_pb2_grpc as pb2_grpc
from infrastructure.grpc.service import AIServiceServicer
from infrastructure.grpc.interceptor import RequestLogInterceptor
from infrastructure.triton.config.task import TritonTask


# ==============================
# CLASSES
# ==============================

class GrpcServer:
    """
    
    """

    # ---- Default ---- #

    def __init__(
        self,
        client_service:IAsyncClientService,
        tasks:Dict[str, TritonTask],
        host:str = "[::]",
        port:int = 8002,
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

        self._client_service:IAsyncClientService = client_service
        pb2_grpc.add_AIServiceServicer_to_server(servicer=AIServiceServicer(
            client_service=self._client_service,tasks=tasks),
            server=self._server
        )



    # ---- Methods ---- #

    async def start(self) -> None:
        """
        
        """

        await self._client_service.startup()
        await self._server.start()

        self._log.info(f"Running server at {self._url}")

        await self._server.wait_for_termination()


    async def stop(self, grace:int=5) -> None:
        """

        """

        await self._server.stop(grace=5)

        self._log.info(f"Server stopped")

        await self._client_service.shutdown()