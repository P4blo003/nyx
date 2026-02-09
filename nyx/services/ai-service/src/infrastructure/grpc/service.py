# ==========================================================================================
# Author: Pablo González García.
# Created: 09/02/2026
# Last edited: 09/02/2026
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
import asyncio
import time
import logging
from typing import Optional

# External:
import grpc
from grpc import aio

# Internal:
from domain.ports.client import IAsyncClient, IAsyncClientService
import infrastructure.grpc.generated.ai_service_pb2 as pb2
import infrastructure.grpc.generated.ai_service_pb2_grpc as pb2_grpc
from infrastructure.triton.client.base import TritonAsyncClient


# ==============================
# CLASSES
# ==============================

class AIServiceServicer(pb2_grpc.AIServiceServicer):
    """
    
    """

    # ---- Default ---- #

    def __init__(
        self,
        client_service:IAsyncClientService[TritonAsyncClient],
        
    ) -> None:
        """
        
        """

        # Initializes the class properties.
        self._client_service = client_service

        self._log:logging.Logger = logging.getLogger("AIServiceServicer")


    # ---- Methods ---- #

    async def load_model(
        self,
        request:pb2.LoadModelRequest,
        context
    ):
        """
        
        """

        try:
            client:Optional[TritonAsyncClient] = self._client_service.get_client(key=request.server)

            if client is None: raise ValueError(f"Client for server '{request.server}' not found.")

            await client.load_model(model_name=request.name, model_version=request.version)

        except Exception as e:

            self._log.error(f"Error loading model '{request.name}:{request.version}' on server '{request.server}': {str(e)}")
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            return pb2.ModelStatus(
                name=request.name,
                version=request.version,
                state=pb2.ModelStatus.ERROR,
                message=str(e)
            )
        
    async def unload_model(
        self,
        request:pb2.UnloadModelRequest,
        context
    ):
        """
        """

        try:
            client:Optional[TritonAsyncClient] = self._client_service.get_client(key=request.server)

            if client is None: raise ValueError(f"Client for server '{request.server}' not found.")

            await client.unload_model(model_name=request.name)

        except Exception as e:

            self._log.error(f"Error unloading model '{request.name}:{request.version}' on server '{request.server}': {str(e)}")
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            return pb2.ModelStatus(
                name=request.name,
                version=request.version,
                state=pb2.ModelStatus.ERROR,
                message=str(e)
            )