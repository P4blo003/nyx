# ==========================================================================================
# Author: Pablo González García.
# Created: 27/01/2026
# Last edited: 27/01/2026
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from abc import ABC
from typing import Dict

# Internal:
from infrastructure.triton.config import Config as TritonConfig
from infrastructure.triton.client.interfaces import IAsyncClient
from infrastructure.triton.client.grpc import GrpcAsyncClient


# ==============================
# CLASSES
# ==============================

class AsyncClientBuilder:
    """
    Factory class responsible for building asynchronous Triton clients based on application
    configuration.
    """

    # ---- Methods ---- #

    @classmethod
    async def build(
        cls,
        config:TritonConfig,
        client_class:str
    ) -> Dict[str, IAsyncClient]:
        """
        Build Triton clients from configuration.

        Args:
            config (TritonConfig): Triton configuration containing server definitions.
            client_class (str): Identifier of the client implementation to build.

        Returns:
            response (Dict[str, IAsyncClient]): If the provided client class is not supported.
        """

        # Variable to save clients.
        clients:Dict[str, IAsyncClient] = {}

        # Iterate over configured Triton servers.
        for key, value in config.servers.items():
            match client_class:
                case 'grpc': 
                    # Create gRPC-base asynchronous Triton client.
                    clients[key] = GrpcAsyncClient(
                        server_name=key,
                        host=value.host,
                        port=value.grpc_port
                    ) 
                
        return clients