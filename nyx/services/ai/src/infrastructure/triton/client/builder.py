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
from infrastructure.triton.config import TritonConfig
from infrastructure.triton.client.base import AsyncClient
from infrastructure.triton.client.grpc import GrpcAsyncClient


# ==============================
# FUNCTIONS
# ==============================

def build(
    config:TritonConfig,
    client_class:str
) -> Dict[str, AsyncClient]:
    """
    Build Triton clients from configuration.

    Args:
        config (TritonConfig): Triton configuration containing server definitions.
        client_class (str): Identifier of the client implementation to build.

    Returns:
        response (Dict[str, IAsyncClient]): If the provided client class is not supported.
    """

    # Variable to save clients.
    clients:Dict[str, AsyncClient] = {}

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