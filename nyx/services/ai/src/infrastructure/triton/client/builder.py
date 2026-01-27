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
from abc import abstractmethod
from typing import Any, Optional, Dict

# External:
from tritonclient.grpc.aio import InferenceServerClient

# Internal:
from infrastructure.triton.config import Config as TritonConfig
from infrastructure.triton.client.interfaces import IAsyncClient
from infrastructure.triton.client.grpc import GrpcAsyncClient


# ==============================
# CLASSES
# ==============================

class AsyncClientBuilder:
    """
    
    """

    # ---- Methods ---- #

    @classmethod
    async def build(
        cls,
        config:TritonConfig,
        client_class:str
    ) -> Dict[str, IAsyncClient]:
        """
        
        """

        # Variable to save clients.
        clients:Dict[str, IAsyncClient] = {}

        for key, value in config.servers.items():
            match client_class:
                case 'grpc': 
                    clients[key] = GrpcAsyncClient(
                        server_name=key,
                        host=value.host,
                        port=value.grpc_port
                    ) 
                
        return clients