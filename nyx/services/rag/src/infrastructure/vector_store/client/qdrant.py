# ==========================================================================================
# Author: Pablo González García.
# Created: 29/01/2026
# Last edited: 29/01/2026
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from abc import ABC
from abc import abstractmethod
from typing import Any, Optional, Dict, List
from typing import Generic, TypeVar

# External:
from qdrant_client import AsyncQdrantClient as QdrantClient

# Internal:
from infrastructure.vector_store.client.interfaces import IAsyncClient


# ==============================
# TYPES
# ==============================

T = TypeVar("T")


# ==============================
# CLASSES
# ==============================

class AsyncQdrantClient(IAsyncClient):
    """
    
    """

    # ---- Default ---- #

    def __init__(
        self,
        url:str,
        http_port:int,
        grpc_port:int,
        api_key:str,
        prefer_grpc:bool = False
    ) -> None:
        """
        Initializes the Qdrant client.

        Args:
            url (str): Qdrant hostname or IP.
            http_port (int): The HTTP endpoint port.
            grpc_port (int): The gRPC endpoint port.
            api_key (str): API key for Qdrant.
            prefer_grpc (bool): `True` if want to use gRPC, `False` otherwise.
        """

        # Initializes the class properties.
        self._client:QdrantClient = QdrantClient(
            url=url,
            grpc_port=grpc_port,
            prefer_grpc=prefer_grpc,
            api_key=api_key
        )


    # ---- Methods ---- #

    async def close(self) -> None:
        """
        Close the client.
        """
        
        # Close the connection.
        await self._client.close()

    async def is_server_alive(self) -> bool:
        """
        Checks if the associated server is alive.

        Returns:
            response (bool): `True` if the server is alive, `False` otherwise.
        """
        
        return True