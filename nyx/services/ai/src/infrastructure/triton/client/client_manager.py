# ==========================================================================================
# Author: Pablo González García.
# Created: 23/01/2026
# Last edited: 23/01/2026
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from typing import Optional, Dict

# Internal:
from infrastructure.triton.client.client_interface import ITritonClientManager, ITritonClient
from infrastructure.triton.client.grpc_client import GrpcClient
from infrastructure.triton.config import TritonConfig


# ==============================
# CLASSES
# ==============================

class TritonClientManager(ITritonClientManager):
    """
    """

    # ---- Default ---- #

    def __init__(
        self,
        triton_config:TritonConfig
    ) -> None:
        """
        Initializes the manager.

        Args:
            triton_config (TritonConfig):
        """

        # Initializes the class properties.
        self._triton_config:TritonConfig = triton_config
        self._clients:Dict[str, ITritonClient] = {}


    # ---- Methods ---- #

    async def startup(self) -> None:
        """
        """
        
        # Iterates over all configured Triton Inference Servers.
        for name, config in self._triton_config.servers.items():
            self._clients[name] = GrpcClient(
                name=name,
                host=config.host,
                port=config.grpc_port
            )

    def get_clients(self) -> Dict[str, ITritonClient]:
        """
        Retrieve all Triton clients managed by this manager.

        Returns:
            response (Dict[str, ITritonClient]): A dictionary mapping container keys to 
                their respective Triton client.
        """

        return self._clients

    def get_client(
        self,
        key: str
    ) -> Optional[ITritonClient]:
        """
        Retrieve a specific Triton client by its key.

        Args:
            key (str): The unique identifier for the Triton client.

        Returns:
            response (Optional[ITritonClient]): The Triton client instance associated with the key, or
                `None` if the key does not exists.
        """

        return self._clients.get(key, None)

    async def shutdown(self):
        """
        Gracefully shutdown all Triton clients managed by this manager.

        This method should close connections and release any resources associated
        with the Triton clients to prevent leaks.
        """

        # Close all active connections.
        for __, client in self._clients.items(): pass