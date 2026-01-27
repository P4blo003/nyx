# ==========================================================================================
# Author: Pablo González García.
# Created: 23/01/2026
# Last edited: 27/01/2026
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
import logging
from logging import Logger
from typing import Optional, Dict

# Internal:
from infrastructure.triton.client.interfaces import IClientManager, IAsyncClient


# ==============================
# CLASSES
# ==============================

class TritonAsyncClientManager(IClientManager[IAsyncClient]):
    """
    
    """

    # ---- Default ---- #

    def __init__(
        self,
        clients:Dict[str, IAsyncClient]
    ) -> None:
        """
        Initializes the manager instance.

        Args:
            config (TritonConfig): Configuration of Triton.
            client_cls (Type[T]): Instance type of the manager clients.
        """

        # Initializes the class properties
        self._clients:Dict[str, IAsyncClient] = clients

        self._log:Logger = logging.getLogger(__name__)


    # ---- Methods ---- #

    async def start(self) -> None:
        """
        Starts all clients. This method checks for each client, if its respective
        server is alive.
        """
        
        # Variables for information.
        servers_alive:int = 0

        # Iterates over all configured servers.
        for __, client in self._clients.items():
            # Checks if the server is alive and increments the value.
            if await client.is_server_alive(): servers_alive += 1

        # Prints information.
        self._log.info(f"TritonClientManager: Manager started with ({servers_alive}/{len(list(self._clients.values()))}) servers alive.")

    async def stop(self) -> None:
        """
        Stops all clients.
        """
        
        # Iterate over all clients and close them.
        for __, client in self._clients.items(): await client.close()
        # Prints information.
        self._log.info("TritonClientManager: Manager stopped. All clients closed.")
    
    def get_clients(self) -> Dict[str, IAsyncClient]:
        """
        Retrieves all clients managed by this manager.

        Returns:
            response (Dict[str, T]): A dictionary mapping containing keys to
                their respective client.
        """

        # Retrieve all registered clients.
        return self._clients
    
    def get_client(
        self,
        key: str
    ) -> Optional[IAsyncClient]:
        """
        Retrieve a specific client by its key.

        Args:
            key (str): The unique identifier.

        Returns:
            response (Optional[T]): The client associated with the key, or `None` it
                the key does not exist.
        """
        
        # Retrieve client assigned to given key.
        return self._clients.get(key, None)