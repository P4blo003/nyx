# ==========================================================================================
# Author: Pablo González García.
# Created: 23/01/2026
# Last edited: 30/01/2026
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
import logging
from typing import Optional
from typing import Dict

# Internal:
from infrastructure.triton.client.base import AsyncClient


# ==============================
# CLASSES
# ==============================

class AsyncClientService:
    """
    
    """

    # ---- Default ---- #

    def __init__(
        self,
        clients:Dict[str, AsyncClient]
    ) -> None:
        """
        Initializes the service.

        Args:
            clients (Dict[str, AsyncClient]): Dictionary containing
                the clients.
        """

        # Initializes the class properties.
        self._clients:Dict[str, AsyncClient] = clients

        self._log:logging.Logger = logging.getLogger(name="ClientService")


    # ---- Methods ---- #

    def get_client(
        self,
        key:str
    ) -> Optional[AsyncClient]:
        """
        Retrieve a specific client by its key.

        Args:
            key (str): The unique identifier.

        Returns:
            response (Optional[T]): The client associated with the key, or `None` it
                the key does not exist.
        """

        return self._clients.get(key, None)
    
    def get_clients(self) -> Dict[str, AsyncClient]:
        """
        Retrieves all clients managed by this manager.

        Returns:
            response (Dict[str, T]): A dictionary mapping containing keys to
                their respective client.
        """

        return self._clients
    
    async def start(self) -> None:
        """
        Starts all clients. This method checks for each client, if its respective
        server is alive.
        """

        # Variables for information.
        alive_servers:int = 0

        # Iterates over all servers.
        for __, client in self._clients.items():
            # Checks if the server is alive.
            if await client.is_server_alive(): alive_servers += 1

        # Prints information.
        self._log.info(f"({alive_servers}/{len(list(self._clients.values()))}) servers are alive")

    async def stop(self) -> None:
        """
        Stops all clients.
        """

        # Iterate over all clients and close them.
        for __, client in self._clients.items(): await client.close()
        # Prints information.
        self._log.info("All clients closed")