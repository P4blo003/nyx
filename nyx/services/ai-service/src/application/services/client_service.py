# ==========================================================================================
# Author: Pablo González García.
# Created: 03/02/2026
# Last edited: 03/02/2026
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
import time
import logging
from typing import Dict
from typing import TypeVar, Generic
from typing import Optional

# Internal:
from domain.ports.client import ISyncClientService, IAsyncClientService
from domain.ports.client import ISyncClient, IAsyncClient


# ==============================
# TYPES
# ==============================

SyncClient = TypeVar("SyncClient", bound=ISyncClient)
AsyncClient = TypeVar("AsyncClient", bound=IAsyncClient)


# ==============================
# CLASSES
# ==============================

class SyncClientService(ISyncClientService, Generic[SyncClient]):
    """
    
    """

    # ---- Default ---- #

    def __init__(
        self,
        clients:Dict[str, SyncClient]
    ) -> None:
        """
        Initializes the service instance.
        """

        # Initializes the class properties.
        self._clients:Dict[str, SyncClient] = clients

        self._log:logging.Logger = logging.getLogger("SyncClientService")


    # ---- Methods ---- #

    def get_client(self, key: str) -> Optional[SyncClient]:
        """
        
        """

        return self._clients.get(key, None)
    
    def startup(self) -> None:
        """
        
        """

        start_time:float = time.perf_counter()

        # TODO: Startup.

        duration:float = time.perf_counter() - start_time

        self._log.info(f"Startup complete in {duration:.2f}s")

    def shutdown(self) -> None:
        """
        
        """

        start_time:float = time.perf_counter()

        # TODO: Shutdown.

        duration:float = time.perf_counter() - start_time

        self._log.info(f"Shutdown complete in {duration:.2f}s")


class AsyncClientService(IAsyncClientService, Generic[AsyncClient]):
    """
    
    """

    # ---- Default ---- #

    def __init__(
        self,
        clients:Dict[str, AsyncClient]
    ) -> None:
        """
        Initializes the service instance.
        """

        # Initializes the class properties.
        self._clients:Dict[str, AsyncClient] = clients

        self._log:logging.Logger = logging.getLogger("AsyncClientService")


    # ---- Methods ---- #

    def get_client(self, key: str) -> Optional[AsyncClient]:
        """
        
        """

        return self._clients.get(key, None)

    async def startup(self) -> None:
        """
        
        """

        start_time:float = time.perf_counter()

        for server, client in self._clients.items():
            self._log.debug(f"Connecting to {server} at {client.get_server_url()} ...")
            await client.connect()
            self._log.info(f"Connected to {server} at {client.get_server_url()}.")

        duration:float = time.perf_counter() - start_time

        self._log.info(f"Startup complete in {duration:.2f}s")

    async def shutdown(self) -> None:
        """
        
        """

        start_time:float = time.perf_counter()

        for server, client in self._clients.items():
            await client.disconnect()
            self._log.info(f"Disconnected from {server} at {client.get_server_url()}.")

        duration:float = time.perf_counter() - start_time

        self._log.info(f"Shutdown complete in {duration:.2f}s")