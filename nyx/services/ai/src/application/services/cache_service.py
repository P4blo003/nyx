# ==========================================================================================
# Author: Pablo González García.
# Created: 27/01/2026
# Last edited: 27/01/2026
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
import asyncio
import logging
from logging import Logger
from asyncio import Task
from typing import Optional, Dict

# Internal:
from application.cache.interfaces import ICache
from domain.models.triton.model import CachedTritonModel
from infrastructure.triton.client.interfaces import IClientManager
from infrastructure.triton.sdk import SDK as TritonSdk


# ==============================
# CLASSES
# ==============================

class CacheService:
    """
    Background service responsible for periodically synchronizing Triton model
    metadata into an application-level cache.

    This service runs as asynchronous loop.

    The lifecycle of the service is explicitly controlled through `start()` and `stop()`
    methods, allowing safe integration with application startup and shutdown hooks.
    """

    # ---- Default ---- #

    def __init__(
        self,
        cache:ICache[CachedTritonModel],
        client_manager:IClientManager,
        interval:int = 10_000    
    ) -> None:
        """
        Initializes the updated instance.

        Args:
            cache (ICache[CachedTritonModel]): Cache implementation used to store Triton
                model information.
            client_manager (IClientManager): Manager responsible for providing initialized
                Triton clients.
            interval (int): Refresh interval in seconds between cache updates.
        """

        # Initializes the class properties.
        self._cache:ICache = cache
        self._client_manager:IClientManager = client_manager
        self._interval:int = interval

        self._task:Optional[Task] = None
        self._logger:Logger = logging.getLogger(__name__)


    # ---- Methods ---- #

    async def _run(self):
        """
        Internal loop that periodically fetches models from Triton and updates the cache.
        """
        
        # Prints information.
        self._logger.info(f"Cache updater started: refreshing Triton mode cache every {self._interval} seconds.")

        # Main loop.
        while True:
            
            try:
                # Get clients from Triton context.
                clients = self._client_manager.get_clients().values()

                # Query all models concurrently.
                tasks = [TritonSdk().get_models(client=client) for client in clients]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Flatten results and convert to CachedTritonModel.
                cached:Dict[str, CachedTritonModel] = {}
                for result in results:
                    if isinstance(result, Exception):
                        raise result
                    if isinstance(result, dict):
                        for server, models in result.items():
                            for model in models:
                                cached[model.name] = CachedTritonModel(
                                    server=server,
                                    model=model
                                )

                # Update cache.
                await self._cache.update(values=cached)
                # Prints information.
                self._logger.info("Cache updated")
            
            # If task.Cancel() is called.
            except asyncio.CancelledError: break

            # If an error occurs.
            except Exception as ex:

                # Prints error.
                self._logger.error(f"Unable to update cache. {ex}")

            await asyncio.sleep(self._interval)

    def start(self):
        """
        Starts the cache synchronization background task.
        
        If the task is already running, this method has no effect. This method
        is safe to call multiple times.
        """

        if self._task is None or self._task.done(): self._task = asyncio.create_task(self._run())

    async def stop(self):
        """
        Stop the cache synchronization background task gracefully.
        """

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
                
        # Prints information.
        self._logger.info("Cache updater stopped.")

    def get_cache(self) -> ICache:
        """
        Retrieve the underlying cache instance.

        Returns:
            response (ICache): The cache storing the latest synchronized Triton model information.
        """
        
        return self._cache