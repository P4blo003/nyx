# ==========================================================================================
# Author: Pablo González García.
# Created: 16/02/2026
# Last edited: 16/02/2026
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
import time
import logging
import asyncio
from typing import Any, Optional
from typing import Callable
from concurrent.futures import ProcessPoolExecutor

# Internal:
from infrastructure.config.base import RunningConfig


# ==============================
# CLASSES
# ==============================

class WorkerService:
    """
    
    """

    # ---- Default ---- #

    def __init__(self, running_config:RunningConfig) -> None:
        """
        
        """

        # Initializes the class properties.
        self._running_config:RunningConfig = running_config
        self._executor:Optional[ProcessPoolExecutor] = None

        self._log:logging.Logger = logging.getLogger("WorkerService")
        
    
    # ---- Methods ---- #

    async def startup(self):
        """
        
        """

        start:float = time.perf_counter()

        self._executor = ProcessPoolExecutor(max_workers=self._running_config.workers.max_workers)

        duration:float = time.perf_counter() - start

        self._log.info(f"Startup complete in {duration:.2f}s with ({self._running_config.workers.max_workers}) max workers.")

    async def submit_task(self, func:Callable[..., Any], *args:Any, **kwargs:Any) -> Any:
        """
        """

        if self._executor is None:
            raise RuntimeError(f"WorkerManager is not started. Call startup() first.")
        
        loop:asyncio.AbstractEventLoop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, func=func, *args, **kwargs)

    async def shutdown(self):
        """
        """

        start:float = time.perf_counter()

        if self._executor:
            self._executor.shutdown(wait=True)

        duration:float = time.perf_counter() - start

        self._log.info(f"Shutdown complete in {duration:.2f}s")