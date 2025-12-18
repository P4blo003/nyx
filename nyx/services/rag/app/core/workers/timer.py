# ==========================================================================================
# Author: Pablo González García.
# Created: 18/12/2025
# Last edited: 18/12/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
import asyncio
from contextlib import suppress
# Internal:
from core.interfaces.worker import IBackgroundWorker


# ==============================
# CLASSES   
# ==============================

class Timer(IBackgroundWorker):
    """
    
    """
    # ---- Default ---- #

    def __init__(
        self,
        notify_event:asyncio.Event,
        delay:float
    ) -> None:
        """
        Initialize the timer worker.
        
        
        """
        # Initialize the properties.
        self._notify_event:asyncio.Event = notify_event
        self._delay:float = delay

        self._is_running:bool = False
        self._task:asyncio.Task|None = None

    
    # ---- Methods ---- #

    async def _run(self) -> None:
        """
        
        """
        # Main loop.
        while self._is_running:
            # Waits for the delay.
            await asyncio.sleep(self._delay)
            # Notify the event.
            self._notify_event.set()
        
    async def start(self) -> None:
        """
        
        """
        # Sets it's running.
        self._is_running = True
        # Starts the timer task.
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        """
        
        """
        # Checks if it's not running.
        if not self._is_running: return

        # Stops the loop.
        self._is_running = False

        # Cancel the task.
        if self._task:
            self._task.cancel()
            # Awaits for the task to finish.
            with suppress(asyncio.CancelledError):
                await self._task