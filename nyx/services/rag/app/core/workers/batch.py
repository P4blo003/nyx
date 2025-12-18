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
from typing import Callable, List
from contextlib import suppress
# Internal:
from core.interfaces.worker import IBackgroundWorker
from core.logging.facade import Log
from dto.models.vector_store import VectorDocument


# ==============================
# CLASSES   
# ==============================

class BatchWorker(IBackgroundWorker):
    """
    Consumes documents from a queue and flushes them in batches.
    """
    # ---- Default ---- #
    
    def __init__(
        self,
        queue:asyncio.Queue,
        notify_event:asyncio.Event,
        batch_size:int,
        callback:Callable
    ) -> None:
        """
        Initialize the worker.

        Args:
            queue (asyncio.Queue): The queue to consume documents from.
            notify_event (asyncio.Event): The event to notify when to flush.
            batch_size (int): The maximum number of documents per batch.
            callback (Callable): The callback function to process the batch.
        """
        # Initialize the properties.
        self._queue:asyncio.Queue = queue
        self._batch_size:int = batch_size
        self._notify_event:asyncio.Event = notify_event
        self._callback:Callable = callback

        self._is_running:bool = False
        self._task:asyncio.Task|None = None

    
    # ---- Methods ---- #

    async def _run(self) -> None:
        """
        Background worker that processes documents in batches.
        Flushes when either batch_size is reached or max_delay expires.
        """
        # Main loop.
        # Main loop.
        while self._is_running:
            # Try-Except to manage errors.
            try:

                # Awaits for flush event.
                await self._notify_event.wait()

                # Checks if the queue is empty.
                if self._queue.empty(): continue

                # Variable to hold the batch.
                batch:List[VectorDocument] = []
                # Process the queue.
                while len(batch) < self._batch_size and not self._queue.empty():
                    batch.append(self._queue.get_nowait())
                    self._queue.task_done()

                # Insert the batch.
                if batch:
                    await self._callback(batch)

                # Reset the flush event.
                self._notify_event.clear()

            # Checks if the task is cancelled.
            except asyncio.CancelledError:
                # Ends loop.
                break

            # If an unexpected error occurred.
            except Exception as ex:
                # Prints information.
                await Log.error("Unable to flush queue", error=ex)

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
