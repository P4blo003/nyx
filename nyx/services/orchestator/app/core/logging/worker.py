# ==========================================================================================
# Author: Pablo González García.
# Created: 16/12/2025
# Last edited: 16/12/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
import threading
import logging
from queue import Queue, Full, Empty
from typing import List
# Internal:
from core.logging.record import LogRecord
from core.logging.handler import LogHandler


# ==============================
# CLASSES
# ==============================

class LogWorker(threading.Thread):
    """
    Background worker consuming log records form a thread-safe queue.
    """
    # ---- Default ---- #

    def __init__(
        self,
        queue:Queue[LogRecord],
        handlers:List[LogHandler],
        worker_id:int
    ) -> None:
        """
        Initialize the worker.

        Args:
            queue (Queue[LogRecord]): Thread-safe queue.
            handlers (List[LogHandler]): Log handlers.
            worker_id (int): Worker identifier.
        """
        # Thread constructor.
        super().__init__(daemon=True, name=f"LogWorker-{worker_id}")
        
        # Initializes the properties.
        self._queue:Queue = queue
        self._handlers:List[LogHandler] = handlers
        self._stop_event:threading.Event = threading.Event()

    
    # ---- Methods ---- #

    def run(self) -> None:
        """
        Main worker loop.
        """
        # Main loop.
        while not self._stop_event.is_set():
            # Try-Except to manage errors.
            try:
                record:LogRecord = self._queue.get()
            
            # If there is not values.
            except Empty: continue

            # Try-Except to manage errors.
            try:
                # Emit log record.
                for handler in self._handlers:
                    handler.emit(record=record)
            
            # If an unexpected error occurred.
            except Exception:
                # Prints information.
                logging.exception("Error while emitting log record", exc_info=True)
    
    def stop(self) -> None:
        """
        Signal worker to stop.
        """
        self._stop_event.set()