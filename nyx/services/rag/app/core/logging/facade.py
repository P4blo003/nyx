# ==========================================================================================
# Author: Pablo González García.
# Created: 17/12/2025
# Last edited: 17/12/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
import time
from queue import Queue, Full
from typing import List, Dict, Any
# Internal:
from core.logging import context as ctx
from core.logging.record import LogLevel, LogRecord
from core.logging.handler import LogHandler
from core.logging.worker import LogWorker


# ==============================
# CLASSES
# ==============================

class Log:
    """
    Global asynchronous logging facade.

    Usage:
        await Log.info("Message", key=value).
    """
    # ---- Attributes ---- #

    _queue:Queue|None = None
    _workers:List[LogWorker] = []
    _initialized:bool = False
    _dropped:int = 0


    # ---- Methods ---- #

    @classmethod
    async def _emit(
        cls,
        level:LogLevel,
        message:str,
        extra:Dict[str, Any]
    ) -> None:
        """
        Emit a log record. Drops records if queue is full (except CRITICAL).

        Args:
            level (LogLevel): Log severity.
            message (str): Log message.
            extra (Dict[str, Any]): Structured metadata.
        """
        # Checks if it's initialized and there is a queue.
        if not cls._initialized or cls._queue is None: return

        # Generates the record.
        record:LogRecord = LogRecord(
            level=level,
            message=message,
            timestamp=time.time(),
            context=ctx._get_log_context(),
            extra=extra
        )

        # Try-Except to manage errors.
        try:
            # Puts the record into the queue.
            cls._queue.put_nowait(record)

        # If the queue is full.
        except Full:
            # Checks if the level is CRITICAL.
            if level < LogLevel.ERROR:
                # Increment number of dropped records.
                cls._dropped += 1
                return
            
            # Add record if it's CRITICAL.
            cls._queue.put(record)

    @classmethod
    def init(
        cls,
        *,
        handlers:List[LogHandler],
        queue_size:int = 10_000,
        num_workers:int = 2
    ) -> None:
        """
        Initialize logging system. Must be called once at application
        startup.

        Args:
            handlers (List[LogHandler]): Log handlers.
            queue_size (int): Maximum queue size.
            num_workers (int): Number of workers threads.
        """
        # Checks if it's already initialized.
        if cls._initialized:
            raise RuntimeError("Logging system already initialized.")
        
        # Initializes the properties.
        cls._queue = Queue(maxsize=queue_size)
        cls._workers = [LogWorker(cls._queue, handlers=handlers, worker_id=i) for i in range(num_workers)]

        # Starts workers.
        for worker in cls._workers: worker.start()

        cls._initialized = True

    @classmethod
    def shutdown(cls) -> None:
        """
        Shutdown logging system gracefully.
        """
        # Stops workers.
        for worker in cls._workers:
            worker.stop()
        # Waits for workers end.
        for worker in cls._workers:
            worker.join(timeout=5)

        cls._initialized = False
    
    @classmethod
    def stats(cls) -> Dict[str, int]:
        """
        Retrieve logging statistics.

        Returns:
            Dict[str, int]: Dictionary with dropped log count.
        """
        return {"dropped": cls._dropped}

    @classmethod
    async def debug(cls, message:str, **extra:Any) -> None:
        await cls._emit(level=LogLevel.DEBUG, message=message, extra=extra)

    @classmethod
    async def info(cls, message:str, **extra:Any) -> None:
        await cls._emit(level=LogLevel.INFO, message=message, extra=extra)

    @classmethod
    async def warning(cls, message:str, **extra:Any) -> None:
        await cls._emit(level=LogLevel.WARNING, message=message, extra=extra)

    @classmethod
    async def error(cls, message:str, **extra:Any) -> None:
        await cls._emit(level=LogLevel.ERROR, message=message, extra=extra)

    @classmethod
    async def critical(cls, message:str, **extra:Any) -> None:
        await cls._emit(level=LogLevel.CRITICAL, message=message, extra=extra)