# ==========================================================================================
# Author: Pablo González García.
# Created: 17/12/2025
# Last edited: 17/12/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
import logging
from abc import ABC
from abc import abstractmethod
from typing import List
# Internal:
from core.logging.record import LogRecord


# ==============================
# CLASSES
# ==============================

class LogHandler(ABC):
    """
    Abstract ase class for log handlers.
    """
    # ---- Methods ---- #

    @abstractmethod
    def emit(self, record:LogRecord) -> None:
        """
        Emit a log record.

        Args:
            record (LogRecord): LogRecord to process.
        """
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """
        Shutdown handler and release resources.
        """
        pass

class StandardLogHandler(LogHandler):
    """
    Log handler delegating to Python's logging handler.
    """
    # ---- Default ---- #

    def __init__(
        self,
        logger_name:str
    ) -> None:
        """
        Initialize the handler.
        Args:
            logger_name (str): Name of the underlying logger.
        """
        # Initialize the properties.
        self._logger:logging.Logger = logging.getLogger(name=logger_name)

    
    # ---- Methods ---- #

    def _format_message(self, record:LogRecord) -> str:
        """
        Format log message with context and extra data.

        Args:
            record (LogRecord): LogRecord to process.
        """
        parts:List = [record.message]
        ctx = record.context

        # Checks if there are any value.
        if any(ctx.values()):
            parts.append(
                "[" + "".join(f"{k}={v}" for k,v in ctx.items() if v) + "]"
            )
        # Checks if there are extra data.
        if record.extra:
            parts.append(
                "{" + "".join(f"{k}={v}" for k,v in record.extra.items()) + "}"
            )

        return "".join(parts)
        
    def emit(self, record: LogRecord) -> None:
        """
        Emit record using standard logging.

        Args:
            record (LogRecord): LogRecord to process.
        """
        self._logger.log(level=record.level, msg=self._format_message(record=record))
    
    def shutdown(self) -> None:
        """
        Flush all handlers.
        """
        for handler in self._logger.handlers: handler.flush()