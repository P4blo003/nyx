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
from enum import IntEnum
from typing import Dict, Any
from dataclasses import dataclass


# ==============================
# ENUMS
# ==============================

class LogLevel(IntEnum):
    """
    Logging severity levels.
    """
    # ---- Attributes ---- #

    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


# ==============================
# CLASSES
# ==============================

@dataclass(frozen=True)
class LogRecord:
    """
    Immutable log record.

    Attributes:
        level (LogLevel): Log severity.
        message (str): Log message.
        timestamp (float): Creation timestamp (epoch seconds).
        context (Dict[str, str|None]): Contextual data captured at emit time.
        extra (Dict[str, Any]): Structured metadata.
    """
    # ---- Attributes ---- #

    level:LogLevel
    message:str
    timestamp:float
    context:Dict[str, str|None]
    extra:Dict[str, Any]