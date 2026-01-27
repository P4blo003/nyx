
# ==========================================================================================
# Author: Pablo González García.
# Created: 27/01/2026
# Last edited: 27/01/2026
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

import logging
import logging.config
import datetime
import json
from typing import Any, Dict, Optional


# ==============================
# CONSTANTS
# ==============================

# Standard text format for development/console
STANDARD_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"


# ==============================
# CLASSES
# ==============================

class JSONFormatter(logging.Formatter):
    """
    Formatter that outputs JSON strings for structured logging.
    """

    # ---- Methods ---- #

    def format(self, record: logging.LogRecord) -> str:
        """

        Args:

        Returns:
        """

        log_obj = {
            "timestamp": datetime.datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "funcName": record.funcName,
            "lineno": record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
            
        return json.dumps(log_obj)
    

# ==============================
# FUNCTIONS
# ==============================

def get_logging_config(formatter_class: str = "default") -> Dict[str, Any]:
    """
    Returns the logging configuration dictionary.

    Args:

    Returns:

    """
    
    config = {
        "version": 1,
        "disable_existing_loggers": False, # We handle cleanup manually to be safe
        "formatters": {
            "default": {
                "format": STANDARD_FORMAT, # FIX: 'format' instead of 'fmt'
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "json": {
                "()": JSONFormatter,
            }
        },
        "handlers": {
            "console": {
                "level": "INFO",
                "class": "logging.StreamHandler",
                "formatter": "default" if formatter_class != "json" else "json",
                "stream": "ext://sys.stdout"
            },
        },
        "loggers": {
            "": {  # Root logger
                "handlers": ["console"],
                "level": "INFO",
                "propagate": True
            },
            "uvicorn": {
                "handlers": [], # Delegate to root
                "level": "INFO", 
                "propagate": True
            },
            "uvicorn.access": {
                "handlers": [], # Delegate to root
                "level": "INFO",
                "propagate": True
            },
            "uvicorn.error": {
                "handlers": [], # Delegate to root
                "level": "INFO", 
                "propagate": True
            },
            "fastapi": {
                "handlers": [], # Delegate to root
                "level": "INFO", 
                "propagate": True
            },
            "application": {
                "handlers": [], # Delegate to root
                "level": "INFO",
                "propagate": True
            }
        }
    }
    return config

def setup(config: Optional[Dict[str, Any]] = None) -> None:
    """
    Configures the application logging.

    Args:

    """
    if config is None:
        config = get_logging_config()

    # Force cleanup of existing handlers on key loggers to avoid duplication
    # especially from uvicorn's default setup
    loggers_to_clean = ["uvicorn", "uvicorn.access", "uvicorn.error", "fastapi"]
    for logger_name in loggers_to_clean:
        logger = logging.getLogger(logger_name)
        logger.handlers = [] # Remove existing handlers
        logger.propagate = True

    logging.config.dictConfig(config)