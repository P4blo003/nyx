
# ==========================================================================================
# Author: Pablo González García.
# Created: 27/01/2026
# Last edited: 29/01/2026
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
import json
import datetime
import logging
import logging.config
from typing import Any, Dict, Optional


# ==============================
# CONSTANTS
# ==============================

STANDARD_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
"""
Default human-readable log format for console/development usage.
"""


# ==============================
# CLASSES
# ==============================

class JSONFormatter(logging.Formatter):
    """
    Custom logging formatter that outputs log records as JSON strings.

    This formatter is intended for structured logging scenarios, where machine-readable
    logs are preferred over plain text.
    """

    # ---- Methods ---- #

    def format(self, record: logging.LogRecord) -> str:
        """
        Format a LogRecord into a JSON-formatted string.

        The output includes standard log metadata such as timestamp, level, logger name,
        source location, and message. Exception information is included when present.

        Args:
            record (logging.LogRecord): The log record to be formatted.

        Returns:
            response (str): A JSON string representing the structured log entry.
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
    Build and return a logging configuration dictionary.

    The configuration is compatible with `logging.config.dictConfig` and supports
    both human-readable and JSON-based log formatting.

    Args:
        formatter_class (str): Formatter type to use for console output.
    
    Returns:
        response (Dict[str, Any]): A complete logging configuration dictionary.
    """
    
    config = {
        "version": 1,
        # Preserve existing loggers; explicit cleanup is handled manually.
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                # Text-based formatter for local development and debugging.
                "format": STANDARD_FORMAT,
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "json": {
                # Custom formatter for structured logging.
                "()": JSONFormatter,
            }
        },
        "handlers": {
            "console": {
                "level": "INFO",
                "class": "logging.StreamHandler",
                # Select formatter based on requested formatter type.
                "formatter": "default" if formatter_class != "json" else "json",
                "stream": "ext://sys.stdout"
            },
        },
        "loggers": {
            "": {  # Root logger configuration.
                "handlers": ["console"],
                "level": "INFO",
                "propagate": True
            },
            # Framework loggers delegate handling to the root logger.
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
    Configure application wide logging.

    This function applies a logging configuration using `logging.config.dictConfig`
    and proactively removes handlers from common framework loggers to avoid duplicated
    log output, specially when running under Uvicorn or similar servers.

    Args:
        config (Optional[Dict[str, Any]]): A logging configuration dictionary. If not provided,
            the default configuration from `get_logging_config` is used.
    """

    # Load default configuration if none is provided.
    if config is None:
        config = get_logging_config()

    # Force cleanup of existing handlers on key loggers to avoid duplication
    # especially from uvicorn's default setup
    loggers_to_clean = ["uvicorn", "uvicorn.access", "uvicorn.error", "fastapi"]
    for logger_name in loggers_to_clean:
        logger = logging.getLogger(logger_name)
        logger.handlers = [] # Remove existing handlers
        logger.propagate = True

    # Apply the logging configuration.
    logging.config.dictConfig(config)