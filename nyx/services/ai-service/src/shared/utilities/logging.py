# ==========================================================================================
# Author: Pablo González García.
# Created: 02/02/2026
# Last edited: 03/02/2026
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
import os
import sys
import logging
import logging.config
from pathlib import Path
from typing import Any, Optional
from typing import Dict


# ==============================
# VARIABLES
# ==============================

# ---- Environment variables ---- #

LOG_LEVEL:str = os.getenv("LOG_LEVEL", "DEBUG").upper()
LOG_DIR:str = os.getenv("LOG_DIR", "./logs")
APP_NAME:str = os.getenv("APP_NAME", "ai-service")


# ==============================
# FUNCTIONS
# ==============================

def get_config(
    log_level:str = LOG_LEVEL,
    log_dir:str = LOG_DIR,
    app_name:str = APP_NAME
) -> Dict[str, Any]:
    """
    Generates a logging configuration dictionary following SOLID principles.

    Returns:
        response (Dict[str, Any]): A dictionary compatible with ``logging.config.dicConfig``.
    """

    return {
        "version": 1,
        "disable_existing_loggers": False,

        "formatters": {
            "standard": {
                "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
            "detailed": {
                "format": (
                    "%(asctime)s [%(levelname)s] [%(process)d:%(threadName)s] "
                    "%(name)s (%(filename)s:%(lineno)d): %(message)s"
                ),
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
            "json_like": {
                "format": (
                    '{"ts": "%(asctime)s", "lvl": "%(levelname)s", '
                    '"logger": "%(name)s", "msg": "%(message)s"}'
                ),
                "datefmt": "%Y-%m-%dT%H:%M:%S%z"
            }
        },

        "handlers": {
            # 1. Console: Targeted at developers and containers orchestrators (Docker/K8s).
            "console": {
                "level": log_level,
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "stream": sys.stdout
            },
            # 2. Main file: Primary audit trail with automatic size-based rotation.
            "file_main": {
                "level": "INFO",
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": "detailed",
                "filename": os.path.join(log_dir, f"{app_name}.log"),
                "maxBytes": 10485760,          # 10MB
                "backupCount": 10,              # Retain up to 100 MB of history.
                "encoding": "utf-8"
            },
            # 3. Error: Isolated high-priority logs por rapid incident response.
            "file_error": {
                "level": "ERROR",
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": "detailed",
                "filename": os.path.join(log_dir, f"{app_name}_error.log"),
                "maxBytes": 5242880,          # 5MB
                "backupCount": 5,              # Retain up to 25 MB of history.
                "encoding": "utf-8"
            }
        },

        "loggers": {
            # Root logger: Handlers everything not explicitly defined.
            "": {
                "handlers": ["console", "file_main", "file_error"],
                "level": log_level,
                "propagate": True
            },
            # Explicit logger for the core application logic.
            "app": {
                "handlers": ["console", "file_main"],
                "level": "DEBUG",
                "propagate": False
            },
            # Suppress noisy third-party libraries.
            "urllib3": {
                "level": "WARNING",
                "handlers": ["console"],
                "propagate": False
            }
        }
    }

def setup(config:Optional[Dict[str, Any]] = None) -> None:
    """
    Initializes the logging configuration using ``logging.config.dictConfig``.

    Args:
        config (Optional[Dict[str, Any]]): Optional configuration to override
            defaults.
    """

    # If no config is given, use default configuration.
    if config is None: config = get_config()

    # Extract filename from handlers to ensure directories to exist.
    for __, handler_config in config.get("handlers", {}).items():
        if "filename" in handler_config:
            log_file = Path(handler_config["filename"])
            # .parent gets the directory, exist_ok=True prevents errors if
            # it exist.
            log_file.parent.mkdir(parents=True, exist_ok=True)

    logging.config.dictConfig(config=config)