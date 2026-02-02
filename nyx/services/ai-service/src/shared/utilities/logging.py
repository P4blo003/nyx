# ==========================================================================================
# Author: Pablo González García.
# Created: 02/02/2026
# Last edited: 02/02/2026
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
import logging
import logging.config
from typing import Any, Optional
from typing import Dict


# ==============================
# VARIABLES
# ==============================

DEFAULT_LOGGING: Dict[str, Any] = {
    "version": 1,
    "disable_existing_loggers": False,
    
    "formatters": {
        "standard": {
            "format": "%(asctime)s - %(levelname)s - %(name)s: %(message)s"
        },
        "detailed": {
            "format": "%(asctime)s - %(levelname)s - %(name)s(%(filename)s:%(lineno)d): %(message)s"
        }
    },

    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "level": "INFO",
        },
    },

    "loggers": {
        "": {  # root logger
            "handlers": ["console"],
            "level": "INFO",
            "propagate": False
        },
        "uvicorn": {
            "handlers": ["console"],
            "level": "INFO",
            "propagate": False
        },
        "uvicorn.error": {
            "handlers": ["console"],
            "level": "INFO",
            "propagate": False
        },
        "uvicorn.access": {
            "handlers": ["console"],
            "level": "INFO",
            "propagate": False
        },
    }
}


# ==============================
# functions
# ==============================

def setup_logging(config:Optional[Dict[str, Any]] = None):
    """
    Initialize Python logging using dictConfig.
    
    Args:
        config (Optional[Dict[str, Any]]): Optional dictConfig to override defaults.
    """
    cfg = config or DEFAULT_LOGGING
    logging.config.dictConfig(cfg)