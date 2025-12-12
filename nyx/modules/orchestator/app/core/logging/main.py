# ==========================================================================================
# Author: Pablo González García.
# Created: 12/12/2025
# Last edited: 12/12/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standar:
import logging
# Internal:
from core.logging.handler import create_console_handler


# ==============================
# FUNCTIONS
# ==============================

def get_logger(name:str, level:int = logging.DEBUG) -> logging.Logger:
    """
    Creates a logger.
    """
    # Initializes the logger.
    logger:logging.Logger = logging.getLogger(name=name)
    logger.handlers.clear()

    # Add the stream handler.
    logger.addHandler(create_console_handler(level=level))

    return logger