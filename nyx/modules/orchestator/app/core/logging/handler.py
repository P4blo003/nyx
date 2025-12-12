# ==========================================================================================
# Author: Pablo González García.
# Created: 12/12/2025
# Last edited: 12/12/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standar:
import sys
import logging
# Internal:
from core.logging.formatter import ConsoleFormatter


# ==============================
# FUNCTIONS
# ==============================

def create_console_handler(level:int = logging.INFO) -> logging.StreamHandler:
    """
    Create a console handler with colored output.

    Args:
        level (int): Minimun log level for this handler.

    Returns:
        logging.StreamHandler: Configure console handler.
    """
    # Creates the handler.
    handler:logging.StreamHandler = logging.StreamHandler(stream=sys.stdout)
    handler.setLevel(level=level)
    handler.setFormatter(ConsoleFormatter(include_timestamp=True))
    return handler