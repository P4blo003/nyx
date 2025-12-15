# ==========================================================================================
# Author: Pablo González García.
# Created: 15/12/2025
# Last edited: 15/12/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from logging import Logger
from pathlib import Path
from typing import Any
# Internal:
from core.config import logger


# ==============================
# CLASSES
# ==============================

class SessionLogger:
    """
    
    """
    # ---- Default ---- #

    def __init__(
        self,
        logger:Logger
    ) -> None:
        """
        Initializes logger properties.
        """
        # Initialize properties.
        self._logger:Logger = logger


    # ---- Methods ---- #

    async def debug(self, msg:str) -> None:
        """
        Prints debug message.

        Args:
            msg (str): Message to show.
        """
        # Prints debug information.
        self._logger.debug(msg=msg)
    
    async def info(self, msg:str) -> None:
        """
        Prints info message.

        Args:
            msg (str): Message to show.
        """
        # Prints debug information.
        self._logger.info(msg=msg)
    
    async def error(self, msg:str) -> None:
        """
        Prints error message.

        Args:
            msg (str): Message to show.
        """
        # Prints debug information.
        self._logger.error(msg=msg)