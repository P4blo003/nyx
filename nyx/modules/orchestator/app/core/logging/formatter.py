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
from typing import Dict


# ==============================
# CLASSES
# ==============================

class  ConsoleFormatter(logging.Formatter):
    """
    Custom formatter forr console output with colors.

    Provides colored output based on log level for better readability.
    """
    # ---- Attributes ---- #

    COLORS:Dict = {
        logging.DEBUG: "\033[36m",      # Cyan
        logging.INFO: "\033[32m",       # Green
        logging.WARNING: "\033[33m",    # Yellow
        logging.ERROR: "\033[31m",      # Red
        logging.CRITICAL: "\033[41m",   # Red background
    }
    RESET = "\033[0m"


    # ---- Default ---- #

    def __init__(self, include_timestamp:bool = True) -> None:
        """
        Initializes the console formatter.
        
        Args:
            include_timestamp (bool): Whether to include timestamp in
                the output.
        """
        # Define format patterns.
        fmt:str = "%(levelname)-8s | %(name)s | %(message)s"
        # Checks if should include timestamp.
        if include_timestamp:
            fmt = "%(asctime)s | " + fmt

        # Formmater constructor.
        super().__init__(fmt, datefmt="%d/%m/%Y %H:%M:%S")
    

    # ---- Methods ---- #

    def format(
        self,
        record: logging.LogRecord
    ) -> str:
        """
        Format the log record with colors.

        Args:
            record (logging.Record): The log record to format.
        
        Returns:
            str: Formatted and colored log message.
        """
        # Get color for this level.
        color = self.COLORS.get(record.levelno, self.RESET)
        # Format the message.
        formatted:str = super().format(record)
        # Apply color.
        return f"{color}{formatted}{self.RESET}"