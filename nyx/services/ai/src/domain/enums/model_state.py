# ==========================================================================================
# Author: Pablo González García.
# Created: 20/01/2025
# Last edited: 20/01/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from enum import StrEnum


# ==============================
# ENUMS
# ==============================

class ModelState(StrEnum):
    """
    Enumeration representing lifecycle and availability states of a Triton
    model.

    Attributes:
        UNKNOWN (str): The model state can not be determined.
        UNAVAILABLE (str): The model is known but not available for inference.
        AVAILABLE (str): The model is available on the server but not yet
            ready for inference.
        READY (str): The model is fully loaded an ready to serve inference request.
    """

    # ---- Attributes ---- #

    UNKNOWN = "UNKNOWN"
    UNAVAILABLE = "UNAVAILABLE"
    AVAILABLE = "AVAILABLE"
    READY = "READY"