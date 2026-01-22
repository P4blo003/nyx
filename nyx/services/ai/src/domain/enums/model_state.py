# ==========================================================================================
# Author: Pablo González García.
# Created: 20/01/2026
# Last edited: 22/01/2026
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
    Represents the operational state of an inference model.

    The state reflects te model's availability and readiness for inference requests as
    reported by the inference backend.

    Attributes:
        UNAVAILABLE (str): Indicates tha the model is not available on the inference server.
        AVAILABLE (str): The model exists and is visible to the server but is not yet ready
            to handle inference requests.
        READY (str): The model is fully loaded into memory and ready to accept inference requests.
            In this state, the model is expected to respond immediately without additional
            initialization.
    """

    # ---- Attributes ---- #

    UNAVAILABLE = "UNAVAILABLE"
    AVAILABLE = "AVAILABLE"
    READY = "READY"