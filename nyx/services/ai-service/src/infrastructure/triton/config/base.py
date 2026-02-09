# ==========================================================================================
# Author: Pablo González García.
# Created: 09/02/2026
# Last edited: 09/02/2026
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from typing import Dict

# External:
from pydantic import BaseModel
from pydantic import Field

# Internal:
from infrastructure.triton.config.endpoint import TritonEndpoint


# ==============================
# CLASSES
# ==============================

class TritonConfig(BaseModel):
    """
    
    """

    # ---- Attributes ---- #

    endpoints:Dict[str, TritonEndpoint] = Field(..., description="")