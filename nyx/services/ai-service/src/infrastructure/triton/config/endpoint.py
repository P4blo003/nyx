# ==========================================================================================
# Author: Pablo González García.
# Created: 09/02/2026
# Last edited: 10/02/2026
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from typing import Dict

# External:
from pydantic import BaseModel
from pydantic import Field


# ==============================
# CLASSES
# ==============================

class TritonEndpoint(BaseModel):
    """
    
    """

    # ---- Attributes ---- #

    host:str = Field(..., description="")
    port:int = Field(..., description="")