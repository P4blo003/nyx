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
from infrastructure.triton.config.connection import TritonConnection
from infrastructure.triton.config.task import TritonTask


# ==============================
# CLASSES
# ==============================

class TritonConfig(BaseModel):
    """
    
    """

    # ---- Attributes ---- #

    connections:Dict[str, TritonConnection] = Field(..., description="")
    tasks:Dict[str, TritonTask] = Field(..., description="")