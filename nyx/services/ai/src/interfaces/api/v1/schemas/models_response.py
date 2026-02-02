# ==========================================================================================
# Author: Pablo González García.
# Created: 27/01/2026
# Last edited: 30/01/2026
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from typing import Optional

# External:
from pydantic import BaseModel, Field


# ==============================
# CLASSES
# ==============================

class ModelSummary(BaseModel):
    """
    
    """

    # ---- Attributes ---- #

    name:str = Field(..., description="")
    server:str = Field(..., description="")
    version:Optional[str] = Field(..., description="")