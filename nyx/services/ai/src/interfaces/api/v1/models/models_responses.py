# ==========================================================================================
# Author: Pablo González García.
# Created: 23/01/2026
# Last edited: 23/01/2026
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# External:
from pydantic import BaseModel, Field

# Internal:
from domain.models.model import TritonModel


# ==============================
# CLASSES
# ==============================

class ModelSummary(BaseModel):
    """
    
    """

    # ---- Attributes ---- #

    name:str = Field(..., description="")
    version:str = Field(..., description="")
    server:str = Field(..., description="")