# ==========================================================================================
# Author: Pablo González García.
# Created: 22/01/2026
# Last edited: 22/01/2026
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from typing import Union, List

# External:
from pydantic import BaseModel, Field


# ==============================
# CLASSES
# ==============================

class InferenceResult(BaseModel):
    """
    
    """

    # ---- Attributes ---- #

    id:str = Field(..., description="")
    embedding:List[float] = Field(..., description="")


class InferenceResponse(BaseModel):
    """
    
    """

    # ---- Attributes ---- #

    results:List[InferenceResult] = Field(..., description="")