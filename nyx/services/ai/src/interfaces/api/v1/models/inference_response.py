# ==========================================================================================
# Author: Pablo González García.
# Created: 27/01/2026
# Last edited: 27/01/2026
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from typing import Any, Optional, List

# External:
from pydantic import BaseModel, Field


# ==============================
# CLASSES
# ==============================

class InferenceOutput(BaseModel):
    """
    
    """

    # ---- Attributes ---- #

    id:str = Field(..., description="Id of the input")
    embedding:List[float] = Field(..., description="")

class InferenceResponse(BaseModel):
    """
    
    """

    # ---- Attributes ---- #

    embeddings:List[InferenceOutput] = Field(..., description="")