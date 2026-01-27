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

class InferenceInput(BaseModel):
    """
    
    """

    # ---- Attributes ---- #

    id:str = Field(..., description="Id of the input")
    content:Any = Field(..., description="Content of the input")

class InferenceRequest(BaseModel):
    """
    
    """

    # ---- Attributes ---- #

    inputs:List[InferenceInput] = Field(..., description="")