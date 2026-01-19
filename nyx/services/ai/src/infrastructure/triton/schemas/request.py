# ==========================================================================================
# Author: Pablo González García.
# Created: 19/01/2025
# Last edited: 19/01/2026
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from typing import List
# External:
from pydantic import BaseModel, Field


# ==============================
# CLASSES
# ==============================

class InferenceRequestInput(BaseModel):
    """
    Represents a single piece of information to process.

    Attributes:
        content (str): Content to process.
    """

    # ---- Attributes ---- #

    content:str = Field(..., description="Content to process")

class InferenceRequest(BaseModel):
    """
    Request schema for inference.

    Attributes:
        inputs (List[InferenceRequestInput]): List of items to be processed.
    """

    # ---- Attributes ---- #

    inputs:List[InferenceRequestInput] = Field(..., description="List of items to be processed")