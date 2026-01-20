# ==========================================================================================
# Author: Pablo González García.
# Created: 20/01/2025
# Last edited: 20/01/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from typing import List, Union

# External:
from pydantic import BaseModel, Field

# Internal:
from domain.inference.inference_task import InferenceTask


# ==============================
# CLASSES
# ==============================

class InferenceRequest(BaseModel):
    """
    Request schema for performing an inference task via the AI service.
    """

    # ---- Attributes ---- #

    task:InferenceTask = Field(..., description="The inference task to perform")
    inputs:List[Union[str, bytes]] = Field(..., description="List of inputs items for the model.")