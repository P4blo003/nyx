# ==========================================================================================
# Author: Pablo González García.
# Created: 20/01/2025
# Last edited: 20/01/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from typing import Dict

# External:
from pydantic import BaseModel, Field

# Internal:
from domain.inference.inference_task import InferenceTask


# ==============================
# CLASSES
# ==============================

class TaskModel(BaseModel):
    """
    Represents a mode capable of handling a specific inference task.
    """

    # ---- Attributes ---- #

    model:str = Field(..., description="Model name registered in Triton")
    server:str = Field(..., description="Inference server responsible for the model")

class TaskModelMapping(BaseModel):
    """
    Central registry mapping inference tasks to their available models.
    """

    # ---- Attributes ---- #

    tasks:Dict[InferenceTask, TaskModel] = Field(..., description="Mapping of inference tasks to supported model entries")