# ==========================================================================================
# Author: Pablo González García.
# Created: 20/01/2025
# Last edited: 20/01/2025
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# External:
from pydantic import BaseModel, Field

# Internal:
from domain.inference.inference_task import InferenceTask


# ==============================
# CLASSES
# ==============================

class LoadModelRequest(BaseModel):
    """
    Request schema for loading a model into the AI service.
    """

    # ---- Attributes ---- #

    task:InferenceTask = Field(..., description="Inference task for which the model should be loaded")

class UnloadModelRequest(BaseModel):
    """
    Request schema for unloading a model into the AI service.
    """

    # ---- Attributes ---- #

    task:InferenceTask = Field(..., description="Inference task for which the model should be unloaded")