# ==========================================================================================
# Author: Pablo González García.
# Created: 27/01/2026
# Last edited: 27/01/2026
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from typing import Any, Optional, Dict

# External:
from pydantic import BaseModel, Field

# Internal:
from domain.enums.model_state import ModelState


# ==============================
# CLASSES
# ==============================

class TritonModel(BaseModel):
    """
    Represents a model deployed or managed by Triton Inference Server.

    This domain model provides a normalized and backend agnostic view of
    a Triton model.

    Attributes:
        name (str): Unique name of the model as registered in Triton Inference Server.
        version (Optional[str]): Model version.
        state (Optional[ModelState]): Currently lifecycle state of the model.
        reason (Optional[str]): Optional human-readable explanation.
        metadata (Optional[Dict[str, Any]]): Additional metadata associated with the model.
        config (Optional[Dict[str, Any]]): Model configuration as defined in Triton.
    """

    # ---- Attributes ---- #

    name:str = Field(..., description="Unique name of the model as registered in Triton Inference Server")
    version:Optional[str] = Field(..., description="Model version")
    state:Optional[ModelState] = Field(..., description="Currently lifecycle state of the model")
    reason:Optional[str] = Field(..., description="Optional human-readable explanation")
    metadata:Optional[Dict[str, Any]] = Field(..., description="Additional metadata associated with the model")
    config:Optional[Dict[str, Any]] = Field(..., description="Model configuration as defined in Triton")

class CachedTritonModel(BaseModel):
    """
    Represents a Triton model stored in the internal cache.

    Attributes:
        server (str): Server where the model is deployed.
        model (TritonModel): Model information.
    """

    # ---- Attributes ---- #

    server:str = Field(..., description="Server where the model is deployed")
    model:TritonModel = Field(..., description="Model information")