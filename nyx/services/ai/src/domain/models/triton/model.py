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


# ==============================
# CLASSES
# ==============================

class TritonModel(BaseModel):
    """
    Represents a model deployed or managed by Triton Inference Server.

    This domain model provides a normalized and backend agnostic view of
    a Triton model.

    Attributes:
        name (str):
        version (Optional[str]):
        state (Optional[str]):
        reason (Optional[str])
        metadata (Optional[Dict[str, Any]]):
        config (Optional[Dict[str, Any]]):
    """

    # ---- Attributes ---- #

    name:str = Field(..., description="")
    version:Optional[str] = Field(..., description="")
    state:Optional[str] = Field(..., description="")
    reason:Optional[str] = Field(..., description="")
    metadata:Optional[Dict[str, Any]] = Field(..., description="")
    config:Optional[Dict[str, Any]] = Field(..., description="")

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