# Author: Pablo González García.
# Created: 20/01/2026
# Last edited: 22/01/2026
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Standard:
from typing import Optional, Dict, Any

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

    This domain model provides a normalized and backend agnostic view of a
    Triton model, suitable for use across application, service, and API
    layers.

    Attributes:
        name (Optional[str]): Unique name of the model.
        version (Optional[str]): Version of the model.
        state (Optional[str]): Current lifecycle state of the model.
        metadata (Optional[Dict[str, Any]]): Arbitrary metadata associated with the model.
        config (Optional[Dict[str, Any]]): Model configuration as defined in the inference backend
    """

    # ---- Attributes ---- #

    name:Optional[str] = Field(..., description="Unique name of the model")
    version:Optional[str] = Field(..., description="Version of the model")
    state:Optional[ModelState] = Field(..., description="Current lifecycle state of the model")
    metadata:Optional[Dict[str, Any]] = Field(..., description="Arbitrary metadata associated with the model")
    config:Optional[Dict[str, Any]] = Field(..., description="Model configuration as defined in the inference backend")