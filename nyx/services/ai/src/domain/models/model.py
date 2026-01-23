# ==========================================================================================
# Author: Pablo González García.
# Created: 23/01/2026
# Last edited: 23/01/2026
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

    This domain model provides a normalized and backend agnostic view of a
    Triton model, suitable for use across application, service, and API layers.

    Attributes:
        name (str): Unique name of the model.
        version (str): Version of the model.

    """

    # ---- Attributes ---- #

    name:str = Field(..., description="")
    version:str = Field(..., description="")

    metadata:Optional[Dict[str, Any]] = Field(..., description="")
    config:Optional[Dict[str, Any]] = Field(..., description="")

class CachedTritonModel(TritonModel):
    """
    Represents a Triton model stored in the internal cache.

    Inherits all attributes from `TritonModel` but adds internal metadata
    used for cache management, such as the server where the model resides.
    This class should never be returned directly to API clients.

    Attributes:
        server (str): Server where the model is deployed.
    """

    # ---- Attributes ---- #

    server:str = Field(..., description="Server where the model is deployed")