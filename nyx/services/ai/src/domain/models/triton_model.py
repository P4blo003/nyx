# Author: Pablo González García.
# Created: 20/01/2025
# Last edited: 20/01/2025
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
    """

    # ---- Attributes ---- #

    name:Optional[str] = Field(..., description="")
    version:Optional[str] = Field(..., description="")
    state:Optional[ModelState] = Field(..., description="")
    metadata:Optional[Dict[str, Any]] = Field(..., description="")
    config:Optional[Dict[str, Any]] = Field(..., description="")