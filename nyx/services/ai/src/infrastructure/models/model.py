# ==========================================================================================
# Author: Pablo González García.
# Created: 27/01/2026
# Last edited: 27/01/2026
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# External:
from pydantic import BaseModel, Field

# Internal:
from domain.models.model import TritonModel


# ==============================
# CLASSES
# ==============================


class CachedTritonModel(BaseModel):
    """
    Represents a Triton model stored in the internal cache.

    Attributes:
        server (str): Server where the model is deployed.
        model (TritonModel): Model information.
    """

    # ---- Attributes ---- #

    server:str = Field(..., description="Server where the model is deployed")
    model:TritonModel = Field(..., description="")