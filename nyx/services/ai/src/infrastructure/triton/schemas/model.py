# ==========================================================================================
# Author: Pablo González García.
# Created: 31/12/2025
# Last edited: 15/01/2026
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Internal:
from typing import Optional, List

# External:
from pydantic import BaseModel
from pydantic import Field


# ==============================
# CLASSES
# ==============================

class ModelIO(BaseModel):
    """
    Represents an input or output tensor of a Triton model.
    """

    # ---- Attributes ---- #

    name:str = Field(..., description="Name of the input or output tensor")
    datatype:str = Field(..., description="Data type of the tensor")
    shape:List[str] = Field(default_factory=list, description="Shape of tensor")

class ModelMetadata(BaseModel):
    """
    Represents the metadata of a model in Triton.
    """

    # ---- Attributes ---- #

    platform:str = Field(..., description="Backend platform")
    versions:Optional[List[str]] = Field(default=None, description="Versions of the model")
    inputs:Optional[List[ModelIO]] = Field(default=None, description="List of input tensors")
    outputs:Optional[List[ModelIO]] = Field(default=None, description="List of output tensors")


class TritonModel(BaseModel):
    """
    Represents a model in Triton.
    """

    # ---- Attributes ---- #

    server:str = Field(..., description="Backend server")
    name:str = Field(..., description="Name of the model")
    version:Optional[str] = Field(default=None, description="Version of the model")
    state:Optional[str] = Field(default=None, description="State of the model")
    metadata:Optional[ModelMetadata] = Field(default=None, description="Model metadata")