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


# ==============================
# CLASSES
# ==============================

class TritonInferenceServer(BaseModel):
    """
    Represents a single Triton Inference Instance.
    """

    # ---- Attributes ---- #

    host:str = Field(..., description="Hostname or IP of the Triton Server")
    http_port:int = Field(..., description="HTTP port for REST inference")
    grpc_port:int = Field(..., description="gRPC port for inference")

class TritonInferenceServerMapping(BaseModel):
    """
    Mapping of available Triton Inference Server by identifier.
    """

    # ---- Attributes ---- #

    servers:Dict[str, TritonInferenceServer] = Field(..., description="Dictionary mapping server identifiers to Triton Servers.")