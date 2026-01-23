# ==========================================================================================
# Author: Pablo González García.
# Created: 23/01/2026
# Last edited: 23/01/2026
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

class TritonServerConfig(BaseModel):
    """
    Represents the configuration of a single Triton Inference Server.

    Attributes:
        host (str): Hostname or IP address of the Triton Inference Server.
        http_port (int): Port number for HTTP endpoint of the Triton Inference Server.
        grpc_port (int): Port number for gRPC endpoint of the Triton Inference Server.
    """

    # ---- Attributes ---- #

    host:str = Field(..., description="Hostname or IP address of the Triton Inference Server")
    http_port:int = Field(..., description="Port number for HTTP endpoint of the Triton Inference Server")
    grpc_port:int = Field(..., description="Port number for gRPC endpoint of the Triton Inference Server")

class TritonConfig(BaseModel):
    """
    Stores the configuration for all Triton Inference Servers.

    Attributes:
        servers (List[TritonServerConfig]): List of TritonServerConfig defining
            each Triton Inference Server.
    """

    # ---- Attributes ---- #

    servers:Dict[str, TritonServerConfig] = Field(..., description="List of TritonServerConfig defining each Triton Inference Server")